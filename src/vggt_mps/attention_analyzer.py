#!/usr/bin/env python3
"""
Attention Layer Importance Analyzer for VGGT

Records per-layer attention weight entropy during a forward pass to determine
which global_block layers carry the most cross-view information.

Key insight (to verify against our own data, cf. Faster VGGT RWTH 2509.07120):
  - Low entropy layers → attention concentrated on few frames → safe to sparsify
  - High entropy layers → attention spread across all frames → need dense attention

This module provides our own empirical evidence for layer-selective sparsity,
independent of RWTH's findings.

Usage:
    analyzer = AttentionEntropyAnalyzer(aggregator)
    analyzer.attach_hooks()
    _ = model(images)
    report = analyzer.get_report()
    analyzer.detach_hooks()
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LayerEntropyStats:
    """Entropy statistics for one global_block layer."""
    layer_idx: int
    # Per-head cross-view entropy (higher = more spread across frames)
    mean_entropy: float = 0.0
    std_entropy: float = 0.0
    # Fraction of attention weight going to non-self frames
    cross_frame_ratio: float = 0.0
    # Effective number of attended frames per query token (exp of entropy)
    effective_frames: float = 0.0
    # Number of samples collected
    n_samples: int = 0


class AttentionEntropyAnalyzer:
    """
    Hooks into VGGT's global_blocks to record attention weight entropy.

    Attaches a forward hook to each global_block's Attention module that
    intercepts the Q, K, V after projection and computes the attention
    weights in frame-aggregated form.

    The key metric is *cross-frame entropy*:
        H(i) = -Σ_j p(i→j) log p(i→j)
    where p(i→j) is the fraction of attention from token-group i to frame j.
    High H → attention is spread → layer needs dense; Low H → safe to sparsify.
    """

    def __init__(self, aggregator, num_heads: int = 16):
        self.aggregator = aggregator
        self.num_heads = num_heads
        self._hooks: List = []
        self._stats: Dict[int, List[torch.Tensor]] = {}  # layer_idx → list of entropy tensors

    def attach_hooks(self):
        """Register forward hooks on every global_block's attention module."""
        self._hooks.clear()
        self._stats.clear()

        if not hasattr(self.aggregator, 'global_blocks'):
            print("Warning: aggregator has no global_blocks — nothing to hook")
            return

        for idx, block in enumerate(self.aggregator.global_blocks):
            if not hasattr(block, 'attn'):
                continue
            self._stats[idx] = []
            hook = block.attn.register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)

        print(f"AttentionEntropyAnalyzer: hooked {len(self._hooks)} global_blocks")

    def detach_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_hook(self, layer_idx: int):
        """Create a forward hook closure for global_block at layer_idx."""
        def hook(module, inputs, output):
            # inputs[0] is the input tensor x: [B, S*P, C]
            x = inputs[0]
            B, N, C = x.shape

            # We need S to aggregate by frame. Infer from stored attention_mask
            # if available; otherwise skip.
            S = getattr(module, '_analyzer_S', None)
            if S is None or N % S != 0:
                return  # Can't infer frame structure

            P = N // S  # patches per frame

            with torch.no_grad():
                # Recompute attention weights (lightweight — no grad, float32)
                x_f = x.float()
                qkv = module.qkv(x_f).reshape(B, N, 3, module.num_heads, module.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)       # [3, B, H, N, D]
                q, k, _ = qkv.unbind(0)                  # [B, H, N, D]
                q = module.q_norm(q)
                k = module.k_norm(k)

                # Attention scores: [B, H, N, N]
                scale = module.head_dim ** -0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale

                # Aggregate over patches → frame-level: [B, H, S, S]
                # Average Q-patches and K-patches within each frame
                attn = attn.reshape(B, module.num_heads, S, P, S, P)
                attn = attn.mean(dim=3).mean(dim=4)  # [B, H, S, S]

                # Softmax over key-frames
                attn_w = F.softmax(attn, dim=-1)  # [B, H, S, S]

                # Cross-frame entropy: treat self-attention (diagonal) separately
                # Zero out diagonal to get pure cross-frame distribution
                mask_off_diag = 1 - torch.eye(S, device=attn_w.device)
                cross_w = attn_w * mask_off_diag.unsqueeze(0).unsqueeze(0)
                cross_w_sum = cross_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                cross_w_norm = cross_w / cross_w_sum   # re-normalize cross-frame

                # Entropy of cross-frame distribution: [B, H, S]
                eps = 1e-9
                entropy = -(cross_w_norm * (cross_w_norm + eps).log()).sum(dim=-1)

                # Cross-frame attention ratio (how much attends to other frames)
                diag_w = attn_w.diagonal(dim1=-2, dim2=-1)  # [B, H, S]
                cross_ratio = 1.0 - diag_w.mean().item()

                # Effective number of frames = exp(entropy) / (S-1) normalized
                effective = entropy.exp().mean().item()

                self._stats[layer_idx].append({
                    'entropy': entropy.mean().item(),
                    'entropy_std': entropy.std().item(),
                    'cross_ratio': cross_ratio,
                    'effective_frames': effective,
                })

        return hook

    def set_S(self, S: int):
        """
        Tell the hooks how many frames S the current batch has.
        Must be called before each forward pass when S changes.
        """
        if not hasattr(self.aggregator, 'global_blocks'):
            return
        for block in self.aggregator.global_blocks:
            if hasattr(block, 'attn'):
                block.attn._analyzer_S = S

    def get_report(self) -> List[LayerEntropyStats]:
        """
        Aggregate collected stats into per-layer entropy report.

        Returns list sorted by layer index. Call after forward pass(es).
        """
        report = []
        for idx in sorted(self._stats.keys()):
            samples = self._stats[idx]
            if not samples:
                continue
            mean_e = sum(s['entropy'] for s in samples) / len(samples)
            std_e = sum(s['entropy_std'] for s in samples) / len(samples)
            cross = sum(s['cross_ratio'] for s in samples) / len(samples)
            eff = sum(s['effective_frames'] for s in samples) / len(samples)
            report.append(LayerEntropyStats(
                layer_idx=idx,
                mean_entropy=mean_e,
                std_entropy=std_e,
                cross_frame_ratio=cross,
                effective_frames=eff,
                n_samples=len(samples),
            ))
        return report

    def print_report(self, report: Optional[List[LayerEntropyStats]] = None):
        """Pretty-print the layer entropy report."""
        if report is None:
            report = self.get_report()
        if not report:
            print("No data collected — run a forward pass first.")
            return

        n_layers = len(report)
        entropies = [r.mean_entropy for r in report]
        max_e = max(entropies) if entropies else 1.0

        print("\n" + "=" * 65)
        print("VGGT Global Block — Attention Entropy per Layer")
        print("(higher entropy = more cross-frame attention = harder to sparsify)")
        print("=" * 65)
        print(f"{'Layer':>5}  {'Entropy':>8}  {'Cross%':>7}  {'EffFrames':>9}  Bar")
        print("-" * 65)
        for r in report:
            bar_len = int(30 * r.mean_entropy / max_e)
            bar = "█" * bar_len
            print(f"  {r.layer_idx:3d}  {r.mean_entropy:8.3f}  "
                  f"{r.cross_frame_ratio*100:6.1f}%  "
                  f"{r.effective_frames:9.2f}  {bar}")

        # Suggest sparse layers (entropy < median)
        median_e = sorted(entropies)[len(entropies) // 2]
        dense_layers = [r.layer_idx for r in report if r.mean_entropy >= median_e]
        sparse_layers = [r.layer_idx for r in report if r.mean_entropy < median_e]
        print("-" * 65)
        print(f"Suggested DENSE layers  (entropy >= median): {dense_layers}")
        print(f"Suggested SPARSE layers (entropy <  median): {sparse_layers}")
        print("=" * 65)

    def recommend_sparse_layers(
        self,
        report: Optional[List[LayerEntropyStats]] = None,
        percentile: float = 0.4,
    ) -> List[int]:
        """
        Return global_block indices that are safe to sparsify.

        Selects the bottom `percentile` fraction by entropy.
        Lower entropy → less cross-frame attention → safer to skip.

        Args:
            report: Pre-computed report (computed if None)
            percentile: Fraction of layers to mark sparse (default 0.4 = 40%)

        Returns:
            List of layer indices recommended for sparse attention
        """
        if report is None:
            report = self.get_report()
        if not report:
            return []

        sorted_layers = sorted(report, key=lambda r: r.mean_entropy)
        n_sparse = max(1, int(len(sorted_layers) * percentile))
        return [r.layer_idx for r in sorted_layers[:n_sparse]]
