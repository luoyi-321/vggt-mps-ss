#!/usr/bin/env python3
"""
Adaptive Covisibility: Per-Frame Attention Budget Allocation

Two novel mechanisms beyond uniform k-nearest:

1. Adaptive-k  (novel contribution A)
   Allocates sparse attention budget proportionally to each frame's
   "isolation score" — frames with few close visual neighbors receive
   more connections; densely overlapping frames receive fewer.

   Fixed total budget: Σ_i k_i = S × k_base
   Allocation: k_i ∝ isolation_i = 1 - max_{j≠i} sim(i,j)

2. Pose-guided covisibility  (novel contribution B)
   After an initial VGGT forward pass (or using GT poses if available),
   replaces visual-similarity-based covisibility with geometric frustum
   overlap derived from camera extrinsics.

   Geometric overlap O(i,j) = f(θ_ij, d_ij)
   where θ_ij = viewing angle between cameras i and j
         d_ij = baseline distance

   This creates a closed-loop: VGGT's geometry prediction guides its
   own attention graph → attention-geometry consistency.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ───────────────────────────────────────────────────────────────
# 1. Adaptive-k allocation
# ───────────────────────────────────────────────────────────────

def compute_adaptive_k(
    similarity_matrix: torch.Tensor,
    k_base: int,
    k_min: int = 1,
    k_max: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute per-frame adaptive k values given a similarity matrix.

    Frames that are visually isolated (low max similarity to any other frame)
    receive a higher k so they can gather information from the widest set of
    neighbors. Densely overlapping frames need fewer connections.

    Budget constraint: Σ_i k_i ≈ S × k_base  (same total as uniform k)

    Args:
        similarity_matrix: [S, S] pairwise cosine similarities (diag = 1.0)
        k_base: Uniform k baseline (target average)
        k_min: Minimum k per frame (connectivity guarantee)
        k_max: Maximum k per frame (None = S-1)

    Returns:
        k_per_frame: [S] integer tensor of per-frame k values
    """
    S = similarity_matrix.shape[0]
    if k_max is None:
        k_max = S - 1

    # Off-diagonal maximum similarity (most similar other frame)
    sim = similarity_matrix.clone()
    sim.fill_diagonal_(-1.0)
    max_sim, _ = sim.max(dim=1)           # [S] — high = densely overlapping

    # Isolation score: 1 - max_sim, high = isolated
    isolation = 1.0 - max_sim.clamp(0.0, 1.0)  # [S]

    # Normalise isolation to get fractional budget allocation
    iso_mean = isolation.mean().clamp(min=1e-6)
    budget_fraction = isolation / iso_mean      # [S], mean ≈ 1.0

    # Scale to k_base average, then round and clamp
    k_float = (budget_fraction * k_base).clamp(k_min, k_max)
    k_int = k_float.round().long()

    # Correct rounding so total budget ≈ S × k_base
    target_total = S * k_base
    current_total = k_int.sum().item()
    diff = target_total - current_total

    if diff > 0:
        # Add to most isolated frames first
        order = isolation.argsort(descending=True)
        for i in range(int(diff)):
            idx = order[i % S]
            if k_int[idx] < k_max:
                k_int[idx] += 1
    elif diff < 0:
        # Remove from most dense frames first
        order = isolation.argsort(descending=False)
        for i in range(int(-diff)):
            idx = order[i % S]
            if k_int[idx] > k_min:
                k_int[idx] -= 1

    return k_int  # [S]


def build_adaptive_covisibility_mask(
    features: torch.Tensor,
    k_base: int,
    k_min: int = 1,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a covisibility mask using per-frame adaptive k values.

    Args:
        features: [S, D] normalised feature vectors
        k_base: Target average k (same total budget as uniform-k)
        k_min: Minimum neighbors per frame
        threshold: Optional minimum similarity threshold (secondary filter)

    Returns:
        mask: [S, S] binary covisibility matrix
        k_per_frame: [S] per-frame k allocation
    """
    S = features.shape[0]
    sim = torch.mm(features, features.t())  # [S, S]

    k_per_frame = compute_adaptive_k(sim, k_base=k_base, k_min=k_min)

    mask = torch.zeros(S, S, device=features.device)
    mask.fill_diagonal_(1.0)   # always self-attend

    for i in range(S):
        ki = int(k_per_frame[i].item())
        sim_i = sim[i].clone()
        sim_i[i] = -1.0   # exclude self
        _, top_idx = sim_i.topk(min(ki, S - 1))
        mask[i, top_idx] = 1.0
        mask[top_idx, i] = 1.0   # symmetric

    return mask, k_per_frame


# ───────────────────────────────────────────────────────────────
# 2. Pose-guided covisibility
# ───────────────────────────────────────────────────────────────

def viewing_angle_overlap(
    extrinsics: torch.Tensor,
    angle_threshold_deg: float = 60.0,
) -> torch.Tensor:
    """
    Compute pairwise covisibility from camera extrinsics (viewing angles).

    Two cameras covisibly overlap if their optical-axis angle is below
    `angle_threshold_deg`.  This is a fast, geometry-principled substitute
    for visual similarity when camera poses are known.

    Args:
        extrinsics: [S, 4, 4] camera-to-world matrices  (VGGT convention)
                    OR [S, 3, 4].  Column 2 of R is the optical axis (+Z).
        angle_threshold_deg: Covisibility angle threshold in degrees.
                             60° matches ~90% scene overlap for typical cameras.

    Returns:
        overlap: [S, S] float tensor in [0, 1]:
                 1.0 = likely covisible, 0.0 = non-covisible
    """
    S = extrinsics.shape[0]

    # Extract optical axes (third column of rotation sub-matrix)
    # extrinsics[i] = [R | t], so R = extrinsics[i, :3, :3]
    R = extrinsics[:, :3, :3]   # [S, 3, 3]
    z_axes = R[:, :, 2]         # [S, 3] — optical axis in world frame
    z_axes = F.normalize(z_axes, dim=-1)

    # Pairwise dot product → cosine of angle between optical axes
    cos_angle = torch.mm(z_axes, z_axes.t())   # [S, S]

    # Convert threshold to cosine
    cos_thresh = np.cos(np.radians(angle_threshold_deg))

    # Soft overlap: sigmoid around threshold for smooth gradients
    # overlap(i,j) = σ((cos_angle - cos_thresh) × 10)
    overlap = torch.sigmoid((cos_angle - cos_thresh) * 10.0)

    # Hard version: threshold at 0.5
    return (overlap > 0.5).float()


def baseline_distance_weight(
    extrinsics: torch.Tensor,
    max_baseline: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute pairwise baseline-distance weight between cameras.

    Nearby cameras share more scene content than far-apart ones.
    Returns a [S, S] weight in [0, 1] where 1 = same position.

    Args:
        extrinsics: [S, 4, 4] or [S, 3, 4] camera-to-world matrices
        max_baseline: Normalisation distance (auto-computed as scene diameter if None)

    Returns:
        weight: [S, S] closeness weight
    """
    # Camera positions = last column of extrinsics
    positions = extrinsics[:, :3, 3]   # [S, 3]

    # Pairwise L2 distances
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)   # [S, S, 3]
    dists = diff.norm(dim=-1)                                 # [S, S]

    if max_baseline is None:
        max_baseline = float(dists.max().item()) + 1e-6

    weight = 1.0 - (dists / max_baseline).clamp(0.0, 1.0)
    return weight


def pose_guided_covisibility(
    extrinsics: torch.Tensor,
    k: int,
    angle_threshold_deg: float = 60.0,
    use_distance: bool = True,
    distance_weight: float = 0.3,
) -> torch.Tensor:
    """
    Build covisibility mask from predicted camera poses.

    Combines two geometric signals:
    - Viewing angle overlap: cameras pointing in similar directions
    - Baseline distance: nearby cameras share more scene

    This is the POSE-GUIDED COVISIBILITY contribution:
    VGGT predicts camera poses → poses define attention graph → attention
    uses geometry-grounded connectivity instead of visual heuristics.

    Args:
        extrinsics: [S, 4, 4] camera-to-world matrices (VGGT output)
        k: Number of neighbors per frame
        angle_threshold_deg: Max viewing angle for covisibility
        use_distance: Blend in baseline-distance signal
        distance_weight: Weight of distance vs. angle signal [0, 1]

    Returns:
        mask: [S, S] binary covisibility matrix
    """
    S = extrinsics.shape[0]

    # Angle-based overlap [S, S]
    angle_score = viewing_angle_overlap(extrinsics, angle_threshold_deg)

    if use_distance:
        dist_score = baseline_distance_weight(extrinsics)
        score = (1 - distance_weight) * angle_score + distance_weight * dist_score
    else:
        score = angle_score

    # K-nearest per frame (ensure minimum connectivity)
    mask = torch.zeros(S, S, device=extrinsics.device)
    mask.fill_diagonal_(1.0)

    score_off = score.clone()
    score_off.fill_diagonal_(-1.0)

    _, top_idx = score_off.topk(min(k, S - 1), dim=1)   # [S, k]
    for i in range(S):
        mask[i, top_idx[i]] = 1.0
        mask[top_idx[i], i] = 1.0   # symmetric

    return mask


# ───────────────────────────────────────────────────────────────
# 3. Two-stage iterative refinement
# ───────────────────────────────────────────────────────────────

def two_stage_covisibility(
    model,
    images: torch.Tensor,
    k: int,
    angle_threshold_deg: float = 60.0,
) -> torch.Tensor:
    """
    Two-stage pose-guided covisibility:
      Stage 1: Run VGGT with dense attention → get predicted camera extrinsics
      Stage 2: Compute geometric covisibility from predicted poses
              → use as attention mask for full inference

    This creates a self-consistency loop:
      VGGT geometry output → attention graph → VGGT geometry output (refined)

    Args:
        model: VGGT model (with or without sparse attention)
        images: [1, S, 3, H, W] input images
        k: Number of neighbors per frame for pose-guided mask
        angle_threshold_deg: Covisibility angle threshold

    Returns:
        mask: [1, S, S] pose-guided covisibility mask
    """
    S = images.shape[1]

    # Stage 1: dense forward to get camera poses
    with torch.no_grad():
        # Temporarily disable sparse attention if active
        aggregator = getattr(model, 'aggregator', None)
        original_mask = None
        if aggregator is not None and hasattr(aggregator, 'attention_mask'):
            original_mask = aggregator.attention_mask
            aggregator.attention_mask = None  # force dense

        preds = model(images)

        if aggregator is not None and original_mask is not None:
            aggregator.attention_mask = original_mask  # restore

    # Extract predicted extrinsics
    # VGGT returns extrinsics as [1, S, 3, 4] or [1, S, 4, 4]
    extrinsics = preds.get('extrinsic', preds.get('extrinsics', None))
    if extrinsics is None:
        raise ValueError(
            "VGGT output has no 'extrinsic'/'extrinsics' key. "
            "Cannot use pose-guided covisibility."
        )

    # Shape: [1, S, 3, 4] → [S, 4, 4] (pad last row)
    ext = extrinsics[0]   # [S, 3, 4]
    if ext.shape[-2] == 3:
        pad = torch.zeros(S, 1, 4, device=ext.device, dtype=ext.dtype)
        pad[:, 0, 3] = 1.0
        ext = torch.cat([ext, pad], dim=1)   # [S, 4, 4]

    # Stage 2: compute geometric covisibility
    mask = pose_guided_covisibility(
        ext, k=k, angle_threshold_deg=angle_threshold_deg
    )
    return mask.unsqueeze(0)   # [1, S, S]


# ───────────────────────────────────────────────────────────────
# Diagnostic utility
# ───────────────────────────────────────────────────────────────

def compare_covisibility_methods(
    features: torch.Tensor,
    extrinsics: Optional[torch.Tensor],
    k: int,
) -> dict:
    """
    Compare visual-similarity vs. pose-guided covisibility masks.

    Returns statistics to quantify how much they agree and differ.

    Args:
        features: [S, D] visual feature vectors
        extrinsics: [S, 4, 4] camera poses (None skips pose comparison)
        k: Neighbors per frame

    Returns:
        dict with agreement statistics
    """
    S = features.shape[0]

    # Visual similarity mask
    sim = torch.mm(features, features.t())
    sim_mask, k_adaptive = build_adaptive_covisibility_mask(features, k_base=k)

    result = {
        'S': S,
        'k': k,
        'visual_mask_density': sim_mask.mean().item(),
        'adaptive_k_mean': k_adaptive.float().mean().item(),
        'adaptive_k_std': k_adaptive.float().std().item(),
        'adaptive_k_min': k_adaptive.min().item(),
        'adaptive_k_max': k_adaptive.max().item(),
    }

    if extrinsics is not None:
        pose_mask = pose_guided_covisibility(extrinsics, k=k)
        agreement = (sim_mask == pose_mask).float().mean().item()
        result['pose_mask_density'] = pose_mask.mean().item()
        result['visual_pose_agreement'] = agreement
        result['visual_only_edges'] = ((sim_mask == 1) & (pose_mask == 0)).sum().item()
        result['pose_only_edges'] = ((sim_mask == 0) & (pose_mask == 1)).sum().item()

    return result
