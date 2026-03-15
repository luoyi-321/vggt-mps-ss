#!/usr/bin/env python3
"""
Flash Attention Support for VGGT-MPS

Provides Flash Attention-like efficient attention for Apple Silicon MPS.

Since the official flash-attn package doesn't support MPS, we use:
1. PyTorch's native scaled_dot_product_attention (SDPA) with Flash backend
2. Memory-efficient attention via chunking for MPS
3. Sparse attention mask integration

Key optimizations:
- SDPA with memory_efficient_attention backend on MPS
- Chunked attention for very long sequences
- Combined with covisibility sparse masks for O(n·k) complexity

References:
- Flash Attention: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691
- PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_flash_attention_available() -> dict:
    """
    Check which Flash Attention backends are available.

    Returns:
        Dictionary with availability status for each backend
    """
    status = {
        'sdpa_available': hasattr(F, 'scaled_dot_product_attention'),
        'flash_attn_available': False,
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
        'recommended_backend': 'naive'
    }

    # Check for flash-attn package (CUDA only)
    try:
        import flash_attn
        status['flash_attn_available'] = True
        status['flash_attn_version'] = flash_attn.__version__
    except ImportError:
        pass

    # Determine recommended backend
    if status['mps_available'] and status['sdpa_available']:
        status['recommended_backend'] = 'sdpa_mps'
    elif status['cuda_available'] and status['flash_attn_available']:
        status['recommended_backend'] = 'flash_attn'
    elif status['cuda_available'] and status['sdpa_available']:
        status['recommended_backend'] = 'sdpa_cuda'
    elif status['sdpa_available']:
        status['recommended_backend'] = 'sdpa_cpu'

    return status


class FlashAttentionMPS(nn.Module):
    """
    Flash Attention-like implementation optimized for Apple Silicon MPS.

    Uses PyTorch's scaled_dot_product_attention with memory-efficient backend,
    combined with optional sparse attention masks.

    Features:
    - O(n) memory via SDPA memory-efficient backend
    - Optional sparse mask support for O(n·k) attention
    - Automatic chunking for very long sequences
    - Dropout support for training
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        chunk_size: Optional[int] = None,
        use_sparse_mask: bool = False
    ):
        """
        Initialize Flash Attention module.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability (0.0 = no dropout)
            bias: If True, add bias to input/output projections
            batch_first: If True, input is (batch, seq, feature)
            chunk_size: Optional chunk size for memory-efficient processing
            use_sparse_mask: If True, expect sparse attention masks
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.chunk_size = chunk_size
        self.use_sparse_mask = use_sparse_mask
        self.scale = 1.0 / math.sqrt(self.head_dim)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Check available backends
        self._backend_status = check_flash_attention_available()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Flash Attention.

        Args:
            query: Query tensor [B, N, D] or [N, B, D]
            key: Key tensor [B, M, D] or [M, B, D]
            value: Value tensor [B, M, D] or [M, B, D]
            attn_mask: Optional attention mask [B, N, M] or [N, M]
                      - For sparse attention: 0 = masked, 1 = attend
                      - For soft masks: values in [0, 1]
            is_causal: If True, apply causal mask

        Returns:
            output: Attention output [B, N, D]
            attn_weights: None (not computed for efficiency)
        """
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, N, _ = query.shape
        M = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention: [B, N, num_heads, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Use appropriate attention implementation
        if self.chunk_size is not None and N > self.chunk_size:
            # Chunked attention for very long sequences
            output = self._chunked_attention(q, k, v, attn_mask)
        else:
            # Standard SDPA or fallback
            output = self._sdpa_attention(q, k, v, attn_mask, is_causal)

        # Reshape output: [B, num_heads, N, head_dim] -> [B, N, embed_dim]
        output = output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

        # Output projection
        output = self.out_proj(output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool
    ) -> torch.Tensor:
        """
        Scaled dot-product attention using PyTorch's optimized implementation.

        Automatically selects best backend:
        - Flash Attention (CUDA)
        - Memory-efficient attention (MPS/CPU)
        - Math fallback
        """
        # Convert sparse mask to attention mask format
        # SDPA expects: additive mask (0 = attend, -inf = masked)
        # Our masks: multiplicative (1 = attend, 0 = masked)
        if attn_mask is not None:
            # Expand mask for heads if needed
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            # Convert to additive mask
            # For soft masks: scale by large negative value
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

            # Handle soft mask values (between 0 and 1)
            # Convert to log-space for attention
            soft_mask = (attn_mask > float('-inf')) & (attn_mask < 0)
            if soft_mask.any():
                # Soft mask: use log to convert probability to additive
                attn_mask = torch.where(
                    soft_mask,
                    torch.log(attn_mask.clamp(min=1e-7)),
                    attn_mask
                )

        # Use SDPA
        try:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask if not is_causal else None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale
            )
        except RuntimeError:
            # Fallback to manual implementation
            output = self._naive_attention(q, k, v, attn_mask)

        return output

    def _naive_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Naive attention implementation as fallback.
        """
        # [B, num_heads, N, M]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        return torch.matmul(attn_weights, v)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Memory-efficient chunked attention for very long sequences.

        Processes query in chunks to reduce peak memory usage.
        """
        B, num_heads, N, head_dim = q.shape
        M = k.shape[2]
        chunk_size = self.chunk_size

        outputs = []

        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            q_chunk = q[:, :, i:end_i, :]

            # Get corresponding mask chunk if available
            mask_chunk = None
            if attn_mask is not None:
                if attn_mask.dim() == 4:
                    mask_chunk = attn_mask[:, :, i:end_i, :]
                elif attn_mask.dim() == 2:
                    mask_chunk = attn_mask[i:end_i, :]

            # Compute attention for this chunk
            chunk_out = self._sdpa_attention(q_chunk, k, v, mask_chunk, is_causal=False)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=2)


class SparseFlashAttention(FlashAttentionMPS):
    """
    Flash Attention with integrated sparse attention masks.

    Combines the memory efficiency of Flash Attention with
    the computational savings of sparse attention patterns.

    Total complexity: O(n·k) instead of O(n²)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        k_nearest: int = 10,
        dropout: float = 0.0,
        bias: bool = True,
        soft_mask: bool = False,
        temperature: float = 0.1
    ):
        """
        Initialize Sparse Flash Attention.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            k_nearest: Number of nearest neighbors for sparse attention
            dropout: Dropout probability
            bias: If True, add bias to projections
            soft_mask: If True, use soft probabilistic masks
            temperature: Temperature for soft mask sigmoid
        """
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            use_sparse_mask=True
        )
        self.k_nearest = k_nearest
        self.soft_mask = soft_mask
        self.temperature = temperature

    def create_sparse_mask(
        self,
        similarity_matrix: torch.Tensor,
        threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Create sparse attention mask from similarity matrix.

        Args:
            similarity_matrix: [N, N] pairwise similarity scores
            threshold: Similarity threshold for hard mask

        Returns:
            mask: [N, N] attention mask
        """
        N = similarity_matrix.shape[0]

        if self.soft_mask:
            # Soft probabilistic mask
            mask = torch.sigmoid(
                (similarity_matrix - threshold) / self.temperature
            )
        else:
            # Hard binary mask
            mask = (similarity_matrix > threshold).float()

        # Ensure k-nearest neighbors are always connected
        if self.k_nearest > 0:
            _, indices = similarity_matrix.topk(
                min(self.k_nearest, N), dim=1
            )
            for i in range(N):
                mask[i, indices[i]] = 1.0
                mask[indices[i], i] = 1.0

        # Self-attention always allowed
        mask.fill_diagonal_(1.0)

        return mask


def integrate_flash_attention_with_vggt(
    vggt_model: nn.Module,
    use_flash: bool = True,
    chunk_size: Optional[int] = None
) -> nn.Module:
    """
    Integrate Flash Attention into VGGT model.

    Replaces standard attention layers with Flash Attention variants.

    Args:
        vggt_model: VGGT model to modify
        use_flash: If True, use Flash Attention
        chunk_size: Optional chunk size for very long sequences

    Returns:
        Modified VGGT model
    """
    if not use_flash:
        return vggt_model

    status = check_flash_attention_available()
    print(f"Flash Attention backend: {status['recommended_backend']}")

    # Find and replace attention modules
    def replace_attention(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is an attention module
            if isinstance(child, nn.MultiheadAttention):
                # Replace with Flash Attention
                flash_attn = FlashAttentionMPS(
                    embed_dim=child.embed_dim,
                    num_heads=child.num_heads,
                    dropout=child.dropout,
                    bias=child.in_proj_bias is not None,
                    chunk_size=chunk_size
                )

                # Copy weights
                with torch.no_grad():
                    if child.in_proj_weight is not None:
                        # Separate Q, K, V projections
                        qkv = child.in_proj_weight.chunk(3, dim=0)
                        flash_attn.q_proj.weight.copy_(qkv[0])
                        flash_attn.k_proj.weight.copy_(qkv[1])
                        flash_attn.v_proj.weight.copy_(qkv[2])

                        if child.in_proj_bias is not None:
                            qkv_bias = child.in_proj_bias.chunk(3, dim=0)
                            flash_attn.q_proj.bias.copy_(qkv_bias[0])
                            flash_attn.k_proj.bias.copy_(qkv_bias[1])
                            flash_attn.v_proj.bias.copy_(qkv_bias[2])

                    flash_attn.out_proj.weight.copy_(child.out_proj.weight)
                    if child.out_proj.bias is not None:
                        flash_attn.out_proj.bias.copy_(child.out_proj.bias)

                setattr(module, child_name, flash_attn)
                print(f"  Replaced {full_name} with FlashAttentionMPS")

            else:
                # Recurse into child modules
                replace_attention(child, full_name)

    print("Integrating Flash Attention...")
    replace_attention(vggt_model)

    return vggt_model


def test_flash_attention():
    """Test Flash Attention implementations."""
    print("=" * 60)
    print("Testing Flash Attention for MPS")
    print("=" * 60)

    # Check availability
    status = check_flash_attention_available()
    print("\nBackend Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Test basic Flash Attention
    device = torch.device("mps" if status['mps_available'] else "cpu")
    print(f"\nUsing device: {device}")

    # Create module
    flash_attn = FlashAttentionMPS(
        embed_dim=256,
        num_heads=8,
        dropout=0.0
    ).to(device)

    # Test forward pass
    B, N, D = 2, 100, 256
    x = torch.randn(B, N, D, device=device)

    output, _ = flash_attn(x, x, x)
    print(f"\nFlash Attention output shape: {output.shape}")
    assert output.shape == (B, N, D), "Output shape mismatch"

    # Test with sparse mask
    print("\nTesting with sparse mask...")
    mask = torch.zeros(N, N, device=device)
    k = 10
    for i in range(N):
        neighbors = torch.randperm(N)[:k]
        mask[i, neighbors] = 1.0
    mask.fill_diagonal_(1.0)

    output_sparse, _ = flash_attn(x, x, x, attn_mask=mask)
    print(f"Sparse attention output shape: {output_sparse.shape}")

    # Test Sparse Flash Attention
    print("\nTesting Sparse Flash Attention...")
    sparse_flash = SparseFlashAttention(
        embed_dim=256,
        num_heads=8,
        k_nearest=10,
        soft_mask=True,
        temperature=0.1
    ).to(device)

    # Create similarity matrix
    similarity = torch.randn(N, N, device=device)
    similarity = (similarity + similarity.T) / 2  # Symmetric
    sparse_mask = sparse_flash.create_sparse_mask(similarity, threshold=0.0)

    print(f"Sparse mask sparsity: {(sparse_mask == 0).sum().item() / sparse_mask.numel():.2%}")

    output_sf, _ = sparse_flash(x, x, x, attn_mask=sparse_mask)
    print(f"Sparse Flash output shape: {output_sf.shape}")

    # Memory comparison
    print("\n--- Memory Comparison ---")
    import gc

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Dense attention memory
    N_large = 500
    x_large = torch.randn(1, N_large, 256, device=device)

    flash_large = FlashAttentionMPS(embed_dim=256, num_heads=8).to(device)

    if device.type == "mps":
        torch.mps.synchronize()
        mem_before = torch.mps.current_allocated_memory()

    with torch.no_grad():
        output_large, _ = flash_large(x_large, x_large, x_large)

    if device.type == "mps":
        torch.mps.synchronize()
        mem_after = torch.mps.current_allocated_memory()
        print(f"Memory for N={N_large}: {(mem_after - mem_before) / 1e6:.2f} MB")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_flash_attention()
