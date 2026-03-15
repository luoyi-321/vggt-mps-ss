#!/usr/bin/env python3
"""
Probabilistic Aggregation for Multi-View Fusion

Implements probabilistic aggregation methods inspired by GaussianFormer-2 (arXiv:2412.04384)
for efficient multi-view 3D reconstruction in VGGT-MPS.

Key concepts:
1. Probabilistic Geometry Aggregation: α(x) = 1 - Π(1 - αᵢ(x))
   - Uses probability multiplication instead of addition
   - Naturally handles overlapping views without unbounded accumulation
   - Any single high-confidence view can confirm occupancy

2. GMM-based Semantic Aggregation: e(x) = Σᵢ p(Gᵢ|x) · cᵢ
   - Gaussian Mixture Model for semantic prediction
   - Normalized logits prevent unnecessary overlap
   - Higher utilization of Gaussian primitives

Reference:
    GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction
    arXiv:2412.04384
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


def probabilistic_geometry_aggregation(
    confidences_per_view: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Probabilistic multiplication aggregation for multi-view geometry.

    Implements the core insight from GaussianFormer-2:
        α(x) = 1 - Πᵢ (1 - αᵢ(x))

    This formulation has key advantages over additive aggregation:
    - Bounded output: α(x) ∈ [0, 1] regardless of number of views
    - Single confirmation: One high-confidence view can confirm occupancy
    - No redundancy penalty: Multiple views don't over-accumulate

    Args:
        confidences_per_view: [N_views, ...] tensor of occupancy probabilities
                              Each value should be in [0, 1]
        eps: Small epsilon for numerical stability

    Returns:
        aggregated: [...] aggregated occupancy probability

    Example:
        >>> # 3 views, each with 256x256 confidence map
        >>> confidences = torch.rand(3, 256, 256)
        >>> aggregated = probabilistic_geometry_aggregation(confidences)
        >>> print(aggregated.shape)  # [256, 256]
    """
    # Clamp to valid probability range
    confidences_per_view = torch.clamp(confidences_per_view, eps, 1 - eps)

    # α(x) = 1 - Π(1 - αᵢ(x))
    complement = 1.0 - confidences_per_view  # [N, ...]
    product = complement.prod(dim=0)  # [...]
    aggregated = 1.0 - product  # [...]

    return aggregated


def probabilistic_depth_aggregation(
    depths_per_view: torch.Tensor,
    confidences_per_view: torch.Tensor,
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Probabilistic depth aggregation with confidence weighting.

    Combines depth estimates from multiple views using probabilistic weighting.
    Higher confidence views contribute more to the final depth estimate.

    The aggregated depth is computed as:
        depth_final(x) = Σᵢ wᵢ(x) · depthᵢ(x) / Σⱼ wⱼ(x)

    where wᵢ(x) = αᵢ(x) / (1 - αᵢ(x)) (odds ratio weighting)

    Args:
        depths_per_view: [N_views, H, W] depth maps from each view
        confidences_per_view: [N_views, H, W] confidence maps in [0, 1]
        eps: Small epsilon for numerical stability

    Returns:
        depth_aggregated: [H, W] aggregated depth map
        confidence_aggregated: [H, W] aggregated confidence map
    """
    # Clamp confidences to avoid division by zero
    confidences = torch.clamp(confidences_per_view, eps, 1 - eps)

    # Compute odds ratio weights: w = α / (1 - α)
    # Higher confidence = higher weight
    weights = confidences / (1 - confidences)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)
    normalized_weights = weights / (weight_sum + eps)

    # Weighted depth aggregation
    depth_aggregated = (normalized_weights * depths_per_view).sum(dim=0)

    # Aggregated confidence using probabilistic formula
    confidence_aggregated = probabilistic_geometry_aggregation(confidences, eps)

    return depth_aggregated, confidence_aggregated


def gmm_semantic_aggregation(
    features_per_view: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    amplitudes: torch.Tensor,
    query_points: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Gaussian Mixture Model based semantic aggregation.

    Implements the GMM semantic prediction from GaussianFormer-2:
        e(x) = Σᵢ p(Gᵢ|x) · c̃ᵢ = [Σᵢ p(x|Gᵢ)aᵢc̃ᵢ] / [Σⱼ p(x|Gⱼ)aⱼ]

    where p(x|Gᵢ) is the Gaussian probability density.

    This normalized formulation:
    - Bounds semantic logits (unlike additive aggregation)
    - Prevents unnecessary Gaussian overlap
    - Improves primitive utilization

    Args:
        features_per_view: [N_gaussians, D] semantic features per Gaussian
        means: [N_gaussians, 3] Gaussian centers (x, y, z)
        covariances: [N_gaussians, 3, 3] or [N_gaussians, 3] covariance matrices or diagonals
        amplitudes: [N_gaussians] amplitude weights
        query_points: [N_points, 3] query positions
        eps: Small epsilon for numerical stability

    Returns:
        semantics: [N_points, D] aggregated semantic features
    """
    N_gaussians = means.shape[0]
    N_points = query_points.shape[0]
    D = features_per_view.shape[1]

    # Compute distances: [N_points, N_gaussians]
    # x - mᵢ for each point-gaussian pair
    diff = query_points.unsqueeze(1) - means.unsqueeze(0)  # [N_points, N_gaussians, 3]

    # Handle covariance (diagonal or full)
    if covariances.dim() == 2:
        # Diagonal covariance: [N_gaussians, 3]
        # Mahalanobis distance: (x-m)ᵀ Σ⁻¹ (x-m) = Σ (xᵢ-mᵢ)² / σᵢ²
        inv_cov = 1.0 / (covariances + eps)  # [N_gaussians, 3]
        mahal_sq = (diff ** 2 * inv_cov.unsqueeze(0)).sum(dim=-1)  # [N_points, N_gaussians]
    else:
        # Full covariance: [N_gaussians, 3, 3]
        # Inverse covariance
        inv_cov = torch.linalg.inv(covariances + eps * torch.eye(3, device=covariances.device))
        # (x-m)ᵀ Σ⁻¹ (x-m)
        mahal_sq = torch.einsum('pgi,gij,pgj->pg', diff, inv_cov, diff)

    # Gaussian probability: p(x|Gᵢ) ∝ exp(-0.5 * mahal²)
    log_prob = -0.5 * mahal_sq  # [N_points, N_gaussians]
    prob = torch.exp(log_prob)  # [N_points, N_gaussians]

    # Weight by amplitude: aᵢ · p(x|Gᵢ)
    weighted_prob = prob * amplitudes.unsqueeze(0)  # [N_points, N_gaussians]

    # Normalize: p(Gᵢ|x) = aᵢ·p(x|Gᵢ) / Σⱼ aⱼ·p(x|Gⱼ)
    posterior = weighted_prob / (weighted_prob.sum(dim=1, keepdim=True) + eps)  # [N_points, N_gaussians]

    # Aggregate semantics: Σᵢ p(Gᵢ|x) · cᵢ
    semantics = torch.mm(posterior, features_per_view)  # [N_points, D]

    return semantics


def alpha_compositing(
    values_per_view: torch.Tensor,
    alphas_per_view: torch.Tensor,
    front_to_back: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Alpha compositing for ordered view aggregation.

    Traditional alpha compositing for front-to-back or back-to-front rendering.

    Front-to-back:
        C_out = C_in + (1 - A_in) · αᵢ · cᵢ
        A_out = A_in + (1 - A_in) · αᵢ

    Args:
        values_per_view: [N_views, ...] values (colors, depths, etc.)
        alphas_per_view: [N_views, ...] alpha values in [0, 1]
        front_to_back: If True, first view is frontmost

    Returns:
        composited: [...] composited values
        accumulated_alpha: [...] accumulated alpha
    """
    N = values_per_view.shape[0]

    if not front_to_back:
        values_per_view = values_per_view.flip(0)
        alphas_per_view = alphas_per_view.flip(0)

    # Initialize
    shape = values_per_view.shape[1:]
    composited = torch.zeros(shape, device=values_per_view.device, dtype=values_per_view.dtype)
    accumulated_alpha = torch.zeros(shape, device=alphas_per_view.device, dtype=alphas_per_view.dtype)

    for i in range(N):
        # Transmittance = 1 - accumulated alpha
        transmittance = 1.0 - accumulated_alpha

        # Accumulate color: C += T · α · c
        composited = composited + transmittance * alphas_per_view[i] * values_per_view[i]

        # Accumulate alpha: A += T · α
        accumulated_alpha = accumulated_alpha + transmittance * alphas_per_view[i]

    return composited, accumulated_alpha


class ProbabilisticMultiViewFusion:
    """
    Multi-view fusion using probabilistic aggregation methods.

    Combines multiple aggregation strategies for comprehensive 3D reconstruction.
    """

    def __init__(
        self,
        geometry_method: str = 'probabilistic',
        depth_method: str = 'weighted',
        semantic_method: str = 'gmm'
    ):
        """
        Initialize fusion module.

        Args:
            geometry_method: 'probabilistic' or 'additive'
            depth_method: 'weighted' or 'mean'
            semantic_method: 'gmm' or 'mean'
        """
        self.geometry_method = geometry_method
        self.depth_method = depth_method
        self.semantic_method = semantic_method

    def fuse_geometry(
        self,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse geometry (occupancy) from multiple views.

        Args:
            confidences: [N_views, ...] occupancy confidences

        Returns:
            fused: [...] fused occupancy
        """
        if self.geometry_method == 'probabilistic':
            return probabilistic_geometry_aggregation(confidences)
        else:
            # Additive (legacy): simple mean
            return confidences.mean(dim=0)

    def fuse_depth(
        self,
        depths: torch.Tensor,
        confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse depth from multiple views.

        Args:
            depths: [N_views, H, W] depth maps
            confidences: [N_views, H, W] confidence maps

        Returns:
            depth: [H, W] fused depth
            confidence: [H, W] fused confidence
        """
        if self.depth_method == 'weighted':
            return probabilistic_depth_aggregation(depths, confidences)
        else:
            # Simple mean
            return depths.mean(dim=0), confidences.mean(dim=0)

    def compute_efficiency_metrics(
        self,
        n_views: int,
        n_gaussians: int
    ) -> Dict[str, Any]:
        """
        Compute efficiency metrics for the fusion configuration.

        Based on GaussianFormer-2 Table 5 metrics.

        Args:
            n_views: Number of views
            n_gaussians: Number of Gaussian primitives

        Returns:
            metrics: Dictionary of efficiency metrics
        """
        # Theoretical metrics (would need actual output for real metrics)
        metrics = {
            'n_views': n_views,
            'n_gaussians': n_gaussians,
            'geometry_method': self.geometry_method,
            'depth_method': self.depth_method,
            'semantic_method': self.semantic_method,
            # Theoretical advantages of probabilistic aggregation
            'expected_overlap_reduction': '3-4x' if self.geometry_method == 'probabilistic' else '1x',
            'expected_utilization_improvement': '~75%' if self.geometry_method == 'probabilistic' else 'baseline'
        }
        return metrics


def test_probabilistic_aggregation():
    """Test probabilistic aggregation functions"""
    print("=" * 60)
    print("Testing Probabilistic Aggregation")
    print("=" * 60)

    # Test probabilistic geometry aggregation
    print("\n--- Probabilistic Geometry Aggregation ---")

    # Create test data: 5 views, 32x32 confidence maps
    n_views = 5
    h, w = 32, 32
    confidences = torch.rand(n_views, h, w) * 0.8  # Random confidences 0-0.8

    # Aggregate
    aggregated = probabilistic_geometry_aggregation(confidences)
    print(f"Input shape: {confidences.shape}")
    print(f"Output shape: {aggregated.shape}")
    print(f"Min aggregated: {aggregated.min():.4f}")
    print(f"Max aggregated: {aggregated.max():.4f}")

    # Verify property: aggregated >= max(individual)
    max_individual = confidences.max(dim=0).values
    assert (aggregated >= max_individual - 1e-6).all(), "Aggregated should be >= max individual"
    print("Property verified: aggregated >= max(individual)")

    # Test depth aggregation
    print("\n--- Probabilistic Depth Aggregation ---")
    depths = torch.rand(n_views, h, w) * 10  # Depths 0-10

    depth_agg, conf_agg = probabilistic_depth_aggregation(depths, confidences)
    print(f"Depth output shape: {depth_agg.shape}")
    print(f"Confidence output shape: {conf_agg.shape}")

    # Test GMM semantic aggregation
    print("\n--- GMM Semantic Aggregation ---")
    n_gaussians = 100
    n_points = 50
    d_features = 32

    features = torch.randn(n_gaussians, d_features)
    means = torch.randn(n_gaussians, 3)
    covariances = torch.rand(n_gaussians, 3) + 0.1  # Diagonal covariances
    amplitudes = F.softmax(torch.randn(n_gaussians), dim=0)
    query_points = torch.randn(n_points, 3)

    semantics = gmm_semantic_aggregation(
        features, means, covariances, amplitudes, query_points
    )
    print(f"Semantics output shape: {semantics.shape}")

    # Test fusion module
    print("\n--- Multi-View Fusion Module ---")
    fusion = ProbabilisticMultiViewFusion(
        geometry_method='probabilistic',
        depth_method='weighted'
    )

    geometry = fusion.fuse_geometry(confidences)
    depth, conf = fusion.fuse_depth(depths, confidences)
    metrics = fusion.compute_efficiency_metrics(n_views, n_gaussians)

    print(f"Geometry shape: {geometry.shape}")
    print(f"Depth shape: {depth.shape}")
    print(f"Efficiency metrics: {metrics}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_probabilistic_aggregation()
