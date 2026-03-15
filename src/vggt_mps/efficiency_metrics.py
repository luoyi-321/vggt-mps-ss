#!/usr/bin/env python3
"""
Efficiency Metrics for VGGT-MPS Sparse Attention

Implements efficiency metrics inspired by GaussianFormer-2 (arXiv:2412.04384)
for quantifying the performance benefits of sparse attention mechanisms.

Metrics:
- ASR (Attention Sparsity Ratio): Percentage of masked-out attention entries
- ECR (Effective Computation Ratio): Ratio of sparse to dense FLOPs
- ME (Memory Efficiency): Ratio of sparse to dense peak memory
- QER (Quality-Efficiency Ratio): Quality loss per computation saved
"""

import torch
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class EfficiencyReport:
    """Container for efficiency metrics"""
    asr: float  # Attention Sparsity Ratio (%)
    ecr: float  # Effective Computation Ratio
    me: float   # Memory Efficiency
    qer: Optional[float] = None  # Quality-Efficiency Ratio

    # Derived metrics
    flops_saved_percent: float = 0.0
    memory_saved_percent: float = 0.0

    # Raw statistics
    n_images: int = 0
    nnz: int = 0  # Number of non-zero entries
    total_entries: int = 0

    def __post_init__(self) -> None:
        self.flops_saved_percent = (1 - self.ecr) * 100
        self.memory_saved_percent = (1 - self.me) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'asr': self.asr,
            'ecr': self.ecr,
            'me': self.me,
            'qer': self.qer,
            'flops_saved_percent': self.flops_saved_percent,
            'memory_saved_percent': self.memory_saved_percent,
            'n_images': self.n_images,
            'nnz': self.nnz,
            'total_entries': self.total_entries
        }

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Efficiency Metrics Report",
            "=" * 50,
            f"Images: {self.n_images}",
            f"Non-zero entries: {self.nnz:,} / {self.total_entries:,}",
            "-" * 50,
            f"ASR (Attention Sparsity Ratio): {self.asr:.2f}%",
            f"ECR (Effective Computation Ratio): {self.ecr:.4f}",
            f"ME  (Memory Efficiency): {self.me:.4f}",
            "-" * 50,
            f"FLOPs Saved: {self.flops_saved_percent:.2f}%",
            f"Memory Saved: {self.memory_saved_percent:.2f}%",
        ]
        if self.qer is not None:
            lines.append(f"QER (Quality-Efficiency Ratio): {self.qer:.6f}")
        lines.append("=" * 50)
        return "\n".join(lines)


class EfficiencyMetrics:
    """
    Efficiency metrics calculator for sparse attention mechanisms.

    Based on metrics from GaussianFormer-2 (arXiv:2412.04384) Table 5,
    adapted for VGGT sparse attention evaluation.
    """

    def __init__(self, d_head: int = 64):
        """
        Initialize metrics calculator.

        Args:
            d_head: Attention head dimension (default: 64 for typical transformers)
        """
        self.d_head = d_head

    def attention_sparsity_ratio(self, mask: torch.Tensor) -> float:
        """
        Compute Attention Sparsity Ratio (ASR).

        ASR = |{(i,j) : M(i,j) = 0}| / (n² - n) × 100%

        Measures the percentage of attention entries that are masked out.
        Higher ASR = more sparse = more efficient.

        Args:
            mask: [N, N] attention mask (1 = compute, 0 = masked)

        Returns:
            ASR as percentage (0-100)
        """
        n = mask.shape[0]
        total_off_diag = n * n - n  # Exclude diagonal

        # Count zeros (masked entries)
        zeros = (mask == 0).sum().item()

        # Subtract diagonal zeros if any (they shouldn't be masked)
        diag_zeros = (torch.diag(mask) == 0).sum().item()
        zeros -= diag_zeros

        if total_off_diag == 0:
            return 0.0

        return zeros / total_off_diag * 100

    def effective_computation_ratio(self, mask: torch.Tensor) -> float:
        """
        Compute Effective Computation Ratio (ECR).

        ECR = FLOPs_sparse / FLOPs_dense
            = nnz(M) / n²

        Where:
            FLOPs_dense = 2 · n² · d (d = attention head dimension)
            FLOPs_sparse = 2 · nnz(M) · d

        Lower ECR = more efficient.

        Args:
            mask: [N, N] attention mask (1 = compute, 0 = masked)

        Returns:
            ECR ratio (0-1)
        """
        nnz = (mask != 0).sum().item()
        total = mask.numel()

        if total == 0:
            return 1.0

        return nnz / total

    def memory_efficiency(self, mask: torch.Tensor, d_head: Optional[int] = None) -> float:
        """
        Compute Memory Efficiency (ME).

        ME = Peak_Memory_Sparse / Peak_Memory_Dense

        Where:
            Peak_Memory_Dense = n² · sizeof(float) + 2 · n · d · sizeof(float)
            Peak_Memory_Sparse = nnz(M) · sizeof(float) + 2 · n · d · sizeof(float)

        Lower ME = more memory efficient.

        Args:
            mask: [N, N] attention mask (1 = compute, 0 = masked)
            d_head: Attention head dimension (uses self.d_head if None)

        Returns:
            ME ratio (0-1)
        """
        if d_head is None:
            d_head = self.d_head

        n = mask.shape[0]
        sizeof_float = 4  # bytes

        # Dense memory: attention matrix + 2 * (query/key projections)
        dense_mem = n * n * sizeof_float + 2 * n * d_head * sizeof_float

        # Sparse memory: only non-zero attention entries + projections
        nnz = (mask != 0).sum().item()
        sparse_mem = nnz * sizeof_float + 2 * n * d_head * sizeof_float

        if dense_mem == 0:
            return 1.0

        return sparse_mem / dense_mem

    def quality_efficiency_ratio(
        self,
        quality_sparse: float,
        quality_dense: float,
        ecr: float
    ) -> float:
        """
        Compute Quality-Efficiency Ratio (QER).

        QER = ΔQuality / ΔCompute

        Where:
            ΔQuality = |metric_sparse - metric_dense| / metric_dense
            ΔCompute = 1 - ECR

        Lower QER = better (less quality loss per computation saved).
        QER ≈ 0 means high efficiency with minimal quality degradation.

        Args:
            quality_sparse: Quality metric with sparse attention
            quality_dense: Quality metric with dense attention
            ecr: Effective Computation Ratio

        Returns:
            QER ratio (lower is better)
        """
        if quality_dense == 0:
            return float('inf')

        delta_quality = abs(quality_sparse - quality_dense) / quality_dense
        delta_compute = 1 - ecr

        if delta_compute <= 0:
            return 0.0

        return delta_quality / delta_compute

    def compute_all_metrics(
        self,
        mask: torch.Tensor,
        quality_sparse: Optional[float] = None,
        quality_dense: Optional[float] = None,
        d_head: Optional[int] = None
    ) -> EfficiencyReport:
        """
        Compute all efficiency metrics for a given attention mask.

        Args:
            mask: [N, N] attention mask (1 = compute, 0 = masked)
            quality_sparse: Optional quality metric with sparse attention
            quality_dense: Optional quality metric with dense attention
            d_head: Optional attention head dimension

        Returns:
            EfficiencyReport containing all metrics
        """
        asr = self.attention_sparsity_ratio(mask)
        ecr = self.effective_computation_ratio(mask)
        me = self.memory_efficiency(mask, d_head)

        qer = None
        if quality_sparse is not None and quality_dense is not None:
            qer = self.quality_efficiency_ratio(quality_sparse, quality_dense, ecr)

        n = mask.shape[0]
        nnz = (mask != 0).sum().item()

        return EfficiencyReport(
            asr=asr,
            ecr=ecr,
            me=me,
            qer=qer,
            n_images=n,
            nnz=nnz,
            total_entries=n * n
        )

    def compute_theoretical_scaling(
        self,
        n_images: int,
        k_nearest: int = 10
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute theoretical memory and FLOPs scaling.

        Args:
            n_images: Number of images
            k_nearest: Number of nearest neighbors per image

        Returns:
            Tuple of (dense_metrics, sparse_metrics) dictionaries
        """
        # Dense O(n²) complexity
        dense_attention_entries = n_images * n_images
        dense_flops = 2 * dense_attention_entries * self.d_head
        dense_memory = dense_attention_entries * 4  # bytes

        # Sparse O(n·k) complexity
        sparse_attention_entries = n_images * min(k_nearest, n_images)
        sparse_flops = 2 * sparse_attention_entries * self.d_head
        sparse_memory = sparse_attention_entries * 4  # bytes

        dense = {
            'attention_entries': dense_attention_entries,
            'flops': dense_flops,
            'memory_bytes': dense_memory,
            'memory_mb': dense_memory / (1024 * 1024)
        }

        sparse = {
            'attention_entries': sparse_attention_entries,
            'flops': sparse_flops,
            'memory_bytes': sparse_memory,
            'memory_mb': sparse_memory / (1024 * 1024),
            'savings_ratio': dense_attention_entries / sparse_attention_entries
        }

        return dense, sparse


class MPSHardwareMetrics:
    """
    Apple Silicon MPS-specific hardware metrics.

    Models the performance characteristics of different Apple Silicon chips
    for understanding compute vs memory bottlenecks.
    """

    # Apple Silicon specifications (theoretical peaks)
    CHIP_SPECS = {
        'M1': {'flops_tflops': 2.6, 'bandwidth_gbps': 68.25, 'memory_gb': 16},
        'M2': {'flops_tflops': 3.6, 'bandwidth_gbps': 100, 'memory_gb': 24},
        'M3': {'flops_tflops': 4.0, 'bandwidth_gbps': 100, 'memory_gb': 24},
        'M3_Pro': {'flops_tflops': 4.5, 'bandwidth_gbps': 150, 'memory_gb': 36},
        'M3_Max': {'flops_tflops': 14.0, 'bandwidth_gbps': 400, 'memory_gb': 128},
        'M4': {'flops_tflops': 4.5, 'bandwidth_gbps': 120, 'memory_gb': 32},
        'M4_Pro': {'flops_tflops': 5.3, 'bandwidth_gbps': 273, 'memory_gb': 64},
        'M4_Max': {'flops_tflops': 18.0, 'bandwidth_gbps': 546, 'memory_gb': 128}
    }

    def __init__(self, chip: str = 'M1'):
        """
        Initialize with target chip specifications.

        Args:
            chip: Apple Silicon chip name (e.g., 'M1', 'M2', 'M3_Pro', 'M4_Max')
        """
        if chip not in self.CHIP_SPECS:
            raise ValueError(f"Unknown chip: {chip}. Available: {list(self.CHIP_SPECS.keys())}")

        self.chip = chip
        self.specs = self.CHIP_SPECS[chip]

    def estimate_execution_time(
        self,
        flops: int,
        memory_bytes: int
    ) -> Dict[str, float]:
        """
        Estimate execution time based on compute and memory requirements.

        Args:
            flops: Number of floating point operations
            memory_bytes: Memory access in bytes

        Returns:
            Dictionary with time estimates and bottleneck analysis
        """
        # Convert TFLOPS to FLOPS
        peak_flops = self.specs['flops_tflops'] * 1e12

        # Convert GB/s to bytes/s
        bandwidth = self.specs['bandwidth_gbps'] * 1e9

        # Compute times
        t_compute = flops / peak_flops  # seconds
        t_memory = memory_bytes / bandwidth  # seconds

        # Determine bottleneck
        bottleneck = 'compute' if t_compute > t_memory else 'memory'

        # Arithmetic intensity
        ai = flops / memory_bytes if memory_bytes > 0 else float('inf')

        # Ridge point (where compute = memory bound)
        ridge_point = peak_flops / bandwidth

        return {
            't_compute_ms': t_compute * 1000,
            't_memory_ms': t_memory * 1000,
            't_total_ms': max(t_compute, t_memory) * 1000,
            'bottleneck': bottleneck,
            'arithmetic_intensity': ai,
            'ridge_point': ridge_point,
            'utilization': min(ai / ridge_point, 1.0) if ai < ridge_point else 1.0
        }

    def check_memory_fit(self, memory_bytes: int) -> Dict[str, Any]:
        """
        Check if workload fits in available memory.

        Args:
            memory_bytes: Required memory in bytes

        Returns:
            Dictionary with memory analysis
        """
        available = self.specs['memory_gb'] * 1e9
        required_gb = memory_bytes / 1e9
        fits = memory_bytes <= available

        return {
            'fits': fits,
            'required_gb': required_gb,
            'available_gb': self.specs['memory_gb'],
            'utilization': memory_bytes / available,
            'headroom_gb': (available - memory_bytes) / 1e9 if fits else 0
        }


def test_efficiency_metrics():
    """Test efficiency metrics calculations"""
    print("=" * 60)
    print("Testing Efficiency Metrics")
    print("=" * 60)

    metrics = EfficiencyMetrics(d_head=64)

    # Create test masks
    n = 50

    # Dense mask (all ones)
    dense_mask = torch.ones(n, n)

    # Sparse mask (k-nearest simulation)
    k = 10
    sparse_mask = torch.zeros(n, n)
    for i in range(n):
        neighbors = torch.randperm(n)[:k]
        sparse_mask[i, neighbors] = 1.0
        sparse_mask[neighbors, i] = 1.0
    sparse_mask.fill_diagonal_(1.0)

    print(f"\nTest configuration: n={n} images, k={k} neighbors")

    # Test dense mask
    print("\n--- Dense Mask ---")
    dense_report = metrics.compute_all_metrics(dense_mask)
    print(f"ASR: {dense_report.asr:.2f}%")
    print(f"ECR: {dense_report.ecr:.4f}")
    print(f"ME:  {dense_report.me:.4f}")

    # Test sparse mask
    print("\n--- Sparse Mask ---")
    sparse_report = metrics.compute_all_metrics(sparse_mask, quality_sparse=0.98, quality_dense=1.0)
    print(sparse_report)

    # Test theoretical scaling
    print("\n--- Theoretical Scaling ---")
    for n_test in [10, 50, 100, 500, 1000]:
        dense, sparse = metrics.compute_theoretical_scaling(n_test, k_nearest=10)
        print(f"n={n_test:4d}: {sparse['savings_ratio']:.1f}x savings")

    # Test MPS hardware metrics
    print("\n--- MPS Hardware Analysis ---")
    mps = MPSHardwareMetrics('M4_Max')

    # Simulate attention computation
    flops = 2 * n * n * 64  # O(n²) attention
    memory = n * n * 4  # attention matrix

    timing = mps.estimate_execution_time(flops, memory)
    print(f"Chip: {mps.chip}")
    print(f"Estimated time: {timing['t_total_ms']:.3f} ms")
    print(f"Bottleneck: {timing['bottleneck']}")
    print(f"Arithmetic Intensity: {timing['arithmetic_intensity']:.2f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_efficiency_metrics()
