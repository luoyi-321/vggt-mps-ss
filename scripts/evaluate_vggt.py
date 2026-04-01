#!/usr/bin/env python3
"""
VGGT Evaluation Script for ECCV 2026 Paper

Computes quality metrics:
- Depth L1 error
- Absolute relative error
- Rotation error (degrees)
- Translation error
- Inference time
- Peak memory usage
- Dense vs Sparse comparison

Usage:
    python scripts/evaluate_vggt.py --image-dir data/real_data/bottle_cap
    python scripts/evaluate_vggt.py --image-dir data/real_data/bottle_cap --mode sparse --k 10
    python scripts/evaluate_vggt.py --compare-dense-sparse --image-dir data/real_data/bottle_cap
"""

import argparse
import glob
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Depth metrics
    depth_l1: float = 0.0
    depth_abs_rel: float = 0.0
    depth_sq_rel: float = 0.0
    depth_rmse: float = 0.0

    # Pose metrics
    rotation_error_deg: float = 0.0
    translation_error: float = 0.0

    # Efficiency metrics
    inference_time_ms: float = 0.0
    peak_memory_mb: float = 0.0

    # Sparsity info
    sparsity_ratio: float = 0.0
    k_nearest: int = 0

    # Additional info
    num_images: int = 0
    mode: str = "dense"

    # Statistics (from multiple runs)
    time_std: float = 0.0
    memory_std: float = 0.0
    n_runs: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "depth_l1": self.depth_l1,
            "depth_abs_rel": self.depth_abs_rel,
            "depth_sq_rel": self.depth_sq_rel,
            "depth_rmse": self.depth_rmse,
            "rotation_error_deg": self.rotation_error_deg,
            "translation_error": self.translation_error,
            "inference_time_ms": self.inference_time_ms,
            "time_std": self.time_std,
            "peak_memory_mb": self.peak_memory_mb,
            "memory_std": self.memory_std,
            "sparsity_ratio": self.sparsity_ratio,
            "k_nearest": self.k_nearest,
            "num_images": self.num_images,
            "mode": self.mode,
            "n_runs": self.n_runs,
        }


def get_device() -> torch.device:
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB"""
    device = get_device()
    if device.type == "mps":
        # MPS memory tracking
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except AttributeError:
            return 0.0
    elif device.type == "cuda":
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def synchronize_device():
    """Synchronize GPU for accurate timing"""
    device = get_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute depth estimation metrics.

    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        valid_mask: Optional mask for valid depth values

    Returns:
        Dictionary of metric values
    """
    if valid_mask is None:
        valid_mask = (gt_depth > 0) & np.isfinite(gt_depth)

    pred = pred_depth[valid_mask]
    gt = gt_depth[valid_mask]

    if len(pred) == 0:
        return {"l1": 0.0, "abs_rel": 0.0, "sq_rel": 0.0, "rmse": 0.0}

    # L1 error
    l1 = np.mean(np.abs(pred - gt))

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    # Squared relative error
    sq_rel = np.mean(((pred - gt) ** 2) / gt)

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    return {
        "l1": float(l1),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
    }


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute rotation error in degrees.

    Formula: arccos((tr(R_pred^T @ R_gt) - 1) / 2)

    Args:
        R_pred: Predicted rotation matrix (3x3)
        R_gt: Ground truth rotation matrix (3x3)

    Returns:
        Rotation error in degrees
    """
    R_diff = R_pred.T @ R_gt
    trace = np.trace(R_diff)
    # Clamp for numerical stability
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos((trace - 1) / 2)
    return float(np.degrees(angle_rad))


def compute_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Compute translation error as L2 norm.

    Args:
        t_pred: Predicted translation vector (3,)
        t_gt: Ground truth translation vector (3,)

    Returns:
        Translation error (L2 norm)
    """
    return float(np.linalg.norm(t_pred - t_gt))


def find_images(image_dir: Path, max_images: int = 10) -> List[Path]:
    """Find image files in directory"""
    image_dir = Path(image_dir)

    # Search patterns
    patterns = ["*.jpg", "*.png", "*.jpeg", "**/*.jpg", "**/*.png"]

    images = []
    for pattern in patterns:
        found = sorted(glob.glob(str(image_dir / pattern), recursive=True))
        images.extend([Path(p) for p in found])
        if len(images) >= max_images:
            break

    # Remove duplicates and limit
    images = list(dict.fromkeys(images))[:max_images]
    return images


def evaluate_chunked(
    image_dir: Path,
    chunk_size: int = 2,
    max_images: int = 10,
    mode: str = "dense",
    k_nearest: int = 10,
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Process many images in memory-safe chunks.
    Aggregates depth/pose predictions across chunks.

    This allows testing with more images on MPS despite memory limits.
    Each chunk reloads the model to free memory between chunks.

    Args:
        image_dir: Directory containing input images
        chunk_size: Number of images per chunk (default: 2 for MPS)
        max_images: Maximum number of images to process
        mode: "dense" or "sparse"
        k_nearest: K for sparse attention
        threshold: Covisibility threshold for sparse mode

    Returns:
        Dictionary with aggregated results
    """
    device = get_device()
    images = find_images(image_dir, max_images)
    n_images = len(images)

    if n_images == 0:
        print("No images found")
        return {}

    print(f"Processing {n_images} images in chunks of {chunk_size}")

    all_times = []
    all_memory = []
    all_depths = []

    # Preload utils once
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError:
        print("Error: Could not import VGGT utils")
        return {}

    # Process in chunks (reload model per chunk to save memory on MPS)
    for i in range(0, n_images, chunk_size):
        chunk_images = images[i:i + chunk_size]
        if len(chunk_images) < 2:
            continue  # VGGT needs at least 2 images

        chunk_num = i // chunk_size + 1
        print(f"\n--- Chunk {chunk_num}: images {i+1}-{i+len(chunk_images)} ---")

        # Load model for this chunk
        clear_memory()
        model = load_model(device, mode=mode, k_nearest=k_nearest, threshold=threshold)

        image_paths = [str(p) for p in chunk_images]
        input_tensor = load_and_preprocess_images(image_paths).to(device)

        # Time inference
        synchronize_device()
        start = time.perf_counter()

        with torch.no_grad():
            preds = model(input_tensor)

        synchronize_device()
        elapsed = (time.perf_counter() - start) * 1000

        all_times.append(elapsed)
        all_memory.append(get_memory_usage())
        all_depths.append(preds['depth'].cpu())

        print(f"  Time: {elapsed:.1f} ms, Memory: {all_memory[-1]:.1f} MB")

        # Free memory for next chunk
        del model, preds, input_tensor
        clear_memory()

    # Aggregate results
    if not all_times:
        print("Warning: No chunks processed (need at least 2 images per chunk)")
        return {"error": "no_chunks_processed", "num_images": n_images}

    total_time = sum(all_times)
    avg_memory = sum(all_memory) / len(all_memory)

    print(f"\n{'='*60}")
    print(f"Chunked Processing Summary ({n_images} images, chunk_size={chunk_size})")
    print(f"{'='*60}")
    print(f"Total inference time: {total_time:.1f} ms")
    print(f"Average memory per chunk: {avg_memory:.1f} MB")
    print(f"Effective throughput: {n_images / (total_time / 1000):.2f} images/sec")

    return {
        "total_time_ms": total_time,
        "avg_memory_mb": avg_memory,
        "num_images": n_images,
        "num_chunks": len(all_times),
        "throughput_img_per_sec": n_images / (total_time / 1000) if total_time > 0 else 0,
        "mode": mode,
    }


def load_model(
    device: torch.device,
    mode: str = "dense",
    k_nearest: int = 10,
    threshold: float = 0.7,
    use_bfloat16: bool = False,
    sparse_layers: Optional[list] = None,
    all_layers: bool = False,
):
    """
    Load VGGT model in dense or sparse mode.

    Args:
        device: PyTorch device
        mode: "dense" or "sparse"
        k_nearest: Number of nearest neighbors for sparse mode
        threshold: Covisibility threshold for sparse mode
        use_bfloat16: Cast model to bfloat16 (VGGT-X technique: ~74% VRAM reduction).
                      Prediction heads stay in float32 for numerical accuracy.
        sparse_layers: Specific global_block indices to sparsify.
                       None = default middle-layer selection [10..18].
        all_layers: Apply sparse attention to all global_blocks.

    Returns:
        Loaded VGGT model
    """
    try:
        from vggt.models.vggt import VGGT
    except ImportError as e:
        print(f"Error: Could not import VGGT: {e}")
        print("Make sure repo/vggt is properly set up.")
        sys.exit(1)

    # Find model weights
    model_path = PROJECT_ROOT / "models" / "model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Download from: https://huggingface.co/facebook/VGGT-1B")
        sys.exit(1)

    # Override flags from environment (set by run_ablations.py CLI)
    import os as _os
    if _os.environ.get("VGGT_BFLOAT16") == "1":
        use_bfloat16 = True
    if _os.environ.get("VGGT_ALL_LAYERS") == "1":
        all_layers = True
    _env_layers = _os.environ.get("VGGT_SPARSE_LAYERS")
    if _env_layers and sparse_layers is None:
        sparse_layers = [int(x) for x in _env_layers.split(",")]

    print(f"Loading model from {model_path}...")
    model = VGGT()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # BFloat16 (VGGT-X paper: 74% VRAM reduction, minimal quality loss)
    # Keep prediction heads in float32 for numerical stability.
    if use_bfloat16 and device.type in ("cuda", "mps"):
        print("Converting to BFloat16 (VGGT-X technique)...")
        model = model.to(torch.bfloat16)
        # Restore float32 for output heads
        head_names = ["depth_head", "camera_head", "track_head", "pts3d_head"]
        for name in head_names:
            head = getattr(model, name, None)
            if head is not None:
                head.to(torch.float32)

    # Convert to sparse if requested
    if mode == "sparse":
        try:
            from vggt_mps.vggt_sparse_attention import make_vggt_sparse
            model = make_vggt_sparse(
                model,
                device=str(device),
                k_nearest=k_nearest,
                threshold=threshold,
                lightweight=True,
                sparse_layers=sparse_layers,
                all_layers=all_layers,
            )
        except ImportError as e:
            print(f"Warning: Could not import sparse attention module: {e}")
            print("Falling back to dense mode.")

    return model


def evaluate(
    image_dir: Path,
    gt_dir: Optional[Path] = None,
    mode: str = "dense",
    k_nearest: int = 10,
    threshold: float = 0.7,
    max_images: int = 10,
    model=None,  # Pre-loaded model (avoids reloading across runs)
) -> EvaluationMetrics:
    """
    Run evaluation on a set of images.

    Args:
        image_dir: Directory containing input images
        gt_dir: Optional directory with ground truth depth/poses
        mode: "dense" or "sparse"
        k_nearest: K for sparse attention
        threshold: Covisibility threshold
        max_images: Maximum number of images to process
        model: Optional pre-loaded model (reused across calls)

    Returns:
        EvaluationMetrics object
    """
    device = get_device()
    print(f"Device: {device}")

    # Find images
    images = find_images(image_dir, max_images)
    if not images:
        print(f"No images found in {image_dir}")
        return EvaluationMetrics()

    n = len(images)
    print(f"Found {n} images")

    # Load model only if not reusing a pre-loaded one
    if model is None:
        model = load_model(device, mode, k_nearest, threshold)

    # Load and preprocess images
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError:
        print("Error: Could not import VGGT utils")
        return EvaluationMetrics()

    image_paths = [str(p) for p in images]
    print(f"Preprocessing {n} images...")
    input_tensor = load_and_preprocess_images(image_paths).to(device)

    # Reset memory tracking
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Run inference with timing (only inference, not feature extraction overhead)
    print("Running inference...")
    synchronize_device()
    start_time = time.perf_counter()

    with torch.no_grad():
        predictions = model(input_tensor)

    synchronize_device()
    end_time = time.perf_counter()

    inference_time_ms = (end_time - start_time) * 1000
    peak_memory_mb = get_memory_usage()

    print(f"Inference time: {inference_time_ms:.1f} ms")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")

    # Extract predictions
    depth_pred = predictions['depth'].cpu().numpy()

    # Compute sparsity ratio for sparse mode
    # Use min(k, n-1) so k never exceeds available neighbors.
    # Denominator is n*(n-1) (off-diagonal pairs only).
    sparsity_ratio = 0.0
    k_eff = 0
    if mode == "sparse":
        k_eff = min(k_nearest, n - 1)  # can't attend to more than n-1 other views
        off_diag = n * (n - 1)
        sparse_pairs = n * k_eff
        sparsity_ratio = max(0.0, 1.0 - (sparse_pairs / off_diag)) if off_diag > 0 else 0.0

    # Initialize metrics
    metrics = EvaluationMetrics(
        inference_time_ms=inference_time_ms,
        peak_memory_mb=peak_memory_mb,
        num_images=n,
        mode=mode,
        k_nearest=k_eff if mode == "sparse" else 0,
        sparsity_ratio=sparsity_ratio,
    )

    # Compute depth metrics only if ground truth is provided
    has_gt = gt_dir is not None and Path(gt_dir).exists()
    if has_gt:
        gt_depths = sorted(glob.glob(str(gt_dir / "*.npy")))
        if gt_depths:
            all_l1, all_abs_rel = [], []
            for i, gt_path in enumerate(gt_depths[:n]):
                gt_depth = np.load(gt_path)
                pred_depth = depth_pred[0, i, :, :, 0]

                # Resize if needed
                if pred_depth.shape != gt_depth.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = (gt_depth.shape[0] / pred_depth.shape[0],
                                   gt_depth.shape[1] / pred_depth.shape[1])
                    pred_depth = zoom(pred_depth, zoom_factors)

                depth_metrics = compute_depth_metrics(pred_depth, gt_depth)
                all_l1.append(depth_metrics["l1"])
                all_abs_rel.append(depth_metrics["abs_rel"])

            metrics.depth_l1 = float(np.mean(all_l1))
            metrics.depth_abs_rel = float(np.mean(all_abs_rel))
    # When no GT is provided, depth_l1 stays at -1.0 to signal N/A
    else:
        metrics.depth_l1 = -1.0
        metrics.depth_abs_rel = -1.0

    return metrics


def clear_memory():
    """Clear GPU memory between runs"""
    import gc
    gc.collect()
    device = get_device()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def compare_dense_vs_sparse(
    image_dir: Path,
    k_values: List[int] = [3, 5, 10],
    max_images: int = 10,
) -> Dict[str, EvaluationMetrics]:
    """
    Compare dense vs sparse VGGT performance.

    Loads and unloads models between runs to avoid MPS OOM.

    Args:
        image_dir: Directory containing input images
        k_values: List of k values to test for sparse attention
        max_images: Maximum number of images to process

    Returns:
        Dictionary mapping config name to metrics
    """
    device = get_device()
    print("=" * 60)
    print("Dense vs Sparse Comparison")
    print("=" * 60)

    results = {}

    # Run dense baseline
    print("\n--- Dense (baseline) ---")
    clear_memory()
    dense_model = load_model(device, mode="dense")
    dense_metrics = evaluate(image_dir, mode="dense", max_images=max_images, model=dense_model)
    results["dense"] = dense_metrics
    # Free memory
    del dense_model
    clear_memory()

    # Run sparse with different k values
    for k in k_values:
        print(f"\n--- Sparse k={k} ---")
        clear_memory()
        sparse_model = load_model(device, mode="sparse", k_nearest=k)
        sparse_metrics = evaluate(
            image_dir, mode="sparse", k_nearest=k, max_images=max_images, model=sparse_model
        )
        results[f"sparse_k{k}"] = sparse_metrics
        # Free memory
        del sparse_model
        clear_memory()

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("Results Summary")
    print("=" * 90)
    print(f"{'Config':<15} {'Sparsity':<10} {'k_eff':<7} {'Time (ms)':<12} {'Memory (MB)':<12} {'Depth L1':<12}")
    print("-" * 90)

    for name, m in results.items():
        sparsity = f"{m.sparsity_ratio * 100:.0f}%" if m.sparsity_ratio > 0 else "0%"
        k_str = str(m.k_nearest) if m.k_nearest > 0 else "-"
        # depth_l1 == -1 means no GT was provided
        depth_str = "N/A" if m.depth_l1 < 0 else f"{m.depth_l1:.4f}"
        print(
            f"{name:<15} {sparsity:<10} {k_str:<7} "
            f"{m.inference_time_ms:<12.1f} {m.peak_memory_mb:<12.1f} {depth_str:<12}"
        )

    print("-" * 90)

    # Compute speedup and memory savings
    if "dense" in results and len(results) > 1:
        dense_time = results["dense"].inference_time_ms
        dense_mem = results["dense"].peak_memory_mb

        print("\nEfficiency Gains vs Dense:")
        for name, m in results.items():
            if name != "dense" and m.inference_time_ms > 0:
                speedup = dense_time / m.inference_time_ms
                mem_reduction = (
                    (dense_mem - m.peak_memory_mb) / dense_mem * 100
                    if dense_mem > 0 else 0
                )
                print(f"  {name}: {speedup:.2f}x speedup, {mem_reduction:.1f}% memory reduction")

    print("\nNote: Depth L1 shows 'N/A' when --gt-dir is not provided (no ground-truth depth).")
    print("      Sparsity is clamped: k_eff = min(k, n-1) where n = number of images.")

    return results


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistics from multiple runs.

    Args:
        values: List of values from multiple runs

    Returns:
        Dictionary with mean, std, 95% CI
    """
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0}

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    # 95% confidence interval
    n = len(arr)
    if n > 1:
        se = std / np.sqrt(n)
        ci_margin = 1.96 * se  # z-score for 95% CI
        ci_lower = mean - ci_margin
        ci_upper = mean + ci_margin
    else:
        ci_lower = ci_upper = mean

    return {
        "mean": mean,
        "std": std,
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n": n
    }


def compute_significance(values_a: List[float], values_b: List[float]) -> Dict[str, float]:
    """
    Compute statistical significance using paired t-test.

    Args:
        values_a: Results from configuration A
        values_b: Results from configuration B

    Returns:
        Dictionary with t-statistic and p-value
    """
    from scipy import stats

    if len(values_a) < 2 or len(values_b) < 2:
        return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}

    t_stat, p_value = stats.ttest_rel(values_a, values_b)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05
    }


def run_scaling_benchmark(
    image_dir: Path,
    n_values: List[int] = [10, 20, 50, 100],
    k_values: List[int] = [5, 10, 15],
    n_runs: int = 3,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Run scaling benchmark across different n (images) and k (neighbors) values.

    Tests how inference time and memory scale with:
    - n: number of input images
    - k: number of nearest neighbors for sparse attention

    Args:
        image_dir: Directory containing input images
        n_values: List of image counts to test
        k_values: List of k values to test
        n_runs: Number of runs per configuration
        threshold: Covisibility threshold

    Returns:
        Dictionary with scaling results
    """
    device = get_device()
    print("=" * 70)
    print("Scaling Benchmark")
    print("=" * 70)
    print(f"Image counts (n): {n_values}")
    print(f"K values: {k_values}")
    print(f"Runs per config: {n_runs}")

    results = {
        "n_values": n_values,
        "k_values": k_values,
        "n_runs": n_runs,
        "configurations": {}
    }

    # Test dense baseline for each n
    print("\n--- Dense Baseline ---")
    for n in n_values:
        config_name = f"dense_n{n}"
        print(f"Testing n={n}...")

        times = []
        memories = []

        for run in range(n_runs):
            clear_memory()
            model = load_model(device, mode="dense")
            metrics = evaluate(image_dir, mode="dense", max_images=n, model=model)

            times.append(metrics.inference_time_ms)
            memories.append(metrics.peak_memory_mb)

            del model
            clear_memory()

        time_stats = compute_statistics(times)
        mem_stats = compute_statistics(memories)

        results["configurations"][config_name] = {
            "n_images": n,
            "mode": "dense",
            "k": 0,
            "time_ms": time_stats,
            "memory_mb": mem_stats,
            "theoretical_complexity": f"O({n}²)"
        }

        print(f"  n={n}: {time_stats['mean']:.1f} ± {time_stats['std']:.1f} ms")

    # Test sparse for each n and k combination
    print("\n--- Sparse Configurations ---")
    for n in n_values:
        for k in k_values:
            if k >= n:
                continue  # k must be less than n

            config_name = f"sparse_n{n}_k{k}"
            print(f"Testing n={n}, k={k}...")

            times = []
            memories = []

            for run in range(n_runs):
                clear_memory()
                model = load_model(device, mode="sparse", k_nearest=k, threshold=threshold)
                metrics = evaluate(
                    image_dir, mode="sparse", k_nearest=k,
                    threshold=threshold, max_images=n, model=model
                )

                times.append(metrics.inference_time_ms)
                memories.append(metrics.peak_memory_mb)

                del model
                clear_memory()

            time_stats = compute_statistics(times)
            mem_stats = compute_statistics(memories)

            results["configurations"][config_name] = {
                "n_images": n,
                "mode": "sparse",
                "k": k,
                "time_ms": time_stats,
                "memory_mb": mem_stats,
                "theoretical_complexity": f"O({n}·{k})"
            }

            print(f"  n={n}, k={k}: {time_stats['mean']:.1f} ± {time_stats['std']:.1f} ms")

    # Compute speedups
    print("\n--- Speedup Analysis ---")
    print(f"{'Config':<20} {'Speedup vs Dense':<20} {'Memory Reduction':<20}")
    print("-" * 60)

    for config_name, config_data in results["configurations"].items():
        if config_data["mode"] == "sparse":
            n = config_data["n_images"]
            dense_key = f"dense_n{n}"

            if dense_key in results["configurations"]:
                dense_time = results["configurations"][dense_key]["time_ms"]["mean"]
                sparse_time = config_data["time_ms"]["mean"]
                speedup = dense_time / sparse_time if sparse_time > 0 else 0

                dense_mem = results["configurations"][dense_key]["memory_mb"]["mean"]
                sparse_mem = config_data["memory_mb"]["mean"]
                mem_reduction = (dense_mem - sparse_mem) / dense_mem * 100 if dense_mem > 0 else 0

                config_data["speedup"] = speedup
                config_data["memory_reduction_pct"] = mem_reduction

                print(f"{config_name:<20} {speedup:.2f}x{'':<14} {mem_reduction:.1f}%")

    print("=" * 70)

    return results


def run_with_statistics(
    image_dir: Path,
    mode: str = "dense",
    k_nearest: int = 10,
    threshold: float = 0.7,
    max_images: int = 10,
    n_runs: int = 5
) -> EvaluationMetrics:
    """
    Run evaluation multiple times and compute statistics.

    Args:
        image_dir: Directory containing input images
        mode: "dense" or "sparse"
        k_nearest: K for sparse attention
        threshold: Covisibility threshold
        max_images: Maximum images to process
        n_runs: Number of runs

    Returns:
        EvaluationMetrics with mean values and standard deviations
    """
    device = get_device()

    all_times = []
    all_memories = []
    all_metrics = []

    # Load model once
    model = load_model(device, mode, k_nearest, threshold)

    for run in range(n_runs):
        metrics = evaluate(
            image_dir,
            mode=mode,
            k_nearest=k_nearest,
            threshold=threshold,
            max_images=max_images,
            model=model
        )
        all_times.append(metrics.inference_time_ms)
        all_memories.append(metrics.peak_memory_mb)
        all_metrics.append(metrics)

    # Clean up
    del model
    clear_memory()

    # Aggregate results
    final = EvaluationMetrics(
        mode=mode,
        k_nearest=k_nearest if mode == "sparse" else 0,
        num_images=all_metrics[0].num_images,
        sparsity_ratio=all_metrics[0].sparsity_ratio,
        n_runs=n_runs
    )

    # Mean values
    final.inference_time_ms = float(np.mean(all_times))
    final.peak_memory_mb = float(np.mean(all_memories))
    final.depth_l1 = float(np.mean([m.depth_l1 for m in all_metrics]))
    final.depth_abs_rel = float(np.mean([m.depth_abs_rel for m in all_metrics]))

    # Standard deviations
    final.time_std = float(np.std(all_times, ddof=1)) if len(all_times) > 1 else 0.0
    final.memory_std = float(np.std(all_memories, ddof=1)) if len(all_memories) > 1 else 0.0

    return final


def print_hardware_info():
    """Print hardware information for reproducibility"""
    import platform

    print("\n" + "=" * 60)
    print("Hardware Information")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    device = get_device()
    print(f"Device: {device}")

    if device.type == "mps":
        print("MPS available: True")
    elif device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VGGT Evaluation Script")
    parser.add_argument("--image-dir", type=Path, required=False,
                       default=PROJECT_ROOT / "data" / "real_data" / "bottle_cap",
                       help="Directory containing input images")
    parser.add_argument("--gt-dir", type=Path, default=None,
                       help="Directory containing ground truth depth/poses")
    parser.add_argument("--mode", choices=["dense", "sparse"], default="dense",
                       help="Attention mode")
    parser.add_argument("--k", type=int, default=10,
                       help="K nearest neighbors for sparse attention")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Covisibility threshold")
    parser.add_argument("--max-images", type=str, default="5",
                       help="Maximum images (single value or comma-separated for scaling)")
    parser.add_argument("--k-values", type=str, default=None,
                       help="Comma-separated k values for scaling benchmark")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of runs for statistics")
    parser.add_argument("--compare-dense-sparse", action="store_true",
                       help="Run dense vs sparse comparison")
    parser.add_argument("--scaling", action="store_true",
                       help="Run scaling benchmark mode")
    parser.add_argument("--chunked", action="store_true",
                       help="Process images in memory-safe chunks (for MPS)")
    parser.add_argument("--chunk-size", type=int, default=2,
                       help="Number of images per chunk (default: 2 for MPS)")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--hardware-info", action="store_true",
                       help="Print hardware info")

    args = parser.parse_args()

    # Parse max-images (can be single value or comma-separated list)
    if "," in args.max_images:
        max_images_list = [int(x.strip()) for x in args.max_images.split(",")]
        max_images = max_images_list[0]  # Use first value for non-scaling modes
    else:
        max_images = int(args.max_images)
        max_images_list = [max_images]

    # Parse k-values
    if args.k_values:
        k_values = [int(x.strip()) for x in args.k_values.split(",")]
    else:
        k_values = [3, 5, 10]

    if args.hardware_info:
        print_hardware_info()

    if args.scaling:
        # Scaling benchmark mode
        results = run_scaling_benchmark(
            args.image_dir,
            n_values=max_images_list,
            k_values=k_values,
            n_runs=args.runs,
            threshold=args.threshold,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    elif args.chunked:
        # Memory-safe chunked processing for many images on MPS
        results = evaluate_chunked(
            args.image_dir,
            chunk_size=args.chunk_size,
            max_images=max_images,
            mode=args.mode,
            k_nearest=args.k,
            threshold=args.threshold,
        )
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    elif args.compare_dense_sparse:
        results = compare_dense_vs_sparse(
            args.image_dir,
            k_values=k_values,
            max_images=max_images
        )

        if args.output:
            output_data = {name: m.to_dict() for name, m in results.items()}
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
    elif args.runs > 1:
        # Run with statistics
        metrics = run_with_statistics(
            args.image_dir,
            mode=args.mode,
            k_nearest=args.k,
            threshold=args.threshold,
            max_images=max_images,
            n_runs=args.runs
        )

        print("\n" + "=" * 60)
        print(f"Evaluation Results ({args.runs} runs)")
        print("=" * 60)
        for key, value in metrics.to_dict().items():
            if isinstance(value, float):
                if value < 0 and key in ("depth_l1", "depth_abs_rel", "depth_sq_rel", "depth_rmse"):
                    print(f"{key}: N/A (no --gt-dir provided)")
                elif key.endswith("_std"):
                    continue  # Print std with mean
                elif key == "inference_time_ms":
                    print(f"{key}: {value:.2f} ± {metrics.time_std:.2f}")
                elif key == "peak_memory_mb":
                    print(f"{key}: {value:.2f} ± {metrics.memory_std:.2f}")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        metrics = evaluate(
            args.image_dir,
            gt_dir=args.gt_dir,
            mode=args.mode,
            k_nearest=args.k,
            threshold=args.threshold,
            max_images=max_images,
        )

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for key, value in metrics.to_dict().items():
            if isinstance(value, float):
                # -1.0 means N/A (no ground-truth provided)
                if value < 0 and key in ("depth_l1", "depth_abs_rel", "depth_sq_rel", "depth_rmse"):
                    print(f"{key}: N/A (no --gt-dir provided)")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
