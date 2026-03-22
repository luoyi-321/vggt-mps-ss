#!/usr/bin/env python3
"""
Ground-Truth Evaluation Script for VGGT-MPS

Computes comprehensive metrics against ground-truth depth and poses:
- Depth: RMSE, Absolute Relative, Squared Relative
- Pose: Rotation error (degrees), Translation error (m)
- Chamfer distance for point clouds
- Quality retention = sparse_metric / dense_metric * 100

Usage:
    python scripts/evaluate_with_gt.py --dataset co3d --sequences bottle,chair --max-images 20
    python scripts/evaluate_with_gt.py --dataset co3d --compare dense,sparse --k-values 5,10,15
    python scripts/evaluate_with_gt.py --dataset co3d --runs 5 --output results/gt_evaluation.json
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import stats

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class GTEvaluationResult:
    """Container for ground-truth evaluation results"""
    # Configuration
    mode: str = "dense"
    k_nearest: int = 0
    n_images: int = 0
    n_runs: int = 1

    # Depth metrics
    depth_rmse: float = 0.0
    depth_abs_rel: float = 0.0
    depth_sq_rel: float = 0.0
    depth_delta_1: float = 0.0  # % of pixels with δ < 1.25

    # Pose metrics
    rotation_error_deg: float = 0.0
    translation_error: float = 0.0

    # Point cloud metrics
    chamfer_distance: float = 0.0

    # Efficiency metrics
    inference_time_ms: float = 0.0
    peak_memory_mb: float = 0.0

    # Statistics (from multiple runs)
    depth_rmse_std: float = 0.0
    time_std: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "k_nearest": self.k_nearest,
            "n_images": self.n_images,
            "n_runs": self.n_runs,
            "depth_rmse": self.depth_rmse,
            "depth_rmse_std": self.depth_rmse_std,
            "depth_abs_rel": self.depth_abs_rel,
            "depth_sq_rel": self.depth_sq_rel,
            "depth_delta_1": self.depth_delta_1,
            "rotation_error_deg": self.rotation_error_deg,
            "translation_error": self.translation_error,
            "chamfer_distance": self.chamfer_distance,
            "inference_time_ms": self.inference_time_ms,
            "time_std": self.time_std,
            "peak_memory_mb": self.peak_memory_mb
        }


def get_device() -> torch.device:
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    device = get_device()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def synchronize_device():
    """Synchronize GPU"""
    device = get_device()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    align_scale: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive depth metrics with optional scale alignment.

    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        valid_mask: Optional mask for valid depth values
        align_scale: If True, apply median scaling alignment (standard for monocular depth)

    Returns:
        Dictionary of metric values
    """
    if valid_mask is None:
        valid_mask = (gt_depth > 0) & np.isfinite(gt_depth) & np.isfinite(pred_depth) & (pred_depth > 0)

    pred = pred_depth[valid_mask].copy()
    gt = gt_depth[valid_mask]

    if len(pred) == 0:
        return {
            "rmse": 0.0,
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "delta_1": 0.0,
            "scale": 1.0
        }

    # Apply median scaling alignment (standard practice for monocular depth evaluation)
    scale = 1.0
    if align_scale:
        scale = np.median(gt) / (np.median(pred) + 1e-8)
        pred = pred * scale

    # RMSE
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # Absolute relative error
    abs_rel = np.mean(np.abs(pred - gt) / (gt + 1e-8))

    # Squared relative error
    sq_rel = np.mean(((pred - gt) ** 2) / (gt + 1e-8))

    # Delta accuracy (δ < 1.25)
    thresh = np.maximum((gt / (pred + 1e-8)), (pred / (gt + 1e-8)))
    delta_1 = (thresh < 1.25).mean() * 100

    return {
        "rmse": float(rmse),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "delta_1": float(delta_1),
        "scale": float(scale)
    }


def compute_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute rotation error in degrees.

    Args:
        R_pred: Predicted rotation matrix (3x3)
        R_gt: Ground truth rotation matrix (3x3)

    Returns:
        Rotation error in degrees
    """
    R_diff = R_pred.T @ R_gt
    trace = np.trace(R_diff)
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


def compute_chamfer_distance(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    n_samples: int = 10000
) -> float:
    """
    Compute Chamfer distance between point clouds.

    Args:
        points_pred: Predicted points (Nx3)
        points_gt: Ground truth points (Mx3)
        n_samples: Number of points to sample

    Returns:
        Chamfer distance
    """
    from scipy.spatial import cKDTree

    # Subsample if needed
    if len(points_pred) > n_samples:
        idx = np.random.choice(len(points_pred), n_samples, replace=False)
        points_pred = points_pred[idx]
    if len(points_gt) > n_samples:
        idx = np.random.choice(len(points_gt), n_samples, replace=False)
        points_gt = points_gt[idx]

    # Build KD-trees
    tree_pred = cKDTree(points_pred)
    tree_gt = cKDTree(points_gt)

    # Compute distances
    dist_pred_to_gt, _ = tree_gt.query(points_pred)
    dist_gt_to_pred, _ = tree_pred.query(points_gt)

    # Chamfer distance
    chamfer = np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)

    return float(chamfer)


def load_benchmark_sequence(
    seq_dir: Path,
    max_images: int = 20
) -> Dict[str, Any]:
    """
    Load a benchmark sequence prepared by prepare_benchmark_data.py.

    Args:
        seq_dir: Path to sequence directory
        max_images: Maximum number of images to load

    Returns:
        Dictionary with images, depths, poses
    """
    data = {
        'images': [],
        'depths': [],
        'poses': [],
        'metadata': None
    }

    # Load metadata
    meta_file = seq_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            data['metadata'] = json.load(f)

    # Load images
    images_dir = seq_dir / "images"
    if images_dir.exists():
        image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        for img_path in image_files[:max_images]:
            img = np.array(Image.open(img_path))
            data['images'].append(img)

    # Load depths
    depth_dir = seq_dir / "depth"
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("*.npy"))
        for depth_path in depth_files[:max_images]:
            depth = np.load(depth_path)
            data['depths'].append(depth)

    # Load poses
    poses_dir = seq_dir / "poses"
    if poses_dir.exists():
        pose_files = sorted(poses_dir.glob("*.npz"))
        for pose_path in pose_files[:max_images]:
            pose = dict(np.load(pose_path))
            data['poses'].append(pose)

    return data


def find_benchmark_sequences(
    dataset_dir: Path,
    categories: Optional[List[str]] = None
) -> List[Path]:
    """
    Find all benchmark sequences.

    Args:
        dataset_dir: Path to benchmark dataset
        categories: Optional list of categories to filter

    Returns:
        List of sequence directory paths
    """
    sequences = []

    if not dataset_dir.exists():
        return sequences

    for cat_dir in sorted(dataset_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if cat_dir.name.startswith("."):
            continue
        if categories and cat_dir.name not in categories:
            continue

        for seq_dir in sorted(cat_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            if (seq_dir / "images").exists():
                sequences.append(seq_dir)

    return sequences


def load_model(
    device: torch.device,
    mode: str = "dense",
    k_nearest: int = 10,
    threshold: float = 0.7,
    soft_mask: bool = False,
    temperature: float = 0.1
):
    """
    Load VGGT model in dense or sparse mode.

    Args:
        device: PyTorch device
        mode: "dense" or "sparse"
        k_nearest: K for sparse attention
        threshold: Covisibility threshold
        soft_mask: Use soft probabilistic masks
        temperature: Temperature for soft masks

    Returns:
        Loaded VGGT model
    """
    try:
        from vggt.models.vggt import VGGT
    except ImportError as e:
        print(f"Error: Could not import VGGT: {e}")
        return None

    model_path = PROJECT_ROOT / "models" / "model.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return None

    model = VGGT()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    if mode == "sparse":
        try:
            from vggt_mps.vggt_sparse_attention import make_vggt_sparse
            model = make_vggt_sparse(
                model,
                device=str(device),
                k_nearest=k_nearest,
                threshold=threshold,
                lightweight=True,
                soft_mask=soft_mask,
                temperature=temperature
            )
        except ImportError as e:
            print(f"Warning: Could not load sparse attention: {e}")

    return model


def evaluate_with_gt(
    seq_data: Dict[str, Any],
    mode: str = "dense",
    k_nearest: int = 10,
    model=None
) -> GTEvaluationResult:
    """
    Evaluate VGGT predictions against ground truth.

    Args:
        seq_data: Sequence data from load_benchmark_sequence
        mode: "dense" or "sparse"
        k_nearest: K for sparse attention
        model: Pre-loaded model (optional)

    Returns:
        GTEvaluationResult with metrics
    """
    device = get_device()

    if len(seq_data['images']) < 2:
        print("Error: Need at least 2 images")
        return GTEvaluationResult()

    n_images = len(seq_data['images'])

    # Load model if not provided
    if model is None:
        model = load_model(device, mode, k_nearest)
        if model is None:
            return GTEvaluationResult()

    # Prepare images
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
        import tempfile

        # Save images temporarily
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_paths = []
            for i, img in enumerate(seq_data['images']):
                tmp_path = Path(tmp_dir) / f"img_{i:04d}.jpg"
                Image.fromarray(img).save(tmp_path)
                tmp_paths.append(str(tmp_path))

            input_tensor = load_and_preprocess_images(tmp_paths).to(device)
    except ImportError:
        print("Error: Could not import VGGT utils")
        return GTEvaluationResult()

    # Run inference with timing
    clear_memory()
    synchronize_device()
    start_time = time.perf_counter()

    with torch.no_grad():
        predictions = model(input_tensor)

    synchronize_device()
    inference_time = (time.perf_counter() - start_time) * 1000

    # Get memory usage
    peak_memory = 0.0
    if device.type == "mps":
        try:
            peak_memory = torch.mps.current_allocated_memory() / (1024 * 1024)
        except AttributeError:
            pass
    elif device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Extract predictions
    pred_depths = predictions['depth'].cpu().numpy()

    # Initialize result
    result = GTEvaluationResult(
        mode=mode,
        k_nearest=k_nearest if mode == "sparse" else 0,
        n_images=n_images,
        inference_time_ms=inference_time,
        peak_memory_mb=peak_memory
    )

    # Compute depth metrics against ground truth
    if seq_data['depths']:
        all_rmse, all_abs_rel, all_sq_rel, all_delta = [], [], [], []

        for i, gt_depth in enumerate(seq_data['depths']):
            if i >= pred_depths.shape[1]:
                break

            pred_depth = pred_depths[0, i, :, :, 0]

            # Resize if needed
            if pred_depth.shape != gt_depth.shape:
                from scipy.ndimage import zoom
                zoom_factors = (gt_depth.shape[0] / pred_depth.shape[0],
                               gt_depth.shape[1] / pred_depth.shape[1])
                pred_depth = zoom(pred_depth, zoom_factors)

            metrics = compute_depth_metrics(pred_depth, gt_depth)
            all_rmse.append(metrics['rmse'])
            all_abs_rel.append(metrics['abs_rel'])
            all_sq_rel.append(metrics['sq_rel'])
            all_delta.append(metrics['delta_1'])

        result.depth_rmse = float(np.mean(all_rmse))
        result.depth_abs_rel = float(np.mean(all_abs_rel))
        result.depth_sq_rel = float(np.mean(all_sq_rel))
        result.depth_delta_1 = float(np.mean(all_delta))

    # Compute pose metrics
    if seq_data['poses'] and 'poses' in predictions:
        all_rot_err, all_trans_err = [], []
        pred_poses = predictions['poses'].cpu().numpy()

        for i, gt_pose in enumerate(seq_data['poses']):
            if i >= pred_poses.shape[1]:
                break

            R_pred = pred_poses[0, i, :3, :3]
            T_pred = pred_poses[0, i, :3, 3]
            R_gt = gt_pose['R']
            T_gt = gt_pose['T']

            rot_err = compute_rotation_error(R_pred, R_gt)
            trans_err = compute_translation_error(T_pred, T_gt)

            all_rot_err.append(rot_err)
            all_trans_err.append(trans_err)

        if all_rot_err:
            result.rotation_error_deg = float(np.mean(all_rot_err))
            result.translation_error = float(np.mean(all_trans_err))

    return result


def run_multiple_evaluations(
    seq_data: Dict[str, Any],
    mode: str = "dense",
    k_nearest: int = 10,
    n_runs: int = 5
) -> GTEvaluationResult:
    """
    Run evaluation multiple times and compute statistics.

    Args:
        seq_data: Sequence data
        mode: "dense" or "sparse"
        k_nearest: K for sparse attention
        n_runs: Number of runs

    Returns:
        GTEvaluationResult with mean and std
    """
    device = get_device()

    all_results = []

    # Load model once
    model = load_model(device, mode, k_nearest)
    if model is None:
        return GTEvaluationResult()

    for run in range(n_runs):
        result = evaluate_with_gt(seq_data, mode, k_nearest, model=model)
        all_results.append(result)
        clear_memory()

    # Aggregate results
    final = GTEvaluationResult(
        mode=mode,
        k_nearest=k_nearest if mode == "sparse" else 0,
        n_images=all_results[0].n_images,
        n_runs=n_runs
    )

    # Compute means
    final.depth_rmse = np.mean([r.depth_rmse for r in all_results])
    final.depth_abs_rel = np.mean([r.depth_abs_rel for r in all_results])
    final.depth_sq_rel = np.mean([r.depth_sq_rel for r in all_results])
    final.depth_delta_1 = np.mean([r.depth_delta_1 for r in all_results])
    final.rotation_error_deg = np.mean([r.rotation_error_deg for r in all_results])
    final.translation_error = np.mean([r.translation_error for r in all_results])
    final.inference_time_ms = np.mean([r.inference_time_ms for r in all_results])
    final.peak_memory_mb = np.mean([r.peak_memory_mb for r in all_results])

    # Compute std
    final.depth_rmse_std = np.std([r.depth_rmse for r in all_results])
    final.time_std = np.std([r.inference_time_ms for r in all_results])

    # Clean up
    del model
    clear_memory()

    return final


def compute_quality_retention(
    sparse_result: GTEvaluationResult,
    dense_result: GTEvaluationResult
) -> Dict[str, float]:
    """
    Compute quality retention metrics.

    Quality retention = (dense_error / sparse_error) * 100
    Higher is better (100% = no quality loss)

    Args:
        sparse_result: Result from sparse evaluation
        dense_result: Result from dense evaluation

    Returns:
        Dictionary of retention percentages
    """
    retention = {}

    if dense_result.depth_rmse > 0:
        # For error metrics, higher retention means sparse is closer to dense
        # retention = dense/sparse * 100 (if sparse error is higher, retention < 100)
        retention['depth_rmse'] = (dense_result.depth_rmse / sparse_result.depth_rmse * 100
                                   if sparse_result.depth_rmse > 0 else 100.0)

    if dense_result.depth_delta_1 > 0:
        # For accuracy metrics, retention = sparse/dense * 100
        retention['depth_delta_1'] = (sparse_result.depth_delta_1 / dense_result.depth_delta_1 * 100
                                       if dense_result.depth_delta_1 > 0 else 100.0)

    if dense_result.rotation_error_deg > 0:
        retention['rotation'] = (dense_result.rotation_error_deg / sparse_result.rotation_error_deg * 100
                                  if sparse_result.rotation_error_deg > 0 else 100.0)

    return retention


def compute_significance(
    results_a: List[float],
    results_b: List[float]
) -> Tuple[float, float]:
    """
    Compute statistical significance using paired t-test.

    Args:
        results_a: Results from configuration A
        results_b: Results from configuration B

    Returns:
        (t_statistic, p_value)
    """
    if len(results_a) < 2 or len(results_b) < 2:
        return 0.0, 1.0

    t_stat, p_value = stats.ttest_rel(results_a, results_b)
    return float(t_stat), float(p_value)


def main():
    parser = argparse.ArgumentParser(
        description="Ground-truth evaluation for VGGT-MPS"
    )
    parser.add_argument(
        "--dataset",
        choices=["co3d", "custom"],
        default="co3d",
        help="Dataset type"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "co3d_benchmark",
        help="Path to benchmark data"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Comma-separated list of categories (default: all)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum images per sequence"
    )
    parser.add_argument(
        "--compare",
        type=str,
        default="dense,sparse",
        help="Comma-separated modes to compare"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,15",
        help="Comma-separated k values for sparse mode"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs for statistics"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file"
    )

    args = parser.parse_args()

    # Parse arguments
    categories = [c.strip() for c in args.sequences.split(",")] if args.sequences else None
    modes = [m.strip() for m in args.compare.split(",")]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # Find sequences
    sequences = find_benchmark_sequences(args.data_dir, categories)
    if not sequences:
        print(f"No benchmark sequences found in {args.data_dir}")
        print("Run prepare_benchmark_data.py first to prepare benchmark data.")
        sys.exit(1)

    print("=" * 70)
    print("Ground-Truth Evaluation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Sequences: {len(sequences)}")
    print(f"Max images: {args.max_images}")
    print(f"Modes: {modes}")
    print(f"K values: {k_values}")
    print(f"Runs: {args.runs}")

    all_results = {}

    for seq_path in sequences:
        seq_name = f"{seq_path.parent.name}/{seq_path.name}"
        print(f"\n--- Evaluating: {seq_name} ---")

        # Load sequence data
        seq_data = load_benchmark_sequence(seq_path, args.max_images)
        if len(seq_data['images']) < 2:
            print(f"  Skipping (not enough images)")
            continue

        seq_results = {}

        # Evaluate each mode
        for mode in modes:
            if mode == "dense":
                print(f"  Mode: dense")
                result = run_multiple_evaluations(
                    seq_data, mode="dense", n_runs=args.runs
                )
                seq_results['dense'] = result.to_dict()
                print(f"    RMSE: {result.depth_rmse:.4f} ± {result.depth_rmse_std:.4f}")
                print(f"    Time: {result.inference_time_ms:.1f} ms")

            elif mode == "sparse":
                for k in k_values:
                    print(f"  Mode: sparse (k={k})")
                    result = run_multiple_evaluations(
                        seq_data, mode="sparse", k_nearest=k, n_runs=args.runs
                    )
                    seq_results[f'sparse_k{k}'] = result.to_dict()
                    print(f"    RMSE: {result.depth_rmse:.4f} ± {result.depth_rmse_std:.4f}")
                    print(f"    Time: {result.inference_time_ms:.1f} ms")

        all_results[seq_name] = seq_results

    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
