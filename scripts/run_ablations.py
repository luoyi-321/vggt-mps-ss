#!/usr/bin/env python3
"""
Ablation Study Driver Script — CO3D Real Image Edition

Runs systematic ablation studies on REAL CO3D dataset images (tv category).
Evaluates both efficiency and depth quality against CO3D ground-truth.

Ablation studies:
1. k-nearest: Test different k values (3, 5, 10, 15, 20, 30)
2. threshold: Test different τ values (0.5, 0.6, 0.7, 0.8, 0.9)
3. mask_type: Compare covisibility, random, and sliding window masks
4. soft_mask: Compare hard vs soft probabilistic masks
5. temperature: Test different soft mask temperatures

Usage:
    # Use real CO3D tv images (default)
    python scripts/run_ablations.py --ablation k_nearest
    python scripts/run_ablations.py --ablation threshold --tau-values 0.5,0.6,0.7,0.8,0.9
    python scripts/run_ablations.py --ablation mask_type
    python scripts/run_ablations.py --ablation soft_mask

    # Specify CO3D category and sequence
    python scripts/run_ablations.py --ablation k_nearest --co3d-category tv --sequence 396_49386_97450

    # Run across multiple CO3D sequences
    python scripts/run_ablations.py --ablation k_nearest --multi-sequence --max-sequences 5

    # Control number of views sampled per sequence
    python scripts/run_ablations.py --ablation k_nearest --num-views 6

    # Legacy: use a custom image directory
    python scripts/run_ablations.py --ablation k_nearest --image-dir data/real_data/bottle_cap --images 50
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


# ═══════════════════════════════════════════════════════════════
# CO3D Dataset Helpers
# ═══════════════════════════════════════════════════════════════

def find_co3d_sequences(co3d_dir: Path, max_sequences: int = 0) -> List[Path]:
    """Find all valid CO3D sequences (directories containing an images/ subfolder)."""
    sequences = []
    for d in sorted(co3d_dir.iterdir()):
        if d.is_dir() and (d / "images").is_dir():
            sequences.append(d)
    if max_sequences > 0:
        sequences = sequences[:max_sequences]
    return sequences


def select_co3d_images(
    sequence_dir: Path, num_views: int = 4
) -> List[Path]:
    """Select evenly-spaced frames from a CO3D sequence."""
    images_dir = sequence_dir / "images"
    all_frames = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not all_frames:
        return []
    if len(all_frames) <= num_views:
        return all_frames
    indices = np.linspace(0, len(all_frames) - 1, num_views, dtype=int)
    return [all_frames[i] for i in indices]


def prepare_co3d_image_dir(
    sequence_dir: Path, num_views: int = 4
) -> Path:
    """
    Create a temporary directory with symlinked CO3D images for evaluate().
    evaluate() expects a flat directory of images, so we symlink selected
    frames into a temp directory.
    """
    import tempfile

    frames = select_co3d_images(sequence_dir, num_views)
    if not frames:
        raise FileNotFoundError(f"No images in {sequence_dir / 'images'}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="co3d_ablation_"))
    for frame_path in frames:
        link = tmp_dir / frame_path.name
        link.symlink_to(frame_path.resolve())

    return tmp_dir


def load_co3d_gt_depths(
    sequence_dir: Path, frame_names: List[str]
) -> List[Optional[np.ndarray]]:
    """Load ground-truth depth maps from CO3D for specified frame names."""
    from PIL import Image

    depth_dir = sequence_dir / "depths"
    if not depth_dir.is_dir():
        return [None] * len(frame_names)

    results = []
    for fname in frame_names:
        # Try to match by frame number
        frame_num = "".join(c for c in Path(fname).stem if c.isdigit())
        found = False
        for df in sorted(depth_dir.iterdir()):
            file_num = "".join(c for c in df.stem if c.isdigit())
            if frame_num and file_num == frame_num:
                try:
                    results.append(np.array(Image.open(df)).astype(np.float32))
                    found = True
                    break
                except Exception:
                    continue
        if not found:
            results.append(None)
    return results


def compute_gt_depth_metrics(
    pred_depths: np.ndarray,
    gt_depths: List[Optional[np.ndarray]],
) -> Dict[str, float]:
    """
    Compute depth quality metrics against CO3D ground-truth.

    Args:
        pred_depths: VGGT predicted depths [1, N, H, W, 1]
        gt_depths: list of GT depth arrays (or None per frame)

    Returns:
        Dict with avg mae, rmse, abs_rel, and count of evaluated frames
    """
    from PIL import Image as PILImage

    all_mae, all_rmse, all_abs_rel = [], [], []
    for i, gt in enumerate(gt_depths):
        if gt is None or gt.max() <= 0:
            continue
        if i >= pred_depths.shape[1]:
            break

        pred = pred_depths[0, i, :, :, 0]

        # Resize prediction to match GT if shapes differ
        if pred.shape != gt.shape:
            pred = np.array(
                PILImage.fromarray(pred).resize(
                    (gt.shape[1], gt.shape[0]), PILImage.BILINEAR
                )
            )

        valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
        if valid.sum() < 10:
            continue

        p, g = pred[valid], gt[valid]
        diff = np.abs(p - g)
        all_mae.append(float(np.mean(diff)))
        all_rmse.append(float(np.sqrt(np.mean(diff**2))))
        all_abs_rel.append(float(np.mean(diff / (g + 1e-8))))

    if not all_mae:
        return {"gt_mae": -1.0, "gt_rmse": -1.0, "gt_abs_rel": -1.0, "gt_frames": 0}

    return {
        "gt_mae": float(np.mean(all_mae)),
        "gt_rmse": float(np.mean(all_rmse)),
        "gt_abs_rel": float(np.mean(all_abs_rel)),
        "gt_frames": len(all_mae),
    }


# ═══════════════════════════════════════════════════════════════
# Evaluate wrapper that returns both metrics + raw predictions
# ═══════════════════════════════════════════════════════════════

def evaluate_co3d(
    image_dir: Path,
    mode: str = "dense",
    k_nearest: int = 10,
    threshold: float = 0.7,
    max_images: int = 10,
    model=None,
) -> Dict[str, Any]:
    """
    Run VGGT on images and return metrics + raw depth predictions.
    Re-uses evaluate_vggt.evaluate() for timing/memory, then also
    returns raw depth for GT comparison.
    """
    from evaluate_vggt import evaluate, get_device, load_model, clear_memory
    import torch

    device = get_device()

    if model is None:
        model = load_model(device, mode, k_nearest, threshold)

    metrics = evaluate(
        image_dir, mode=mode, k_nearest=k_nearest,
        threshold=threshold, max_images=max_images, model=model
    )

    # Also get raw predictions for GT comparison
    from vggt.utils.load_fn import load_and_preprocess_images
    images = sorted(
        list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    )[:max_images]
    image_paths = [str(p) for p in images]
    input_tensor = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        preds = model(input_tensor)

    raw_depth = preds["depth"].cpu().numpy()
    del input_tensor, preds
    clear_memory()

    return {
        "metrics": metrics,
        "raw_depth": raw_depth,
        "image_names": [p.name for p in images],
    }


# ═══════════════════════════════════════════════════════════════
# Ablation runners — now CO3D aware
# ═══════════════════════════════════════════════════════════════

def run_k_nearest_ablation(
    image_dir: Path,
    k_values: List[int],
    max_images: int = 50,
    n_runs: int = 3,
    output_file: Optional[Path] = None,
    sequence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ablation study on k-nearest neighbors parameter.
    When sequence_dir is provided, also evaluates against CO3D GT depth.
    """
    from evaluate_vggt import run_with_statistics, clear_memory

    print("=" * 70)
    print("K-Nearest Ablation Study (CO3D)")
    print("=" * 70)
    print(f"K values: {k_values}")
    print(f"Images: {max_images}")
    print(f"Runs: {n_runs}")
    if sequence_dir:
        print(f"CO3D sequence: {sequence_dir.name}")

    results = {
        "ablation": "k_nearest",
        "k_values": k_values,
        "max_images": max_images,
        "n_runs": n_runs,
        "data_source": str(sequence_dir) if sequence_dir else str(image_dir),
        "configurations": {},
    }

    # --- Dense baseline ---
    print("\n--- Dense baseline ---")
    dense_result = evaluate_co3d(image_dir, mode="dense", max_images=max_images)
    dense_m = dense_result["metrics"]
    gt_metrics_dense = {"gt_mae": -1.0}
    if sequence_dir:
        gt_depths = load_co3d_gt_depths(sequence_dir, dense_result["image_names"])
        gt_metrics_dense = compute_gt_depth_metrics(dense_result["raw_depth"], gt_depths)

    results["configurations"]["dense"] = {
        "k": 0,
        "mode": "dense",
        "time_ms": dense_m.inference_time_ms,
        "memory_mb": dense_m.peak_memory_mb,
        "depth_l1": dense_m.depth_l1,
        **gt_metrics_dense,
    }
    print(f"  Time: {dense_m.inference_time_ms:.1f} ms | Memory: {dense_m.peak_memory_mb:.1f} MB")
    if gt_metrics_dense["gt_mae"] >= 0:
        print(f"  GT MAE: {gt_metrics_dense['gt_mae']:.4f} | GT RMSE: {gt_metrics_dense['gt_rmse']:.4f}")
    clear_memory()

    # --- Each k value ---
    for k in k_values:
        print(f"\n--- Sparse k={k} ---")

        metrics = run_with_statistics(
            image_dir, mode="sparse", k_nearest=k,
            max_images=max_images, n_runs=n_runs,
        )

        config = {
            "k": k,
            "mode": "sparse",
            "time_ms": metrics.inference_time_ms,
            "time_std": metrics.time_std,
            "memory_mb": metrics.peak_memory_mb,
            "sparsity": metrics.sparsity_ratio,
            "depth_l1": metrics.depth_l1,
        }

        # GT comparison
        if sequence_dir:
            sparse_result = evaluate_co3d(
                image_dir, mode="sparse", k_nearest=k, max_images=max_images
            )
            gt_depths = load_co3d_gt_depths(sequence_dir, sparse_result["image_names"])
            gt_m = compute_gt_depth_metrics(sparse_result["raw_depth"], gt_depths)
            config.update(gt_m)
            if gt_m["gt_mae"] >= 0:
                print(f"  GT MAE: {gt_m['gt_mae']:.4f} | GT RMSE: {gt_m['gt_rmse']:.4f}")

        results["configurations"][f"k={k}"] = config

        print(f"  Time: {metrics.inference_time_ms:.1f} ± {metrics.time_std:.1f} ms")
        print(f"  Memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"  Sparsity: {metrics.sparsity_ratio * 100:.1f}%")

        clear_memory()

    # Optimal k
    configs = results["configurations"]
    sparse_configs = {k: v for k, v in configs.items() if k != "dense"}
    if sparse_configs:
        best_k = min(sparse_configs.keys(), key=lambda x: sparse_configs[x]["time_ms"])
        results["optimal_k"] = sparse_configs[best_k]["k"]

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def run_threshold_ablation(
    image_dir: Path,
    tau_values: List[float],
    k_nearest: int = 10,
    max_images: int = 50,
    n_runs: int = 3,
    output_file: Optional[Path] = None,
    sequence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ablation study on covisibility threshold parameter.
    """
    from evaluate_vggt import evaluate, load_model, get_device, clear_memory

    print("=" * 70)
    print("Threshold Ablation Study (CO3D)")
    print("=" * 70)
    print(f"Tau values: {tau_values}")
    print(f"K: {k_nearest}")
    print(f"Images: {max_images}")
    if sequence_dir:
        print(f"CO3D sequence: {sequence_dir.name}")

    device = get_device()
    results = {
        "ablation": "threshold",
        "tau_values": tau_values,
        "k_nearest": k_nearest,
        "max_images": max_images,
        "data_source": str(sequence_dir) if sequence_dir else str(image_dir),
        "configurations": {},
    }

    for tau in tau_values:
        print(f"\n--- Testing τ={tau} ---")

        times = []
        sparsities = []

        for run in range(n_runs):
            clear_memory()
            model = load_model(device, mode="sparse", k_nearest=k_nearest, threshold=tau)
            metrics = evaluate(
                image_dir, mode="sparse", k_nearest=k_nearest,
                threshold=tau, max_images=max_images, model=model,
            )
            times.append(metrics.inference_time_ms)
            sparsities.append(metrics.sparsity_ratio)
            del model
            clear_memory()

        config = {
            "threshold": tau,
            "time_ms": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "sparsity": float(np.mean(sparsities)),
        }

        # GT comparison
        if sequence_dir:
            clear_memory()
            sparse_result = evaluate_co3d(
                image_dir, mode="sparse", k_nearest=k_nearest,
                threshold=tau, max_images=max_images,
            )
            gt_depths = load_co3d_gt_depths(sequence_dir, sparse_result["image_names"])
            gt_m = compute_gt_depth_metrics(sparse_result["raw_depth"], gt_depths)
            config.update(gt_m)
            if gt_m["gt_mae"] >= 0:
                print(f"  GT MAE: {gt_m['gt_mae']:.4f}")

        results["configurations"][f"tau={tau}"] = config

        print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} ms")
        print(f"  Sparsity: {np.mean(sparsities) * 100:.1f}%")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def run_mask_type_ablation(
    image_dir: Path,
    mask_types: List[str],
    k_nearest: int = 10,
    max_images: int = 50,
    n_runs: int = 3,
    output_file: Optional[Path] = None,
    sequence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ablation study comparing different mask types."""
    print("=" * 70)
    print("Mask Type Ablation Study (CO3D)")
    print("=" * 70)
    print(f"Mask types: {mask_types}")
    print(f"K: {k_nearest}, Images: {max_images}")
    if sequence_dir:
        print(f"CO3D sequence: {sequence_dir.name}")

    results = {
        "ablation": "mask_type",
        "mask_types": mask_types,
        "k_nearest": k_nearest,
        "max_images": max_images,
        "data_source": str(sequence_dir) if sequence_dir else str(image_dir),
        "configurations": {},
    }

    for mask_type in mask_types:
        print(f"\n--- Testing {mask_type} mask ---")

        if mask_type == "covisibility":
            from evaluate_vggt import run_with_statistics, clear_memory

            metrics = run_with_statistics(
                image_dir, mode="sparse", k_nearest=k_nearest,
                max_images=max_images, n_runs=n_runs,
            )
            config = {
                "type": mask_type,
                "time_ms": metrics.inference_time_ms,
                "time_std": metrics.time_std,
                "memory_mb": metrics.peak_memory_mb,
                "sparsity": metrics.sparsity_ratio,
            }

            if sequence_dir:
                sparse_result = evaluate_co3d(
                    image_dir, mode="sparse", k_nearest=k_nearest, max_images=max_images
                )
                gt_depths = load_co3d_gt_depths(sequence_dir, sparse_result["image_names"])
                gt_m = compute_gt_depth_metrics(sparse_result["raw_depth"], gt_depths)
                config.update(gt_m)

            results["configurations"][mask_type] = config
            clear_memory()

        elif mask_type == "random":
            n = max_images
            k = k_nearest
            sparsity = 1.0 - (n * k) / (n * (n - 1)) if n > 1 else 0.0
            results["configurations"][mask_type] = {
                "type": mask_type,
                "time_ms": 0.0,
                "sparsity": sparsity,
                "note": "Theoretical values — requires random mask implementation",
            }

        elif mask_type == "sliding_window":
            n = max_images
            w = k_nearest // 2
            connections = min(2 * w + 1, n)
            sparsity = 1.0 - (n * connections) / (n * (n - 1)) if n > 1 else 0.0
            results["configurations"][mask_type] = {
                "type": mask_type,
                "window_size": w,
                "time_ms": 0.0,
                "sparsity": sparsity,
                "note": "Theoretical values — requires sliding window implementation",
            }

        print(f"  Sparsity: {results['configurations'][mask_type].get('sparsity', 0) * 100:.1f}%")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def run_soft_mask_ablation(
    image_dir: Path,
    temperatures: List[float] = [0.05, 0.1, 0.2, 0.5],
    k_nearest: int = 10,
    max_images: int = 50,
    n_runs: int = 3,
    output_file: Optional[Path] = None,
    sequence_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ablation study comparing hard vs soft masks at different temperatures."""
    from evaluate_vggt import evaluate, load_model, get_device, clear_memory

    print("=" * 70)
    print("Soft Mask Ablation Study (CO3D)")
    print("=" * 70)
    print(f"Testing: hard mask vs soft mask at T={temperatures}")
    print(f"K: {k_nearest}, Images: {max_images}")
    if sequence_dir:
        print(f"CO3D sequence: {sequence_dir.name}")

    device = get_device()
    results = {
        "ablation": "soft_mask",
        "temperatures": temperatures,
        "k_nearest": k_nearest,
        "max_images": max_images,
        "data_source": str(sequence_dir) if sequence_dir else str(image_dir),
        "configurations": {},
    }

    # --- Hard mask ---
    print("\n--- Testing hard mask ---")
    times = []
    for run in range(n_runs):
        clear_memory()
        model = load_model(device, mode="sparse", k_nearest=k_nearest)
        metrics = evaluate(
            image_dir, mode="sparse", k_nearest=k_nearest,
            max_images=max_images, model=model,
        )
        times.append(metrics.inference_time_ms)
        del model
        clear_memory()

    config_hard = {
        "soft_mask": False,
        "temperature": None,
        "time_ms": float(np.mean(times)),
        "time_std": float(np.std(times)),
        "sparsity": metrics.sparsity_ratio,
    }

    if sequence_dir:
        hard_result = evaluate_co3d(
            image_dir, mode="sparse", k_nearest=k_nearest, max_images=max_images
        )
        gt_depths = load_co3d_gt_depths(sequence_dir, hard_result["image_names"])
        gt_m = compute_gt_depth_metrics(hard_result["raw_depth"], gt_depths)
        config_hard.update(gt_m)
        if gt_m["gt_mae"] >= 0:
            print(f"  GT MAE: {gt_m['gt_mae']:.4f}")

    results["configurations"]["hard"] = config_hard
    print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} ms")

    # --- Soft masks at each temperature ---
    for temp in temperatures:
        print(f"\n--- Testing soft mask (T={temp}) ---")

        times = []
        for run in range(n_runs):
            clear_memory()
            try:
                from vggt_mps.vggt_sparse_attention import make_vggt_sparse
                from vggt.models.vggt import VGGT
                import torch

                model_path = PROJECT_ROOT / "models" / "model.pt"
                if not model_path.exists():
                    print(f"  Model not found at {model_path}")
                    continue

                model = VGGT()
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint)
                model = model.to(device).eval()

                model = make_vggt_sparse(
                    model, device=str(device), k_nearest=k_nearest,
                    lightweight=True, soft_mask=True, temperature=temp,
                )

                metrics = evaluate(
                    image_dir, mode="sparse", k_nearest=k_nearest,
                    max_images=max_images, model=model,
                )
                times.append(metrics.inference_time_ms)

                del model
            except ImportError as e:
                print(f"  Import error: {e}")
                break

            clear_memory()

        if times:
            config_soft = {
                "soft_mask": True,
                "temperature": temp,
                "time_ms": float(np.mean(times)),
                "time_std": float(np.std(times)) if len(times) > 1 else 0.0,
            }

            if sequence_dir:
                try:
                    clear_memory()
                    import torch
                    from vggt_mps.vggt_sparse_attention import make_vggt_sparse
                    from vggt.models.vggt import VGGT

                    model = VGGT()
                    ckpt = torch.load(
                        PROJECT_ROOT / "models" / "model.pt",
                        map_location=device, weights_only=True,
                    )
                    model.load_state_dict(ckpt)
                    model = model.to(device).eval()
                    model = make_vggt_sparse(
                        model, device=str(device), k_nearest=k_nearest,
                        lightweight=True, soft_mask=True, temperature=temp,
                    )
                    soft_result = evaluate_co3d(
                        image_dir, mode="sparse", k_nearest=k_nearest, model=model,
                        max_images=max_images,
                    )
                    gt_depths = load_co3d_gt_depths(sequence_dir, soft_result["image_names"])
                    gt_m = compute_gt_depth_metrics(soft_result["raw_depth"], gt_depths)
                    config_soft.update(gt_m)
                    del model
                    clear_memory()
                except Exception as e:
                    print(f"  GT eval error: {e}")

            results["configurations"][f"soft_T={temp}"] = config_soft
            print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} ms")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


# ═══════════════════════════════════════════════════════════════
# Multi-sequence aggregation
# ═══════════════════════════════════════════════════════════════

def run_multi_sequence_ablation(
    ablation_func,
    co3d_dir: Path,
    max_sequences: int = 5,
    num_views: int = 4,
    output_file: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run an ablation across multiple CO3D sequences and aggregate results.
    """
    sequences = find_co3d_sequences(co3d_dir, max_sequences)
    if not sequences:
        print(f"❌ No CO3D sequences found in {co3d_dir}")
        sys.exit(1)

    print(f"\n🔁 Running ablation across {len(sequences)} CO3D sequences")
    all_seq_results = {}

    for seq in sequences:
        print(f"\n{'═' * 70}")
        print(f"📂 Sequence: {seq.name}")
        print(f"{'═' * 70}")

        try:
            tmp_image_dir = prepare_co3d_image_dir(seq, num_views)
            seq_result = ablation_func(
                image_dir=tmp_image_dir,
                sequence_dir=seq,
                max_images=num_views,
                **kwargs,
            )
            all_seq_results[seq.name] = seq_result

            # Cleanup temp dir
            import shutil
            shutil.rmtree(tmp_image_dir, ignore_errors=True)
        except Exception as e:
            print(f"  ⚠️ Error on {seq.name}: {e}")
            continue

    # Aggregate across sequences
    aggregated = {"sequences": all_seq_results, "n_sequences": len(all_seq_results)}

    # Compute per-config averages
    if all_seq_results:
        config_keys = set()
        for sr in all_seq_results.values():
            config_keys.update(sr.get("configurations", {}).keys())

        averages = {}
        for ck in config_keys:
            values = {}
            for sr in all_seq_results.values():
                cfg = sr.get("configurations", {}).get(ck, {})
                for metric_key in ["time_ms", "memory_mb", "sparsity", "gt_mae", "gt_rmse"]:
                    if metric_key in cfg and cfg[metric_key] is not None and cfg[metric_key] >= 0:
                        values.setdefault(metric_key, []).append(cfg[metric_key])
            averages[ck] = {k: float(np.mean(v)) for k, v in values.items()}

        aggregated["averaged_configs"] = averages

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\n📄 Aggregated results saved to: {output_file}")

    # Print summary table
    print(f"\n{'═' * 70}")
    print("📊 Aggregated Results Across Sequences")
    print(f"{'═' * 70}")
    print(f"{'Config':<20} {'Time (ms)':<12} {'Memory (MB)':<12} {'GT MAE':<10} {'GT RMSE':<10}")
    print("-" * 70)
    for ck, avg in aggregated.get("averaged_configs", {}).items():
        t = avg.get("time_ms", -1)
        m = avg.get("memory_mb", -1)
        mae = avg.get("gt_mae", -1)
        rmse = avg.get("gt_rmse", -1)
        t_s = f"{t:.1f}" if t >= 0 else "N/A"
        m_s = f"{m:.1f}" if m >= 0 else "N/A"
        mae_s = f"{mae:.4f}" if mae >= 0 else "N/A"
        rmse_s = f"{rmse:.4f}" if rmse >= 0 else "N/A"
        print(f"{ck:<20} {t_s:<12} {m_s:<12} {mae_s:<10} {rmse_s:<10}")
    print("-" * 70)

    return aggregated


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study driver for VGGT-MPS on CO3D dataset"
    )
    parser.add_argument(
        "--ablation", type=str, required=True,
        choices=["k_nearest", "threshold", "mask_type", "soft_mask"],
        help="Type of ablation study to run",
    )

    # ── CO3D options ───────────────────────────────────────────
    parser.add_argument(
        "--co3d-category", type=str, default="tv",
        help="CO3D category to use (default: tv)",
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="Specific CO3D sequence name (default: first available)",
    )
    parser.add_argument(
        "--num-views", type=int, default=4,
        help="Number of views to sample per sequence (default: 4)",
    )
    parser.add_argument(
        "--multi-sequence", action="store_true",
        help="Run across multiple sequences and aggregate",
    )
    parser.add_argument(
        "--max-sequences", type=int, default=5,
        help="Max sequences for --multi-sequence (default: 5)",
    )

    # ── Legacy / fallback ──────────────────────────────────────
    parser.add_argument(
        "--image-dir", type=Path, default=None,
        help="Custom image directory (overrides CO3D)",
    )
    parser.add_argument(
        "--images", type=int, default=50,
        help="Number of images to process (for --image-dir mode)",
    )

    # ── Shared options ─────────────────────────────────────────
    parser.add_argument("--runs", type=int, default=3, help="Runs per configuration")
    parser.add_argument("--k-values", type=str, default="3,5,10,15,20,30")
    parser.add_argument("--tau-values", type=str, default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--mask-types", type=str, default="covisibility,random,sliding_window")
    parser.add_argument("--temperatures", type=str, default="0.05,0.1,0.2,0.5")
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    # Parse lists
    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    tau_values = [float(x.strip()) for x in args.tau_values.split(",")]
    mask_types = [x.strip() for x in args.mask_types.split(",")]
    temperatures = [float(x.strip()) for x in args.temperatures.split(",")]

    # Default output
    if args.output is None:
        args.output = PROJECT_ROOT / "results" / f"ablation_{args.ablation}_co3d.json"

    # ── Resolve image source ───────────────────────────────────
    sequence_dir = None  # CO3D sequence for GT comparison

    if args.image_dir:
        # Legacy mode: custom directory
        image_dir = args.image_dir
        max_images = args.images
        print(f"📂 Using custom images: {image_dir}")
    else:
        # CO3D mode
        co3d_dir = PROJECT_ROOT / "co3d-main" / args.co3d_category
        if not co3d_dir.exists():
            print(f"❌ CO3D category not found: {co3d_dir}")
            print("   Download first: cd co3d-main && python co3d/download_dataset.py ...")
            sys.exit(1)

        if args.multi_sequence:
            # Multi-sequence mode — delegate to aggregation wrapper
            ablation_funcs = {
                "k_nearest": run_k_nearest_ablation,
                "threshold": run_threshold_ablation,
                "mask_type": run_mask_type_ablation,
                "soft_mask": run_soft_mask_ablation,
            }
            extra_kwargs = {
                "k_nearest": {"k_values": k_values, "n_runs": args.runs},
                "threshold": {"tau_values": tau_values, "k_nearest": k_values[0] if k_values else 10, "n_runs": args.runs},
                "mask_type": {"mask_types": mask_types, "k_nearest": k_values[0] if k_values else 10, "n_runs": args.runs},
                "soft_mask": {"temperatures": temperatures, "k_nearest": k_values[0] if k_values else 10, "n_runs": args.runs},
            }

            run_multi_sequence_ablation(
                ablation_funcs[args.ablation],
                co3d_dir=co3d_dir,
                max_sequences=args.max_sequences,
                num_views=args.num_views,
                output_file=args.output,
                **extra_kwargs[args.ablation],
            )
            return

        # Single sequence mode
        sequences = find_co3d_sequences(co3d_dir)
        if not sequences:
            print(f"❌ No sequences in {co3d_dir}")
            sys.exit(1)

        if args.sequence:
            sequence_dir = co3d_dir / args.sequence
            if not (sequence_dir / "images").is_dir():
                print(f"❌ Sequence not found: {args.sequence}")
                sys.exit(1)
        else:
            sequence_dir = sequences[0]

        print(f"📂 CO3D category: {args.co3d_category}")
        print(f"🎯 Sequence: {sequence_dir.name}")
        print(f"📸 Sampling {args.num_views} views")

        image_dir = prepare_co3d_image_dir(sequence_dir, args.num_views)
        max_images = args.num_views

    # ── Run the ablation ───────────────────────────────────────
    if args.ablation == "k_nearest":
        run_k_nearest_ablation(
            image_dir, k_values=k_values, max_images=max_images,
            n_runs=args.runs, output_file=args.output, sequence_dir=sequence_dir,
        )
    elif args.ablation == "threshold":
        run_threshold_ablation(
            image_dir, tau_values=tau_values,
            max_images=max_images, n_runs=args.runs,
            output_file=args.output, sequence_dir=sequence_dir,
        )
    elif args.ablation == "mask_type":
        run_mask_type_ablation(
            image_dir, mask_types=mask_types,
            max_images=max_images, n_runs=args.runs,
            output_file=args.output, sequence_dir=sequence_dir,
        )
    elif args.ablation == "soft_mask":
        run_soft_mask_ablation(
            image_dir, temperatures=temperatures,
            max_images=max_images, n_runs=args.runs,
            output_file=args.output, sequence_dir=sequence_dir,
        )

    # Cleanup temp dir if using CO3D
    if not args.image_dir and sequence_dir:
        import shutil
        shutil.rmtree(image_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
