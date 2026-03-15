#!/usr/bin/env python3
"""
Ablation Study Driver Script

Thin wrapper around benchmark.py ablation modes for systematic ablation studies.

Ablation studies:
1. k-nearest: Test different k values (3, 5, 10, 15, 20, 30)
2. threshold: Test different τ values (0.5, 0.6, 0.7, 0.8, 0.9)
3. mask_type: Compare covisibility, random, and sliding window masks
4. soft_mask: Compare hard vs soft probabilistic masks
5. temperature: Test different soft mask temperatures

Usage:
    python scripts/run_ablations.py --ablation k_nearest --k-values 3,5,10,15,20,30 --images 50
    python scripts/run_ablations.py --ablation threshold --tau-values 0.5,0.6,0.7,0.8,0.9 --images 50
    python scripts/run_ablations.py --ablation mask_type --mask-types covisibility,random,sliding_window --images 50
    python scripts/run_ablations.py --ablation soft_mask --images 50
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


def run_k_nearest_ablation(
    image_dir: Path,
    k_values: List[int],
    max_images: int = 50,
    n_runs: int = 3,
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ablation study on k-nearest neighbors parameter.

    Args:
        image_dir: Directory with input images
        k_values: List of k values to test
        max_images: Number of images to process
        n_runs: Number of runs per configuration
        output_file: Optional output file for results

    Returns:
        Dictionary with ablation results
    """
    from evaluate_vggt import run_with_statistics, clear_memory

    print("=" * 70)
    print("K-Nearest Ablation Study")
    print("=" * 70)
    print(f"K values: {k_values}")
    print(f"Images: {max_images}")
    print(f"Runs: {n_runs}")

    results = {
        "ablation": "k_nearest",
        "k_values": k_values,
        "max_images": max_images,
        "n_runs": n_runs,
        "configurations": {}
    }

    for k in k_values:
        print(f"\n--- Testing k={k} ---")

        metrics = run_with_statistics(
            image_dir,
            mode="sparse",
            k_nearest=k,
            max_images=max_images,
            n_runs=n_runs
        )

        results["configurations"][f"k={k}"] = {
            "k": k,
            "time_ms": metrics.inference_time_ms,
            "time_std": metrics.time_std,
            "memory_mb": metrics.peak_memory_mb,
            "sparsity": metrics.sparsity_ratio,
            "depth_l1": metrics.depth_l1
        }

        print(f"  Time: {metrics.inference_time_ms:.1f} ± {metrics.time_std:.1f} ms")
        print(f"  Memory: {metrics.peak_memory_mb:.1f} MB")
        print(f"  Sparsity: {metrics.sparsity_ratio*100:.1f}%")

        clear_memory()

    # Compute optimal k (best time-quality tradeoff)
    configs = results["configurations"]
    best_k = min(configs.keys(), key=lambda x: configs[x]["time_ms"])
    results["optimal_k"] = configs[best_k]["k"]

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
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ablation study on covisibility threshold parameter.

    Args:
        image_dir: Directory with input images
        tau_values: List of threshold values to test
        k_nearest: K value for sparse attention
        max_images: Number of images to process
        n_runs: Number of runs per configuration
        output_file: Optional output file for results

    Returns:
        Dictionary with ablation results
    """
    from evaluate_vggt import evaluate, load_model, get_device, clear_memory

    print("=" * 70)
    print("Threshold Ablation Study")
    print("=" * 70)
    print(f"Tau values: {tau_values}")
    print(f"K: {k_nearest}")
    print(f"Images: {max_images}")

    device = get_device()
    results = {
        "ablation": "threshold",
        "tau_values": tau_values,
        "k_nearest": k_nearest,
        "max_images": max_images,
        "configurations": {}
    }

    for tau in tau_values:
        print(f"\n--- Testing τ={tau} ---")

        times = []
        sparsities = []

        for run in range(n_runs):
            clear_memory()
            model = load_model(device, mode="sparse", k_nearest=k_nearest, threshold=tau)
            metrics = evaluate(
                image_dir,
                mode="sparse",
                k_nearest=k_nearest,
                threshold=tau,
                max_images=max_images,
                model=model
            )
            times.append(metrics.inference_time_ms)
            sparsities.append(metrics.sparsity_ratio)
            del model
            clear_memory()

        results["configurations"][f"tau={tau}"] = {
            "threshold": tau,
            "time_ms": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "sparsity": float(np.mean(sparsities))
        }

        print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} ms")
        print(f"  Sparsity: {np.mean(sparsities)*100:.1f}%")

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
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ablation study comparing different mask types.

    Mask types:
    - covisibility: Our method using MegaLoc features
    - random: Random k connections per image
    - sliding_window: Fixed temporal window (± w frames)

    Args:
        image_dir: Directory with input images
        mask_types: List of mask types to test
        k_nearest: K value for sparse attention
        max_images: Number of images to process
        n_runs: Number of runs per configuration
        output_file: Optional output file for results

    Returns:
        Dictionary with ablation results
    """
    print("=" * 70)
    print("Mask Type Ablation Study")
    print("=" * 70)
    print(f"Mask types: {mask_types}")
    print(f"K: {k_nearest}")
    print(f"Images: {max_images}")

    results = {
        "ablation": "mask_type",
        "mask_types": mask_types,
        "k_nearest": k_nearest,
        "max_images": max_images,
        "configurations": {}
    }

    for mask_type in mask_types:
        print(f"\n--- Testing {mask_type} mask ---")

        if mask_type == "covisibility":
            # Use our sparse attention (MegaLoc-based)
            from evaluate_vggt import run_with_statistics, clear_memory
            metrics = run_with_statistics(
                image_dir,
                mode="sparse",
                k_nearest=k_nearest,
                max_images=max_images,
                n_runs=n_runs
            )
            results["configurations"][mask_type] = {
                "type": mask_type,
                "time_ms": metrics.inference_time_ms,
                "time_std": metrics.time_std,
                "memory_mb": metrics.peak_memory_mb,
                "sparsity": metrics.sparsity_ratio
            }
            clear_memory()

        elif mask_type == "random":
            # Simulate random mask (for comparison baseline)
            # This would require modifying the sparse attention module
            print("  [Random mask simulation - using theoretical values]")
            n = max_images
            k = k_nearest
            sparsity = 1.0 - (n * k) / (n * (n - 1))
            results["configurations"][mask_type] = {
                "type": mask_type,
                "time_ms": 0.0,  # Would need actual implementation
                "sparsity": sparsity,
                "note": "Theoretical values - requires implementation"
            }

        elif mask_type == "sliding_window":
            # Simulate sliding window mask
            print("  [Sliding window simulation - using theoretical values]")
            n = max_images
            w = k_nearest // 2  # Window size
            connections_per_image = min(2 * w + 1, n)
            sparsity = 1.0 - (n * connections_per_image) / (n * (n - 1))
            results["configurations"][mask_type] = {
                "type": mask_type,
                "window_size": w,
                "time_ms": 0.0,
                "sparsity": sparsity,
                "note": "Theoretical values - requires implementation"
            }

        print(f"  Sparsity: {results['configurations'][mask_type].get('sparsity', 0)*100:.1f}%")

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
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Ablation study comparing hard vs soft masks at different temperatures.

    Args:
        image_dir: Directory with input images
        temperatures: List of temperature values for soft mask
        k_nearest: K value for sparse attention
        max_images: Number of images to process
        n_runs: Number of runs per configuration
        output_file: Optional output file for results

    Returns:
        Dictionary with ablation results
    """
    from evaluate_vggt import evaluate, load_model, get_device, clear_memory

    print("=" * 70)
    print("Soft Mask Ablation Study")
    print("=" * 70)
    print(f"Testing: hard mask vs soft mask at T={temperatures}")
    print(f"K: {k_nearest}")
    print(f"Images: {max_images}")

    device = get_device()
    results = {
        "ablation": "soft_mask",
        "temperatures": temperatures,
        "k_nearest": k_nearest,
        "max_images": max_images,
        "configurations": {}
    }

    # Test hard mask
    print("\n--- Testing hard mask ---")
    times = []
    for run in range(n_runs):
        clear_memory()
        model = load_model(device, mode="sparse", k_nearest=k_nearest)
        metrics = evaluate(
            image_dir,
            mode="sparse",
            k_nearest=k_nearest,
            max_images=max_images,
            model=model
        )
        times.append(metrics.inference_time_ms)
        del model
        clear_memory()

    results["configurations"]["hard"] = {
        "soft_mask": False,
        "temperature": None,
        "time_ms": float(np.mean(times)),
        "time_std": float(np.std(times)),
        "sparsity": metrics.sparsity_ratio
    }
    print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} ms")

    # Test soft masks at different temperatures
    for temp in temperatures:
        print(f"\n--- Testing soft mask (T={temp}) ---")

        times = []
        for run in range(n_runs):
            clear_memory()
            # Load model with soft mask enabled
            try:
                from vggt_mps.vggt_sparse_attention import make_vggt_sparse
                from vggt.models.vggt import VGGT

                model_path = PROJECT_ROOT / "models" / "model.pt"
                if not model_path.exists():
                    print(f"  Model not found at {model_path}")
                    continue

                import torch
                model = VGGT()
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                model.load_state_dict(checkpoint)
                model = model.to(device)
                model.eval()

                model = make_vggt_sparse(
                    model,
                    device=str(device),
                    k_nearest=k_nearest,
                    lightweight=True,
                    soft_mask=True,
                    temperature=temp
                )

                metrics = evaluate(
                    image_dir,
                    mode="sparse",
                    k_nearest=k_nearest,
                    max_images=max_images,
                    model=model
                )
                times.append(metrics.inference_time_ms)

                del model
            except ImportError as e:
                print(f"  Import error: {e}")
                break

            clear_memory()

        if times:
            results["configurations"][f"soft_T={temp}"] = {
                "soft_mask": True,
                "temperature": temp,
                "time_ms": float(np.mean(times)),
                "time_std": float(np.std(times)) if len(times) > 1 else 0.0
            }
            print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} ms")

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study driver for VGGT-MPS"
    )
    parser.add_argument(
        "--ablation",
        type=str,
        required=True,
        choices=["k_nearest", "threshold", "mask_type", "soft_mask"],
        help="Type of ablation study to run"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_data" / "bottle_cap",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=50,
        help="Number of images to process"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per configuration"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="3,5,10,15,20,30",
        help="Comma-separated k values for k_nearest ablation"
    )
    parser.add_argument(
        "--tau-values",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated threshold values for threshold ablation"
    )
    parser.add_argument(
        "--mask-types",
        type=str,
        default="covisibility,random,sliding_window",
        help="Comma-separated mask types for mask_type ablation"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.05,0.1,0.2,0.5",
        help="Comma-separated temperatures for soft_mask ablation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Parse list arguments
    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    tau_values = [float(x.strip()) for x in args.tau_values.split(",")]
    mask_types = [x.strip() for x in args.mask_types.split(",")]
    temperatures = [float(x.strip()) for x in args.temperatures.split(",")]

    # Default output file
    if args.output is None:
        args.output = PROJECT_ROOT / "results" / f"ablation_{args.ablation}.json"

    # Run appropriate ablation
    if args.ablation == "k_nearest":
        run_k_nearest_ablation(
            args.image_dir,
            k_values=k_values,
            max_images=args.images,
            n_runs=args.runs,
            output_file=args.output
        )
    elif args.ablation == "threshold":
        run_threshold_ablation(
            args.image_dir,
            tau_values=tau_values,
            max_images=args.images,
            n_runs=args.runs,
            output_file=args.output
        )
    elif args.ablation == "mask_type":
        run_mask_type_ablation(
            args.image_dir,
            mask_types=mask_types,
            max_images=args.images,
            n_runs=args.runs,
            output_file=args.output
        )
    elif args.ablation == "soft_mask":
        run_soft_mask_ablation(
            args.image_dir,
            temperatures=temperatures,
            max_images=args.images,
            n_runs=args.runs,
            output_file=args.output
        )


if __name__ == "__main__":
    main()
