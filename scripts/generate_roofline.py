#!/usr/bin/env python3
"""
Roofline Analysis Plot Generator

Generates roofline analysis plots using MPSHardwareMetrics for understanding
compute vs memory bottlenecks in VGGT sparse attention.

Usage:
    python scripts/generate_roofline.py --chip M4_Max --output docs/figures/roofline.pdf
    python scripts/generate_roofline.py --chip M1 --n-values 10,50,100,500 --output docs/figures/roofline_m1.pdf
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_roofline_data(
    chip: str,
    n_values: List[int],
    k_values: List[int],
    d_head: int = 64
) -> Dict[str, Any]:
    """
    Generate roofline analysis data for given configurations.

    Args:
        chip: Apple Silicon chip name
        n_values: List of image counts to analyze
        k_values: List of k values for sparse attention
        d_head: Attention head dimension

    Returns:
        Dictionary with roofline data
    """
    from vggt_mps.efficiency_metrics import MPSHardwareMetrics, EfficiencyMetrics

    mps = MPSHardwareMetrics(chip)
    efficiency = EfficiencyMetrics(d_head)

    data = {
        "chip": chip,
        "specs": mps.specs,
        "ridge_point": mps.specs['flops_tflops'] * 1e12 / (mps.specs['bandwidth_gbps'] * 1e9),
        "configurations": []
    }

    # Analyze dense configurations
    for n in n_values:
        dense, sparse = efficiency.compute_theoretical_scaling(n, k_nearest=10)

        # Dense analysis
        dense_timing = mps.estimate_execution_time(
            dense['flops'],
            dense['memory_bytes']
        )
        data["configurations"].append({
            "name": f"Dense n={n}",
            "mode": "dense",
            "n": n,
            "k": n,
            "flops": dense['flops'],
            "memory_bytes": dense['memory_bytes'],
            "arithmetic_intensity": dense_timing['arithmetic_intensity'],
            "attainable_tflops": min(
                mps.specs['flops_tflops'],
                dense_timing['arithmetic_intensity'] * mps.specs['bandwidth_gbps'] / 1000
            ),
            "bottleneck": dense_timing['bottleneck'],
            "utilization": dense_timing['utilization']
        })

        # Sparse analysis for each k
        for k in k_values:
            if k >= n:
                continue

            _, sparse = efficiency.compute_theoretical_scaling(n, k_nearest=k)
            sparse_timing = mps.estimate_execution_time(
                sparse['flops'],
                sparse['memory_bytes']
            )
            data["configurations"].append({
                "name": f"Sparse n={n}, k={k}",
                "mode": "sparse",
                "n": n,
                "k": k,
                "flops": sparse['flops'],
                "memory_bytes": sparse['memory_bytes'],
                "arithmetic_intensity": sparse_timing['arithmetic_intensity'],
                "attainable_tflops": min(
                    mps.specs['flops_tflops'],
                    sparse_timing['arithmetic_intensity'] * mps.specs['bandwidth_gbps'] / 1000
                ),
                "bottleneck": sparse_timing['bottleneck'],
                "utilization": sparse_timing['utilization'],
                "savings_ratio": sparse['savings_ratio']
            })

    return data


def plot_roofline(
    data: Dict[str, Any],
    output_file: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Generate roofline plot.

    Args:
        data: Roofline data from generate_roofline_data
        output_file: Path to save figure
        show: If True, display plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Error: matplotlib required for plotting")
        print("Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Extract specs
    specs = data['specs']
    peak_tflops = specs['flops_tflops']
    bandwidth_gbps = specs['bandwidth_gbps']
    ridge_point = data['ridge_point']

    # Create roofline
    ai_range = np.logspace(-2, 4, 1000)
    memory_bound = ai_range * bandwidth_gbps / 1000  # Convert to TFLOPS
    compute_bound = np.full_like(ai_range, peak_tflops)
    roofline = np.minimum(memory_bound, compute_bound)

    # Plot roofline
    ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
    ax.axvline(x=ridge_point, color='gray', linestyle='--', alpha=0.5,
               label=f'Ridge Point ({ridge_point:.1f})')

    # Plot configurations
    colors = {'dense': 'red', 'sparse': 'blue'}
    markers = {'dense': 'o', 'sparse': 's'}

    for config in data['configurations']:
        ai = config['arithmetic_intensity']
        attainable = config['attainable_tflops']
        mode = config['mode']

        ax.scatter(ai, attainable, c=colors[mode], marker=markers[mode],
                   s=100, alpha=0.7, edgecolors='black', linewidth=1)

        # Add label for key points
        if config['n'] in [50, 100] and config.get('k', 0) in [0, 10]:
            ax.annotate(config['name'],
                       (ai, attainable),
                       textcoords="offset points",
                       xytext=(5, 5),
                       fontsize=8)

    # Labels and legend
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Attainable Performance (TFLOPS)', fontsize=12)
    ax.set_title(f'Roofline Analysis - {data["chip"]}', fontsize=14)

    # Create legend
    dense_patch = mpatches.Patch(color='red', label='Dense O(n²)')
    sparse_patch = mpatches.Patch(color='blue', label='Sparse O(n·k)')
    ax.legend(handles=[dense_patch, sparse_patch], loc='lower right')

    # Set limits
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(0.001, peak_tflops * 2)

    # Add annotations
    ax.text(0.02, peak_tflops * 0.9, f'Peak: {peak_tflops} TFLOPS',
            fontsize=10, va='top')
    ax.text(0.02, peak_tflops * 0.6, f'Bandwidth: {bandwidth_gbps} GB/s',
            fontsize=10, va='top')

    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved roofline plot to: {output_file}")

    if show:
        plt.show()

    plt.close()


def plot_scaling_comparison(
    data: Dict[str, Any],
    output_file: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Generate scaling comparison plot (memory vs n).

    Args:
        data: Roofline data
        output_file: Path to save figure
        show: If True, display plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib required for plotting")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data
    dense_configs = [c for c in data['configurations'] if c['mode'] == 'dense']
    sparse_configs = [c for c in data['configurations'] if c['mode'] == 'sparse']

    # Plot 1: Memory scaling
    n_dense = [c['n'] for c in dense_configs]
    mem_dense = [c['memory_bytes'] / 1e6 for c in dense_configs]  # MB

    ax1.plot(n_dense, mem_dense, 'ro-', linewidth=2, markersize=8, label='Dense O(n²)')

    # Group sparse by k
    k_values = sorted(set(c['k'] for c in sparse_configs))
    for k in k_values[:3]:  # Plot top 3 k values
        configs_k = [c for c in sparse_configs if c['k'] == k]
        n_sparse = [c['n'] for c in configs_k]
        mem_sparse = [c['memory_bytes'] / 1e6 for c in configs_k]
        ax1.plot(n_sparse, mem_sparse, 's--', linewidth=2, markersize=6,
                 label=f'Sparse k={k}')

    ax1.set_xlabel('Number of Images (n)', fontsize=12)
    ax1.set_ylabel('Memory (MB)', fontsize=12)
    ax1.set_title('Memory Scaling', fontsize=14)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Savings ratio
    for k in k_values[:3]:
        configs_k = [c for c in sparse_configs if c['k'] == k]
        n_sparse = [c['n'] for c in configs_k]
        savings = [c.get('savings_ratio', 1.0) for c in configs_k]
        ax2.plot(n_sparse, savings, 's-', linewidth=2, markersize=6,
                 label=f'k={k}')

    ax2.set_xlabel('Number of Images (n)', fontsize=12)
    ax2.set_ylabel('Memory Savings Ratio (×)', fontsize=12)
    ax2.set_title('Sparse vs Dense Savings', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved scaling plot to: {output_file}")

    if show:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate roofline analysis plots"
    )
    parser.add_argument(
        "--chip",
        type=str,
        default="M4_Max",
        choices=["M1", "M2", "M3", "M3_Pro", "M3_Max", "M4", "M4_Pro", "M4_Max"],
        help="Apple Silicon chip to analyze"
    )
    parser.add_argument(
        "--n-values",
        type=str,
        default="10,50,100,200,500",
        help="Comma-separated image counts to analyze"
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,20",
        help="Comma-separated k values for sparse attention"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "roofline.pdf",
        help="Output file for roofline plot"
    )
    parser.add_argument(
        "--scaling-output",
        type=Path,
        default=None,
        help="Output file for scaling comparison plot"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also save raw data as JSON"
    )

    args = parser.parse_args()

    # Parse arguments
    n_values = [int(x.strip()) for x in args.n_values.split(",")]
    k_values = [int(x.strip()) for x in args.k_values.split(",")]

    print("=" * 60)
    print("Roofline Analysis Generator")
    print("=" * 60)
    print(f"Chip: {args.chip}")
    print(f"N values: {n_values}")
    print(f"K values: {k_values}")

    # Generate data
    data = generate_roofline_data(args.chip, n_values, k_values)

    # Print summary
    print(f"\nChip Specifications:")
    print(f"  Peak FLOPs: {data['specs']['flops_tflops']} TFLOPS")
    print(f"  Memory Bandwidth: {data['specs']['bandwidth_gbps']} GB/s")
    print(f"  Ridge Point: {data['ridge_point']:.2f} FLOPs/Byte")

    print(f"\nConfigurations analyzed: {len(data['configurations'])}")

    # Save JSON if requested
    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nData saved to: {args.json}")

    # Generate plots
    plot_roofline(data, args.output, args.show)

    if args.scaling_output:
        plot_scaling_comparison(data, args.scaling_output, args.show)
    else:
        # Generate scaling plot with default name
        scaling_output = args.output.parent / "scaling_comparison.pdf"
        plot_scaling_comparison(data, scaling_output, args.show)


if __name__ == "__main__":
    main()
