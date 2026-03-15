#!/bin/bash
# =============================================================================
# VGGT-MPS Paper Benchmark Runner
# =============================================================================
# This script runs all benchmarks needed for the paper and saves results
# to the results/ directory.
#
# Usage: ./scripts/run_paper_benchmarks.sh
# =============================================================================

set -e

echo "============================================================"
echo "VGGT-MPS Paper Benchmark Suite"
echo "============================================================"
echo ""

# Create results directory
mkdir -p results/figures

# -----------------------------------------------------------------------------
# Experiment 1: Scaling Performance (Table 1 + Figure 2)
# -----------------------------------------------------------------------------
echo "[1/7] Running Scaling Benchmark..."
echo "      This generates data for Table 1 and Figure 2"
vggt benchmark --mode scaling \
    --images 10,20,30,50,100,200 \
    --methods dense,sparse \
    --sparse-k 5,10,20 \
    --output results/scaling_benchmark.json
echo "      Done! Results: results/scaling_benchmark.json"
echo ""

# -----------------------------------------------------------------------------
# Experiment 2: Output Consistency (Table 2)
# -----------------------------------------------------------------------------
echo "[2/7] Running Consistency Benchmark..."
echo "      This generates data for Table 2"
vggt benchmark --mode consistency \
    --images 5,10,20,30 \
    --compare dense,sparse \
    --metrics depth_l1,pose_rotation,pose_translation,chamfer \
    --output results/consistency.json
echo "      Done! Results: results/consistency.json"
echo ""

# -----------------------------------------------------------------------------
# Experiment 3a: k-Nearest Ablation (Table 3a)
# -----------------------------------------------------------------------------
echo "[3/7] Running k-Nearest Ablation..."
echo "      This generates data for Table 3a"
vggt benchmark --mode ablation-k \
    --images 30 \
    --sparse-k 3,5,10,15,20,30 \
    --output results/ablation_k.json
echo "      Done! Results: results/ablation_k.json"
echo ""

# -----------------------------------------------------------------------------
# Experiment 3b: Threshold Ablation (Table 3b)
# -----------------------------------------------------------------------------
echo "[4/7] Running Threshold Ablation..."
echo "      This generates data for Table 3b"
vggt benchmark --mode ablation-tau \
    --images 30 \
    --threshold 0.3,0.5,0.7,0.8,0.9 \
    --output results/ablation_tau.json
echo "      Done! Results: results/ablation_tau.json"
echo ""

# -----------------------------------------------------------------------------
# Experiment 3c: Mask Type Ablation (Table 3c + Figure 3)
# -----------------------------------------------------------------------------
echo "[5/7] Running Mask Type Ablation..."
echo "      This generates data for Table 3c and Figure 3"
vggt benchmark --mode ablation-mask \
    --images 30 \
    --mask-types covisibility,random,sliding_window \
    --sparsity 0.56 \
    --output results/ablation_mask.json
echo "      Done! Results: results/ablation_mask.json"
echo ""

# -----------------------------------------------------------------------------
# Experiment 4: Full Method Comparison (Table 4)
# -----------------------------------------------------------------------------
echo "[6/7] Running Method Comparison..."
echo "      This generates data for Table 4"
vggt benchmark --mode compare-methods \
    --images 30 \
    --methods dense,covisibility,sliding_window,random \
    --sparsity 0.5,0.6,0.7 \
    --output results/method_comparison.json
echo "      Done! Results: results/method_comparison.json"
echo ""

# -----------------------------------------------------------------------------
# Visualization: Generate All Figures
# -----------------------------------------------------------------------------
echo "[7/7] Generating Visualization Figures..."
echo "      This generates Figure 1-5"
vggt benchmark --mode visualize \
    --images 30,100 \
    --output-dir results/figures/
echo "      Done! Figures: results/figures/"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "============================================================"
echo "All Benchmarks Complete!"
echo "============================================================"
echo ""
echo "Generated Files:"
echo "  Tables:"
echo "    - results/scaling_benchmark.json    (Table 1)"
echo "    - results/consistency.json          (Table 2)"
echo "    - results/ablation_k.json           (Table 3a)"
echo "    - results/ablation_tau.json         (Table 3b)"
echo "    - results/ablation_mask.json        (Table 3c)"
echo "    - results/method_comparison.json    (Table 4)"
echo ""
echo "  Figures:"
echo "    - results/figures/fig1_memory_scaling.png"
echo "    - results/figures/fig2_sparsity_patterns.png"
echo "    - results/figures/fig3_mask_comparison.png"
echo "    - results/figures/fig4_speedup_analysis.png"
echo "    - results/figures/fig5_ablation_study.png"
echo ""
echo "Next Steps:"
echo "  1. Review JSON results in results/*.json"
echo "  2. Copy data to docs/PAPER_DRAFT.md tables"
echo "  3. Include figures in your paper"
echo "============================================================"
