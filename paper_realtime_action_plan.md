# Paper Real-Time Action Plan — What To Do Now

> Your CO3D tv data is downloading. Here's exactly what to run once it finishes.

## Current Situation

| Item | Status |
|------|--------|
| CO3D tv data | ⏳ Downloading now |
| Model weights ([models/model.pt](file:///Users/sulixay/thridPerson_file/suvan/vggt-mps/models/model.pt)) | ❓ Need to verify (5GB) |
| Existing scaling results | ⚠️ **Simulated only** — not real model runs |
| GT evaluation on CO3D | ❌ Not done yet |
| Real ablations on CO3D | ❌ Not done yet |
| Statistical significance | ❌ No error bars yet |

> [!CAUTION]
> Your current [results/scaling_benchmark.json](file:///Users/sulixay/thridPerson_file/suvan/vggt-mps/results/scaling_benchmark.json) has **simulated** data (sub-millisecond latencies, sub-MB memory). Reviewers will reject this. You need **real VGGT-1B inference** results.

---

## Step 0: Verify Model Weights (Do This First!)

```bash
# Check if model exists
ls -lh models/model.pt

# If missing, download (~5GB):
python main.py download
# OR manually:
# wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt -O models/model.pt
```

Without the 5GB model weights, none of the following steps will work.

---

## Step 1: Verify CO3D Data Loaded ✅

Once the download finishes:

```bash
# Quick check — should show 27+ sequences
python examples/demo_co3d.py --dry_run
```

---

## Step 2: Real VGGT on CO3D (Table 1 — Main Result)

Run VGGT inference on real CO3D images. This gives you **real** timing + depth numbers.

```bash
# Test with 4 views first (quick sanity check, ~2-5 minutes)
python examples/demo_co3d.py --num_views 4

# Then the full comparison: Base vs MPS
python examples/compare_base_vs_mps.py --num_views 4

# Skip CPU if too slow, just MPS + sparse
python examples/compare_base_vs_mps.py --skip_cpu --include_sparse --k_nearest 10
```

📁 Results → `outputs/comparison/comparison_report.json`

---

## Step 3: GT Evaluation on CO3D (Table 2 — Quality Retention)

This is the **most critical** missing piece. Run on several sequences:

```bash
# Single sequence with GT depth comparison
python scripts/evaluate_with_gt.py \
    --dataset co3d \
    --data-dir co3d-main/tv \
    --max-images 4 \
    --compare dense,sparse \
    --k-values 5,10,15 \
    --runs 3 \
    --output results/gt_evaluation_co3d.json
```

This produces the depth RMSE, AbsRel, and quality retention numbers.

---

## Step 4: Ablation Studies on CO3D (Tables 3a-3d)

All updated to use real CO3D data:

```bash
# 4a. k-nearest ablation (most important)
python scripts/run_ablations.py --ablation k_nearest \
    --k-values 3,5,10,15,20 --num-views 4 --runs 3

# 4b. Threshold ablation
python scripts/run_ablations.py --ablation threshold \
    --tau-values 0.5,0.6,0.7,0.8,0.9 --num-views 4

# 4c. Mask type ablation
python scripts/run_ablations.py --ablation mask_type --num-views 4

# 4d. Soft mask temperature ablation
python scripts/run_ablations.py --ablation soft_mask \
    --temperatures 0.05,0.1,0.2,0.5 --num-views 4

# 🔥 BONUS: Multi-sequence (runs across 5 sequences, averages results)
python scripts/run_ablations.py --ablation k_nearest \
    --multi-sequence --max-sequences 5 --num-views 4 --runs 3
```

📁 Results → `results/ablation_*_co3d.json`

---

## Step 5: Statistical Significance (Error Bars)

Use `--runs 5` (or 10 for final paper) on the key experiments:

```bash
# 10 runs for the main comparison
python examples/compare_base_vs_mps.py --num_views 4 --skip_cpu --include_sparse

# Ablation with 5 runs
python scripts/run_ablations.py --ablation k_nearest \
    --multi-sequence --max-sequences 5 --runs 5
```

---

## Step 6: Generate Figures

```bash
python scripts/generate_paper_figures.py \
    --input-dir results/ \
    --output-dir docs/figures/ \
    --format pdf --style camera_ready
```

---

## Priority Order (What To Run Tonight)

| Priority | Command | Time | Paper Section |
|:--------:|---------|:----:|:-------------:|
| 🔴 1 | Verify model weights exist | 1 min | — |
| 🔴 2 | `demo_co3d.py --dry_run` | 1 min | Verify data |
| 🔴 3 | `demo_co3d.py --num_views 4` | 5 min | Real inference proof |
| 🟠 4 | `compare_base_vs_mps.py --skip_cpu --include_sparse` | 10 min | Table 1 |
| 🟠 5 | `run_ablations.py --ablation k_nearest` | 30 min | Table 3a |
| 🟡 6 | `run_ablations.py --ablation k_nearest --multi-sequence` | 2 hr | Robust Table 3a |
| 🟡 7 | Other ablations (threshold, mask, soft_mask) | 2 hr | Tables 3b-3d |
| ⚪ 8 | Generate figures | 5 min | Figures 1-8 |

> [!TIP]
> Start with priorities 1-5 tonight. You can run multi-sequence ablations overnight.
