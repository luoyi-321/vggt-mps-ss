# 🔴 P0: Real VGGT Validation — Preparation Checklist

> **Date:** 2026-02-17  
> **Goal:** Move from simulated/dummy results to real VGGT model inference on MPS, producing publishable quality metrics for the ECCV 2026 paper.  
> **Deadline:** Feb 19, 2026 (per ECCV submission timeline)

---

## 1. Current Project Status

### ✅ What's Already Ready

| Item | Status | Location |
|------|--------|----------|
| VGGT model weights (5GB) | ✅ Downloaded | `models/model.pt` |
| VGGT source code | ✅ Cloned | `repo/vggt/` |
| Core processor code | ✅ Written | `src/vggt_mps/vggt_core.py` |
| Sparse attention module | ✅ Written | `src/vggt_mps/vggt_sparse_attention.py` |
| MegaLoc covisibility module | ✅ Written | `src/vggt_mps/megaloc_mps.py` |
| Real data (5 objects) | ✅ Exists | `data/real_data/` (audiojack, bottle_cap, button_battery, end_cap, eraser) |
| Dependencies file | ✅ Exists | `requirements.txt` |
| MPS device detection | ✅ Working | `src/vggt_mps/config.py` |

### ❌ What's Missing

| Item | Impact | Effort |
|------|--------|--------|
| Model path mismatch (tests look at wrong path) | Blocks all testing | 5 min |
| Real multi-view test images (tests use solid-color dummies) | No paper-quality results | 30-60 min |
| Quality metrics script (depth L1, pose error) | No publishable numbers | 2-3 hrs |
| Dense vs sparse comparison on real data | Core paper claim unverified | 1-2 hrs |
| Standard benchmark data (CO3Dv2 / DTU) | Weak comparison vs RWTH paper | 30-60 min |

---

## 2. Fix Model Path Mismatch

**Problem:** Model weights live at `models/model.pt` but test scripts look at `repo/vggt/vggt_model.pt`.

**Solution (pick one):**

```bash
# Option A: Create symlink (quick)
ln -s ../../models/model.pt repo/vggt/vggt_model.pt

# Option B: Update vggt_core.py load_model() — add models/model.pt to search paths
```

---

## 3. Prepare Real Multi-View Test Images

Current test scripts (`tests/test_real_quick.py`) create synthetic solid-color images:
```python
img = Image.new('RGB', (640, 480), color=(100 + i*50, 150, 200 - i*30))
```

This produces **meaningless** depth/pose results. You need real multi-view photos.

### Recommended Datasets

| Dataset | Why | Download |
|---------|-----|----------|
| **CO3Dv2** | RWTH paper uses it; standard benchmark | [github.com/facebookresearch/co3d](https://github.com/facebookresearch/co3d) |
| **DTU** | Multi-view stereo benchmark with GT depth | [roboimagedata.compute.dtu.dk](https://roboimagedata.compute.dtu.dk/) |
| **ETH3D** | High-quality GT, used by RWTH | [eth3d.net](https://www.eth3d.net/) |
| **Your real_data/** | Already available, no GT depth though | `data/real_data/bottle_cap/OK/S0001/*.jpg` (5 camera views per scene) |

> **Minimum for paper:** Download at least a small subset of CO3Dv2 or DTU (5-10 scenes).

---

## 4. Verify End-to-End Pipeline

Run these commands **in order** to confirm real VGGT works on your machine.

> **Prerequisites:** Activate conda environment first:
> ```bash
> conda activate vggt-env
> cd /Users/sulixay/thridPerson_file/suvan/vggt-mps
> ```

### Step 1: Verify VGGT Module Import

```bash
cd /Users/sulixay/thridPerson_file/suvan/vggt-mps

python -c "
import sys
sys.path.insert(0, 'repo/vggt')
from vggt.models.vggt import VGGT
print('✅ VGGT module importable')
"
```

### Step 2: Verify Model Loads on MPS

```bash
python -c "
import torch, sys
sys.path.insert(0, 'repo/vggt')
from vggt.models.vggt import VGGT

model = VGGT()
ckpt = torch.load('models/model.pt', map_location='mps', weights_only=True)
model.load_state_dict(ckpt)
model.to('mps').eval()

print('✅ Model loaded on MPS')
print(f'Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
"
```

### Step 3: Run Real Inference (2 Images)

> **Note:** MPS memory limits restrict batch size to 2 images. See Section 9 for details.

```bash
python -c "
import torch, sys, glob
torch.mps.empty_cache()  # Clear cache first
sys.path.insert(0, 'repo/vggt')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Load model
model = VGGT()
ckpt = torch.load('models/model.pt', map_location='mps', weights_only=True)
model.load_state_dict(ckpt)
model.to('mps').eval()

# Use only 2 images (MPS memory limit)
images = sorted(glob.glob('data/real_data/bottle_cap/OK/S0001/*.jpg'))[:2]
print(f'Using {len(images)} images: {images}')

# Preprocess and run
imgs = load_and_preprocess_images(images).to('mps')
with torch.no_grad():
    preds = model(imgs)

print('✅ Real inference complete!')
for k, v in preds.items():
    if isinstance(v, torch.Tensor):
        print(f'  {k}: {v.shape}')
"
```

**Expected output (tested 2026-03-05):**
```
Using 2 images: ['data/real_data/bottle_cap/OK/S0001/bottle_cap_0001_OK_C1_20230925164816.jpg', ...]
✅ Real inference complete!
  pose_enc: torch.Size([1, 2, 9])
  depth: torch.Size([1, 2, 518, 518, 1])
  depth_conf: torch.Size([1, 2, 518, 518])
  world_points: torch.Size([1, 2, 518, 518, 3])
  world_points_conf: torch.Size([1, 2, 518, 518])
  images: torch.Size([1, 2, 3, 518, 518])
```

---

## 5. Build Evaluation Script

**Create:** `scripts/evaluate_vggt.py`

### Required Metrics

| Metric | What It Measures | Formula |
|--------|-----------------|---------|
| **Depth L1 ↓** | Depth accuracy | `mean(|pred_depth - gt_depth|)` |
| **Abs Rel ↓** | Relative depth error | `mean(|pred - gt| / gt)` |
| **Rotation Error ↓** | Pose accuracy (degrees) | `arccos((tr(R_pred^T · R_gt) - 1) / 2)` |
| **Translation Error ↓** | Pose accuracy | `||t_pred - t_gt||₂` |
| **Inference Time** | Speed | `torch.mps.synchronize()` + `time.time()` |
| **Peak Memory** | Memory usage | `torch.mps.current_allocated_memory()` |

### Script Pseudo-structure

```python
def evaluate(image_dir, gt_dir=None, mode='dense'):
    """
    1. Load images from image_dir
    2. Run VGGT (dense or sparse mode)
    3. If gt_dir provided: compute depth L1, pose error
    4. Record timing and memory
    5. Output results table
    """
    pass

def compare_dense_vs_sparse(image_dir):
    """
    1. Run dense VGGT → depth_dense, pose_dense
    2. Run sparse VGGT (covisibility) → depth_sparse, pose_sparse
    3. Compute: |depth_sparse - depth_dense|
    4. Report: degradation from sparsification
    """
    pass
```

---

## 6. Dense vs Sparse Comparison

This is the **core claim** of the paper. You need to show:

1. **Sparse ≈ Dense in quality** (depth, pose metrics within acceptable margin)
2. **Sparse >> Dense in efficiency** (memory, time savings)

### What to Compare

| Config | Sparsity | Depth L1 | Pose Error | Memory | Time |
|--------|----------|----------|------------|--------|------|
| Dense (baseline) | 0% | _measure_ | _measure_ | _measure_ | _measure_ |
| Sparse k=10 | ~50% | _measure_ | _measure_ | _measure_ | _measure_ |
| Sparse k=5 | ~75% | _measure_ | _measure_ | _measure_ | _measure_ |
| Sparse k=3 | ~85% | _measure_ | _measure_ | _measure_ | _measure_ |

---

## 7. Hardware Info to Record

```bash
# Run these and record the output
system_profiler SPHardwareDataType | grep -E "Chip|Memory"
sw_vers
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

---

## 8. Step-by-Step Priority Order

| # | Task | Status | Priority |
|---|------|--------|----------|
| 1 | Fix model path (symlink) | ⏭️ Not needed (using `models/model.pt` directly) | 🔴 Critical |
| 2 | `conda activate vggt-env` | ✅ Done | 🔴 Critical |
| 3 | Verify VGGT imports & loads on MPS | ✅ Done (2026-03-05) | 🔴 Critical |
| 4 | Run inference on 2 real images | ✅ Done (2026-03-05) | 🔴 Critical |
| 5 | Download CO3Dv2 or DTU subset | ✅ CO3D downloaded | 🟡 Important |
| 6 | Build evaluation script with metrics | ✅ `scripts/evaluate_vggt.py` exists | 🟡 Important |
| 7 | Run dense vs sparse comparison | ✅ Done (2026-03-05) | 🟡 Important |
| 8 | Record hardware benchmark numbers | ✅ Done (2026-03-05) | 🟢 Nice to have |

---

## 10. Benchmark Results (2026-03-05)

### Hardware

| Component | Value |
|-----------|-------|
| Platform | Darwin 25.3.0 (macOS) |
| Processor | Apple Silicon (arm) |
| Python | 3.10.19 |
| PyTorch | 2.10.0 |
| Device | MPS |

### Dense vs Sparse Comparison (2 images)

| Config | k_eff | Time (ms) | Memory (MB) | Speedup |
|--------|-------|-----------|-------------|---------|
| Dense (baseline) | - | 5287.5 | 4844.4 | 1.00x |
| Sparse k=3 | 1 | 4956.6 | 4880.0 | 1.07x |
| Sparse k=5 | 1 | 4631.1 | 4880.0 | 1.14x |
| Sparse k=10 | 1 | 4584.2 | 4880.0 | 1.15x |

### Dense vs Sparse Comparison (3 images)

| Config | k_eff | Time (ms) | Memory (MB) | Speedup |
|--------|-------|-----------|-------------|---------|
| Dense (baseline) | - | 14414.5 | 4870.0 | 1.00x |
| Sparse k=3 | 2 | 16497.5 | 4905.4 | 0.87x |
| Sparse k=5 | 2 | 26360.2 | 4905.4 | 0.55x |
| Sparse k=10 | 2 | 20424.9 | 4905.4 | 0.71x |

### Key Findings

1. **MPS memory limit:** Max 2-3 images before OOM (~20GB limit)
2. **Small n performance:** With few images (n≤3), dense attention is faster than sparse
3. **Sparse attention overhead:** Covisibility mask computation adds latency
4. **Sparse benefits require large n:** Expected benefits appear when n>10 images
5. **For paper claims:** Need to test on CUDA with more images to show sparse scaling benefits

> **Recommendation:** For MPS with memory constraints, use dense attention for ≤5 images. Sparse attention is designed for large image sets (10+) where O(n²) becomes prohibitive.

---

## 9. Potential Blockers

| Blocker | Risk | Workaround |
|---------|------|------------|
| MPS out-of-memory on >2 images | **Confirmed** | Use 2 images max per batch (see below) |
| MPS unsupported ops in VGGT | Medium | Fall back to CPU for specific ops |
| No ground-truth depth for `real_data/` | High | Use CO3Dv2/DTU which have GT depth |
| Slow inference on MPS vs CUDA | Medium | Expected — document the time, this is the platform story |

### ⚠️ MPS Memory Limits (Tested 2026-03-05)

| Images | Status | Notes |
|--------|--------|-------|
| 2 | ✅ Works | Recommended batch size |
| 5 | ❌ OOM | Needs ~20GB, exceeds MPS limit |

**Workaround:** Process images in batches of 2, then aggregate results.

---

*Created: 2026-02-17*  
*Related: [ECCV_SUBMISSION_PLAN.md](./ECCV_SUBMISSION_PLAN.md)*
