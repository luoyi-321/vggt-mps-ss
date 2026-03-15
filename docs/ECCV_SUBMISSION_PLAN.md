# ECCV 2026 Submission Plan: Positioning Against Prior Work

> **Date:** 2026-02-10
> **Status:** Planning Phase
> **Target:** ECCV 2026 (Main Conference or Workshop)

---

## 1. Prior Work Analysis: RWTH Aachen Paper

### Paper Details

| Field | Information |
|-------|-------------|
| **Title** | Faster VGGT with Block-Sparse Global Attention |
| **Authors** | Chung-Shien Brian Wang, Christian Schmidt, Jens Piekenbrinck, Bastian Leibe |
| **Institution** | RWTH Aachen University |
| **arXiv** | 2509.07120v1 |
| **Date** | September 8, 2025 |
| **Venue** | Preprint (likely targeting CVPR/ECCV 2026) |

### RWTH Paper Summary

**Problem:** VGGT's global attention has O(n²) complexity, becoming the bottleneck for large image collections.

**Key Observations:**
- Global attention matrices are highly sparse
- Only a small subset of patch-patch interactions carry significant probability mass
- Middle layers of the aggregator are most critical (layer-drop ablation shows this)
- Special tokens (camera, register) behave differently from patch tokens

**Method:**
1. Pool queries Q and keys K to create low-resolution representations
2. Compute similarity matrix S = P^b(Q) · P^b(K)^T
3. Apply softmax to get block importance distribution
4. Select top-k blocks based on CDF threshold τ and sparse ratio ρ
5. Use block-sparse attention kernels (SpargeAttention) for computation

**Results:**
- Up to **4× faster inference** on H100 GPUs
- Tested on VGGT and π³ models
- Comprehensive evaluation on CO3Dv2, Real Estate 10K, TUM, ScanNet, 7Scenes, NRGBD, DTU, ETH3D, Tanks & Temples
- Processes sequences exceeding 512K tokens at >75% sparsity

---

## 2. Comparison: RWTH Paper vs Our Project

| Aspect | RWTH Paper | Our Project (vggt-mps) |
|--------|-----------|------------------------|
| **Core Problem** | O(n²) attention bottleneck in VGGT | Same |
| **Approach** | Block-sparse attention using internal Q/K pooling | Covisibility-guided sparsity using MegaLoc/DINOv2 |
| **Sparsity Source** | Data-driven (learned from attention patterns) | Task-specific (geometric covisibility priors) |
| **Sparsity Prediction** | Pooled queries/keys → low-res attention map | External visual similarity features |
| **Granularity** | Patch-level blocks (fine-grained) | Image-level covisibility (coarse-grained) |
| **Training Required** | Training-free | Training-free |
| **Speedup Claimed** | Up to 4× faster inference | Up to 50-100× memory savings |
| **Platform** | H100 GPUs (CUDA, SpargeAttention) | Apple Silicon (MPS backend) |
| **Models Tested** | VGGT, π³ | VGGT |
| **Special Token Handling** | Separate attention for special vs patch tokens | Included in covisibility mask |

### Overlap (Similarities)

1. Both identify O(n²) attention as the bottleneck
2. Both propose training-free solutions
3. Both exploit sparsity in attention patterns
4. Both target multi-view 3D reconstruction with VGGT

### Differentiation (Our Unique Contributions)

1. **Geometric Prior vs Data-Driven:**
   - RWTH learns which blocks matter from internal Q/K similarity
   - We use **external covisibility priors** based on visual localization features
   - Our approach is more **interpretable** (we know WHY connections are pruned)

2. **Feature Reuse:**
   - MegaLoc/DINOv2 features can be reused for other tasks (place recognition, loop closure)
   - RWTH's pooled Q/K are internal to VGGT, single-use

3. **Platform Democratization:**
   - We target **consumer hardware** (Apple Silicon)
   - RWTH requires datacenter GPUs with specialized kernels

4. **Complexity Claim:**
   - We claim O(n·k) complexity with explicit k parameter
   - RWTH uses adaptive sparsity ratio without explicit complexity bound

5. **Complementary Nature:**
   - Our image-level covisibility can be **combined with** their patch-level block-sparse
   - Two-stage sparsification: covisibility (which images) → block-sparse (which patches)

---

## 3. Revised Paper Positioning

### Option A: Main Conference Paper (Higher Risk, Higher Reward)

**Revised Title:**
"Covisibility-Guided Attention Sparsification for Scalable Multi-View 3D Reconstruction"

**Revised Abstract Framing:**
> Recent work [RWTH] has shown that VGGT's global attention can be sparsified using internal query-key pooling. However, this data-driven approach lacks geometric interpretability and requires specialized GPU kernels. We propose a complementary, **geometry-aware** sparsification method that exploits **external covisibility priors** from visual localization features. Our key insight is that non-covisible image pairs share no geometric constraints and thus require no mutual attention. Unlike patch-level block selection, our image-level covisibility mask provides principled, interpretable sparsity that can be combined with existing block-sparse methods for further acceleration.

**Key Claims to Make:**
1. First to use **external covisibility priors** for VGGT attention sparsification
2. **Geometric interpretability:** Explains why certain connections are pruned
3. **Orthogonal contribution:** Can be combined with block-sparse methods
4. **Consumer hardware focus:** Democratizes large-scale reconstruction on Apple Silicon
5. **Explicit complexity bound:** O(n·k) with tunable k parameter

### Option B: Workshop Paper (Lower Risk, Safer)

**Target Workshops:**
- ECCV 2026 Workshop on "Efficient Deep Learning for Computer Vision"
- ECCV 2026 Workshop on "3D Vision"
- ECCV 2026 Workshop on "Mobile and Embedded Vision"

**Advantages:**
- Lower novelty bar (contribution can be more incremental)
- Platform-specific contribution (MPS) is acceptable
- Faster review cycle
- Still counts as ECCV publication

**Workshop-Specific Framing:**
> "Efficient Multi-View Reconstruction on Consumer Hardware: Covisibility-Guided Sparse Attention for Apple Silicon"

---

## 4. ECCV 2026 Timeline

### Key Dates (Estimated)

| Milestone | Date (Estimated) |
|-----------|------------------|
| Paper Submission Deadline | ~March 7, 2026 |
| Supplementary Material Deadline | ~March 14, 2026 |
| Rebuttal Period | ~May 15-22, 2026 |
| Author Notification | ~June 15, 2026 |
| Camera-Ready Deadline | ~July 15, 2026 |
| Conference | ~September 28 - October 3, 2026 |

### Remaining Time

**From today (Feb 10) to submission (Mar 7): ~25 days**

---

## 5. Action Plan

### Week 1: Feb 10-16 (Foundation)

- [ ] **Update Related Work section**
  - Add citation to RWTH paper (arXiv:2509.07120)
  - Position our work as complementary
  - Acknowledge overlap, emphasize differentiation

- [ ] **Run real VGGT validation**
  - Move from simulated mode to actual model
  - Verify depth/pose quality metrics

- [ ] **Prepare comparison experiment**
  - Implement RWTH-style internal Q/K pooling baseline
  - Compare at equivalent sparsity levels

### Week 2: Feb 17-23 (Experiments)

- [ ] **Critical ablation: Covisibility vs Internal Pooling**
  - Same sparsity ratio, compare reconstruction quality
  - Hypothesis: Covisibility preserves geometric accuracy better

- [ ] **Combination study**
  - Covisibility (image-level) + Block-sparse (patch-level)
  - Show methods are complementary, not competing

- [ ] **Visualization**
  - Show which connections are pruned by covisibility
  - Demonstrate geometric interpretability

### Week 3: Feb 24 - Mar 2 (Writing)

- [ ] **Finalize all figures and tables**
- [ ] **Complete camera-ready quality draft**
- [ ] **Internal review and revision**

### Week 4: Mar 3-7 (Submission)

- [ ] **Final polish and proofread**
- [ ] **Prepare supplementary material**
- [ ] **Submit paper**
- [ ] **Code release preparation**

---

## 6. Critical Experiments to Add

### Experiment A: Direct Method Comparison

**Goal:** Show covisibility-guided sparsity vs RWTH-style internal pooling

| Method | Sparsity | Depth L1 ↓ | Pose Error ↓ | Interpretability |
|--------|----------|------------|--------------|------------------|
| Dense (baseline) | 0% | X.XXX | X.XX° | N/A |
| Internal Q/K Pooling (RWTH-style) | 50% | X.XXX | X.XX° | Low |
| **Covisibility (Ours)** | 50% | X.XXX | X.XX° | **High** |
| Combined (Both) | 70% | X.XXX | X.XX° | Medium |

### Experiment B: Interpretability Study

**Goal:** Demonstrate that covisibility-guided pruning is geometrically meaningful

1. For each pruned image pair (i, j), compute:
   - Visual overlap score (from MegaLoc)
   - Actual geometric overlap (from ground truth poses)
   - Show high correlation

2. Visualize:
   - Covisibility matrix heatmap
   - Corresponding 3D scene structure
   - Which pairs share common 3D points

### Experiment C: Combined Sparsification

**Goal:** Show our method is orthogonal and can be combined with block-sparse

```
Stage 1: Covisibility mask (image-level)
         - Remove non-covisible image pairs
         - Reduces from n² to ~n·k image pairs

Stage 2: Block-sparse (patch-level, RWTH-style)
         - Within covisible pairs, apply block selection
         - Further reduces computation

Combined Sparsity = Stage1_Sparsity + Stage2_Sparsity * (1 - Stage1_Sparsity)
```

### Experiment D: Consumer Hardware Benchmark

**Goal:** Demonstrate practical impact on Apple Silicon

| Config | Images | Dense Time | Dense Memory | Sparse Time | Sparse Memory | Status |
|--------|--------|------------|--------------|-------------|---------------|--------|
| M1 16GB | 50 | X.Xs | X GB | X.Xs | X GB | ✓ |
| M1 16GB | 100 | OOM | - | X.Xs | X GB | ✓ |
| M1 16GB | 200 | OOM | - | X.Xs | X GB | ✓ |
| M1 16GB | 500 | OOM | - | X.Xs | X GB | ✓ |

---

## 7. Updated Related Work Section Draft

```markdown
## Related Work

### Efficient Attention for Multi-View Reconstruction

The quadratic complexity of global attention in Vision Geometry Transformers has
been recently addressed by Wang et al. [RWTH], who propose block-sparse attention
based on internal query-key pooling. Their method achieves up to 4× speedup by
predicting block importance from low-resolution attention approximations.

Our work differs in three key aspects:

1. **Sparsity Source:** We use *external* covisibility features from visual
   localization (MegaLoc/DINOv2), rather than internal Q/K pooling. This provides
   geometric interpretability—we know that pruned connections correspond to
   non-covisible image pairs that share no 3D points.

2. **Granularity:** Our image-level covisibility mask operates at a coarser
   granularity than patch-level block selection. The two approaches are
   *orthogonal* and can be combined: covisibility first removes non-overlapping
   image pairs, then block-sparse attention can further sparsify within
   covisible pairs.

3. **Platform:** We target consumer hardware (Apple Silicon with MPS backend),
   democratizing large-scale multi-view reconstruction beyond datacenter GPUs.
```

---

## 8. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Reviewer sees as incremental over RWTH | 40% | High | Emphasize complementary nature, combination experiments |
| Real VGGT validation fails | 20% | Critical | Start validation immediately, have backup metrics |
| Time runs out before deadline | 30% | High | Prioritize critical experiments, cut optional ones |
| Apple Silicon specific seen as niche | 25% | Medium | Frame as democratization, add theoretical analysis |

---

## 9. Backup Plan: If Main Conference Rejected

1. **ECCV Workshop submission** (deadline usually later, ~June 2026)
2. **WACV 2027** (submission ~August 2026)
3. **CVPR 2027** (submission ~November 2026)
4. **arXiv preprint** + open source release for community impact

---

## 10. References to Add

```bibtex
@article{wang2025faster,
  title={Faster VGGT with Block-Sparse Global Attention},
  author={Wang, Chung-Shien Brian and Schmidt, Christian and Piekenbrinck, Jens and Leibe, Bastian},
  journal={arXiv preprint arXiv:2509.07120},
  year={2025}
}

@inproceedings{zhang2025spargeattention,
  title={SpargeAttention: Accurate and Training-Free Sparse Attention Accelerating Any Model Inference},
  author={Zhang, Jintao and others},
  booktitle={ICML},
  year={2025}
}
```

---

*Document created: 2026-02-10*
*Last updated: 2026-02-10*
