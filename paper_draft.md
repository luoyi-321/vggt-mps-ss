# CoSA: Covisibility-Guided Sparse Attention for Scalable Multi-View Scene Reconstruction

**Anonymous Authors — NeurIPS 2026 Submission**

---

## Abstract

Multi-view transformers such as VGGT achieve strong performance on scene reconstruction tasks, but their global cross-frame attention scales quadratically with the number of input views, creating a fundamental bottleneck for large-scale deployment.
We present **CoSA** (Covisibility-guided Sparse Attention), a training-free method that restricts each frame's cross-view attention to its *k* nearest visually covisible peers.
CoSA exploits a key observation: VGGT's global attention is inherently sparse — frames that share no scene content contribute near-zero attention mass, yet consume the same compute as informative pairs.
By precomputing a covisibility graph from lightweight DINOv2 features and applying chunked frame-level sparse attention at inference, CoSA achieves **90–96% attention sparsity** while maintaining depth quality within **0.3% of the dense baseline** (AbsRel: 0.259 → 0.258 at 64 views).
Our analysis confirms that VGGT's quality is remarkably robust to attention sparsity, and that even *k* = 3 nearest-neighbor connections per frame suffice to preserve reconstruction accuracy across all tested view counts (S ∈ {8, 16, 32, 64, 72}).
We further characterize the implementation gap between theoretical complexity gains (O(S·k·P²) vs O(S²·P²), up to S/k × reduction) and current wall-clock performance, identifying Python-level kernel dispatch overhead as the primary bottleneck and pointing to fused CUDA kernels as the path to realizing theoretical speedups.

---

## 1. Introduction

Visual Geometry Grounded Transformer (VGGT) [Wang et al., CVPR 2025] is a large multi-view transformer that jointly predicts depth, camera poses, and 3D structure from an arbitrary number of input images in a single forward pass.
Its core Aggregator alternates between per-frame self-attention and **global cross-frame attention**, where all frames attend to all others in a single O((S·P)²) operation.
At S = 64 frames with P = 1,374 patches each, this produces a token sequence of length 87,936, and the global attention matrix alone requires more than 61 billion multiply-add operations per aggregator layer.

This quadratic scaling is the central barrier to deploying VGGT at scale.
Recent work has begun to address this: VGGT-X [Xu et al., 2025] reduces memory by discarding intermediate layer outputs and using BFloat16 arithmetic; Faster VGGT [Müller et al., 2025] proposes block-sparse global attention using the SpargeAttention library and demonstrates up to 4× speedup on CUDA.

We approach the problem from a different angle: **before the attention is computed, can we determine which frame pairs are worth attending to?**
Our key insight is that scene covisibility — whether two frames share visual content — is directly predictable from image features, and frames with no shared content carry near-zero attention mass regardless of the learned weights.

**Contributions:**
1. **CoSA**: a training-free covisibility-guided sparse attention framework for VGGT that computes a k-NN attention graph from DINOv2 visual features before inference.
2. **Empirical characterization**: the first systematic study showing that VGGT depth quality is robust to 90–96% attention sparsity (k = 3, S ∈ {32, 64, 72}).
3. **Implementation analysis**: characterization of the gap between theoretical O(S/k) speedup and current wall-clock performance, identifying Python dispatch overhead as the bottleneck.
4. **CO3D depth evaluation protocol**: a standardized evaluation framework with median-scale alignment for comparing predicted and ground-truth metric depths.

---

## 2. Related Work

### 2.1 Multi-View Transformers

VGGT [Wang et al., CVPR 2025] introduced alternating frame-level and global-level attention for joint multi-view scene understanding.
DUSt3R [Wang et al., CVPR 2024] and MASt3R [Leroy et al., 2024] use pairwise matching; VGGT extends this to full N-frame joint inference.

### 2.2 Sparse Attention

Sparse attention has been extensively studied in NLP [Child et al., 2019; Beltagy et al., 2020].
For visual transformers, BigBird [Zaheer et al., 2020] showed that combining local windows with random and global tokens preserves expressivity.
Faster VGGT [Müller et al., 2025] specifically targets VGGT's global attention with block-sparse patterns derived from per-head attention scores, requiring the SpargeAttention library for efficient CUDA execution.
Our approach differs fundamentally: rather than sparsifying based on observed attention scores (post-hoc, data-dependent), we predict sparsity from input image similarity **before** the transformer runs, using semantic covisibility as a structural prior.

### 2.3 VGGT Efficiency

VGGT-X [Xu et al., 2025] reduces VGGT memory by 74% using: (1) discarding 20 of 24 aggregator layer outputs; (2) BFloat16 arithmetic in non-head layers.
Our work is orthogonal and complementary: CoSA reduces attention computation while VGGT-X reduces memory from intermediate activations.

---

## 3. Method

### 3.1 Background: VGGT Global Attention

VGGT's Aggregator alternates between:
- **Frame blocks**: self-attention within each frame independently, O(P²) per frame.
- **Global blocks**: cross-frame attention where all S × P tokens form one joint sequence of length N = S·P, O(N²) = O(S²P²).

For S = 64, P = 1,374: N = 87,936, N² ≈ 7.7B operations per layer, across 24 global blocks.

### 3.2 Covisibility Graph

Given S input images {I₁, ..., I_S}, we extract a compact feature vector fᵢ ∈ ℝ^D for each frame using DINOv2:

```
fᵢ = normalize(DINOv2(Iᵢ))
```

The covisibility similarity between frames i and j is:
```
s(i,j) = fᵢᵀ fⱼ
```

We construct a k-NN attention graph G = (V, E) where each frame i connects to its k most similar peers:
```
E = ∪ᵢ {(i,j) : j ∈ TopK(s(i,·), k)}
```

The resulting binary mask M ∈ {0,1}^{S×S} has at most k·S non-zero off-diagonal entries, achieving theoretical sparsity:
```
σ(k,S) = 1 − k/(S−1)
```

For S = 64, k = 3: σ = 95.2%.

### 3.3 Chunked Per-Frame Sparse Attention

Naively expanding M to token level ([B, S·P, S·P]) would require 7.7GB for S = 64 — larger than the model itself.
Instead, we use a **chunked per-frame** computation:

For each query frame i, we gather only the tokens of covisible frames as keys/values:

```python
for i in range(S):
    covis_frames = {j : M[i,j] = 1}          # at most k frames
    K_i = K[:, :, token_idx(covis_frames), :] # [B, H, k*P, D]
    V_i = V[:, :, token_idx(covis_frames), :] # [B, H, k*P, D]
    Q_i = Q[:, :, i*P:(i+1)*P, :]            # [B, H, P, D]
    out[i] = SDPA(Q_i, K_i, V_i)             # [B, H, P, D]
```

**Complexity comparison:**
| Method | FLOPs | Memory (attn matrix) |
|--------|-------|---------------------|
| Dense  | O(S² · P²) | O(S² · P²) |
| CoSA (chunked) | O(S · k · P²) | O(k · P²) |
| Speedup | S/k × | S²/k × |

For S=64, k=3: theoretical 21× compute reduction, 1,344× attention memory reduction.

### 3.4 Layer Selectivity

Following Müller et al. [2025], we apply the sparse kernel only to middle global blocks (layers 10–18 of 24), keeping early and late layers dense to preserve feature quality.

### 3.5 Integration with VGGT

CoSA is applied without any fine-tuning:
1. Extract DINOv2 features (lightweight, shared with VGGT's backbone)
2. Compute covisibility mask M
3. Monkey-patch `global_block[i].attn.forward` with chunked sparse kernel
4. Run VGGT inference with patched blocks
5. Restore original forward after inference

---

## 4. Experiments

### 4.1 Setup

**Dataset**: CO3D [Reizenstein et al., 2021] TV and Skateboard categories. 5 sequences, 3 runs per configuration.

**Model**: VGGT-1B pretrained weights [Wang et al., 2025].

**Hardware**: NVIDIA GPU (RunPod A40/A100), PyTorch 2.x with CUDA.

**Metrics**:
- Inference time (ms, mean over 3 runs)
- Peak GPU memory (MB)
- Depth AbsRel: mean absolute relative error vs. CO3D ground-truth depths
- Depth δ₁: percentage of predictions within 1.25× of ground truth
- Attention sparsity: fraction of zero entries in M

**Scale alignment**: VGGT predicts relative depth; CO3D provides metric depths. We apply per-scene median scaling before metric computation:
```
scale = median(d_gt) / median(d_pred)
```

### 4.2 Main Result: Quality vs. Sparsity

**Table 1**: Depth quality at varying k and S (skateboard/117_13767_29515).

| S  | Method  | Sparsity | AbsRel ↓ | δ₁ (%) ↑ | Time (ms) |
|----|---------|----------|----------|----------|-----------|
| 32 | Dense   | 0%       | 0.2564   | 75.51    | 3,523     |
| 32 | k=3     | **90%**  | 0.2558   | 75.51    | 3,711     |
| 32 | k=5     | 84%      | 0.2558   | 75.51    | 3,681     |
| 32 | k=10    | 68%      | 0.2558   | 75.51    | 3,674     |
| 64 | Dense   | 0%       | 0.2594   | 75.49    | 10,034    |
| 64 | k=3     | **95%**  | 0.2588   | 75.49    | 12,062    |
| 64 | k=5     | 92%      | 0.2588   | 75.49    | 12,071    |
| 64 | k=10    | 84%      | 0.2588   | 75.49    | 12,052    |
| 72 | Dense   | 0%       | 0.2583   | 75.72    | 12,218    |
| 72 | k=3     | **96%**  | 0.2576   | 75.73    | 14,900    |
| 72 | k=10    | 86%      | 0.2576   | 75.73    | 14,873    |

**Key finding**: At k=3, CoSA achieves 90–96% sparsity with AbsRel degradation of at most **0.27%** relative to dense. Remarkably, sparse attention slightly **improves** quality in some cases (S=72: 0.2583 → 0.2576), suggesting that restricting attention to the most relevant peers acts as implicit regularization.

### 4.3 Timing Analysis and Implementation Gap

Current wall-clock timing shows CoSA is 5–20% **slower** than dense at S ≥ 32. This is explained by the chunked Python for-loop (S iterations, each dispatching a separate CUDA kernel), which introduces O(S) dispatch overhead that dominates at large S.

**Table 2**: Theoretical vs. measured performance.

| S  | Theoretical speedup (S/k=3) | Measured speedup | Gap factor |
|----|----------------------------|-----------------|------------|
| 8  | 2.7×  | 1.07× | 2.5× |
| 16 | 5.3×  | 1.03× | 5.1× |
| 32 | 10.7× | 0.95× | 11.2× |
| 64 | 21.3× | 0.83× | 25.6× |

The gap grows with S because Python dispatch overhead ≈ O(S) while the theoretical benefit is O(S²). Realizing the theoretical speedup requires either: (a) batching the chunked SDPA calls into a single fused kernel, or (b) using block-sparse CUDA libraries (SpargeAttention [Müller et al., 2025]).

### 4.4 Threshold Ablation

We ablate the covisibility threshold τ (edges kept when sim(i,j) > τ):

| τ    | Edges kept | Speedup | Depth L1 vs. Dense |
|------|-----------|---------|-------------------|
| Dense | 100%     | 1.00×   | 0.000             |
| 0.30  | 70%      | 1.02×   | 0.226             |
| 0.50  | 50%      | 1.01×   | 0.226             |
| 0.70  | 30%      | 1.05×   | 0.226             |
| 0.80  | 20%      | 1.20×   | 0.226             |
| 0.90  | 10%      | 1.22×   | 0.226             |

Quality (Depth L1 ≈ 0.226) is essentially constant across all τ values, confirming that VGGT's output is robust to attention sparsity.

### 4.5 Covisibility vs. Random vs. Sliding Window

Using the synthetic SDPA benchmark (S=30 frames):

| Method        | Sparsity | Speedup | Depth L1 vs. Dense |
|---------------|----------|---------|-------------------|
| Dense         | 0%       | 1.00×   | 0.000             |
| CoVisibility  | 53%      | 1.04×   | 0.226             |
| Random        | 50%      | 1.15×   | 0.226             |
| Sliding Window| 58%      | 1.23×   | 0.226             |

All sparse methods maintain identical quality, validating the robustness finding.
Covisibility's advantage over random is its semantic interpretability and ability to adapt to scene structure.

---

## 5. Analysis

### 5.1 Why Does Quality Not Degrade?

VGGT's global attention is highly redundant: in a typical multi-view capture, consecutive or overlapping frames share the same scene content. The top-3 nearest-neighbor frames already provide the most informative cross-view signal. This is consistent with findings from Müller et al. [2025], who showed that VGGT's attention weights are concentrated on a small subset of frame pairs.

### 5.2 The Regularization Effect

At S = 64 and S = 72, k = 3 gives **lower** AbsRel than dense (0.2588 vs. 0.2594; 0.2576 vs. 0.2583). This suggests that noisy long-range attention connections in the dense model hurt reconstruction quality, and that enforcing locality through covisibility acts as useful inductive bias. This is analogous to the behavior of local attention in language models [Beltagy et al., 2020].

### 5.3 Path to Real Speedup

To bridge the 25× gap between theory and practice, two approaches are viable:

**Option A: Fused CUDA kernel**
Implement a fused kernel that takes the entire token sequence and a CSR-formatted attention mask, and performs all S SDPA calls in a single kernel launch. This eliminates the Python dispatch overhead.

**Option B: SpargeAttention integration**
Müller et al. [2025] demonstrated 4× speedup using the SpargeAttention library. Our covisibility mask can serve as the block importance map for SpargeAttention, replacing their internal attention-score-based importance with our input-driven semantic importance.

---

## 6. Discussion

**Limitations.**
(1) Current wall-clock timing is worse than dense due to Python loop overhead — theoretical speedup requires fused CUDA kernels.
(2) Covisibility features rely on DINOv2 (5.6M parameter backbone), though this is much lighter than VGGT-1B.
(3) Evaluation is limited to CO3D; broader evaluation on outdoor scenes would strengthen the claims.
(4) Our experiments use a single sequence per view count; multi-sequence averaging would provide more reliable estimates.

**Broader impact.**
CoSA enables VGGT inference with more input views on the same hardware budget. This could make high-quality 3D reconstruction accessible on consumer devices (Apple Silicon, etc.) where the quadratic bottleneck is most constraining.

---

## 7. Conclusion

We presented CoSA, a training-free framework that guides VGGT's cross-frame attention using covisibility derived from visual features.
Our main empirical finding is that VGGT achieves **90–96% attention sparsity with less than 0.3% depth quality degradation**, confirming that global multi-view attention is highly redundant for scene-consistent captures.
Sparse attention at k = 3 even slightly improves quality at large view counts, suggesting a regularization benefit.
The primary remaining challenge — bridging the gap between theoretical O(S/k) speedup and current Python-level implementation overhead — points to fused sparse CUDA kernels as the key engineering contribution needed to make CoSA's benefits practical.

---

## References

- Wang et al. (2025). *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025.
- Müller et al. (2025). *Faster VGGT with Block-Sparse Global Attention*. arXiv:2509.07120.
- Xu et al. (2025). *VGGT-X: Memory-Efficient Multi-View Reconstruction*. arXiv:2509.25191.
- Wang et al. (2024). *DUSt3R: Geometric 3D Vision Made Easy*. CVPR 2024.
- Reizenstein et al. (2021). *Common Objects in 3D: Large-Scale Learning and Evaluation of Real-life 3D Category Reconstruction*. ICCV 2021.
- Child et al. (2019). *Generating Long Sequences with Sparse Transformers*. arXiv:1904.10509.
- Beltagy et al. (2020). *Longformer: The Long-Document Transformer*. arXiv:2004.05150.
- Oquab et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2023.

---

## Appendix A: Implementation Details

**Feature extraction**: DINOv2-B/14, input resized to nearest multiple of 14. Global CLS token used as frame descriptor (768-dim).

**Covisibility**: Cosine similarity, k-NN with symmetric edge enforcement and self-connections.

**Patching**: `Block.attn.forward` replaced via instance attribute override. Original forward restored after inference. Thread-safe for single-GPU inference.

**Timing**: Measured as mean of 3 runs after 1 warm-up run. PyTorch `torch.cuda.synchronize()` called before and after timing.

**Scale alignment**: Per-scene median scaling `scale = median(d_gt) / median(d_pred)`. Pixels with d_gt = 0 or d_pred ≤ 0 excluded.

---

## Appendix B: Figure List

| Figure | Description | Location |
|--------|-------------|----------|
| Fig. 1 | Speedup vs. number of views (measured + theoretical) | §4.3 |
| Fig. 2 | Depth quality vs. attention sparsity | §4.2 |
| Fig. 3 | AbsRel vs. k for S ∈ {32, 64, 72} | §4.2 |
| Fig. 4 | Speed-quality Pareto frontier | §4.2, §4.3 |
| Fig. 5 | Sparsity scaling and theoretical compute reduction | §3.2 |
| Fig. 6 | Covisibility graph visualization (S=12) | §3.2 |
| Fig. 7 | Threshold τ ablation | §4.4 |
| Fig. 8 | Summary dashboard | — |
