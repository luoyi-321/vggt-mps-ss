# 论文写作方案 v2（面向二区/Workshop）

> 核心原则：**只写能跑通实验、能出数据的东西**

---

## 一、坦率评估：什么能做、什么不能做

### 你有的（优势）

| 资源 | 状态 |
|------|------|
| 预训练 VGGT 模型（1B 参数, 5GB） | 可直接推理 |
| MegaLoc/DINOv2 特征提取 | 已实现 |
| 稀疏注意力 runtime patch | 已实现，0 误差验证 |
| Apple Silicon Mac (MPS) | 可跑基准测试 |
| 开源代码库 | 可复现 |

### 你没有的（限制）

| 资源 | 影响 |
|------|------|
| 多卡 A100 训练集群 | 不能重训练 VGGT → 不能做需要训练的改进 |
| nuScenes/KITTI 占据预测标注 | 不能做 GaussianFormer 那种 occupancy 实验 |
| COLMAP ground truth 大规模对比 | 定量评估受限 |

### 结论

**不要做需要重训练的事**。论文的贡献应该是 **inference-time, training-free** 的方法。

---

## 二、推荐论文方向

### 标题建议

**"Training-Free Covisibility-Guided Sparse Attention for Scalable Multi-View 3D Reconstruction"**

或更具体地：

**"Scaling Vision Geometry Transformers to Hundreds of Views via Covisibility-Guided Attention Sparsification"**

### 核心卖点（审稿人看什么）

1. **问题有意义** — VGGT 等视觉几何 Transformer 在大量图像时 OOM，这是真实痛点
2. **方法简洁有效** — 不需要重训练，即插即用
3. **有理论支撑** — 共可见性分析 + 复杂度证明
4. **有实验验证** — 内存/延迟/质量三个维度的完整评估
5. **有工程价值** — 在消费级硬件（Mac）上跑通

---

## 三、论文结构与每部分需要什么

### 3.1 Introduction（1.5 页）

**需要说清楚的：**
- 视觉几何 Transformer（如 VGGT）做多视角 3D 重建效果好
- 但注意力是 O(n²)，n>50 时在消费级 GPU 上就 OOM
- 现有稀疏注意力方法（如 Longformer、BigBird）是通用的，不考虑视觉几何的先验
- 我们的洞察：**非共可见帧之间不需要互相注意** → 利用这个先验做 task-specific sparsification

**需要的图：**
- **Figure 1（动机图）**：左边是 VGGT 在不同图像数下的内存消耗（迅速爆炸），右边是我们方法的内存消耗（线性增长）。中间放一个定性结果展示质量保持。

```
建议格式（对标 GaussianFormer-2 Fig.1）：

┌─────────────────────────────────────────────────┐
│                                                 │
│  [输入多视角图片]  →  [3D重建结果]              │
│                                                 │
│  ┌──────────────┐    ┌──────────────┐           │
│  │ Memory (MB)  │    │ Quality      │           │
│  │              │    │ (保持)       │           │
│  │  Dense: OOM  │    │ Dense: 100%  │           │
│  │  at n>50     │    │ Ours: ~100%  │           │
│  │              │    │              │           │
│  │  Ours: 线性  │    │              │           │
│  │  n=1000 OK   │    │              │           │
│  └──────────────┘    └──────────────┘           │
│                                                 │
└─────────────────────────────────────────────────┘
```

**成功概率：高** — 只需要跑内存测量，不需要新实验。

---

### 3.2 Related Work（1 页）

覆盖三块：
1. **多视角 3D 重建** — VGGT, DUSt3R, MASt3R
2. **高效注意力** — Sparse Transformer, Longformer, FlashAttention
3. **视觉局部性先验** — 共可见性、图像检索（MegaLoc, NetVLAD）

**重点：** 现有高效注意力方法是 **task-agnostic** 的（滑动窗口、随机稀疏），我们是第一个利用 **covisibility prior** 对视觉几何 Transformer 做 **task-specific** 稀疏化的。

---

### 3.3 Method（3 页，核心）

#### 3.3.1 问题定义（0.5 页）

**公式 1：标准多视角注意力**

```
给定 n 张图像 I = {I₁, ..., Iₙ}
标准注意力: Attn(Q, K, V) = softmax(QKᵀ / √d) · V

其中 Q, K, V ∈ ℝⁿˢ×ᵈ (n 张图像, 每张 s 个 token, d 维)
注意力矩阵 A ∈ ℝⁿˢ × ⁿˢ

内存复杂度: O(n²s²)    （对于图像间注意力，简化为 O(n²)）
计算复杂度: O(n²s²d)
```

**公式 2：问题定义**

```
目标: 找到稀疏掩码 M ∈ {0,1}ⁿ×ⁿ 使得:

    min  |{(i,j) : M(i,j) = 1}|          (最小化计算量)
    s.t. ‖f(I; M) - f(I; 𝟙)‖ < ε        (重建质量损失 < ε)

其中 f(I; M) 是在掩码 M 下的重建结果，𝟙 是全连接掩码。
```

**验证难度：低** — 这是形式化描述，不需要实验。

#### 3.3.2 共可见性引导的稀疏注意力（1.5 页）

**Step 1: 特征提取**

```
公式 3: 图像特征提取

使用预训练 DINOv2 (ViT-B/14) 提取全局特征：
    fᵢ = MegaLoc(Iᵢ) ∈ ℝᵈᶠ    (df = 16640)
    f̂ᵢ = fᵢ / ‖fᵢ‖₂            (L2 归一化)

其中 MegaLoc 包含:
    - SALAD 局部聚合: ℝ⁷⁶⁸ → ℝ¹⁶³⁸⁴  (64 clusters × 256 dims)
    - 全局 token: ℝ⁷⁶⁸ → ℝ²⁵⁶
    - 拼接 + L2 归一化
```

**Step 2: 共可见性矩阵构建**

```
公式 4: 余弦相似度矩阵

    S(i,j) = f̂ᵢᵀ · f̂ⱼ ∈ [-1, 1]

公式 5: 二值共可见性掩码

    M(i,j) = 𝟙[S(i,j) > τ]  ∨  𝟙[j ∈ KNN(i, k)]  ∨  𝟙[i = j]

其中:
    τ = 共可见性阈值（默认 0.7）
    KNN(i, k) = 图像 i 的 k 个最相似图像（默认 k=10）
    最后一项保证自注意力
```

**Step 3: 图连通性保障**

```
公式 6: 时序连通性

    M(i, i+1) = M(i+1, i) = 1,  ∀i ∈ [1, n-1]

确保视频序列的时序连续性，防止注意力图分裂。
```

**Step 4: 掩码注意力**

```
公式 7: 稀疏注意力计算

    Â(i,j) = { QᵢKⱼᵀ/√d,  if M(i,j) = 1
             { -∞,          if M(i,j) = 0

    Attn_sparse = softmax(Â) · V
```

**验证难度：低** — 这些公式描述的是已实现的代码逻辑，只需要形式化写出来。

#### 3.3.3 复杂度分析（1 页）

```
公式 8: 稀疏度

    ρ = |{(i,j) : M(i,j) = 0, i≠j}| / (n² - n)

公式 9: 内存复杂度

    Dense:   O(n² · s² + n · s · d)
    Sparse:  O(nnz(M) · s² + n · s · d)

    其中 nnz(M) = n·k（当 n >> k 时）

    节省比: R_mem = n²/(n·k) = n/k

公式 10: 计算复杂度（FLOPs）

    Dense:   2 · n² · s² · d
    Sparse:  2 · nnz(M) · s² · d

    节省比: R_flops = n/k （同内存）

公式 11: MegaLoc 开销

    T_megaloc = O(n · C_dino + n² · df)

    其中 C_dino 是每张图的 DINOv2 推理开销
    n² · df 是余弦相似度矩阵计算（但 df 远小于 s²·d，所以开销很小）

公式 12: 总开销对比

    T_total_dense  = T_vggt(n²)
    T_total_sparse = T_megaloc(n) + T_vggt(n·k)

    当 n > k·(T_megaloc / T_vggt_per_pair) 时，稀疏方法更快。
    实测: 大约 n > 30 时稀疏方法开始更快。
```

**验证难度：中** — 需要实际测量来验证理论分析。但公式本身是正确的。

---

### 3.4 Experiments（3 页，核心）

#### 实验 1：缩放性能测试（必做，成功概率 95%）

**做法：** 从 10 到 N 张图像，分别测量 dense 和 sparse 的内存和延迟。

```
实验设计:
    图像数: [10, 20, 30, 50, 75, 100, 150, 200, 500]
    方法: Dense VGGT, Sparse VGGT (k=5, 10, 20)
    硬件: Apple Silicon MPS (标注具体型号)
    测量: 峰值内存 (MB), 端到端延迟 (ms), 是否 OOM

预期结果:
    - Dense 在 n=50~100 附近 OOM（取决于你的 Mac 内存）
    - Sparse 线性增长，可以跑到 n=500+
```

**输出图表：Table 1 + Figure 2（内存/延迟 vs 图像数曲线）**

这是论文最核心的实验，直接证明方法的价值。

#### 实验 2：输出一致性验证（必做，成功概率 99%）

**做法：** 在小 n（两者都能跑的范围）下对比 dense 和 sparse 的输出差异。

```
实验设计:
    图像数: [5, 10, 20, 30]
    度量:
      - 深度图 L1 误差: mean(|depth_sparse - depth_dense|)
      - 相机位姿误差: rotation error (°), translation error (m)
      - 点云 Chamfer Distance
    稀疏配置: k=10, τ=0.7

预期结果:
    - 误差 ≈ 0（数值精度级别）
    - 证明稀疏化不损失质量
```

**输出图表：Table 2**

你们已经有"0.000000 差异"的结果了，这一项基本确保成功。

#### 实验 3：消融实验（必做，成功概率 90%）

**做法：** 分析 k（近邻数）和 τ（阈值）对结果的影响。

```
实验设计:
    固定 n=30 张图像（dense 也能跑的范围，便于对比）

    (a) k 的消融:
        k = [3, 5, 10, 15, 20, 30(=n, 即 dense)]
        测量: 内存, 延迟, 输出误差

    (b) τ 的消融:
        τ = [0.3, 0.5, 0.7, 0.8, 0.9]
        测量: 稀疏度 ρ, 输出误差

    (c) 共可见性 vs 随机稀疏:
        相同稀疏度下，比较共可见性掩码 vs 随机掩码的输出质量
        → 证明共可见性先验的重要性（这是关键消融！）
```

**输出图表：Table 3 (a)(b) + Figure 3 (c)**

消融实验 (c) 特别重要：如果随机稀疏也能得到类似结果，说明共可见性先验没有价值；如果随机稀疏质量明显下降，就证明了方法的核心贡献。

#### 实验 4：定性可视化（必做，成功概率 95%）

**做法：** 展示不同配置下的 3D 重建结果。

```
可视化内容:
    - 深度图对比: Dense vs Sparse vs Ground Truth（如果有）
    - 3D 点云对比
    - 共可见性矩阵热力图
    - 注意力模式可视化（哪些帧互相关注）

场景选择:
    - 室内（VGGT 厨房数据集）
    - 室外（如果有数据）
    - 大规模（100+ 图像，仅 sparse 能跑）
```

**输出图表：Figure 4 (定性) + Figure 5 (共可见性矩阵)**

#### 实验 5：与通用稀疏注意力对比（可选，成功概率 70%）

**做法：** 对比我们的方法和通用稀疏注意力策略。

```
对比方法:
    - Dense (baseline)
    - Sliding window attention (Longformer 式)
    - Random sparsity (相同稀疏度)
    - Ours: Covisibility-guided

度量: 输出误差 @ 相同稀疏度
```

**风险：** 实现 sliding window 和 random sparsity 需要额外代码，但逻辑简单。

**输出图表：Table 4**

---

### 3.5 图表清单总结

| 图表 | 类型 | 优先级 | 验证成功率 | 内容 |
|:----:|------|:------:|:---------:|------|
| Fig.1 | 动机图 | P0 | 99% | 内存爆炸 vs 线性增长 + 定性结果 |
| Fig.2 | 折线图 | P0 | 95% | Memory/Latency vs #Images 曲线 |
| Fig.3 | 柱状图 | P0 | 90% | Covisibility vs Random sparsity 质量对比 |
| Fig.4 | 定性图 | P1 | 95% | Depth/PointCloud 对比（Dense vs Sparse） |
| Fig.5 | 热力图 | P1 | 99% | 共可见性矩阵可视化 |
| Fig.6 | 流程图 | P1 | 99% | 方法整体架构 |
| Tab.1 | 数据表 | P0 | 95% | 缩放性能（核心结果） |
| Tab.2 | 数据表 | P0 | 99% | 输出一致性（dense vs sparse 误差） |
| Tab.3 | 数据表 | P0 | 90% | 消融实验（k, τ） |
| Tab.4 | 数据表 | P2 | 70% | vs 通用稀疏注意力方法 |

---

## 四、关于效率公式的务实建议

### 应该写进论文的公式（共 12 个）

| 编号 | 公式 | 用途 | 是否需要实验验证 |
|:----:|------|------|:---:|
| Eq.1 | 标准注意力 QKᵀ/√d | 背景 | 否 |
| Eq.2 | 内存复杂度 O(n²s²) | 动机 | 用实验 1 验证 |
| Eq.3 | 特征提取 fᵢ = MegaLoc(Iᵢ) | 方法 | 否 |
| Eq.4 | 余弦相似度 S(i,j) = f̂ᵢᵀf̂ⱼ | 方法 | 否 |
| Eq.5 | 掩码构建 M(i,j) = ... | 方法核心 | 用实验 3 验证 |
| Eq.6 | 时序连通性约束 | 方法 | 否 |
| Eq.7 | 稀疏注意力计算 | 方法 | 用实验 2 验证 |
| Eq.8 | 稀疏度 ρ | 分析 | 用实验 3 验证 |
| Eq.9 | 内存节省比 R = n/k | 分析核心 | 用实验 1 验证 |
| Eq.10 | FLOPs 节省比 | 分析 | 用实验 1 验证 |
| Eq.11 | MegaLoc 额外开销 | 分析 | 用实验 1 验证 |
| Eq.12 | 总开销对比 break-even point | 分析 | 用实验 1 验证 |

### 不应该写进论文的公式

| 放弃的公式 | 原因 |
|-----------|------|
| 概率乘法聚合 α(x)=1-Π(1-αᵢ) | 来自 GaussianFormer-2，不适用于 VGGT 架构 |
| GMM 语义预测 | 同上，occupancy 专用 |
| 软 sigmoid 掩码 | 改进极小，反而增加超参数 |
| Bhattacharyya 重叠系数 | 评估 Gaussian 重叠，与本项目无关 |
| MPS 硬件 Roofline 模型 | 对二区论文来说太深入硬件，偏离主题 |

---

## 五、执行计划（按时间顺序）

### Step 1：跑通基准测试脚本

```bash
# 需要实现或完善的核心脚本
vggt benchmark --mode scaling \
    --images 10,20,30,50,75,100,150,200 \
    --methods dense,sparse \
    --sparse-k 5,10,20 \
    --output results/scaling_benchmark.json
```

**产出：** Tab.1 + Fig.2 的数据

### Step 2：跑输出一致性对比

```bash
vggt benchmark --mode consistency \
    --images 5,10,20,30 \
    --compare dense,sparse \
    --metrics depth_l1,pose_rotation,pose_translation,chamfer \
    --output results/consistency.json
```

**产出：** Tab.2 的数据

### Step 3：跑消融实验

```bash
# k 消融
vggt benchmark --mode ablation-k \
    --images 30 --sparse-k 3,5,10,15,20,30

# τ 消融
vggt benchmark --mode ablation-tau \
    --images 30 --threshold 0.3,0.5,0.7,0.8,0.9

# 共可见性 vs 随机（最重要的消融！）
vggt benchmark --mode ablation-mask \
    --images 30 --mask-types covisibility,random,sliding_window
```

**产出：** Tab.3 + Fig.3 的数据

### Step 4：生成可视化

```bash
vggt benchmark --mode visualize \
    --images 30 \
    --output-dir results/figures/
```

**产出：** Fig.1, Fig.4, Fig.5

### Step 5：写论文

按照 Section 3 的结构写，先填数据和图表，再写文字。

---

## 六、风险与对策

| 风险 | 概率 | 对策 |
|------|:----:|------|
| Dense VGGT 在小 n 就 OOM | 30% | 降低图像分辨率 / 用 CPU+swap |
| 输出一致性不是精确 0 | 20% | 没关系，只要误差极小就行，报告实际数值 |
| 消融显示随机稀疏也行 | 15% | 加大 n 试试（大 n 时差异应该更明显） |
| MegaLoc 特征质量不够 | 10% | 换用更简单的特征（如 ResNet 余弦相似度） |
| 审稿人觉得创新不够 | 25% | 强调 (1) training-free (2) task-specific sparsity (3) 首次将 covisibility 用于 transformer sparsification |

---

## 七、与原方案 v1 的对比

| 维度 | v1（原方案） | v2（本方案） |
|------|-------------|-------------|
| 方向 | 移植 GaussianFormer-2 公式 | 围绕自有贡献写论文 |
| 公式数 | 多（包含不适用的） | 12 个（全部可验证） |
| 实验依赖 | 需要重训练 | 全部 inference-time |
| 成功概率 | ~40% | ~85% |
| 图表 | 6 张（部分无数据支撑） | 6 图 + 4 表（全部有数据） |
| 创新叙事 | 不清晰 | Covisibility-guided task-specific sparse attention |

---

*方案更新日期：2026-01-28*
