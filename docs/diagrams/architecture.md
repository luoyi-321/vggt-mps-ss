# VGGT-MPS Architecture Diagrams

This directory contains architecture diagrams for VGGT-MPS.

## Diagram 1: Overall Architecture Pipeline

```mermaid
flowchart TB
    subgraph Input["Input Stage"]
        A["Multi-view Images<br/>[B, S, C, H, W]"]
    end

    subgraph Feature["Feature Extraction"]
        B["DINOv2 ViT-B/14<br/>[B, S, N, 768]"]
        C["MegaLoc SALAD<br/>[B, S, 16640]"]
    end

    subgraph Covis["Covisibility Detection"]
        D["Similarity Matrix<br/>[S, S]"]
        E{"Soft Mask?"}
        F["Hard Binary Mask<br/>M(i,j) ∈ {0,1}"]
        G["Soft Probability Mask<br/>M(i,j) = σ((sim-τ)/T)"]
        H["K-Nearest Guarantee"]
    end

    subgraph Attention["Sparse VGGT Transformer"]
        I["Masked Attention<br/>O(n·k) vs O(n²)"]
        J["Cross-View Aggregation"]
    end

    subgraph Output["3D Outputs"]
        K["Depth Maps<br/>[B, S, H, W]"]
        L["Camera Poses<br/>[B, S, 4, 4]"]
        M["Point Cloud<br/>[B, N, 3]"]
        N["Confidence<br/>[B, S, H, W]"]
    end

    A --> B --> C
    C --> D
    D --> E
    E -->|No| F
    E -->|Yes| G
    F --> H
    G --> H
    H --> I
    I --> J
    J --> K & L & M & N

    style Input fill:#e1f5fe
    style Feature fill:#fff3e0
    style Covis fill:#f3e5f5
    style Attention fill:#e8f5e9
    style Output fill:#fce4ec
```

## Diagram 2: Efficiency Comparison (Conceptual)

```mermaid
graph LR
    subgraph Dense["Dense Attention O(n²)"]
        DA["n=100<br/>10,000 ops"]
        DB["n=500<br/>250,000 ops"]
        DC["n=1000<br/>1,000,000 ops"]
    end

    subgraph Sparse["Sparse Attention O(n·k)"]
        SA["n=100, k=10<br/>1,000 ops"]
        SB["n=500, k=10<br/>5,000 ops"]
        SC["n=1000, k=10<br/>10,000 ops"]
    end

    DA -.->|"10x savings"| SA
    DB -.->|"50x savings"| SB
    DC -.->|"100x savings"| SC

    style Dense fill:#ffcdd2
    style Sparse fill:#c8e6c9
```

## Diagram 3: Probabilistic Aggregation

```mermaid
flowchart LR
    subgraph Additive["Additive Aggregation (Legacy)"]
        A1["View 1: α₁"]
        A2["View 2: α₂"]
        A3["View 3: α₃"]
        AR["Result: Σαᵢ<br/>(Unbounded)"]
        A1 & A2 & A3 -->|"+"| AR
    end

    subgraph Probabilistic["Probabilistic Aggregation"]
        P1["View 1: α₁"]
        P2["View 2: α₂"]
        P3["View 3: α₃"]
        PR["Result: 1-Π(1-αᵢ)<br/>(Bounded ≤1)"]
        P1 & P2 & P3 -->|"1-Π(1-·)"| PR
    end

    style Additive fill:#ffcdd2
    style Probabilistic fill:#c8e6c9
```

## Diagram 4: MPS Hardware Pipeline

```mermaid
flowchart TB
    subgraph SoC["Apple Silicon SoC"]
        subgraph CPU["CPU (ARM)"]
            C1["Preprocessing"]
            C2["I/O Management"]
        end

        subgraph GPU["GPU (Metal/MPS)"]
            G1["DINOv2 Features"]
            G2["MegaLoc Similarity"]
            G3["VGGT Transformer"]
            G4["3D Reconstruction"]
        end

        subgraph NPU["Neural Engine"]
            N1["(Reserved)"]
        end

        subgraph Memory["Unified Memory"]
            M1["Zero-Copy CPU↔GPU"]
            M2["16/32/64/128 GB"]
        end
    end

    C1 --> G1
    G1 --> G2 --> G3 --> G4
    G1 & G2 & G3 & G4 <--> M1

    subgraph Bottleneck["Bottleneck Analysis"]
        B1["MegaLoc: ~20% Compute"]
        B2["Attention: ~60% Memory"]
        B3["Reconstruction: ~20% Compute"]
    end

    style CPU fill:#e3f2fd
    style GPU fill:#fff3e0
    style NPU fill:#f5f5f5
    style Memory fill:#e8f5e9
```

## Diagram 5: Efficiency Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **ASR** | \|{M(i,j)=0}\| / (n²-n) | Attention Sparsity Ratio |
| **ECR** | nnz(M) / n² | Effective Computation Ratio |
| **ME** | Mem_sparse / Mem_dense | Memory Efficiency |
| **QER** | ΔQuality / ΔCompute | Quality-Efficiency Ratio |

## Diagram 6: Covisibility Matrix Pattern

```
     Covisibility Matrix (n=50, k=10)

  Images  1  5  10 15 20 25 30 35 40 45 50
    1    [█  █  █  ·  ·  ·  ·  ·  ·  ·  · ]
    5    [█  █  █  █  ·  ·  ·  ·  ·  ·  · ]
   10    [█  █  █  █  █  ·  ·  ·  ·  ·  · ]
   15    [·  █  █  █  █  █  ·  ·  ·  ·  · ]
   20    [·  ·  █  █  █  █  █  ·  ·  ·  · ]
   25    [·  ·  ·  █  █  █  █  █  ·  ·  · ]
   30    [·  ·  ·  ·  █  █  █  █  █  ·  · ]
   35    [·  ·  ·  ·  ·  █  █  █  █  █  · ]
   40    [·  ·  ·  ·  ·  ·  █  █  █  █  █ ]
   45    [·  ·  ·  ·  ·  ·  ·  █  █  █  █ ]
   50    [·  ·  ·  ·  ·  ·  ·  ·  █  █  █ ]

         █ = covisible (attention computed)
         · = not covisible (masked out)

         Sparsity: ~56%
         FLOPs saved: ~56%
```
