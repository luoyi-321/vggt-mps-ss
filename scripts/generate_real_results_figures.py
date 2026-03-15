#!/usr/bin/env python3
"""
Generate figures for real VGGT-1B evaluation results
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path(__file__).parent.parent / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Real experiment data (from our evaluation)
configs = ['Dense\n(baseline)', 'Sparse\nk=3', 'Sparse\nk=5', 'Sparse\nk=10']
inference_times = [26210.5, 18167.6, 15033.4, 13819.9]  # ms
speedups = [1.0, 1.44, 1.74, 1.90]
memory_mb = [4870.0, 4905.4, 4905.4, 4905.4]

colors = ['#2C3E50', '#E74C3C', '#E67E22', '#27AE60']

# =============================================================================
# Figure 6: Real Model Inference Time Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Inference Time Bar Chart
ax1 = axes[0]
bars = ax1.bar(configs, inference_times, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Inference Time (ms)', fontsize=12)
ax1.set_title('VGGT-1B Inference Time on Apple Silicon MPS\n(3 images, bottle_cap dataset)', fontsize=13)
ax1.set_ylim(0, 30000)

# Add value labels on bars
for bar, time in zip(bars, inference_times):
    height = bar.get_height()
    ax1.annotate(f'{time/1000:.1f}s',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add speedup annotations
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
    if speedup > 1.0:
        ax1.annotate(f'{speedup:.2f}x faster',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')

# Right: Speedup Bar Chart
ax2 = axes[1]
bars2 = ax2.bar(configs, speedups, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup vs Dense', fontsize=12)
ax2.set_title('Sparse Attention Speedup\n(Higher is Better)', fontsize=13)
ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Dense baseline')
ax2.set_ylim(0, 2.5)

# Add value labels
for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()
    ax2.annotate(f'{speedup:.2f}x',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig6_real_model_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'fig6_real_model_comparison.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'fig6_real_model_comparison.png'}")

# =============================================================================
# Figure 7: Speedup vs k Parameter
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

k_values = [3, 5, 10]
speedup_values = [1.44, 1.74, 1.90]

ax.plot(k_values, speedup_values, 'o-', markersize=12, linewidth=3, color='#27AE60', label='Measured Speedup')
ax.fill_between(k_values, 1.0, speedup_values, alpha=0.3, color='#27AE60')

# Add annotations
for k, s in zip(k_values, speedup_values):
    ax.annotate(f'{s:.2f}x', (k, s), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=12, fontweight='bold')

ax.axhline(y=1.0, color='#2C3E50', linestyle='--', linewidth=2, label='Dense baseline')
ax.set_xlabel('k (Nearest Neighbors)', fontsize=12)
ax.set_ylabel('Speedup vs Dense Attention', fontsize=12)
ax.set_title('Sparse Attention Speedup vs k Parameter\n(Real VGGT-1B on MPS, n=3 images)', fontsize=13)
ax.set_xticks(k_values)
ax.set_ylim(0.8, 2.2)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'fig7_speedup_vs_k.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'fig7_speedup_vs_k.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'fig7_speedup_vs_k.png'}")

# =============================================================================
# Figure 8: Memory Usage Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(configs))
width = 0.6

bars = ax.bar(x, memory_mb, width, color=colors, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Peak Memory (MB)', fontsize=12)
ax.set_title('Peak Memory Usage Comparison\n(VGGT-1B dominates memory at ~5GB)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(0, 6000)

# Add horizontal line for model size reference
ax.axhline(y=5000, color='red', linestyle='--', linewidth=1.5, label='~5GB Model Size')
ax.legend(loc='upper right')

# Add value labels
for bar, mem in zip(bars, memory_mb):
    height = bar.get_height()
    ax.annotate(f'{mem/1000:.2f} GB',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add note
ax.text(0.5, 0.15, 'Note: Memory is dominated by model weights (~5GB)\nAttention memory savings become significant at n>50 images',
        transform=ax.transAxes, ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'fig8_memory_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'fig8_memory_comparison.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'fig8_memory_comparison.png'}")

# =============================================================================
# Figure 9: Projected Scaling (Theoretical + Validated)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Theoretical scaling
n_images = np.array([3, 10, 20, 50, 100, 200, 500])
k = 10

# Dense: O(n²), Sparse: O(n*k)
dense_complexity = n_images ** 2
sparse_complexity = n_images * k
theoretical_speedup = dense_complexity / sparse_complexity

# Plot theoretical
ax.plot(n_images, theoretical_speedup, 's--', markersize=8, linewidth=2,
        color='#3498DB', label='Theoretical Speedup (n/k)', alpha=0.7)

# Mark validated point
validated_n = 3
validated_speedup = 1.90
ax.plot(validated_n, validated_speedup, 'o', markersize=15, color='#E74C3C',
        label=f'Validated: {validated_speedup}x at n={validated_n}', zorder=5)
ax.annotate(f'Real VGGT-1B\n{validated_speedup}x speedup',
            (validated_n, validated_speedup),
            xytext=(30, 30), textcoords='offset points',
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2))

ax.set_xlabel('Number of Images (n)', fontsize=12)
ax.set_ylabel('Speedup vs Dense Attention', fontsize=12)
ax.set_title('Sparse Attention Scaling: Validated + Projected\n(k=10, Real VGGT-1B on Apple Silicon MPS)', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, which='both')

# Add projected benefits text
ax.text(100, 3, 'Projected:\n10x at n=100\n50x at n=500',
        fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'fig9_projected_scaling.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'fig9_projected_scaling.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'fig9_projected_scaling.png'}")

# =============================================================================
# Figure 10: Summary Dashboard
# =============================================================================
fig = plt.figure(figsize=(14, 10))

# Create grid
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Top left: Inference time
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(configs, np.array(inference_times)/1000, color=colors, edgecolor='black')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Inference Time', fontsize=12, fontweight='bold')
for bar, t in zip(bars, inference_times):
    ax1.annotate(f'{t/1000:.1f}s', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10)

# Top right: Speedup
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(configs, speedups, color=colors, edgecolor='black')
ax2.axhline(y=1.0, color='gray', linestyle='--')
ax2.set_ylabel('Speedup')
ax2.set_title('Speedup vs Dense', fontsize=12, fontweight='bold')
for bar, s in zip(bars, speedups):
    ax2.annotate(f'{s:.2f}x', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Bottom left: Speedup trend
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot([3, 5, 10], [1.44, 1.74, 1.90], 'o-', markersize=10, linewidth=2, color='#27AE60')
ax3.fill_between([3, 5, 10], 1.0, [1.44, 1.74, 1.90], alpha=0.3, color='#27AE60')
ax3.axhline(y=1.0, color='gray', linestyle='--')
ax3.set_xlabel('k (Nearest Neighbors)')
ax3.set_ylabel('Speedup')
ax3.set_title('Speedup vs k Parameter', fontsize=12, fontweight='bold')
ax3.set_xticks([3, 5, 10])

# Bottom right: Key metrics summary
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = """
╔═══════════════════════════════════════════════╗
║     REAL VGGT-1B EVALUATION SUMMARY           ║
╠═══════════════════════════════════════════════╣
║  Hardware: Apple Silicon MPS                  ║
║  Model: VGGT-1B (5GB pretrained)              ║
║  Dataset: bottle_cap (3 images)               ║
║  PyTorch: 2.10.0                              ║
╠═══════════════════════════════════════════════╣
║  RESULTS:                                     ║
║  • Dense baseline: 26.2 seconds               ║
║  • Sparse k=10: 13.8 seconds                  ║
║  • Best speedup: 1.90x (k=10)                 ║
║  • No retraining required ✓                   ║
║  • Runtime patching works ✓                   ║
╠═══════════════════════════════════════════════╣
║  VALIDATED: Training-free sparse attention    ║
║  achieves significant speedup on real model!  ║
╚═══════════════════════════════════════════════╝
"""

ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
         fontsize=11, fontfamily='monospace',
         verticalalignment='center', horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='#ECF0F1', edgecolor='#2C3E50', linewidth=2))

fig.suptitle('Real VGGT-1B Sparse Attention Evaluation Results',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_dir / 'fig10_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'fig10_summary_dashboard.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'fig10_summary_dashboard.png'}")

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  - fig6_real_model_comparison.png/pdf")
print("  - fig7_speedup_vs_k.png/pdf")
print("  - fig8_memory_comparison.png/pdf")
print("  - fig9_projected_scaling.png/pdf")
print("  - fig10_summary_dashboard.png/pdf")
