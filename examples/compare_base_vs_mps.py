#!/usr/bin/env python3
"""
Compare Base VGGT vs VGGT-MPS on Real CO3D Images

Runs the SAME model weights in 3 modes and compares:
  1. Base VGGT  → CPU (float32)  — the reference baseline
  2. VGGT-MPS   → MPS (float32)  — Apple Silicon accelerated
  3. VGGT-MPS + Sparse Attn → MPS — sparse attention mode (optional)

Metrics:
  - Depth MAE / RMSE vs base
  - Camera pose difference
  - Inference time (ms)
  - Peak memory (MB)
  - GT depth error (if CO3D ground-truth available)

Usage:
    # Default: compare on first CO3D tv sequence
    python examples/compare_base_vs_mps.py

    # Specify sequence and views
    python examples/compare_base_vs_mps.py --sequence 396_49386_97450 --num_views 4

    # Skip CPU baseline (too slow for large inputs)
    python examples/compare_base_vs_mps.py --skip_cpu

    # Include sparse attention mode
    python examples/compare_base_vs_mps.py --include_sparse --k_nearest 3

    # Use your own images instead of CO3D
    python examples/compare_base_vs_mps.py --images_dir ./data/my_images
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt"))


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def find_images(source_dir: Path, num_views: int) -> list[str]:
    """Find evenly-spaced images from a directory."""
    all_imgs = sorted(source_dir.glob("*.jpg")) + sorted(source_dir.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No images in {source_dir}")
    if len(all_imgs) <= num_views:
        return [str(p) for p in all_imgs]
    idx = np.linspace(0, len(all_imgs) - 1, num_views, dtype=int)
    return [str(all_imgs[i]) for i in idx]


def load_gt_depth(seq_dir: Path, frame_name: str) -> np.ndarray | None:
    """Load CO3D ground-truth depth if available."""
    depth_dir = seq_dir / "depths"
    if not depth_dir.is_dir():
        return None
    frame_num = "".join(c for c in Path(frame_name).stem if c.isdigit())
    for df in sorted(depth_dir.iterdir()):
        file_num = "".join(c for c in df.stem if c.isdigit())
        if frame_num and file_num == frame_num:
            try:
                return np.array(Image.open(df)).astype(np.float32)
            except Exception:
                continue
    return None


def compute_depth_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute depth comparison metrics between two depth maps."""
    # Resize pred to match ref if needed
    if pred.shape != ref.shape:
        from PIL import Image as PILImage
        pred = np.array(PILImage.fromarray(pred).resize(
            (ref.shape[1], ref.shape[0]), PILImage.BILINEAR
        ))

    valid = np.isfinite(pred) & np.isfinite(ref) & (ref > 0)
    if valid.sum() < 10:
        return {"mae": float("nan"), "rmse": float("nan"), "abs_rel": float("nan")}

    p, r = pred[valid], ref[valid]
    diff = np.abs(p - r)
    return {
        "mae": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "abs_rel": float(np.mean(diff / (r + 1e-8))),
        "max_diff": float(np.max(diff)),
        "corr": float(np.corrcoef(p.flatten(), r.flatten())[0, 1]),
    }


def measure_memory():
    """Get current memory usage in MB."""
    if torch.backends.mps.is_available():
        # MPS doesn't have detailed memory tracking, use process memory
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS returns bytes
    elif torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


# ═══════════════════════════════════════════════════════════════
# Inference runners
# ═══════════════════════════════════════════════════════════════

def run_inference(model, image_paths: list[str], device: torch.device, label: str):
    """Run VGGT inference and measure time + memory."""
    from vggt.utils.load_fn import load_and_preprocess_images

    print(f"\n{'─' * 50}")
    print(f"🔄 [{label}] on {device} ...")

    # Preprocess
    input_tensor = load_and_preprocess_images(image_paths).to(device)
    print(f"   Input: {input_tensor.shape} → {device}")

    # Warm-up (for MPS/CUDA)
    if device.type != "cpu":
        with torch.no_grad():
            _ = model(input_tensor)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # Timed run
    mem_before = measure_memory()
    t0 = time.perf_counter()
    with torch.no_grad():
        predictions = model(input_tensor)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    mem_after = measure_memory()

    # Move everything to CPU numpy
    results = {}
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            results[key] = val.cpu().numpy()

    print(f"   ⏱  {elapsed_ms:.0f} ms")
    print(f"   📦 Memory: {mem_after:.0f} MB")
    for key, val in results.items():
        print(f"   📐 {key}: {list(val.shape)}")

    return results, elapsed_ms, mem_after


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Compare Base VGGT vs VGGT-MPS")
    parser.add_argument("--co3d_dir", type=str, default=str(PROJECT_ROOT / "co3d-main" / "tv"))
    parser.add_argument("--images_dir", type=str, default=None, help="Custom images dir (overrides CO3D)")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--skip_cpu", action="store_true", help="Skip CPU baseline (faster)")
    parser.add_argument("--include_sparse", action="store_true", help="Also test sparse attention")
    parser.add_argument("--k_nearest", type=int, default=3, help="k for sparse attention")
    args = parser.parse_args()

    print("═" * 60)
    print("⚖️  Base VGGT vs VGGT-MPS — Real CO3D Comparison")
    print("═" * 60)

    # ── 1. Find images ─────────────────────────────────────────
    seq_dir = None
    if args.images_dir:
        images_dir = Path(args.images_dir)
        print(f"📂 Using custom images: {images_dir}")
    else:
        co3d_dir = Path(args.co3d_dir)
        sequences = sorted(
            [d for d in co3d_dir.iterdir() if d.is_dir() and (d / "images").is_dir()]
        )
        if not sequences:
            print(f"❌ No CO3D sequences in {co3d_dir}")
            sys.exit(1)
        seq_dir = co3d_dir / args.sequence if args.sequence else sequences[0]
        images_dir = seq_dir / "images"
        print(f"📂 CO3D sequence: {seq_dir.name}")

    image_paths = find_images(images_dir, args.num_views)
    print(f"📸 {len(image_paths)} images selected")
    for p in image_paths:
        print(f"   • {Path(p).name}")

    # ── 2. Load model (once, shared weights) ───────────────────
    print(f"\n📥 Loading VGGT model...")
    from vggt.models.vggt import VGGT

    model_candidates = [
        PROJECT_ROOT / "models" / "model.pt",
        PROJECT_ROOT / "models" / "vggt_model.pt",
        PROJECT_ROOT / "repo" / "vggt" / "vggt_model.pt",
    ]
    model_path = None
    for mp in model_candidates:
        if mp.exists() and mp.stat().st_size > 1_000_000:
            model_path = mp
            break

    if model_path:
        print(f"   Loading from: {model_path} ({model_path.stat().st_size / 1e9:.1f} GB)")
        model = VGGT()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
    else:
        print("   Downloading from HuggingFace...")
        model = VGGT.from_pretrained("facebook/VGGT-1B")

    model.eval()
    print("✅ Model loaded (shared weights for all modes)")

    # ── 3. Run each mode ───────────────────────────────────────
    all_results = {}

    # Mode A: Base VGGT on CPU
    if not args.skip_cpu:
        model_cpu = model.to("cpu")
        preds_cpu, time_cpu, mem_cpu = run_inference(
            model_cpu, image_paths, torch.device("cpu"), "Base VGGT (CPU)"
        )
        all_results["base_cpu"] = {
            "preds": preds_cpu,
            "time_ms": time_cpu,
            "memory_mb": mem_cpu
        }
        # Free CPU model before loading MPS
        del model_cpu
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # Mode B: VGGT-MPS
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        model_mps = model.to(mps_device)
        preds_mps, time_mps, mem_mps = run_inference(
            model_mps, image_paths, mps_device, "VGGT-MPS (Apple Silicon)"
        )
        all_results["mps"] = {
            "preds": preds_mps,
            "time_ms": time_mps,
            "memory_mb": mem_mps
        }

        # Mode C: VGGT-MPS + Sparse Attention
        if args.include_sparse:
            try:
                from vggt_mps.vggt_sparse_attention import make_vggt_sparse  # noqa: E402
                sparse_model = make_vggt_sparse(
                    model_mps, device="mps",
                    k_nearest=args.k_nearest, lightweight=True
                )
                preds_sparse, time_sparse, mem_sparse = run_inference(
                    sparse_model, image_paths, mps_device,
                    f"VGGT-MPS + Sparse (k={args.k_nearest})"
                )
                all_results["mps_sparse"] = {
                    "preds": preds_sparse,
                    "time_ms": time_sparse,
                    "memory_mb": mem_sparse
                }
            except Exception as e:
                print(f"⚠️  Sparse attention failed: {e}")
    else:
        print("⚠️  MPS not available — only CPU baseline will run")

    # ── 4. Compare outputs ─────────────────────────────────────
    print("\n" + "═" * 60)
    print("📊 COMPARISON RESULTS")
    print("═" * 60)

    # Timing table
    print("\n⏱  Inference Time:")
    print(f"   {'Mode':<30} {'Time (ms)':>10} {'Speedup':>10}")
    print(f"   {'─' * 52}")
    base_time = all_results.get("base_cpu", {}).get("time_ms", None)
    for mode, data in all_results.items():
        speedup = f"{base_time / data['time_ms']:.1f}x" if base_time else "—"
        print(f"   {mode:<30} {data['time_ms']:>10.0f} {speedup:>10}")

    # Memory table
    print("\n📦 Memory Usage:")
    for mode, data in all_results.items():
        print(f"   {mode:<30} {data['memory_mb']:>8.0f} MB")

    # Depth comparison (MPS vs CPU)
    if "base_cpu" in all_results and "mps" in all_results:
        print("\n📐 Depth Accuracy (MPS vs Base CPU):")
        cpu_depth = all_results["base_cpu"]["preds"]["depth"]
        mps_depth = all_results["mps"]["preds"]["depth"]
        metrics = compute_depth_metrics(mps_depth.flatten(), cpu_depth.flatten())
        print(f"   MAE:     {metrics['mae']:.6f}")
        print(f"   RMSE:    {metrics['rmse']:.6f}")
        print(f"   AbsRel:  {metrics['abs_rel']:.6f}")
        print(f"   MaxDiff: {metrics['max_diff']:.6f}")
        print(f"   Corr:    {metrics['corr']:.6f}")

        if metrics["mae"] < 1e-4:
            print("   ✅ Outputs are effectively IDENTICAL (< 1e-4 MAE)")
        elif metrics["mae"] < 1e-2:
            print("   ✅ Outputs are very close (< 1e-2 MAE)")
        else:
            print("   ⚠️  Noticeable difference — check precision settings")

    # Sparse vs Dense comparison
    if "mps" in all_results and "mps_sparse" in all_results:
        print(f"\n📐 Depth Accuracy (Sparse k={args.k_nearest} vs Dense MPS):")
        dense_depth = all_results["mps"]["preds"]["depth"]
        sparse_depth = all_results["mps_sparse"]["preds"]["depth"]
        metrics = compute_depth_metrics(sparse_depth.flatten(), dense_depth.flatten())
        print(f"   MAE:     {metrics['mae']:.6f}")
        print(f"   RMSE:    {metrics['rmse']:.6f}")
        print(f"   Corr:    {metrics['corr']:.6f}")

    # Compare with CO3D ground-truth depth
    if seq_dir is not None:
        print("\n🎯 Ground-Truth Comparison (CO3D depth):")
        for mode, data in all_results.items():
            pred_depths = data["preds"]["depth"]  # [1, N, H, W, 1]
            errors = []
            for i, img_path in enumerate(image_paths):
                gt = load_gt_depth(seq_dir, Path(img_path).name)
                if gt is not None and gt.max() > 0:
                    pred_d = pred_depths[0, i, :, :, 0]
                    m = compute_depth_metrics(pred_d, gt)
                    errors.append(m)
            if errors:
                avg_mae = np.mean([e["mae"] for e in errors if np.isfinite(e["mae"])])
                avg_rmse = np.mean([e["rmse"] for e in errors if np.isfinite(e["rmse"])])
                print(f"   {mode:<30} MAE={avg_mae:.4f}  RMSE={avg_rmse:.4f}  ({len(errors)} frames)")
            else:
                print(f"   {mode:<30} (no valid GT depth found)")

    # ── 5. Visualize ───────────────────────────────────────────
    output_dir = PROJECT_ROOT / "outputs" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        modes = list(all_results.keys())
        n_views = len(image_paths)
        n_rows = 1 + len(modes)  # input row + one depth row per mode

        fig, axes = plt.subplots(n_rows, n_views, figsize=(5 * n_views, 4 * n_rows))
        if n_views == 1:
            axes = axes[:, np.newaxis]

        # Row 0: Input images
        for j in range(n_views):
            img = Image.open(image_paths[j])
            axes[0, j].imshow(img)
            axes[0, j].set_title(Path(image_paths[j]).name, fontsize=8)
            axes[0, j].axis("off")

        # Subsequent rows: depth per mode
        for row, mode in enumerate(modes, start=1):
            preds = all_results[mode]["preds"]
            for j in range(n_views):
                d = preds["depth"][0, j, :, :, 0]
                im = axes[row, j].imshow(d, cmap="viridis")
                axes[row, j].axis("off")
                if j == 0:
                    axes[row, j].set_ylabel(mode, fontsize=10, rotation=0, labelpad=80)
                plt.colorbar(im, ax=axes[row, j], fraction=0.04)

        plt.suptitle(
            f"VGGT Comparison — {seq_dir.name if seq_dir else 'Custom Images'}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        fig_path = output_dir / "base_vs_mps_comparison.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n🖼️  Saved visual comparison: {fig_path}")
    except Exception as e:
        print(f"⚠️  Visualization skipped: {e}")

    # ── 6. Save JSON report ────────────────────────────────────
    report = {
        "sequence": seq_dir.name if seq_dir else str(args.images_dir),
        "num_views": len(image_paths),
        "modes": {}
    }
    for mode, data in all_results.items():
        report["modes"][mode] = {
            "time_ms": round(data["time_ms"], 1),
            "memory_mb": round(data["memory_mb"], 1),
            "depth_shape": list(data["preds"]["depth"].shape),
        }
    # Add cross-mode metrics
    if "base_cpu" in all_results and "mps" in all_results:
        cpu_d = all_results["base_cpu"]["preds"]["depth"]
        mps_d = all_results["mps"]["preds"]["depth"]
        report["mps_vs_cpu"] = compute_depth_metrics(mps_d.flatten(), cpu_d.flatten())

    json_path = output_dir / "comparison_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Saved report: {json_path}")

    print("\n" + "═" * 60)
    print("✅ Comparison Complete!")
    print("═" * 60)


if __name__ == "__main__":
    main()
