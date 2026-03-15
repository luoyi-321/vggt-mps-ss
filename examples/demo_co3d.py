#!/usr/bin/env python3
"""
Test VGGT 3D Reconstruction with Real CO3D Dataset Images

This script:
1. Loads real images from a downloaded CO3D sequence (e.g. tv category)
2. Runs VGGT inference to predict depth maps, camera poses, and 3D points
3. Visualizes results and optionally compares with CO3D ground-truth depth
4. Exports the reconstructed point cloud as PLY

Usage:
    # Use default CO3D path (co3d-main/tv/)
    python examples/demo_co3d.py

    # Specify a custom CO3D category folder
    python examples/demo_co3d.py --co3d_dir /path/to/co3d/category

    # Specify number of views and a specific sequence
    python examples/demo_co3d.py --num_views 4 --sequence 396_49386_97450

    # Skip model inference (just verify data loading)
    python examples/demo_co3d.py --dry_run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_co3d_sequences(co3d_dir: Path) -> list[Path]:
    """Find all valid CO3D sequences (directories containing an images/ subfolder)."""
    sequences = []
    for d in sorted(co3d_dir.iterdir()):
        if d.is_dir() and (d / "images").is_dir():
            sequences.append(d)
    return sequences


def load_co3d_frames(sequence_dir: Path, num_views: int = 4) -> list[Path]:
    """Load evenly-spaced frames from a CO3D sequence."""
    images_dir = sequence_dir / "images"
    all_frames = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not all_frames:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Pick evenly-spaced frames
    if len(all_frames) <= num_views:
        return all_frames

    indices = np.linspace(0, len(all_frames) - 1, num_views, dtype=int)
    return [all_frames[i] for i in indices]


def load_gt_depth(sequence_dir: Path, frame_name: str) -> np.ndarray | None:
    """Load ground-truth depth map from CO3D if available."""
    depth_dir = sequence_dir / "depths"
    if not depth_dir.is_dir():
        return None

    # CO3D depth files typically match frame filenames but with different extension
    depth_candidates = [
        depth_dir / frame_name.replace(".jpg", ".png"),
        depth_dir / frame_name.replace(".jpg", ".exr"),
        depth_dir / frame_name,  # same name
    ]
    # Also look for .jpg.geometric.png pattern
    depth_candidates.append(depth_dir / f"{frame_name}.geometric.png")

    for dp in depth_candidates:
        if dp.exists():
            try:
                depth_img = Image.open(dp)
                return np.array(depth_img).astype(np.float32)
            except Exception:
                continue

    # Try to find any depth file matching the frame number
    frame_num = "".join(c for c in Path(frame_name).stem if c.isdigit())
    for df in sorted(depth_dir.iterdir()):
        file_num = "".join(c for c in df.stem if c.isdigit())
        if frame_num and file_num == frame_num:
            try:
                depth_img = Image.open(df)
                return np.array(depth_img).astype(np.float32)
            except Exception:
                continue

    return None


def main():
    parser = argparse.ArgumentParser(description="Test VGGT with CO3D dataset")
    parser.add_argument(
        "--co3d_dir",
        type=str,
        default=str(PROJECT_ROOT / "co3d-main" / "tv"),
        help="Path to CO3D category directory (e.g. co3d-main/tv/)",
    )
    parser.add_argument("--num_views", type=int, default=4, help="Number of views to use")
    parser.add_argument("--sequence", type=str, default=None, help="Specific sequence name")
    parser.add_argument("--dry_run", action="store_true", help="Only load data, skip inference")
    parser.add_argument("--save_vis", action="store_true", default=True, help="Save visualizations")
    args = parser.parse_args()

    print("=" * 60)
    print("🎬 VGGT × CO3D Dataset Test")
    print("=" * 60)

    # ── 1. Find CO3D data ──────────────────────────────────────
    co3d_dir = Path(args.co3d_dir)
    if not co3d_dir.exists():
        print(f"❌ CO3D directory not found: {co3d_dir}")
        print("   Download it first:")
        print("   cd co3d-main && python co3d/download_dataset.py --download_folder ./ --n_download_workers 4")
        sys.exit(1)

    sequences = find_co3d_sequences(co3d_dir)
    if not sequences:
        print(f"❌ No valid sequences found in {co3d_dir}")
        print("   Each sequence should have an 'images/' subdirectory.")
        sys.exit(1)

    print(f"📂 CO3D directory: {co3d_dir}")
    print(f"📁 Found {len(sequences)} sequences")

    # Select sequence
    if args.sequence:
        seq_dir = co3d_dir / args.sequence
        if not seq_dir.exists():
            print(f"❌ Sequence not found: {args.sequence}")
            print(f"   Available: {[s.name for s in sequences[:10]]}...")
            sys.exit(1)
    else:
        seq_dir = sequences[0]

    print(f"🎯 Using sequence: {seq_dir.name}")
    print("-" * 60)

    # ── 2. Load frames ─────────────────────────────────────────
    frame_paths = load_co3d_frames(seq_dir, num_views=args.num_views)
    print(f"📸 Selected {len(frame_paths)} frames:")
    for fp in frame_paths:
        print(f"   • {fp.name}")

    # Load images
    images = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        images.append(img)
        print(f"   ✅ {fp.name}: {img.size[0]}×{img.size[1]}")

    # Check for ground-truth depth
    gt_depths = []
    for fp in frame_paths:
        gt_depth = load_gt_depth(seq_dir, fp.name)
        gt_depths.append(gt_depth)
        if gt_depth is not None:
            print(f"   📏 GT depth for {fp.name}: shape={gt_depth.shape}")

    has_gt = any(d is not None for d in gt_depths)

    # Check for ground-truth point cloud
    gt_ply = seq_dir / "pointcloud.ply"
    if gt_ply.exists():
        print(f"   ☁️  GT point cloud: {gt_ply} ({gt_ply.stat().st_size / 1e6:.1f} MB)")

    if args.dry_run:
        print("\n✅ Dry run complete — data loads successfully!")
        print("   Remove --dry_run to run VGGT inference.")
        return

    # ── 3. Set up device ───────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n✅ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("\n✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("\n⚠️  Using CPU (this will be slow)")

    # ── 4. Load VGGT model ─────────────────────────────────────
    print("\n📥 Loading VGGT model...")

    # Try to import from the repo
    vggt_repo = PROJECT_ROOT / "repo" / "vggt"
    sys.path.insert(0, str(vggt_repo))

    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except ImportError:
        print("❌ Could not import VGGT. Make sure repo/vggt/ exists with the model code.")
        print("   See README.md for setup instructions.")
        sys.exit(1)

    # Load model weights
    model_path = vggt_repo / "vggt_model.pt"
    model_path_alt = PROJECT_ROOT / "models" / "vggt_model.pt"

    if model_path.exists():
        print(f"📂 Loading from: {model_path}")
        model = VGGT()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    elif model_path_alt.exists():
        print(f"📂 Loading from: {model_path_alt}")
        model = VGGT()
        checkpoint = torch.load(model_path_alt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    else:
        print("📥 Downloading from HuggingFace...")
        model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = model.to(device).eval()
    print("✅ Model loaded!")

    # ── 5. Preprocess and run inference ────────────────────────
    print("\n🖼️  Preprocessing CO3D images...")

    # Save frames temporarily for VGGT's loader (expects file paths)
    output_dir = PROJECT_ROOT / "outputs" / "co3d_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_paths = []
    for i, img in enumerate(images):
        temp_path = output_dir / f"_temp_co3d_{i:03d}.jpg"
        img.save(temp_path, quality=95)
        temp_paths.append(str(temp_path))

    # Use VGGT's own image loader
    input_tensor = load_and_preprocess_images(temp_paths).to(device)
    print(f"   Input tensor shape: {input_tensor.shape}")  # [1, N, 3, H, W]

    print("🧠 Running VGGT inference on real CO3D images...")
    with torch.no_grad():
        if device.type == "mps":
            predictions = model(input_tensor)
        else:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predictions = model(input_tensor)

    print("✅ Inference complete!")

    # Clean up temp files
    for tp in temp_paths:
        Path(tp).unlink(missing_ok=True)

    # ── 6. Extract and display results ─────────────────────────
    print("\n📊 Results:")
    for key, val in predictions.items():
        if isinstance(val, torch.Tensor):
            print(f"   {key}: shape={list(val.shape)}, dtype={val.dtype}")

    # Extract predictions
    pred_depths = predictions["depth"].cpu().numpy()     # [1, N, H, W, 1]
    pred_points = predictions.get("world_points_xyz")    # [1, N, H, W, 3] if available
    pred_conf = predictions.get("world_points_confidence")  # confidence

    # ── 7. Visualize ───────────────────────────────────────────
    if args.save_vis:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(frame_paths)
        rows = 3 if has_gt else 2

        fig, axes = plt.subplots(rows, n, figsize=(5 * n, 5 * rows))
        if n == 1:
            axes = axes[:, np.newaxis]

        for i in range(n):
            # Row 1: Input images
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f"CO3D: {frame_paths[i].name}", fontsize=9)
            axes[0, i].axis("off")

            # Row 2: Predicted depth
            depth_i = pred_depths[0, i, :, :, 0]
            im = axes[1, i].imshow(depth_i, cmap="viridis")
            axes[1, i].set_title(f"Predicted Depth", fontsize=9)
            axes[1, i].axis("off")
            plt.colorbar(im, ax=axes[1, i], fraction=0.046)

            # Row 3: GT depth (if available)
            if has_gt:
                if gt_depths[i] is not None:
                    axes[2, i].imshow(gt_depths[i], cmap="viridis")
                    axes[2, i].set_title("GT Depth", fontsize=9)
                else:
                    axes[2, i].text(0.5, 0.5, "No GT", ha="center", va="center",
                                    transform=axes[2, i].transAxes)
                    axes[2, i].set_title("GT Depth (N/A)", fontsize=9)
                axes[2, i].axis("off")

        plt.suptitle(f"VGGT on CO3D — Sequence: {seq_dir.name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        vis_path = output_dir / "co3d_depth_comparison.png"
        plt.savefig(vis_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n🖼️  Saved depth visualization: {vis_path}")

        # ── 8. Generate and export point cloud ─────────────────
        if pred_points is not None:
            pts = pred_points[0].cpu().numpy()       # [N, H, W, 3]
            conf = pred_conf[0].cpu().numpy() if pred_conf is not None else None  # [N, H, W, 1]
        else:
            # Fallback: back-project from depth with simple pinhole
            print("   ℹ️  Using simple back-projection (no world_points_xyz in output)")
            pts_list = []
            for i in range(n):
                depth_i = pred_depths[0, i, :, :, 0]
                h, w = depth_i.shape
                fx = fy = max(h, w)
                cx, cy = w / 2, h / 2
                xx, yy = np.meshgrid(np.arange(w), np.arange(h))
                z = depth_i
                x = (xx - cx) * z / fx
                y = (yy - cy) * z / fy
                pts_list.append(np.stack([x, y, z], axis=-1))
            pts = np.stack(pts_list, axis=0)
            conf = None

        # Collect colored points
        all_points = []
        all_colors = []
        step = 4  # downsample
        for i in range(n):
            img_np = np.array(images[i].resize((pts.shape[2], pts.shape[1])))
            p = pts[i, ::step, ::step].reshape(-1, 3)
            c = img_np[::step, ::step].reshape(-1, 3)

            # Filter by confidence if available
            if conf is not None:
                c_i = conf[i, ::step, ::step].reshape(-1)
                mask = c_i > 0.3
                p = p[mask]
                c = c[mask]

            # Remove invalid points
            valid = np.isfinite(p).all(axis=1) & (np.abs(p) < 100).all(axis=1)
            all_points.append(p[valid])
            all_colors.append(c[valid])

        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)
        print(f"   ☁️  Total 3D points: {len(combined_points):,}")

        # Save PLY
        ply_path = output_dir / f"co3d_{seq_dir.name}_reconstruction.ply"
        with open(ply_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(combined_points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for pt, cl in zip(combined_points, combined_colors):
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(cl[0])} {int(cl[1])} {int(cl[2])}\n")

        print(f"   💾 Saved point cloud: {ply_path}")

        # 3D scatter preview
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        idx = np.random.choice(len(combined_points), min(8000, len(combined_points)), replace=False)
        ax.scatter(
            combined_points[idx, 0],
            combined_points[idx, 1],
            combined_points[idx, 2],
            c=combined_colors[idx] / 255.0,
            s=0.5,
            alpha=0.6,
        )
        ax.set_title(f"3D Reconstruction — CO3D {seq_dir.name}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        scatter_path = output_dir / "co3d_3d_scatter.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   📸 Saved 3D scatter: {scatter_path}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ CO3D Test Complete!")
    print(f"   Sequence: {seq_dir.name}")
    print(f"   Views:    {len(frame_paths)}")
    print(f"   Device:   {device}")
    print(f"   Results:  {output_dir}")
    if gt_ply.exists():
        print(f"   GT PLY:   {gt_ply}")
        print("   💡 Tip: Open both PLYs in MeshLab/CloudCompare to compare!")
    print("=" * 60)


if __name__ == "__main__":
    main()
