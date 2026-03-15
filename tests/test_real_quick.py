#!/usr/bin/env python3
"""Test real VGGT with actual model weights on MPS"""

import torch
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import glob

print("=" * 60)
print("🍎 VGGT Real Model Test on MPS")
print("=" * 60)

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add VGGT to path
sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Check if model file exists (correct path: models/model.pt)
model_path = PROJECT_ROOT / "models" / "model.pt"
if not model_path.exists():
    print(f"❌ Model file not found at {model_path}")
    print("Please download from: https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
    exit(1)

print(f"✅ Model file found: {model_path} ({model_path.stat().st_size / 1e9:.1f} GB)")

# Load model
print("\n🔄 Loading model weights...")
model = VGGT()
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()
print("✅ Model loaded!")

# Find real test images from data/real_data/
real_data_dir = PROJECT_ROOT / "data" / "real_data"
test_images = []

# Search for real images in bottle_cap directory (or any available object)
for obj_dir in ["bottle_cap", "audiojack", "end_cap", "button_battery", "eraser"]:
    obj_path = real_data_dir / obj_dir
    if obj_path.exists():
        # Find jpg images
        found_images = sorted(glob.glob(str(obj_path / "**" / "*.jpg"), recursive=True))[:2]
        if found_images:
            test_images = [Path(p) for p in found_images]
            print(f"\n📸 Using real images from {obj_dir}/")
            break

# Fallback to synthetic images if no real data found
if not test_images:
    print("\n⚠️ No real images found, creating synthetic test images...")
    test_dir = Path(__file__).parent / "tmp" / "inputs"
    test_dir.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        img = Image.new('RGB', (640, 480), color=(100 + i*50, 150, 200 - i*30))
        img_path = test_dir / f"quick_test_{i+1}.jpg"
        img.save(img_path)
        test_images.append(img_path)

# Convert to strings
image_paths = [str(p) for p in test_images]
print(f"\n🖼️ Using images: {image_paths}")

# Load and preprocess
print("\n🔄 Preprocessing images...")
input_images = load_and_preprocess_images(image_paths).to(device)
print(f"✅ Input shape: {input_images.shape}")

# Run inference
print("\n🧠 Running VGGT inference on MPS...")
with torch.no_grad():
    predictions = model(input_images)

print("\n✅ Inference complete!")
print(f"   - Depth: {predictions['depth'].shape}")
print(f"   - Camera poses: {predictions['pose_enc'].shape}")
print(f"   - World points: {predictions['world_points'].shape}")

# Save one depth map
depth = predictions['depth'][0, 0, :, :, 0].cpu().numpy()
output_dir = Path(__file__).parent / "tmp" / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap='viridis')
plt.colorbar()
plt.title('VGGT Depth Map (Real Model on MPS)')
plt.savefig(output_dir / "real_depth_mps.png", dpi=100, bbox_inches='tight')
print(f"\n💾 Saved depth map to {output_dir}/real_depth_mps.png")

print("\n" + "=" * 60)
print("✅ VGGT Real Model Test Complete!")
print("🍎 Running on Apple Silicon with MPS acceleration")
print("=" * 60)