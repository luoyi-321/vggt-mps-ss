#!/usr/bin/env python3
"""
CO3D Benchmark Data Preparation Script

Extracts CO3D sequences with ground-truth depth and poses for benchmarking
VGGT-MPS sparse attention efficiency.

Usage:
    python scripts/prepare_benchmark_data.py --source co3d-main/ --output data/co3d_benchmark/
    python scripts/prepare_benchmark_data.py --source co3d-main/ --output data/co3d_benchmark/ --dry-run
    python scripts/prepare_benchmark_data.py --source co3d-main/ --output data/co3d_benchmark/ --categories bottle,chair
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_co3d_links(source_dir: Path) -> Dict[str, List[str]]:
    """
    Load CO3D download links from links.json.

    Args:
        source_dir: Path to CO3D source directory

    Returns:
        Dictionary mapping category names to download URLs
    """
    links_file = source_dir / "co3d" / "links.json"
    if not links_file.exists():
        raise FileNotFoundError(f"CO3D links.json not found at {links_file}")

    with open(links_file, "r") as f:
        links = json.load(f)

    # Return full dataset links (not singlesequence)
    return links.get("full", {})


def get_available_categories(source_dir: Path) -> List[str]:
    """
    Get list of available CO3D categories.

    Args:
        source_dir: Path to CO3D source directory

    Returns:
        List of category names
    """
    links = load_co3d_links(source_dir)
    # Filter out METADATA entry
    return [cat for cat in links.keys() if cat != "METADATA"]


def find_sequence_data(
    source_dir: Path,
    category: str
) -> List[Dict[str, Any]]:
    """
    Find sequence data for a given category.

    CO3D v2 structure:
    - category/
        - sequence_name/
            - images/
            - depth_maps/
            - frame_annotations.json
            - sequence_annotations.json

    Args:
        source_dir: Path to CO3D source directory
        category: Category name (e.g., 'bottle', 'chair')

    Returns:
        List of sequence info dictionaries
    """
    category_dir = source_dir / category
    if not category_dir.exists():
        # Check if we need to look in a different location
        return []

    sequences = []
    for seq_dir in sorted(category_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        seq_info = {
            'name': seq_dir.name,
            'path': seq_dir,
            'images': [],
            'depths': [],
            'annotations': None
        }

        # Find images
        images_dir = seq_dir / "images"
        if images_dir.exists():
            seq_info['images'] = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

        # Find depth maps
        depth_dir = seq_dir / "depth_maps"
        if depth_dir.exists():
            seq_info['depths'] = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.npy"))

        # Find annotations
        annot_file = seq_dir / "frame_annotations.json"
        if annot_file.exists():
            seq_info['annotations'] = annot_file

        if seq_info['images']:
            sequences.append(seq_info)

    return sequences


def load_frame_annotations(annot_path: Path) -> List[Dict[str, Any]]:
    """
    Load frame annotations from CO3D JSON file.

    Args:
        annot_path: Path to frame_annotations.json

    Returns:
        List of frame annotation dictionaries
    """
    with open(annot_path, "r") as f:
        data = json.load(f)
    return data


def extract_camera_pose(frame_annot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract camera pose from frame annotation.

    CO3D provides camera parameters including:
    - R: 3x3 rotation matrix
    - T: 3D translation vector
    - focal_length: (fx, fy)
    - principal_point: (cx, cy)

    Args:
        frame_annot: Frame annotation dictionary

    Returns:
        Camera pose dictionary or None if not available
    """
    viewpoint = frame_annot.get("viewpoint")
    if viewpoint is None:
        return None

    pose = {
        'R': np.array(viewpoint.get("R", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        'T': np.array(viewpoint.get("T", [0, 0, 0])),
        'focal_length': viewpoint.get("focal_length", [1.0, 1.0]),
        'principal_point': viewpoint.get("principal_point", [0.5, 0.5])
    }
    return pose


def prepare_sequence(
    seq_info: Dict[str, Any],
    output_dir: Path,
    max_frames: int = 50,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Prepare a single sequence for benchmarking.

    Args:
        seq_info: Sequence information dictionary
        output_dir: Output directory for prepared data
        max_frames: Maximum number of frames to extract
        dry_run: If True, don't actually copy files

    Returns:
        Dictionary with preparation results
    """
    seq_name = seq_info['name']
    seq_output = output_dir / seq_name

    result = {
        'name': seq_name,
        'n_images': len(seq_info['images']),
        'n_depths': len(seq_info['depths']),
        'n_extracted': 0,
        'has_poses': False,
        'output_dir': str(seq_output)
    }

    if dry_run:
        result['n_extracted'] = min(len(seq_info['images']), max_frames)
        result['has_poses'] = seq_info['annotations'] is not None
        return result

    # Create output directories
    seq_output.mkdir(parents=True, exist_ok=True)
    (seq_output / "images").mkdir(exist_ok=True)
    (seq_output / "depth").mkdir(exist_ok=True)
    (seq_output / "poses").mkdir(exist_ok=True)

    # Select frames (evenly spaced if more than max_frames)
    images = seq_info['images'][:max_frames]
    n_frames = len(images)

    # Load annotations if available
    annotations = []
    if seq_info['annotations']:
        annotations = load_frame_annotations(seq_info['annotations'])
        result['has_poses'] = True

    # Copy images and extract data
    for i, img_path in enumerate(images):
        frame_name = f"frame_{i:04d}"

        # Copy image
        dst_img = seq_output / "images" / f"{frame_name}.jpg"
        shutil.copy2(img_path, dst_img)

        # Copy depth if available
        if i < len(seq_info['depths']):
            depth_path = seq_info['depths'][i]
            dst_depth = seq_output / "depth" / f"{frame_name}.npy"
            if depth_path.suffix == ".npy":
                shutil.copy2(depth_path, dst_depth)
            else:
                # Convert PNG depth to numpy
                try:
                    from PIL import Image
                    depth_img = Image.open(depth_path)
                    depth_array = np.array(depth_img).astype(np.float32)
                    # Normalize depth (CO3D uses 16-bit PNG, scale factor 1000)
                    depth_array = depth_array / 1000.0
                    np.save(dst_depth, depth_array)
                except Exception as e:
                    print(f"Warning: Could not convert depth {depth_path}: {e}")

        # Extract pose if available
        if i < len(annotations):
            pose = extract_camera_pose(annotations[i])
            if pose:
                pose_file = seq_output / "poses" / f"{frame_name}.npz"
                np.savez(pose_file, **pose)

        result['n_extracted'] += 1

    # Save metadata
    metadata = {
        'sequence_name': seq_name,
        'n_frames': n_frames,
        'source_path': str(seq_info['path']),
        'has_depth': len(seq_info['depths']) > 0,
        'has_poses': result['has_poses']
    }
    with open(seq_output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return result


def prepare_benchmark_data(
    source_dir: Path,
    output_dir: Path,
    categories: Optional[List[str]] = None,
    sequences_per_category: int = 5,
    max_frames: int = 50,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Prepare CO3D benchmark data.

    Args:
        source_dir: Path to CO3D source directory
        output_dir: Output directory for benchmark data
        categories: List of categories to extract (None = all)
        sequences_per_category: Number of sequences per category
        max_frames: Maximum frames per sequence
        dry_run: If True, only show what would be done

    Returns:
        Summary of prepared data
    """
    print("=" * 60)
    print("CO3D Benchmark Data Preparation")
    print("=" * 60)

    # Get available categories
    available_cats = get_available_categories(source_dir)
    print(f"Available categories: {len(available_cats)}")

    if categories:
        # Filter to requested categories
        selected_cats = [c for c in categories if c in available_cats]
        missing = [c for c in categories if c not in available_cats]
        if missing:
            print(f"Warning: Categories not found: {missing}")
    else:
        # Use a default selection of diverse categories
        default_cats = ['bottle', 'chair', 'car', 'cup', 'plant', 'vase', 'book', 'ball']
        selected_cats = [c for c in default_cats if c in available_cats]

    print(f"Selected categories: {selected_cats}")

    if dry_run:
        print("\n[DRY RUN - No files will be copied]")

    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'categories': {},
        'total_sequences': 0,
        'total_frames': 0
    }

    for cat in selected_cats:
        print(f"\n--- Processing: {cat} ---")
        cat_output = output_dir / cat

        # Find sequences
        sequences = find_sequence_data(source_dir, cat)
        if not sequences:
            print(f"  No sequences found for {cat}")
            print(f"  (Category may need to be downloaded first)")
            continue

        print(f"  Found {len(sequences)} sequences")

        # Select sequences
        selected_seqs = sequences[:sequences_per_category]
        cat_results = []

        for seq in selected_seqs:
            result = prepare_sequence(
                seq,
                cat_output,
                max_frames=max_frames,
                dry_run=dry_run
            )
            cat_results.append(result)
            print(f"  - {result['name']}: {result['n_extracted']} frames, poses={result['has_poses']}")

        summary['categories'][cat] = {
            'n_sequences': len(cat_results),
            'total_frames': sum(r['n_extracted'] for r in cat_results),
            'sequences': cat_results
        }
        summary['total_sequences'] += len(cat_results)
        summary['total_frames'] += sum(r['n_extracted'] for r in cat_results)

    # Save summary
    if not dry_run and summary['total_sequences'] > 0:
        summary_file = output_dir / "benchmark_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")

    print("\n" + "=" * 60)
    print("Preparation Summary")
    print("=" * 60)
    print(f"Categories: {len(summary['categories'])}")
    print(f"Sequences: {summary['total_sequences']}")
    print(f"Total frames: {summary['total_frames']}")

    if dry_run:
        print("\n[DRY RUN COMPLETE - Run without --dry-run to extract data]")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CO3D benchmark data for VGGT evaluation"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "co3d-main",
        help="Path to CO3D source directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "co3d_benchmark",
        help="Output directory for benchmark data"
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated list of categories (default: auto-select diverse set)"
    )
    parser.add_argument(
        "--sequences",
        type=int,
        default=5,
        help="Number of sequences per category (default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum frames per sequence (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files"
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List available categories and exit"
    )

    args = parser.parse_args()

    # List categories mode
    if args.list_categories:
        cats = get_available_categories(args.source)
        print("Available CO3D categories:")
        for cat in sorted(cats):
            print(f"  - {cat}")
        print(f"\nTotal: {len(cats)} categories")
        return

    # Parse categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    # Run preparation
    prepare_benchmark_data(
        source_dir=args.source,
        output_dir=args.output,
        categories=categories,
        sequences_per_category=args.sequences,
        max_frames=args.max_frames,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
