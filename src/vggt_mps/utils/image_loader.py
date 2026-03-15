"""
Image loading utilities for VGGT-MPS benchmarking.

Provides functions to load real images from directories for more realistic
benchmark testing compared to synthetic random images.
"""

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def parse_image_size(size_str: str) -> Tuple[int, int]:
    """
    Parse image size string in WxH format.

    Args:
        size_str: Size string like "640x480" or "1024x768"

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_image_size("640x480")
        (640, 480)
        >>> parse_image_size("1920x1080")
        (1920, 1080)
    """
    if 'x' not in size_str.lower():
        raise ValueError(
            f"Invalid size format '{size_str}'. Expected format: WxH (e.g., 640x480)"
        )

    parts = size_str.lower().split('x')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format '{size_str}'. Expected format: WxH (e.g., 640x480)"
        )

    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
    except ValueError:
        raise ValueError(
            f"Invalid size format '{size_str}'. Width and height must be integers."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid size '{size_str}'. Width and height must be positive integers."
        )

    return (width, height)


def get_image_paths(
    image_dir: Union[str, Path],
    recursive: bool = False
) -> List[Path]:
    """
    Get all image file paths from a directory.

    Args:
        image_dir: Directory to search for images
        recursive: If True, search subdirectories recursively

    Returns:
        List of Path objects for image files, sorted by name

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If directory contains no images
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not image_dir.is_dir():
        raise ValueError(f"Path is not a directory: {image_dir}")

    # Collect image paths
    image_paths = []

    if recursive:
        for ext in SUPPORTED_FORMATS:
            image_paths.extend(image_dir.rglob(f'*{ext}'))
            image_paths.extend(image_dir.rglob(f'*{ext.upper()}'))
    else:
        for ext in SUPPORTED_FORMATS:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'*{ext.upper()}'))

    # Remove duplicates and sort
    image_paths = sorted(set(image_paths))

    return image_paths


def load_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load a single image and optionally resize it.

    Args:
        image_path: Path to image file
        target_size: Optional (width, height) to resize to

    Returns:
        Image as numpy array of shape (H, W, 3) with dtype uint8

    Raises:
        FileNotFoundError: If image file doesn't exist
        IOError: If image cannot be loaded
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = Image.open(image_path)

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if target size specified
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)

        return img_array

    except Exception as e:
        raise IOError(f"Failed to load image '{image_path}': {e}")


def load_images_from_directory(
    image_dir: Union[str, Path],
    max_images: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    recursive: bool = False,
    skip_corrupted: bool = True
) -> List[np.ndarray]:
    """
    Load images from a directory.

    Args:
        image_dir: Directory containing images
        max_images: Maximum number of images to load (None for all)
        target_size: Optional (width, height) to resize images to.
                     If None, images are loaded at original size.
        recursive: If True, search subdirectories recursively
        skip_corrupted: If True, skip corrupted images with a warning.
                       If False, raise an error on corrupted images.

    Returns:
        List of numpy arrays, each of shape (H, W, 3) with dtype uint8

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid images found

    Examples:
        >>> images = load_images_from_directory("./data/scenes", max_images=10)
        >>> images = load_images_from_directory(
        ...     "./data/scenes",
        ...     target_size=(640, 480),
        ...     recursive=True
        ... )
    """
    image_dir = Path(image_dir)

    # Get all image paths
    image_paths = get_image_paths(image_dir, recursive=recursive)

    if not image_paths:
        supported = ', '.join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"No images found in '{image_dir}'.\n"
            f"Supported formats: {supported}\n"
            f"Use --recursive to search subdirectories."
        )

    # Limit number of images if specified
    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    # Load images
    images = []
    skipped = 0

    for path in image_paths:
        try:
            img = load_image(path, target_size=target_size)
            images.append(img)
        except (IOError, OSError) as e:
            if skip_corrupted:
                warnings.warn(f"Skipping corrupted image: {path} ({e})")
                skipped += 1
            else:
                raise

    if not images:
        raise ValueError(
            f"No valid images could be loaded from '{image_dir}'.\n"
            f"All {skipped} images were corrupted or unreadable."
        )

    if skipped > 0:
        print(f"Warning: Skipped {skipped} corrupted images")

    return images


def create_synthetic_images(
    n_images: int,
    size: Tuple[int, int] = (640, 480)
) -> List[np.ndarray]:
    """
    Create synthetic random images for benchmarking.

    This is the original behavior - creates random noise images
    when no real image directory is specified.

    Args:
        n_images: Number of images to create
        size: Image size as (width, height)

    Returns:
        List of numpy arrays, each of shape (H, W, 3) with dtype uint8
    """
    width, height = size
    images = []

    for _ in range(n_images):
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        images.append(img_array)

    return images
