"""
Unit tests for image_loader utilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vggt_mps.utils.image_loader import (
    parse_image_size,
    get_image_paths,
    load_image,
    load_images_from_directory,
    create_synthetic_images,
    SUPPORTED_FORMATS
)


class TestParseImageSize:
    """Tests for parse_image_size function."""

    def test_standard_format(self):
        """Test standard WxH format."""
        assert parse_image_size("640x480") == (640, 480)
        assert parse_image_size("1920x1080") == (1920, 1080)
        assert parse_image_size("1024x768") == (1024, 768)

    def test_case_insensitive(self):
        """Test that 'x' is case-insensitive."""
        assert parse_image_size("640X480") == (640, 480)

    def test_whitespace_handling(self):
        """Test handling of whitespace around numbers."""
        assert parse_image_size("640 x 480") == (640, 480)
        assert parse_image_size(" 640x480 ") == (640, 480)

    def test_invalid_format_no_x(self):
        """Test that missing 'x' raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size format"):
            parse_image_size("640-480")

    def test_invalid_format_non_numeric(self):
        """Test that non-numeric values raise ValueError."""
        with pytest.raises(ValueError, match="must be integers"):
            parse_image_size("abcx480")

    def test_invalid_format_negative(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_image_size("-640x480")

    def test_invalid_format_zero(self):
        """Test that zero values raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_image_size("0x480")


class TestGetImagePaths:
    """Tests for get_image_paths function."""

    def test_find_images(self, tmp_path):
        """Test finding images in a directory."""
        # Create test images
        for i, ext in enumerate(['.jpg', '.png', '.jpeg']):
            img = Image.new('RGB', (10, 10), color='red')
            img.save(tmp_path / f"test_{i}{ext}")

        paths = get_image_paths(tmp_path)
        assert len(paths) == 3

    def test_recursive_search(self, tmp_path):
        """Test recursive subdirectory search."""
        # Create subdirectory with images
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        img = Image.new('RGB', (10, 10), color='red')
        img.save(tmp_path / "root.jpg")
        img.save(subdir / "nested.jpg")

        # Non-recursive should find only root
        paths = get_image_paths(tmp_path, recursive=False)
        assert len(paths) == 1

        # Recursive should find both
        paths = get_image_paths(tmp_path, recursive=True)
        assert len(paths) == 2

    def test_nonexistent_directory(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_image_paths("/nonexistent/path")

    def test_not_a_directory(self, tmp_path):
        """Test that file path raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            get_image_paths(file_path)

    def test_case_insensitive_extensions(self, tmp_path):
        """Test that uppercase extensions are found."""
        img = Image.new('RGB', (10, 10), color='red')
        img.save(tmp_path / "test.JPG")
        img.save(tmp_path / "test.PNG")

        paths = get_image_paths(tmp_path)
        assert len(paths) == 2


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_rgb_image(self, tmp_path):
        """Test loading RGB image."""
        img_path = tmp_path / "test.jpg"
        original = Image.new('RGB', (100, 100), color='blue')
        original.save(img_path)

        loaded = load_image(img_path)

        assert loaded.shape == (100, 100, 3)
        assert loaded.dtype == np.uint8

    def test_load_rgba_image(self, tmp_path):
        """Test loading RGBA image converts to RGB."""
        img_path = tmp_path / "test.png"
        original = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        original.save(img_path)

        loaded = load_image(img_path)

        assert loaded.shape == (100, 100, 3)

    def test_load_grayscale_image(self, tmp_path):
        """Test loading grayscale image converts to RGB."""
        img_path = tmp_path / "test.png"
        original = Image.new('L', (100, 100), color=128)
        original.save(img_path)

        loaded = load_image(img_path)

        assert loaded.shape == (100, 100, 3)

    def test_resize_image(self, tmp_path):
        """Test resizing image to target size."""
        img_path = tmp_path / "test.jpg"
        original = Image.new('RGB', (1000, 800), color='green')
        original.save(img_path)

        loaded = load_image(img_path, target_size=(320, 240))

        assert loaded.shape == (240, 320, 3)  # H, W, C

    def test_nonexistent_file(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")


class TestLoadImagesFromDirectory:
    """Tests for load_images_from_directory function."""

    def test_load_all_images(self, tmp_path):
        """Test loading all images from directory."""
        # Create 5 test images
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i * 50, 0, 0))
            img.save(tmp_path / f"img_{i}.jpg")

        images = load_images_from_directory(tmp_path)

        assert len(images) == 5
        assert all(img.shape == (100, 100, 3) for img in images)

    def test_max_images_limit(self, tmp_path):
        """Test that max_images limits the number loaded."""
        # Create 10 test images
        for i in range(10):
            img = Image.new('RGB', (50, 50), color='red')
            img.save(tmp_path / f"img_{i:02d}.jpg")

        images = load_images_from_directory(tmp_path, max_images=3)

        assert len(images) == 3

    def test_custom_size(self, tmp_path):
        """Test resizing to target dimensions."""
        # Create images of various sizes
        for i, size in enumerate([(100, 100), (200, 150), (300, 200)]):
            img = Image.new('RGB', size, color='blue')
            img.save(tmp_path / f"img_{i}.jpg")

        images = load_images_from_directory(tmp_path, target_size=(64, 48))

        assert len(images) == 3
        assert all(img.shape == (48, 64, 3) for img in images)  # H, W, C

    def test_empty_directory(self, tmp_path):
        """Test that empty directory raises ValueError."""
        with pytest.raises(ValueError, match="No images found"):
            load_images_from_directory(tmp_path)

    def test_skip_corrupted_images(self, tmp_path):
        """Test that corrupted images are skipped with warning."""
        # Create a valid image
        img = Image.new('RGB', (50, 50), color='red')
        img.save(tmp_path / "valid.jpg")

        # Create a corrupted "image" file
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"not an image")

        # Should load 1 valid image and skip the corrupted one
        images = load_images_from_directory(tmp_path, skip_corrupted=True)

        assert len(images) == 1

    def test_recursive_loading(self, tmp_path):
        """Test recursive loading from subdirectories."""
        # Create images in root and subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        img = Image.new('RGB', (50, 50), color='red')
        img.save(tmp_path / "root.jpg")
        img.save(subdir / "nested.jpg")

        images = load_images_from_directory(tmp_path, recursive=True)

        assert len(images) == 2


class TestCreateSyntheticImages:
    """Tests for create_synthetic_images function."""

    def test_create_correct_count(self):
        """Test creating correct number of images."""
        images = create_synthetic_images(5)
        assert len(images) == 5

    def test_default_size(self):
        """Test default image size."""
        images = create_synthetic_images(1)
        assert images[0].shape == (480, 640, 3)  # H, W, C

    def test_custom_size(self):
        """Test custom image size."""
        images = create_synthetic_images(1, size=(320, 240))
        assert images[0].shape == (240, 320, 3)  # H, W, C

    def test_dtype(self):
        """Test output dtype is uint8."""
        images = create_synthetic_images(1)
        assert images[0].dtype == np.uint8

    def test_value_range(self):
        """Test values are in valid range [0, 255]."""
        images = create_synthetic_images(10)
        for img in images:
            assert img.min() >= 0
            assert img.max() <= 255


class TestSupportedFormats:
    """Tests for supported image formats."""

    def test_all_formats_loadable(self, tmp_path):
        """Test that all supported formats can be loaded."""
        for ext in SUPPORTED_FORMATS:
            img = Image.new('RGB', (50, 50), color='red')
            path = tmp_path / f"test{ext}"

            # Some formats need special handling
            if ext in {'.tiff', '.tif'}:
                img.save(path, 'TIFF')
            elif ext == '.webp':
                try:
                    img.save(path, 'WEBP')
                except OSError:
                    # WebP may not be available on all systems
                    continue
            else:
                img.save(path)

            loaded = load_image(path)
            assert loaded.shape == (50, 50, 3)
