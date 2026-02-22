"""Unit tests for data preprocessing utilities (M3 requirement)."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catsml.data import get_transforms, preprocess_single_image, CLASS_NAMES, IMG_SIZE


class TestGetTransforms:
    def test_train_transform_returns_compose(self):
        t = get_transforms(train=True)
        assert t is not None

    def test_val_transform_returns_compose(self):
        t = get_transforms(train=False)
        assert t is not None

    def test_train_transform_output_shape(self):
        """Train transform should produce (3, 224, 224) tensor from PIL image."""
        from PIL import Image
        import torch
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        t = get_transforms(train=True)
        out = t(img)
        assert out.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_transform_output_shape(self):
        from PIL import Image
        import torch
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        t = get_transforms(train=False)
        out = t(img)
        assert out.shape == (3, IMG_SIZE, IMG_SIZE)

    def test_val_transform_is_deterministic(self):
        """Validation transforms should give same result for same input."""
        from PIL import Image
        import torch
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        t = get_transforms(train=False)
        out1 = t(img)
        out2 = t(img)
        assert torch.allclose(out1, out2)

    def test_normalisation_applied(self):
        """Normalised output should have values roughly in [-3, 3]."""
        from PIL import Image
        img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        t = get_transforms(train=False)
        out = t(img)
        assert out.min().item() > -4.0
        assert out.max().item() < 4.0


class TestPreprocessSingleImage:
    def test_output_shape(self, tmp_path):
        """preprocess_single_image should return (1, 3, 224, 224) tensor."""
        from PIL import Image
        import torch
        arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        Image.fromarray(arr).save(img_path)

        tensor = preprocess_single_image(str(img_path))
        assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)

    def test_output_dtype(self, tmp_path):
        from PIL import Image
        import torch
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_path = tmp_path / "cat.jpg"
        Image.fromarray(arr).save(img_path)

        tensor = preprocess_single_image(str(img_path))
        assert tensor.dtype == torch.float32

    def test_handles_rgba_image(self, tmp_path):
        """RGBA images should be converted to RGB without error."""
        from PIL import Image
        arr = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        img_path = tmp_path / "rgba.png"
        Image.fromarray(arr, mode="RGBA").save(img_path)

        tensor = preprocess_single_image(str(img_path))
        assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)


class TestClassNames:
    def test_binary_classes(self):
        assert len(CLASS_NAMES) == 2
        assert "Cat" in CLASS_NAMES
        assert "Dog" in CLASS_NAMES
