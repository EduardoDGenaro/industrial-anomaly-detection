"""Unit tests for MVTec AD dataset loaders and parsers."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.dataset.mvtec import (
    MVTEC_CATEGORIES,
    MVTEC_OBJECTS,
    MVTEC_TEXTURES,
    MVTecDataset,
)


def test_mvtec_categories_defined():
    """Verify that all 15 MVTec AD categories are defined without duplicates."""
    assert len(MVTEC_CATEGORIES) == 15
    assert len(MVTEC_TEXTURES) == 5
    assert len(MVTEC_OBJECTS) == 10
    assert len(set(MVTEC_CATEGORIES)) == 15


def test_invalid_category_raises_error():
    """Verify error on invalid category."""
    with pytest.raises(ValueError, match="Unknown category"):
        MVTecDataset(root_dir="./data", category="invalid_item", split="train")


def test_dataset_indexing_with_mock_files(tmp_path: Path):
    """Test dataset indexing and item fetching with a synthetic category structure."""
    cat_dir = tmp_path / "bottle"
    train_good = cat_dir / "train" / "good"
    test_good = cat_dir / "test" / "good"
    test_crack = cat_dir / "test" / "broken_large"
    gt_crack = cat_dir / "ground_truth" / "broken_large"

    for d in [train_good, test_good, test_crack, gt_crack]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy images
    dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    dummy_img.save(train_good / "000.png")
    dummy_img.save(test_good / "000.png")
    dummy_img.save(test_crack / "000.png")

    dummy_mask = Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255)
    dummy_mask.save(gt_crack / "000_mask.png")

    # Train dataset (unsupervised, only good)
    train_ds = MVTecDataset(root_dir=tmp_path, category="bottle", split="train", image_size=(64, 64))
    assert len(train_ds) == 1
    sample = train_ds[0]
    assert sample["label"] == 0
    assert sample["defect_type"] == "good"
    assert sample["image"].shape == (64, 64, 3)
    assert np.all(sample["mask"] == 0.0)

    # Test dataset (good + broken)
    test_ds = MVTecDataset(root_dir=tmp_path, category="bottle", split="test", image_size=(64, 64))
    assert len(test_ds) == 2
    labels = [test_ds[i]["label"] for i in range(len(test_ds))]
    assert 0 in labels
    assert 1 in labels

