"""Smoke test for the 'bottle' pilot category of MVTec AD."""

from pathlib import Path
import pytest
import numpy as np

from src.dataset.mvtec import MVTecDataset, validate_category

DATA_ROOT = Path("data/mvtec")
BOTTLE_DIR = DATA_ROOT / "bottle"


@pytest.mark.skipif(not BOTTLE_DIR.exists(), reason="MVTec 'bottle' dataset not yet downloaded.")
def test_bottle_integrity_and_counts():
    """Verify official folder structure and image counts for 'bottle'."""
    report = validate_category(DATA_ROOT, "bottle")
    assert report.ok, f"Integrity check failed:\n{report.summary()}"
    assert report.counts["train"]["good"] == 209
    assert report.counts["test"]["good"] == 20
    assert report.num_masks == 63


@pytest.mark.skipif(not BOTTLE_DIR.exists(), reason="MVTec 'bottle' dataset not yet downloaded.")
def test_bottle_dataset_loader():
    """Verify PyTorch loader retrieves images and masks correctly."""
    train_ds = MVTecDataset(DATA_ROOT, category="bottle", split="train", image_size=(224, 224))
    assert len(train_ds) == 209
    sample_train = train_ds[0]
    assert sample_train["label"] == 0
    assert sample_train["defect_type"] == "good"
    assert sample_train["image"].shape == (224, 224, 3)
    assert np.all(sample_train["mask"] == 0.0)

    test_ds = MVTecDataset(DATA_ROOT, category="bottle", split="test", image_size=(224, 224))
    assert len(test_ds) == 83
    # Check that defective samples have active mask pixels
    defective_samples = [s for s in test_ds if s["label"] == 1]
    assert len(defective_samples) == 63
    for s in defective_samples:
        assert s["mask"].shape == (224, 224)
        assert np.sum(s["mask"]) > 0, f"Defect mask for {s['image_path']} is empty!"

