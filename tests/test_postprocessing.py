"""Unit tests for postprocessing, morphological filtering, and heatmap visualization."""

import numpy as np
import pytest

from src.postprocessing.postprocessor import AnomalyPostProcessor


@pytest.fixture
def post_processor():
    return AnomalyPostProcessor(
        blur_sigma=1.0,
        opening_kernel_size=3,
        closing_kernel_size=3,
        min_defect_area_pixels=5,
    )


def test_normalize_anomaly_map(post_processor):
    """Test min-max normalization to [0, 1]."""
    raw_map = np.array([[10.0, 20.0], [30.0, 50.0]], dtype=np.float32)
    norm = post_processor.normalize_anomaly_map(raw_map)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)
    assert norm.shape == raw_map.shape


def test_hysteresis_suppresses_isolated_weak_noise(post_processor):
    """Test that hysteresis discards weak anomalies that have no connection to a strong seed."""
    norm_map = np.zeros((30, 30), dtype=np.float32)

    # Component 1: strong defect (has high score)
    norm_map[5:10, 5:10] = 0.9

    # Component 2: isolated weak noise (e.g. sensor dust)
    norm_map[20:23, 20:23] = 0.55

    # Run hysteresis: high=0.8, low=0.5
    mask = post_processor.apply_hysteresis_threshold(norm_map, high_threshold=0.8, low_threshold=0.5)

    # Strong component must be kept
    assert np.all(mask[5:10, 5:10] == 255)
    # Isolated weak component must be suppressed
    assert np.all(mask[20:23, 20:23] == 0)


def test_morphology_removes_single_pixel_noise(post_processor):
    """Test opening filter removes single pixel artifacts."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2, 2] = 255  # isolated pixel
    mask[10:15, 10:15] = 255  # valid blob

    filtered = post_processor.filter_morphology(mask)
    assert filtered[2, 2] == 0
    assert filtered[12, 12] == 255


def test_extract_defect_regions(post_processor):
    """Test contour and bounding box extraction for defects."""
    binary_mask = np.zeros((50, 50), dtype=np.uint8)
    anomaly_map = np.zeros((50, 50), dtype=np.float32)

    # Create defect blob
    binary_mask[10:20, 15:30] = 255
    anomaly_map[10:20, 15:30] = 0.95

    regions = post_processor.extract_defect_regions(binary_mask, anomaly_map)
    assert len(regions) == 1
    reg = regions[0]
    assert reg.area >= 100
    assert reg.max_anomaly_score == pytest.approx(0.95, abs=1e-3)
    assert reg.bbox[0] == 15  # x
    assert reg.bbox[1] == 10  # y


def test_create_heatmap_overlay(post_processor):
    """Test thermal heatmap generation and alpha blending."""
    img = np.ones((50, 50, 3), dtype=np.float32)
    norm_map = np.zeros((50, 50), dtype=np.float32)
    norm_map[10:20, 10:20] = 1.0

    blended = post_processor.create_heatmap_overlay(img, norm_map, alpha=0.5)
    assert blended.shape == (50, 50, 3)
    assert blended.dtype == np.uint8

