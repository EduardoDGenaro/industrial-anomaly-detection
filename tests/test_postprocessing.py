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


def test_otsu_foreground_mask_and_segment_defects(post_processor):
    """Test Otsu foreground extraction and decoupled pixel-threshold segmentation."""
    # Simulate a dark screw on a uniform bright (220) background
    img = np.full((100, 100, 3), 220, dtype=np.uint8)
    img[25:75, 40:60] = 60  # screw shaft silhouette

    raw_map = np.full((100, 100), 1.30, dtype=np.float32)  # background noise
    raw_map[25:75, 40:60] = 1.40  # normal screw body
    raw_map[65:74, 44:56] = 1.85  # localized defect at screw tip

    norm_map, filtered_mask, fg_mask, regions = post_processor.segment_defects(
        raw_map=raw_map,
        image_rgb=img,
        pixel_baseline=1.40,
        pixel_threshold=1.55,
        tau_high=0.68,
        tau_low=0.45,
        use_foreground_mask=True,
    )

    # Background outside screw silhouette must be forced to 0.0
    assert norm_map[5, 5] == pytest.approx(0.0)
    assert fg_mask[5, 5] == 0
    assert fg_mask[50, 50] == 255

    # Strictly 1 localized defect at the screw tip (not the whole screw or background)
    assert len(regions) == 1
    x, y, w, h = regions[0].bbox
    assert 40 <= x <= 46
    assert 60 <= y <= 67
    assert w <= 20 and h <= 16


