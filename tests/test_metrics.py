"""Unit tests for industrial evaluation metrics (AUROC, PRO, Cost Threshold)."""

import numpy as np
import pytest

from src.utils.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_metric,
    find_cost_sensitive_threshold,
    find_f1_optimal_threshold,
)


def test_compute_image_auroc_perfect():
    """Test perfect image AUROC separation."""
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.15, 0.8, 0.9, 0.85]
    auroc = compute_image_auroc(labels, scores)
    assert auroc == pytest.approx(1.0, abs=1e-4)


def test_compute_pixel_auroc_perfect():
    """Test perfect pixel AUROC separation."""
    masks = np.zeros((2, 10, 10), dtype=np.uint8)
    anomaly_maps = np.zeros((2, 10, 10), dtype=np.float32)

    # Place defect
    masks[1, 2:5, 2:5] = 1
    anomaly_maps[1, 2:5, 2:5] = 1.0

    auroc = compute_pixel_auroc(masks, anomaly_maps, subsample_ratio=1.0)
    assert auroc == pytest.approx(1.0, abs=1e-4)


def test_compute_pro_metric_perfect_overlap():
    """Test PRO metric with complete prediction alignment."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1  # 10x10 defect component

    # Prediction matches the defect region perfectly
    pred_map = np.zeros((20, 20), dtype=np.float32)
    pred_map[5:15, 5:15] = 1.0

    pro = compute_pro_metric([mask], [pred_map], max_fpr=0.30, num_thresholds=20)
    assert pro >= 0.95


def test_compute_pro_metric_zero_overlap():
    """Test PRO metric with disjoint prediction."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:5, 0:5] = 1

    pred_map = np.zeros((20, 20), dtype=np.float32)
    # Anomaly placed in completely wrong region
    pred_map[15:20, 15:20] = 1.0

    pro = compute_pro_metric([mask], [pred_map], max_fpr=0.30, num_thresholds=20)
    assert pro == pytest.approx(0.0, abs=1e-3)


def test_find_cost_sensitive_threshold():
    """Test that cost-sensitive threshold achieves required minimum recall."""
    labels = [0] * 50 + [1] * 50
    # Normal scores [0.0 - 0.4], Anomalous scores [0.3 - 0.9]
    scores = [float(i) / 150.0 for i in range(50)] + [float(i) / 80.0 for i in range(25, 75)]

    res = find_cost_sensitive_threshold(
        labels, scores, cost_fn=10000.0, cost_fp=1.0, min_recall_constraint=0.99
    )

    assert "calibrated_threshold" in res
    assert res["achieved_recall"] >= 0.99


def test_find_f1_optimal_threshold():
    """Test F1 optimal threshold finder."""
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    res = find_f1_optimal_threshold(labels, scores)
    assert res["max_f1"] == pytest.approx(1.0, abs=1e-3)
    assert 0.2 < res["optimal_threshold"] <= 0.8

