"""Industrial metrics evaluation for anomaly detection.

Implements Image-level AUROC, Pixel-level AUROC, Per-Region Overlap (PRO),
and cost-sensitive asymmetric operational threshold calibration.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


def compute_image_auroc(
    labels: Union[np.ndarray, Sequence[int]],
    scores: Union[np.ndarray, Sequence[float]],
) -> float:
    """Compute Image-level Area Under the ROC Curve (AUROC).

    Args:
        labels: 1D array of ground truth labels (0 = normal, 1 = anomalous).
        scores: 1D array of continuous anomaly scores.

    Returns:
        Float AUROC in [0.0, 1.0].
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    if len(np.unique(labels_arr)) < 2:
        return 1.0  # Perfect if single class present

    return float(roc_auc_score(labels_arr, scores_arr))


def compute_pixel_auroc(
    masks: Union[np.ndarray, Sequence[np.ndarray]],
    anomaly_maps: Union[np.ndarray, Sequence[np.ndarray]],
    subsample_ratio: float = 0.1,
) -> float:
    """Compute Pixel-level Area Under the ROC Curve (AUROC).

    Args:
        masks: Ground truth masks of shape (N, H, W) or (H, W) in {0, 1}.
        anomaly_maps: Predicted heatmaps of shape (N, H, W) or (H, W).
        subsample_ratio: Fraction of pixels sampled for speed optimization.

    Returns:
        Float Pixel-AUROC in [0.0, 1.0].
    """
    flat_masks = np.asarray(masks, dtype=np.uint8).ravel()
    flat_maps = np.asarray(anomaly_maps, dtype=np.float32).ravel()

    # Subsample if necessary to prevent OOM / slow execution on millions of pixels
    if 0.0 < subsample_ratio < 1.0 and flat_masks.size > 100_000:
        idx = np.random.choice(
            flat_masks.size, size=int(flat_masks.size * subsample_ratio), replace=False
        )
        flat_masks = flat_masks[idx]
        flat_maps = flat_maps[idx]

    if len(np.unique(flat_masks)) < 2:
        return 1.0

    return float(roc_auc_score(flat_masks, flat_maps))


def compute_pro_metric(
    masks: Union[np.ndarray, List[np.ndarray]],
    anomaly_maps: Union[np.ndarray, List[np.ndarray]],
    max_fpr: float = 0.30,
    num_thresholds: int = 100,
) -> float:
    """Compute Per-Region Overlap (PRO) metric for localized defect evaluation.

    The PRO metric treats each connected anomaly component in ground-truth masks
    as an independent defect and computes average overlap up to a max FPR (default 0.30).

    Args:
        masks: List or array of ground-truth binary masks (0 = normal, 1 = defect).
        anomaly_maps: List or array of predicted anomaly maps in [0, 1].
        max_fpr: Integration limit for False Positive Rate (standard MVTec is 0.30).
        num_thresholds: Number of evaluation thresholds.

    Returns:
        Float normalized PRO score in [0.0, 1.0].
    """
    masks_list = [np.asarray(m, dtype=np.uint8) for m in masks]
    maps_list = [np.asarray(a, dtype=np.float32) for a in anomaly_maps]

    # Pre-extract connected defect components across all images
    defect_components: List[Tuple[int, np.ndarray, int]] = []
    total_normal_pixels: int = 0

    for img_idx, mask in enumerate(masks_list):
        normal_pixels = int(np.sum(mask == 0))
        total_normal_pixels += normal_pixels

        labeled_mask, num_features = ndimage.label(mask)
        for comp_id in range(1, num_features + 1):
            comp_pixels = (labeled_mask == comp_id)
            comp_area = int(np.sum(comp_pixels))
            if comp_area > 0:
                defect_components.append((img_idx, comp_pixels, comp_area))

    if not defect_components or total_normal_pixels == 0:
        return 1.0

    # Determine threshold candidates
    all_scores = np.concatenate([m.ravel() for m in maps_list])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num=num_thresholds)

    pro_curve: List[float] = []
    fpr_curve: List[float] = []

    for th in reversed(thresholds):
        # 1. Compute false positive rate on normal pixels
        total_fp = 0
        for img_idx, mask in enumerate(masks_list):
            pred_binary = maps_list[img_idx] >= th
            fp = np.sum((pred_binary == 1) & (mask == 0))
            total_fp += int(fp)

        current_fpr = total_fp / total_normal_pixels
        if current_fpr > max_fpr:
            break

        # 2. Compute overlap for each ground-truth defect component
        overlaps: List[float] = []
        for img_idx, comp_pixels, comp_area in defect_components:
            pred_binary = maps_list[img_idx] >= th
            intersection = np.sum(pred_binary & comp_pixels)
            overlaps.append(float(intersection / comp_area))

        mean_pro = float(np.mean(overlaps)) if overlaps else 0.0

        fpr_curve.append(current_fpr)
        pro_curve.append(mean_pro)

    if not fpr_curve:
        return 0.0

    # Normalize AUC up to max_fpr
    fpr_arr = np.array(fpr_curve)
    pro_arr = np.array(pro_curve)

    # Sort in ascending order of FPR
    sort_idx = np.argsort(fpr_arr)
    fpr_sorted = fpr_arr[sort_idx]
    pro_sorted = pro_arr[sort_idx]

    # Ensure integration spans the full [0, max_fpr] domain
    fpr_list = list(fpr_sorted)
    pro_list = list(pro_sorted)

    if not fpr_list:
        return 0.0

    if fpr_list[0] > 0.0:
        fpr_list.insert(0, 0.0)
        pro_list.insert(0, pro_list[0])

    if fpr_list[-1] < max_fpr:
        fpr_list.append(max_fpr)
        pro_list.append(pro_list[-1])

    fpr_final = np.array(fpr_list)
    pro_final = np.array(pro_list)

    # Integrate via trapezoidal rule (compatible with NumPy 1.x and 2.x)
    if hasattr(np, "trapezoid"):
        raw_auc = float(np.trapezoid(pro_final, fpr_final))
    else:
        from scipy.integrate import trapezoid
        raw_auc = float(trapezoid(pro_final, fpr_final))

    normalized_pro = raw_auc / max_fpr
    return float(np.clip(normalized_pro, 0.0, 1.0))


def find_f1_optimal_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Dict[str, float]:
    """Find threshold that maximizes the F1-Score."""
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    precisions, recalls, thresholds = precision_recall_curve(labels_arr, scores_arr)
    numerator = 2 * (precisions * recalls)
    denominator = precisions + recalls
    f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(scores_arr.mean())

    return {
        "optimal_threshold": best_th,
        "max_f1": float(f1_scores[best_idx]),
        "precision_at_optimal": float(precisions[best_idx]),
        "recall_at_optimal": float(recalls[best_idx]),
    }


def find_cost_sensitive_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    cost_fn: float = 10000.0,
    cost_fp: float = 1.0,
    min_recall_constraint: float = 0.995,
) -> Dict[str, float]:
    """Calculate operational threshold based on asymmetric industrial cost matrix.

    In production: Cost(FN) >> Cost(FP).
    Minimizes Cost_total = Cost_FN * FN + Cost_FP * FP, subject to Recall >= min_recall_constraint.
    """
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    num_pos = np.sum(labels_arr == 1)
    num_neg = np.sum(labels_arr == 0)

    best_threshold = float(scores_arr.min())
    lowest_cost = float("inf")
    best_recall = 0.0
    best_precision = 0.0

    for i, th in enumerate(thresholds):
        preds = scores_arr >= th
        tp = np.sum((preds == 1) & (labels_arr == 1))
        fp = np.sum((preds == 1) & (labels_arr == 0))
        fn = np.sum((preds == 0) & (labels_arr == 1))

        recall = tp / num_pos if num_pos > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if recall >= min_recall_constraint:
            total_cost = (cost_fn * fn) + (cost_fp * fp)
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_threshold = float(th)
                best_recall = float(recall)
                best_precision = float(precision)

    return {
        "calibrated_threshold": best_threshold,
        "minimum_cost": lowest_cost,
        "achieved_recall": best_recall,
        "achieved_precision": best_precision,
    }

