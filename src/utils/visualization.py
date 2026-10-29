"""Visualization tools for industrial visual anomaly inspection."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_inspection_results(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    binary_mask: Optional[np.ndarray] = None,
    heatmap_overlay: Optional[np.ndarray] = None,
    defect_type: str = "unknown",
    anomaly_score: float = 0.0,
    threshold: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a 5-panel industrial QA inspection dashboard figure.

    Panels:
        1. Original Inspection Image
        2. Ground Truth Defect Mask
        3. Raw Anomaly Intensity Heatmap
        4. Binarized Defect Mask with Bounding Boxes
        5. Thermal Overlay on Part

    Args:
        image_rgb: (H, W, 3) normalized [0, 1] or [0, 255].
        anomaly_map: (H, W) float anomaly map.
        ground_truth_mask: Optional (H, W) binary mask.
        binary_mask: Optional (H, W) post-processed binary mask.
        heatmap_overlay: Optional (H, W, 3) blended overlay.
        defect_type: Defect name / label.
        anomaly_score: Image-level prediction score.
        threshold: Decision threshold.
        save_path: Optional file path to save figure.

    Returns:
        matplotlib.figure.Figure object.
    """
    is_rejected = anomaly_score >= threshold
    status_str = "FAIL (REJECT)" if is_rejected else "PASS (APPROVED)"
    title_color = "red" if is_rejected else "green"

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle(
        f"Industrial QA Decision: [{status_str}] | Type: {defect_type} | "
        f"Score: {anomaly_score:.3f} (Threshold: {threshold:.3f})",
        fontsize=14,
        fontweight="bold",
        color=title_color,
    )

    # 1. Original Image
    img_disp = np.clip(image_rgb, 0.0, 1.0) if image_rgb.max() <= 1.0 else image_rgb / 255.0
    axes[0].imshow(img_disp)
    axes[0].set_title("1. Original Part")
    axes[0].axis("off")

    # 2. Ground Truth Mask
    if ground_truth_mask is not None:
        axes[1].imshow(ground_truth_mask, cmap="gray")
        axes[1].set_title("2. Ground Truth Mask")
    else:
        axes[1].text(0.5, 0.5, "N/A (Normal Part)", ha="center", va="center")
        axes[1].set_title("2. Ground Truth")
    axes[1].axis("off")

    # 3. Anomaly Heatmap
    im_heat = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("3. Anomaly Heatmap")
    axes[2].axis("off")
    fig.colorbar(im_heat, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Binary Mask with Defect Bounding Boxes
    if binary_mask is not None:
        mask_disp = cv2.cvtColor(binary_mask.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # Find contours and draw red bounding boxes
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(mask_disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
        axes[3].imshow(mask_disp)
        axes[3].set_title(f"4. Segmented ({len(contours)} defects)")
    else:
        axes[3].text(0.5, 0.5, "No Seg Mask", ha="center", va="center")
        axes[3].set_title("4. Segmented Mask")
    axes[3].axis("off")

    # 5. Thermal Overlay
    if heatmap_overlay is not None:
        axes[5 - 1].imshow(heatmap_overlay)
        axes[5 - 1].set_title("5. Thermal QA Overlay")
    else:
        axes[4].imshow(img_disp)
        axes[4].set_title("5. Overlay N/A")
    axes[4].axis("off")

    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=150, bbox_inches="tight")

    return fig

