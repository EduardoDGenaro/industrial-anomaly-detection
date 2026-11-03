"""Post-processing and morphological filtering pipeline for industrial inspection.

Handles anomaly map smoothing, normalization, hysteresis thresholding,
sensor noise suppression, contour extraction, and RGB heatmap overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DefectRegion:
    """Represents a localized defect region on an inspected part."""

    contour: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: float
    max_anomaly_score: float
    centroid: Tuple[int, int]


class AnomalyPostProcessor:
    """Post-processor for raw anomaly maps from PatchCore / Autoencoders."""

    def __init__(
        self,
        blur_sigma: float = 1.5,
        opening_kernel_size: int = 3,
        closing_kernel_size: int = 5,
        min_defect_area_pixels: int = 24,
    ) -> None:
        """Initialize post-processor.

        Args:
            blur_sigma: Sigma for Gaussian spatial smoothing (1.5-2.0 prevents bleeding).
            opening_kernel_size: Kernel size for morphological opening (removes dust/thread noise).
            closing_kernel_size: Kernel size for morphological closing (connects scratches).
            min_defect_area_pixels: Minimum blob size in pixels to consider as a defect.
        """
        self.blur_sigma = blur_sigma
        self.opening_kernel_size = opening_kernel_size
        self.closing_kernel_size = closing_kernel_size
        self.min_defect_area_pixels = min_defect_area_pixels

        # Morphological kernels
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size)
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size)
        )
        self._fg_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._fg_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def smooth_anomaly_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to smooth patch interpolation artifacts."""
        if self.blur_sigma <= 0:
            return anomaly_map
        ksize = int(2 * round(self.blur_sigma * 2) + 1)
        return cv2.GaussianBlur(anomaly_map, (ksize, ksize), self.blur_sigma)

    def extract_foreground_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Extract binary object silhouette (0 or 255) via inverted Otsu thresholding.

        For objects placed on a uniform bright background (e.g. MVTec 'screw'),
        segments the part silhouette, fills internal specular reflections, and
        applies a small boundary dilation so edge defects (such as a damaged tip)
        are preserved while background pixels are forced to zero.
        """
        if image_rgb.ndim == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_rgb.copy()

        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if float(border.mean()) < 160.0 or float(border.std()) > 32.0:
            return np.full(gray.shape, 255, dtype=np.uint8)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._fg_close_kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.full(gray.shape, 255, dtype=np.uint8)

        fg_filled = np.zeros_like(fg)
        max_area = max(cv2.contourArea(c) for c in contours)
        for cnt in contours:
            if cv2.contourArea(cnt) >= max(64.0, 0.15 * max_area):
                cv2.drawContours(fg_filled, [cnt], -1, 255, thickness=cv2.FILLED)

        fg_filled = cv2.dilate(fg_filled, self._fg_dilate_kernel, iterations=1)
        return fg_filled

    def normalize_anomaly_map(
        self,
        anomaly_map: np.ndarray,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> np.ndarray:
        """Normalize anomaly map values to [0.0, 1.0]."""
        v_min = float(anomaly_map.min()) if min_val is None else min_val
        v_max = float(anomaly_map.max()) if max_val is None else max_val

        if np.isclose(v_min, v_max) or v_max <= v_min:
            return np.zeros_like(anomaly_map, dtype=np.float32)

        norm_map = (anomaly_map - v_min) / (v_max - v_min)
        return np.clip(norm_map, 0.0, 1.0).astype(np.float32)

    def segment_defects(
        self,
        raw_map: np.ndarray,
        image_rgb: np.ndarray,
        pixel_baseline: float,
        pixel_threshold: float,
        tau_high: float = 0.68,
        tau_low: float = 0.45,
        use_foreground_mask: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[DefectRegion]]:
        """Calibrated pixel-level spatial segmentation pipeline.

        Strictly decouples pixel segmentation from the image-level classification
        threshold, suppresses background via Otsu foreground mask, applies
        dual-threshold hysteresis (tau_high / tau_low), 3x3 morphological opening,
        and minimum area filtering.

        Returns:
            Tuple of (norm_map, filtered_mask, fg_mask, defect_regions).
        """
        fg_mask = (
            self.extract_foreground_mask(image_rgb)
            if use_foreground_mask
            else np.full(raw_map.shape, 255, dtype=np.uint8)
        )
        fg_vals = raw_map[fg_mask > 0] if np.any(fg_mask > 0) else raw_map.ravel()

        eff_lo = max(float(pixel_baseline), float(np.percentile(fg_vals, 75.0)))
        eff_hi = max(float(pixel_threshold) * 1.08, float(fg_vals.max()), eff_lo + 1e-4)

        norm_map = np.clip((raw_map - eff_lo) / (eff_hi - eff_lo), 0.0, 1.0).astype(np.float32)
        norm_map[fg_mask == 0] = 0.0

        binary_mask = self.apply_hysteresis_threshold(
            norm_map, high_threshold=tau_high, low_threshold=tau_low
        )
        binary_mask[fg_mask == 0] = 0
        filtered_mask = self.filter_morphology(binary_mask)
        regions = self.extract_defect_regions(filtered_mask, norm_map)
        return norm_map, filtered_mask, fg_mask, regions

    def apply_hysteresis_threshold(
        self,
        normalized_map: np.ndarray,
        high_threshold: float,
        low_threshold: float,
    ) -> np.ndarray:
        """Apply dual-threshold hysteresis to segment defects cleanly.

        Pixels >= high_threshold are strong defects.
        Pixels >= low_threshold are weak defects, kept only if connected to strong defects.

        Args:
            normalized_map: 2D array normalized to [0, 1].
            high_threshold: Upper threshold for confirmed defect.
            low_threshold: Lower threshold for spatial continuity.

        Returns:
            Binary mask (0 or 255) as uint8.
        """
        strong = (normalized_map >= high_threshold).astype(np.uint8) * 255
        weak = (normalized_map >= low_threshold).astype(np.uint8) * 255

        # Connected component analysis starting from strong seeds
        num_labels, labels = cv2.connectedComponents(weak)
        strong_labels = np.unique(labels[strong > 0])

        # Keep components that contain at least one strong seed
        mask = np.isin(labels, strong_labels[strong_labels > 0]).astype(np.uint8) * 255
        return mask

    def filter_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening followed by closing."""
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.open_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.close_kernel)
        return closed

    def extract_defect_regions(
        self,
        binary_mask: np.ndarray,
        anomaly_map: np.ndarray,
    ) -> List[DefectRegion]:
        """Extract contours, bounding boxes, and statistics for detected defects."""
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions: List[DefectRegion] = []

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_defect_area_pixels:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Find max anomaly score inside this defect contour
            roi = anomaly_map[y : y + h, x : x + w]
            max_score = float(roi.max()) if roi.size > 0 else 0.0

            # Compute centroid
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            regions.append(
                DefectRegion(
                    contour=cnt,
                    bbox=(x, y, w, h),
                    area=area,
                    max_anomaly_score=max_score,
                    centroid=(cx, cy),
                )
            )

        return regions

    def create_heatmap_overlay(
        self,
        image_rgb: np.ndarray,
        normalized_map: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Generate thermal heatmap overlay over original inspection image.

        Args:
            image_rgb: Float [0, 1] or uint8 [0, 255] RGB image.
            normalized_map: Float [0, 1] 2D anomaly map.
            alpha: Heatmap blend weight (0 = image only, 1 = heatmap only).
            colormap: OpenCV colormap enum (e.g. cv2.COLORMAP_JET, cv2.COLORMAP_TURBO).

        Returns:
            Blended uint8 RGB image.
        """
        # Convert RGB image to uint8 if needed
        if image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1.0:
                base_img = (image_rgb * 255.0).astype(np.uint8)
            else:
                base_img = image_rgb.astype(np.uint8)
        else:
            base_img = image_rgb.copy()

        # Generate false-color heatmap
        heatmap_uint8 = (normalized_map * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        # OpenCV applyColorMap returns BGR, convert to RGB
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Alpha blend
        blended = cv2.addWeighted(
            base_img, 1.0 - alpha, heatmap_color_rgb, alpha, 0
        )
        return blended

