"""MVTec Anomaly Detection (MVTec AD) Dataset Loader and Parser.

This module provides an industrial-grade dataset parser, PyTorch Dataset wrapper,
and automated download utilities for the 15 categories of the MVTec AD benchmark.
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# 15 Standard Industrial Categories in MVTec AD
MVTEC_TEXTURES: Tuple[str, ...] = (
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
)

MVTEC_OBJECTS: Tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)

MVTEC_CATEGORIES: Tuple[str, ...] = MVTEC_TEXTURES + MVTEC_OBJECTS

MVTEC_DOWNLOAD_URL: str = (
    "https://www.mydrive.ch/shares/38536/3830184030e49c74741a77d64795a35f/download/420938113-1629952094/{category}.tar.xz"
)


@dataclass
class MVTecSample:
    """Represents a single inspection sample from MVTec AD."""

    image_path: Path
    mask_path: Optional[Path]
    category: str
    split: str  # 'train' or 'test'
    defect_type: str  # 'good' or defect name (e.g. 'scratch', 'crack')
    label: int  # 0 for normal/good, 1 for anomalous


def download_and_extract_mvtec(
    category: str,
    root_dir: Union[str, Path] = "./data/mvtec_anomaly_detection",
    force_download: bool = False,
) -> Path:
    """Download and extract a specific MVTec AD category archive.

    Args:
        category: Name of the category (e.g. 'bottle', 'screw').
        root_dir: Target root directory for storage.
        force_download: Whether to re-download even if folder exists.

    Returns:
        Path to the extracted category directory.
    """
    if category not in MVTEC_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {MVTEC_CATEGORIES}"
        )

    root = Path(root_dir).resolve()
    target_dir = root / category

    if target_dir.exists() and not force_download:
        # Check if train and test folders exist
        if (target_dir / "train").exists() and (target_dir / "test").exists():
            return target_dir

    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / f"{category}.tar.xz"
    url = MVTEC_DOWNLOAD_URL.format(category=category)

    print(f"[INFO] Downloading MVTec AD category '{category}' from official source...")
    try:
        urllib.request.urlretrieve(url, archive_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MVTec AD category '{category}' from {url}. "
            "Please check network connection or manually place files in 'data/'."
        ) from exc

    print(f"[INFO] Extracting archive: {archive_path}...")
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(path=root)

    # Clean up archive file after extraction
    if archive_path.exists():
        archive_path.unlink()

    print(f"[INFO] Successfully prepared category '{category}' at {target_dir}")
    return target_dir


class MVTecDataset:
    """PyTorch-compatible dataset for MVTec Anomaly Detection.

    Adheres strictly to the unsupervised learning paradigm:
    - 'train' split loads ONLY normal samples ('good').
    - 'test' split loads both normal ('good') and anomalous samples with ground-truth masks.
    - Does NOT apply destructive texturing transforms (CutOut, Salt&Pepper, heavy noise).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        category: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize the MVTecDataset.

        Args:
            root_dir: Path to directory containing MVTec category folders.
            category: One of the 15 MVTec AD classes.
            split: 'train' or 'test'.
            image_size: Target (height, width) for resizing.
            transform: Callable transform for RGB images.
            target_transform: Callable transform for binary segmentation masks.
        """
        if category not in MVTEC_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. Available: {MVTEC_CATEGORIES}"
            )
        if split not in ("train", "test"):
            raise ValueError(f"Split must be 'train' or 'test', got '{split}'.")

        self.root_dir = Path(root_dir).resolve()
        self.category = category
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform

        self.category_dir = self.root_dir / category
        self.samples: List[MVTecSample] = self._index_dataset()

    def _index_dataset(self) -> List[MVTecSample]:
        """Index images and ground truth masks from the filesystem."""
        samples: List[MVTecSample] = []
        split_dir = self.category_dir / self.split
        gt_dir = self.category_dir / "ground_truth"

        if not split_dir.exists():
            # Return empty if directory not populated yet (e.g. prior to download)
            return samples

        # In 'train', strictly index 'good' (normal)
        if self.split == "train":
            good_dir = split_dir / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png")):
                    samples.append(
                        MVTecSample(
                            image_path=img_path,
                            mask_path=None,
                            category=self.category,
                            split="train",
                            defect_type="good",
                            label=0,
                        )
                    )
        else:
            # In 'test', index all subdirectories (good + defect types)
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name
                is_anomalous = defect_type != "good"
                label = 1 if is_anomalous else 0

                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path: Optional[Path] = None
                    if is_anomalous and gt_dir.exists():
                        # GT mask filename typically matches <stem>_mask.png
                        expected_mask = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                        if expected_mask.exists():
                            mask_path = expected_mask

                    samples.append(
                        MVTecSample(
                            image_path=img_path,
                            mask_path=mask_path,
                            category=self.category,
                            split="test",
                            defect_type=defect_type,
                            label=label,
                        )
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str, int]]:
        """Fetch item at index.

        Returns:
            Dict containing:
                - 'image': RGB image array shaped (H, W, 3) normalized [0, 1]
                - 'mask': Binary ground-truth mask shaped (H, W) in {0, 1}
                - 'label': Int 0 (normal) or 1 (anomalous)
                - 'defect_type': String identifier of defect
                - 'image_path': String absolute path to image
        """
        sample = self.samples[idx]

        # Load RGB image
        with Image.open(sample.image_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            if self.image_size:
                pil_img = pil_img.resize(self.image_size, Image.BILINEAR)
            img_array = np.array(pil_img, dtype=np.float32) / 255.0

        # Load or synthesize Ground Truth Mask
        if sample.mask_path and sample.mask_path.exists():
            with Image.open(sample.mask_path) as pil_mask:
                pil_mask = pil_mask.convert("L")
                if self.image_size:
                    pil_mask = pil_mask.resize(self.image_size, Image.NEAREST)
                mask_array = (np.array(pil_mask, dtype=np.float32) > 127).astype(np.float32)
        else:
            # Normal sample: mask is all zeros
            h, w = self.image_size if self.image_size else (img_array.shape[0], img_array.shape[1])
            mask_array = np.zeros((h, w), dtype=np.float32)

        if self.transform is not None:
            img_array = self.transform(img_array)
        if self.target_transform is not None:
            mask_array = self.target_transform(mask_array)

        return {
            "image": img_array,
            "mask": mask_array,
            "label": sample.label,
            "defect_type": sample.defect_type,
            "image_path": str(sample.image_path),
        }

