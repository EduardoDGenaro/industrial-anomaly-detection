"""MVTec Anomaly Detection (MVTec AD) Dataset Loader, Parser and Validator.

Provides:
- Category constants for the 15 MVTec AD classes.
- ``validate_category``: integrity check (official counts, 1:1 image/mask pairing).
- ``extract_archive``: safe local extraction of an official ``<category>.tar.xz``
  or a Kaggle ``.zip`` (path-traversal protected).
- ``MVTecDataset``: unsupervised-paradigm dataset (train = only ``good``).

Data acquisition (the official MVTec mirror requires accepting the license form,
so it cannot be automated):
- Kaggle mirror:  ``python scripts/download_kaggle.py --category bottle``
- Manual:         download ``bottle.tar.xz`` from
                  https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
                  then ``python -m src.dataset.mvtec --category bottle --archive data/bottle.tar.xz``

License: MVTec AD is CC BY-NC-SA 4.0 (non-commercial use).
"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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

MVTEC_OFFICIAL_PAGE: str = "https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads"

# Official image counts per split/defect (used for integrity validation).
# Extend as new pilot categories are onboarded.
MVTEC_EXPECTED_COUNTS: Dict[str, Dict[str, Dict[str, int]]] = {
    "bottle": {
        "train": {"good": 209},
        "test": {"good": 20, "broken_large": 20, "broken_small": 22, "contamination": 21},
    },
}

IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class MVTecSample:
    """Represents a single inspection sample from MVTec AD."""

    image_path: Path
    mask_path: Optional[Path]
    category: str
    split: str  # 'train' or 'test'
    defect_type: str  # 'good' or defect name (e.g. 'scratch', 'crack')
    label: int  # 0 for normal/good, 1 for anomalous


@dataclass
class ValidationReport:
    """Result of a category integrity validation."""

    category: str
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    num_masks: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"[VALIDATION] category='{self.category}'"]
        for split, per_defect in self.counts.items():
            for defect, n in sorted(per_defect.items()):
                lines.append(f"   {split:<5} / {defect:<15} {n:>4} images")
        lines.append(f"   ground_truth masks       {self.num_masks:>4}")
        if self.ok:
            lines.append("[VALIDATION] OK - integrity checks passed.")
        else:
            lines.append(f"[VALIDATION] FAILED - {len(self.errors)} problem(s):")
            lines.extend(f"   - {e}" for e in self.errors[:20])
        return "\n".join(lines)


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def validate_category(root_dir: Union[str, Path], category: str) -> ValidationReport:
    """Validate the on-disk structure and integrity of an MVTec AD category.

    Checks: required folders exist, official image counts (when known), every
    anomalous test image has exactly one ground-truth mask, no orphan masks,
    and train contains only ``good`` samples.
    """
    root = Path(root_dir).resolve()
    cat_dir = root / category
    report = ValidationReport(category=category)

    if category not in MVTEC_CATEGORIES:
        report.errors.append(f"Unknown category '{category}'.")
        return report
    for sub in ("train", "test", "ground_truth"):
        if not (cat_dir / sub).is_dir():
            report.errors.append(f"Missing folder: {cat_dir / sub}")
    if report.errors:
        return report

    for split in ("train", "test"):
        report.counts[split] = {
            d.name: len(_list_images(d)) for d in sorted((cat_dir / split).iterdir()) if d.is_dir()
        }

    train_defects = set(report.counts["train"]) - {"good"}
    if train_defects:
        report.errors.append(f"train/ must contain only 'good', found: {sorted(train_defects)}")

    expected = MVTEC_EXPECTED_COUNTS.get(category)
    if expected:
        for split, per_defect in expected.items():
            for defect, n_exp in per_defect.items():
                n_got = report.counts.get(split, {}).get(defect, 0)
                if n_got != n_exp:
                    report.errors.append(f"{split}/{defect}: expected {n_exp} images, found {n_got}")

    # 1:1 image <-> mask pairing for anomalous test images
    gt_dir = cat_dir / "ground_truth"
    for defect in report.counts["test"]:
        if defect == "good":
            continue
        images = _list_images(cat_dir / "test" / defect)
        masks = _list_images(gt_dir / defect)
        report.num_masks += len(masks)
        img_stems = {p.stem for p in images}
        mask_stems = {p.stem.removesuffix("_mask") for p in masks}
        for stem in sorted(img_stems - mask_stems):
            report.errors.append(f"Missing mask for test/{defect}/{stem}")
        for stem in sorted(mask_stems - img_stems):
            report.errors.append(f"Orphan mask ground_truth/{defect}/{stem}_mask")

    return report


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_archive(archive: Union[str, Path], root_dir: Union[str, Path], category: str) -> Path:
    """Safely extract a category archive (.tar.xz from MVTec, or .zip from Kaggle).

    Only members belonging to ``category/`` are extracted. Members that would
    escape ``root_dir`` (path traversal) are rejected.
    """
    archive = Path(archive)
    root = Path(root_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        raise FileNotFoundError(
            f"Archive not found: {archive}. Download it from {MVTEC_OFFICIAL_PAGE} "
            "or use scripts/download_kaggle.py."
        )

    def _wanted(name: str) -> Optional[str]:
        # Accept 'bottle/...' or 'mvtec_anomaly_detection/bottle/...'
        parts = Path(name).parts
        if category in parts:
            idx = parts.index(category)
            return str(Path(*parts[idx:]))
        return None

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                rel = _wanted(info.filename)
                if rel is None or info.is_dir():
                    continue
                dest = root / rel
                if not _is_within(root, dest):
                    raise RuntimeError(f"Unsafe path in archive: {info.filename}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
    else:
        with tarfile.open(archive, "r:*") as tar:
            for member in tar.getmembers():
                rel = _wanted(member.name)
                if rel is None or not member.isfile():
                    continue
                dest = root / rel
                if not _is_within(root, dest):
                    raise RuntimeError(f"Unsafe path in archive: {member.name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is None:
                    continue
                with src, open(dest, "wb") as dst:
                    dst.write(src.read())

    return root / category


def load_mask(mask_path: Optional[Path], size_hw: Tuple[int, int]) -> np.ndarray:
    """Load a binary mask {0,1} resized (nearest) to (H, W); zeros if no mask."""
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros(size_hw, dtype=np.uint8)
    with Image.open(mask_path) as m:
        m = m.convert("L").resize((size_hw[1], size_hw[0]), Image.NEAREST)
        return (np.asarray(m) > 127).astype(np.uint8)


class MVTecDataset:
    """Dataset for MVTec Anomaly Detection (unsupervised paradigm).

    - 'train' split loads ONLY normal samples ('good').
    - 'test' split loads normal and anomalous samples with ground-truth masks.
    - Does NOT apply destructive texture transforms (CutOut, Salt&Pepper, heavy noise).
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
        if category not in MVTEC_CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Available: {MVTEC_CATEGORIES}")
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
        samples: List[MVTecSample] = []
        split_dir = self.category_dir / self.split
        gt_dir = self.category_dir / "ground_truth"
        if not split_dir.exists():
            return samples

        defect_dirs = [split_dir / "good"] if self.split == "train" else sorted(
            d for d in split_dir.iterdir() if d.is_dir()
        )
        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            is_anomalous = defect_type != "good"
            for img_path in _list_images(defect_dir):
                mask_path: Optional[Path] = None
                if is_anomalous:
                    candidate = gt_dir / defect_type / f"{img_path.stem}_mask.png"
                    mask_path = candidate if candidate.exists() else None
                samples.append(
                    MVTecSample(
                        image_path=img_path,
                        mask_path=mask_path,
                        category=self.category,
                        split=self.split,
                        defect_type=defect_type,
                        label=int(is_anomalous),
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str, int]]:
        """Return dict with 'image' (H,W,3 float [0,1]), 'mask' (H,W float {0,1}),
        'label', 'defect_type', 'image_path'."""
        sample = self.samples[idx]
        h, w = self.image_size

        with Image.open(sample.image_path) as pil_img:
            pil_img = pil_img.convert("RGB").resize((w, h), Image.BILINEAR)
            img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

        mask_array = load_mask(sample.mask_path, (h, w)).astype(np.float32)

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


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Extract and/or validate an MVTec AD category.")
    parser.add_argument("--category", required=True, choices=MVTEC_CATEGORIES)
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--archive", default=None, help="Local .tar.xz (official) or .zip (Kaggle) to extract.")
    args = parser.parse_args()

    if args.archive:
        print(f"[INFO] Extracting '{args.category}' from {args.archive} ...")
        extract_archive(args.archive, args.root, args.category)

    report = validate_category(args.root, args.category)
    print(report.summary())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
