"""Train (fit) PatchCore memory bank on normal images of an MVTec category.

Usage:
    .venv\\Scripts\\python.exe scripts\\train.py --config configs\\patchcore_mvtec.yaml --category bottle --out weights\\patchcore_bottle.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.mvtec import IMAGE_EXTENSIONS, validate_category
from src.models.patchcore_wrapper import (
    PatchCoreConfig,
    PatchCoreInferenceEngine,
    load_config_yaml,
)


def get_train_images(root: Path, category: str) -> List[Path]:
    train_good = root / category / "train" / "good"
    if not train_good.exists():
        raise FileNotFoundError(f"Training directory not found: {train_good}")
    images = sorted(p for p in train_good.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No valid images found in {train_good}")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit PatchCore on an MVTec category.")
    parser.add_argument("--config", default="configs/patchcore_mvtec.yaml", help="Path to config YAML")
    parser.add_argument("--category", default="bottle", help="Category name")
    parser.add_argument("--data-root", default="data/mvtec", help="Path to MVTec dataset root")
    parser.add_argument("--out", default="weights/patchcore_bottle.pt", help="Output path for weights")
    parser.add_argument("--backbone", default=None, help="Override backbone (e.g. wide_resnet50_2 or resnet18)")
    parser.add_argument("--coreset-ratio", type=float, default=None, help="Override coreset ratio")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    cfg, _ = load_config_yaml(args.config)
    if args.backbone:
        cfg.backbone = args.backbone
    if args.coreset_ratio is not None:
        cfg.coreset_sampling_ratio = args.coreset_ratio

    data_root = Path(args.data_root).resolve()
    print(f"[INFO] Validating category '{args.category}' in {data_root}...")
    report = validate_category(data_root, args.category)
    if not report.ok:
        print(f"[WARNING] Validation issues detected:\n{report.summary()}")
    else:
        print("[INFO] Dataset validation passed.")

    train_imgs = get_train_images(data_root, args.category)
    print(f"[INFO] Initializing PatchCore with backbone={cfg.backbone}, layers={cfg.layers}, coreset_ratio={cfg.coreset_sampling_ratio}...")
    engine = PatchCoreInferenceEngine(config=cfg, device=args.device)

    print(f"[INFO] Fitting memory bank on {len(train_imgs)} normal training images...")
    t_start = time.perf_counter()
    stats = engine.fit(train_imgs, batch_size=8, verbose=True)
    elapsed = time.perf_counter() - t_start

    # Compute baseline score statistics on normal images for calibration
    print("[INFO] Calibrating score baseline on normal training images...")
    scores: List[float] = []
    map_maxes: List[float] = []
    for img_p in train_imgs[:30]:  # sample subset for fast calibration
        res = engine.predict(img_p)
        scores.append(res["anomaly_score"])
        map_maxes.append(float(res["anomaly_map"].max()))

    engine.calibration = {
        "nominal_score_mean": float(sum(scores) / len(scores)),
        "nominal_score_max": float(max(scores)),
        "map_min": 0.0,
        "map_max": float(max(map_maxes) * 1.5),
        "category": args.category,
    }
    engine.metadata.update({
        "category": args.category,
        "training_time_s": round(elapsed, 2),
    })

    out_path = Path(args.out).resolve()
    engine.save(out_path)
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"[SUCCESS] PatchCore model trained and saved successfully!")
    print(f"   Destination: {out_path} ({file_mb:.2f} MB)")
    print(f"   Memory Bank: {engine.memory_bank.shape[0]} vectors x {engine.memory_bank.shape[1]} dims")
    print(f"   Training time: {elapsed:.2f}s ({stats.get('feature_extraction_s', 0)}s features, {stats.get('coreset_selection_s', 0)}s coreset)")
    print(f"=======================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

