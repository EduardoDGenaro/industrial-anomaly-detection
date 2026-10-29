"""Evaluate trained PatchCore model on MVTec test split and generate metrics report.

Computes:
- Image-level AUROC
- Pixel-level AUROC
- PRO Metric (Per-Region Overlap at FPR <= 0.30)
- Latency statistics (mean, P50, P95 ms per image)
- Cost-sensitive and F1-optimal thresholds

Outputs:
- Terminal report
- JSON artifact: docs/benchmark_<category>.json
- Updates calibrated thresholds inside the weights artifact

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate.py --weights weights\\patchcore_bottle.pt --category bottle --out docs\\benchmark_bottle.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.mvtec import MVTecDataset
from src.models.patchcore_wrapper import PatchCoreInferenceEngine, preprocess_mask
from src.utils.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_metric,
    find_cost_sensitive_threshold,
    find_f1_optimal_threshold,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PatchCore model on MVTec category.")
    parser.add_argument("--weights", default="weights/patchcore_bottle.pt", help="Path to weights .pt")
    parser.add_argument("--category", default="bottle", help="Category name")
    parser.add_argument("--data-root", default="data/mvtec", help="Path to dataset root")
    parser.add_argument("--out", default="docs/benchmark_bottle.json", help="Path to output JSON")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    args = parser.parse_args()

    weights_p = Path(args.weights).resolve()
    if not weights_p.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_p}. Please run scripts/train.py first.")

    print(f"[INFO] Loading model weights from {weights_p}...")
    engine = PatchCoreInferenceEngine.load(weights_p, device=args.device)
    cfg = engine.config

    data_root = Path(args.data_root).resolve()
    print(f"[INFO] Indexing test dataset for category '{args.category}' from {data_root}...")
    test_ds = MVTecDataset(data_root, category=args.category, split="test", image_size=(cfg.crop, cfg.crop))
    if len(test_ds) == 0:
        raise ValueError(f"No test samples found for category '{args.category}' in {data_root}")

    print(f"[INFO] Total test samples: {len(test_ds)} (Good: {sum(1 for s in test_ds.samples if s.label == 0)}, Anomalous: {sum(1 for s in test_ds.samples if s.label == 1)})")

    # Warmup
    print(f"[INFO] Warming up hardware cache ({args.warmup} passes)...")
    for i in range(min(args.warmup, len(test_ds))):
        _ = engine.predict(test_ds.samples[i].image_path)

    # Full evaluation loop
    print("[INFO] Running evaluation on full test split...")
    labels: List[int] = []
    scores: List[float] = []
    latencies: List[float] = []
    anomaly_maps: List[np.ndarray] = []
    ground_truth_masks: List[np.ndarray] = []
    per_defect_scores: Dict[str, List[float]] = {}

    for i, sample in enumerate(test_ds.samples):
        res = engine.predict(sample.image_path)
        gt_mask = preprocess_mask(sample.mask_path, cfg.resize, cfg.crop)

        labels.append(sample.label)
        scores.append(res["anomaly_score"])
        latencies.append(res["latency_ms"])
        anomaly_maps.append(res["anomaly_map"])
        ground_truth_masks.append(gt_mask)

        per_defect_scores.setdefault(sample.defect_type, []).append(res["anomaly_score"])

        if (i + 1) % 20 == 0 or (i + 1) == len(test_ds):
            print(f"   Evaluated {i + 1}/{len(test_ds)} images...")

    # Calculate metrics
    print("\n[INFO] Calculating performance metrics...")
    img_auroc = compute_image_auroc(labels, scores)
    pixel_auroc = compute_pixel_auroc(ground_truth_masks, anomaly_maps)
    pro_metric = compute_pro_metric(ground_truth_masks, anomaly_maps, max_fpr=0.30, num_thresholds=100)

    f1_opt = find_f1_optimal_threshold(labels, scores)
    cost_opt = find_cost_sensitive_threshold(
        labels, scores, cost_fn=10000.0, cost_fp=1.0, min_recall_constraint=0.995
    )

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # Normalization bounds for heatmap rendering
    all_maps = np.stack(anomaly_maps)
    map_min = float(np.percentile(all_maps, 1.0))
    map_max = float(np.percentile(all_maps, 99.0))

    # Update calibration inside weights artifact
    engine.calibration.update({
        "image_threshold": float(cost_opt["calibrated_threshold"]),
        "f1_threshold": float(f1_opt["optimal_threshold"]),
        "map_min": map_min,
        "map_max": map_max,
        "image_auroc": float(img_auroc),
        "pixel_auroc": float(pixel_auroc),
        "pro_metric": float(pro_metric),
        "mean_latency_ms": round(mean_lat, 2),
    })
    engine.save(weights_p)

    # Build benchmark results report
    benchmark_data = {
        "category": args.category,
        "backbone": cfg.backbone,
        "coreset_sampling_ratio": cfg.coreset_sampling_ratio,
        "image_size": [cfg.crop, cfg.crop],
        "device": str(engine.device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {
            "image_auroc": round(img_auroc, 4),
            "pixel_auroc": round(pixel_auroc, 4),
            "pro_metric_fpr_030": round(pro_metric, 4),
            "f1_optimal": {
                "threshold": round(f1_opt["optimal_threshold"], 4),
                "max_f1": round(f1_opt["max_f1"], 4),
                "precision": round(f1_opt["precision_at_optimal"], 4),
                "recall": round(f1_opt["recall_at_optimal"], 4),
            },
            "cost_sensitive": {
                "threshold": round(cost_opt["calibrated_threshold"], 4),
                "cost_fn": 10000.0,
                "cost_fp": 1.0,
                "achieved_recall": round(cost_opt["achieved_recall"], 4),
                "achieved_precision": round(cost_opt["achieved_precision"], 4),
            },
        },
        "latency": {
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "fps": round(fps, 1),
            "target_sla_50ms_met": p95_lat < 50.0,
        },
        "test_breakdown": {
            k: {
                "count": len(v),
                "mean_score": round(float(np.mean(v)), 4),
                "min_score": round(float(np.min(v)), 4),
                "max_score": round(float(np.max(v)), 4),
            }
            for k, v in per_defect_scores.items()
        },
    }

    out_json = Path(args.out).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print(f"[BENCHMARK REPORT] MVTec AD [{args.category.upper()}]")
    print("=" * 65)
    print(f"   Backbone:               {cfg.backbone}")
    print(f"   Coreset ratio:          {cfg.coreset_sampling_ratio * 100:.1f}%")
    print(f"   Image-level AUROC:      {img_auroc * 100:.2f}%  (Target: >= 99.0%)")
    print(f"   Pixel-level AUROC:      {pixel_auroc * 100:.2f}%  (Target: >= 97.5%)")
    print(f"   PRO Metric (FPR <= 0.3):{pro_metric * 100:.2f}%  (Target: >= 93.5%)")
    print("-" * 65)
    print(f"   Operational Threshold:  {cost_opt['calibrated_threshold']:.4f} (Recall: {cost_opt['achieved_recall'] * 100:.1f}%)")
    print(f"   F1-Max Threshold:       {f1_opt['optimal_threshold']:.4f} (Max F1: {f1_opt['max_f1']:.3f})")
    print("-" * 65)
    print(f"   Latency Mean:           {mean_lat:.1f} ms")
    print(f"   Latency P95:            {p95_lat:.1f} ms  (Target: < 50 ms)")
    print(f"   Throughput:             {fps:.1f} FPS")
    print("=" * 65)
    print(f"Report exported to: {out_json}")
    print(f"Weights updated with calibration: {weights_p}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

