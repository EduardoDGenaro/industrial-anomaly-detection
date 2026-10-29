"""Evaluate trained PatchCore model on MVTec test split and generate metrics report.

Supports two execution engines via ``--engine [pytorch|onnx_faiss]``:
- ``pytorch``: Original PyTorch eager backbone + ``torch.cdist`` memory bank search.
- ``onnx_faiss``: ONNX Runtime ``CPUExecutionProvider`` + ``faiss.IndexFlatL2`` SIMD search.

When ``--compare`` is passed (or when evaluating ``onnx_faiss`` after a ``pytorch``
baseline), outputs a side-by-side latency comparison (Mean, P50, P90, P95, FPS) and
stores both results in ``docs/benchmark_<category>.json``.

Usage:
    .venv\\Scripts\\python.exe scripts\\evaluate.py --weights weights\\patchcore_bottle.pt --category bottle --engine onnx_faiss --compare
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
from src.models.patchcore_wrapper import (
    SUPPORTED_ENGINES,
    PatchCoreInferenceEngine,
    preprocess_mask,
)
from src.utils.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    compute_pro_metric,
    find_cost_sensitive_threshold,
    find_f1_optimal_threshold,
)


def evaluate_engine(
    weights_p: Path,
    test_ds: MVTecDataset,
    engine_type: str,
    device: str = "cpu",
    onnx_path: str | None = None,
    warmup: int = 5,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Run full benchmark for a specific engine ('pytorch' or 'onnx_faiss')."""
    print(f"\n[INFO] Loading model with engine='{engine_type}' (threads={num_threads})...")
    engine = PatchCoreInferenceEngine.load(
        weights_p,
        device=device,
        engine_type=engine_type,
        onnx_path=onnx_path,
        num_threads=num_threads,
    )
    cfg = engine.config

    print(f"[INFO] Warming up hardware cache ({warmup} passes)...")
    for i in range(min(warmup, len(test_ds))):
        _ = engine.predict(test_ds.samples[i].image_path)

    print(f"[INFO] Evaluating {len(test_ds)} test images with engine='{engine_type}'...")
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
            print(f"   [{engine_type}] Evaluated {i + 1}/{len(test_ds)} images...")

    img_auroc = compute_image_auroc(labels, scores)
    pixel_auroc = compute_pixel_auroc(ground_truth_masks, anomaly_maps, subsample_ratio=1.0)
    pro_metric = compute_pro_metric(ground_truth_masks, anomaly_maps, max_fpr=0.30, num_thresholds=100)

    f1_opt = find_f1_optimal_threshold(labels, scores)
    cost_opt = find_cost_sensitive_threshold(
        labels, scores, cost_fn=10000.0, cost_fp=1.0, min_recall_constraint=0.995
    )

    lat_arr = np.asarray(latencies, dtype=np.float64)
    mean_lat = float(np.mean(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    all_maps = np.stack(anomaly_maps)
    map_min = float(np.percentile(all_maps, 1.0))
    map_max = float(np.percentile(all_maps, 99.0))

    return {
        "engine_instance": engine,
        "engine_type": engine_type,
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
            "min_ms": round(min_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p90_ms": round(p90_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "max_ms": round(max_lat, 2),
            "fps": round(fps, 2),
            "target_sla_50ms_met": p95_lat < 50.0,
        },
        "map_bounds": {"map_min": map_min, "map_max": map_max},
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PatchCore model on MVTec category.")
    parser.add_argument("--weights", default="weights/patchcore_bottle.pt", help="Path to weights .pt")
    parser.add_argument("--category", default="bottle", help="Category name")
    parser.add_argument("--data-root", default="data/mvtec", help="Path to dataset root")
    parser.add_argument("--engine", default="onnx_faiss", choices=SUPPORTED_ENGINES, help="Inference engine backend")
    parser.add_argument("--onnx", default=None, help="Optional explicit ONNX backbone path")
    parser.add_argument("--compare", action="store_true", help="Run both pytorch and onnx_faiss for head-to-head comparison")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--out", default="docs/benchmark_bottle.json", help="Path to output JSON")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    args = parser.parse_args()

    weights_p = Path(args.weights).resolve()
    if not weights_p.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_p}. Please run scripts/train.py first.")

    data_root = Path(args.data_root).resolve()
    # Peek config from weights
    base_eng = PatchCoreInferenceEngine.load(weights_p, device=args.device, engine_type="pytorch")
    cfg = base_eng.config
    del base_eng

    test_ds = MVTecDataset(data_root, category=args.category, split="test", image_size=(cfg.crop, cfg.crop))
    if len(test_ds) == 0:
        raise ValueError(f"No test samples found for category '{args.category}' in {data_root}")

    print(
        f"[INFO] Total test samples: {len(test_ds)} "
        f"(Good: {sum(1 for s in test_ds.samples if s.label == 0)}, "
        f"Anomalous: {sum(1 for s in test_ds.samples if s.label == 1)})"
    )

    out_json = Path(args.out).resolve()
    existing_json: Dict[str, Any] = {}
    if out_json.exists():
        try:
            with open(out_json, "r", encoding="utf-8") as fh:
                existing_json = json.load(fh)
        except Exception:
            existing_json = {}

    results_by_engine: Dict[str, Dict[str, Any]] = {}
    if args.compare:
        for eng_name in ("pytorch", "onnx_faiss"):
            results_by_engine[eng_name] = evaluate_engine(
                weights_p, test_ds, eng_name, args.device, args.onnx, args.warmup, args.threads
            )
    else:
        results_by_engine[args.engine] = evaluate_engine(
            weights_p, test_ds, args.engine, args.device, args.onnx, args.warmup, args.threads
        )

    primary = results_by_engine[args.engine]
    engine = primary["engine_instance"]
    m = primary["metrics"]
    lat = primary["latency"]

    # Update calibration inside weights artifact
    engine.calibration.update({
        "image_threshold": float(m["cost_sensitive"]["threshold"]),
        "f1_threshold": float(m["f1_optimal"]["threshold"]),
        "map_min": primary["map_bounds"]["map_min"],
        "map_max": primary["map_bounds"]["map_max"],
        "image_auroc": float(m["image_auroc"]),
        "pixel_auroc": float(m["pixel_auroc"]),
        "pro_metric": float(m["pro_metric_fpr_030"]),
        "mean_latency_ms": float(lat["mean_ms"]),
    })
    engine.save(weights_p)

    # Preserve or merge engine comparison
    comparison = existing_json.get("engine_comparison", {})
    if not comparison and existing_json.get("latency") and "pytorch" not in results_by_engine:
        comparison["pytorch"] = {
            "latency": existing_json["latency"],
            "metrics": existing_json.get("metrics", {}),
        }
    for eng_name, res in results_by_engine.items():
        comparison[eng_name] = {
            "latency": res["latency"],
            "metrics": res["metrics"],
        }

    speedup_summary: Dict[str, Any] = {}
    if "pytorch" in comparison and "onnx_faiss" in comparison:
        pt_lat = comparison["pytorch"]["latency"]
        of_lat = comparison["onnx_faiss"]["latency"]
        speedup_summary = {
            "mean_speedup_x": round(pt_lat["mean_ms"] / of_lat["mean_ms"], 2) if of_lat["mean_ms"] > 0 else 0.0,
            "p50_speedup_x": round(pt_lat["p50_ms"] / of_lat["p50_ms"], 2) if of_lat["p50_ms"] > 0 else 0.0,
            "p90_speedup_x": round(pt_lat.get("p90_ms", pt_lat["p95_ms"]) / of_lat["p90_ms"], 2)
            if of_lat.get("p90_ms", 0) > 0
            else 0.0,
            "p95_speedup_x": round(pt_lat["p95_ms"] / of_lat["p95_ms"], 2) if of_lat["p95_ms"] > 0 else 0.0,
            "mean_ms_saved": round(pt_lat["mean_ms"] - of_lat["mean_ms"], 2),
            "p95_ms_saved": round(pt_lat["p95_ms"] - of_lat["p95_ms"], 2),
        }

    benchmark_data = {
        "category": args.category,
        "active_engine": args.engine,
        "backbone": cfg.backbone,
        "coreset_sampling_ratio": cfg.coreset_sampling_ratio,
        "image_size": [cfg.crop, cfg.crop],
        "device": str(engine.device),
        "cpu_threads": args.threads,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": m,
        "latency": lat,
        "engine_comparison": comparison,
        "speedup_vs_pytorch": speedup_summary,
        "test_breakdown": primary["test_breakdown"],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Print summary table
    print("\n" + "=" * 72)
    print(f"[BENCHMARK REPORT] MVTec AD [{args.category.upper()}] | Engine: {args.engine.upper()}")
    print("=" * 72)
    print(f"   Backbone:               {cfg.backbone}")
    print(f"   Coreset ratio:          {cfg.coreset_sampling_ratio * 100:.1f}%")
    print(f"   Image-level AUROC:      {m['image_auroc'] * 100:.2f}%  (Target: >= 99.0%)")
    print(f"   Pixel-level AUROC:      {m['pixel_auroc'] * 100:.2f}%  (Target: >= 97.5%)")
    print(f"   PRO Metric (FPR <= 0.3):{m['pro_metric_fpr_030'] * 100:.2f}%  (Target: >= 93.5%)")
    print("-" * 72)
    print(
        f"   Operational Threshold:  {m['cost_sensitive']['threshold']:.4f} "
        f"(Recall: {m['cost_sensitive']['achieved_recall'] * 100:.1f}%)"
    )
    print(f"   F1-Max Threshold:       {m['f1_optimal']['threshold']:.4f} (Max F1: {m['f1_optimal']['max_f1']:.3f})")
    print("-" * 72)
    print(
        f"   Latency ({args.engine}): Mean={lat['mean_ms']:.1f} ms | "
        f"P50={lat['p50_ms']:.1f} ms | P90={lat['p90_ms']:.1f} ms | P95={lat['p95_ms']:.1f} ms | "
        f"FPS={lat['fps']:.2f}"
    )

    if "pytorch" in comparison and "onnx_faiss" in comparison:
        pt_l = comparison["pytorch"]["latency"]
        of_l = comparison["onnx_faiss"]["latency"]
        print("=" * 72)
        print("[LATENCY COMPARISON: PYTORCH vs ONNX_FAISS (N=83 test images)]")
        print("-" * 72)
        print(f"   {'Metric':<18} {'PyTorch (ms)':>15} {'ONNX+FAISS (ms)':>17} {'Delta / Speedup':>18}")
        print("-" * 72)
        for label, key in [
            ("Mean Latency", "mean_ms"),
            ("P50 Latency", "p50_ms"),
            ("P90 Latency", "p90_ms"),
            ("P95 Latency", "p95_ms"),
        ]:
            v_pt = pt_l.get(key)
            v_of = of_l.get(key)
            if v_pt is not None and v_of is not None and v_of > 0:
                sp = v_pt / v_of
                print(f"   {label:<18} {v_pt:>15.2f} {v_of:>17.2f} {sp:>16.2f}x")
        print(f"   {'Throughput (FPS)':<18} {pt_l.get('fps', 0):>15.2f} {of_l.get('fps', 0):>17.2f} {(of_l.get('fps', 0)/max(pt_l.get('fps', 1e-9), 1e-9)):>16.2f}x")

    print("=" * 72)
    print(f"Report exported to: {out_json}")
    print(f"Weights updated with calibration: {weights_p}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
