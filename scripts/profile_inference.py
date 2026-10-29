"""Stage-by-stage latency profiler for the PatchCore inference pipeline.

Supports both ``--engine pytorch`` (baseline) and ``--engine onnx_faiss``
(ONNX Runtime CPUExecutionProvider + FAISS IndexFlatL2).

Usage:
    .venv\\Scripts\\python.exe scripts\\profile_inference.py --weights weights\\patchcore_bottle.pt --engine onnx_faiss
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.mvtec import MVTecDataset
from src.models.patchcore_wrapper import (
    SUPPORTED_ENGINES,
    PatchCoreInferenceEngine,
    load_rgb,
    preprocess,
    preprocess_numpy,
)
from src.postprocessing.postprocessor import AnomalyPostProcessor


def stats_summary(values_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Profile PatchCore inference stages.")
    parser.add_argument("--weights", default="weights/patchcore_bottle.pt")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data-root", default="data/mvtec")
    parser.add_argument("--engine", default="pytorch", choices=SUPPORTED_ENGINES)
    parser.add_argument("--onnx", default=None, help="Optional explicit ONNX backbone path")
    parser.add_argument("--num-images", type=int, default=30, help="Number of test images to profile (max 83)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_file = args.out or f"docs/profile_{args.engine}_{args.category}.json"
    weights_p = Path(args.weights).resolve()
    engine = PatchCoreInferenceEngine.load(
        weights_p,
        device="cpu",
        engine_type=args.engine,
        onnx_path=args.onnx,
        num_threads=4,
    )
    cfg = engine.config
    post = AnomalyPostProcessor(
        blur_sigma=cfg.blur_sigma,
        opening_kernel_size=3,
        closing_kernel_size=5,
    )

    ds = MVTecDataset(Path(args.data_root), category=args.category, split="test", image_size=(cfg.crop, cfg.crop))
    samples = ds.samples[: min(args.num_images, len(ds.samples))]

    print(f"[INFO] Engine: {args.engine} | Torch: {torch.__version__} | CPU threads: {engine.num_threads}")
    print(f"[INFO] Memory bank: {tuple(engine.memory_bank.shape)} ({engine.memory_bank.dtype})")
    print(f"[INFO] Warming up ({args.warmup} passes)...")
    for s in samples[: args.warmup]:
        _ = engine.predict(s.image_path)

    preloaded_pil = [load_rgb(s.image_path) for s in samples]

    timings: Dict[str, List[float]] = {
        "1_preprocess_with_io": [],
        "1a_disk_decode_only": [],
        "1b_preprocess_in_memory": [],
        "2_backbone_total": [],
        "3_nn_search_total": [],
        "3a_nn_patch_search": [],
        "3b_nn_reweight_eq7": [],
        "4_postprocess_total": [],
        "4a_upsample_and_blur": [],
        "4b_hysteresis_morph_bbox": [],
        "4c_heatmap_overlay": [],
        "total_e2e_with_post": [],
        "total_predict_only": [],
    }

    thr = float(engine.calibration.get("image_threshold", 1.0624))
    nom_max = float(engine.calibration.get("nominal_score_max", 0.6276))
    th_norm = float(np.clip(thr / (nom_max * 1.5 + 1e-5), 0.1, 0.95))

    print(f"[INFO] Profiling {len(samples)} test images stage-by-stage ({args.engine})...")
    for idx, s in enumerate(samples):
        # --- Stage 1: Pre-processing ---
        t0 = time.perf_counter()
        pil_disk = load_rgb(s.image_path)
        t_io = time.perf_counter()
        if args.engine == "onnx_faiss":
            batch_np, rgb = preprocess_numpy(pil_disk, cfg.resize, cfg.crop)
        else:
            tensor, rgb = preprocess(pil_disk, cfg.resize, cfg.crop)
        t1 = time.perf_counter()

        t_mem0 = time.perf_counter()
        if args.engine == "onnx_faiss":
            _, _ = preprocess_numpy(preloaded_pil[idx], cfg.resize, cfg.crop)
        else:
            _, _ = preprocess(preloaded_pil[idx], cfg.resize, cfg.crop)
        t_mem1 = time.perf_counter()

        # --- Stage 2: Backbone Forward Pass ---
        t2_0 = time.perf_counter()
        if args.engine == "onnx_faiss" and engine.ort_session is not None:
            emb_np = engine.embed_onnx(batch_np)
            patches_np = np.ascontiguousarray(
                emb_np[0].transpose(1, 2, 0).reshape(-1, emb_np.shape[1]),
                dtype=np.float32,
            )
        else:
            emb = engine.embed(tensor)
            patches = emb[0].permute(1, 2, 0).reshape(-1, emb.shape[1])
        t2_1 = time.perf_counter()

        # --- Stage 3: Nearest Neighbor Search ---
        t3_0 = time.perf_counter()
        if args.engine == "onnx_faiss" and engine.faiss_index is not None:
            k = max(1, int(cfg.n_neighbors))
            d2, _ = engine.faiss_index.search(patches_np, k)
            patch_scores = np.sqrt(np.maximum(d2[:, 0], 0.0)).astype(np.float32)
            t3_search = time.perf_counter()

            star = int(np.argmax(patch_scores))
            d_nb = np.sqrt(np.maximum(d2[star], 0.0))
            exp_d = np.exp(d_nb - np.max(d_nb))
            w = 1.0 - float(exp_d[0] / np.sum(exp_d))
            _ = float(w * float(patch_scores[star]))
        else:
            patch_scores, nn_idx = engine._nearest(patches)
            t3_search = time.perf_counter()
            _ = engine._image_score(patches, patch_scores, nn_idx)
        t3_1 = time.perf_counter()

        # --- Stage 4: Post-processing ---
        t4_0 = time.perf_counter()
        amap = engine._anomaly_map(patch_scores, (cfg.crop, cfg.crop))
        t4_blur = time.perf_counter()

        norm_map = engine.normalize_map(amap)
        binary_mask = post.apply_hysteresis_threshold(
            norm_map, high_threshold=th_norm, low_threshold=th_norm * 0.70
        )
        filtered_mask = post.filter_morphology(binary_mask)
        _ = post.extract_defect_regions(filtered_mask, norm_map)
        t4_morph = time.perf_counter()

        _ = post.create_heatmap_overlay(rgb, norm_map, alpha=0.45)
        t4_1 = time.perf_counter()

        d_pre_io = (t1 - t0) * 1000.0
        d_io = (t_io - t0) * 1000.0
        d_pre_mem = (t_mem1 - t_mem0) * 1000.0
        d_bb = (t2_1 - t2_0) * 1000.0
        d_nn = (t3_1 - t3_0) * 1000.0
        d_nn_search = (t3_search - t3_0) * 1000.0
        d_nn_eq7 = (t3_1 - t3_search) * 1000.0
        d_post = (t4_1 - t4_0) * 1000.0
        d_post_blur = (t4_blur - t4_0) * 1000.0
        d_post_morph = (t4_morph - t4_blur) * 1000.0
        d_post_ovl = (t4_1 - t4_morph) * 1000.0

        timings["1_preprocess_with_io"].append(d_pre_io)
        timings["1a_disk_decode_only"].append(d_io)
        timings["1b_preprocess_in_memory"].append(d_pre_mem)
        timings["2_backbone_total"].append(d_bb)
        timings["3_nn_search_total"].append(d_nn)
        timings["3a_nn_patch_search"].append(d_nn_search)
        timings["3b_nn_reweight_eq7"].append(d_nn_eq7)
        timings["4_postprocess_total"].append(d_post)
        timings["4a_upsample_and_blur"].append(d_post_blur)
        timings["4b_hysteresis_morph_bbox"].append(d_post_morph)
        timings["4c_heatmap_overlay"].append(d_post_ovl)
        timings["total_predict_only"].append(d_pre_io + d_bb + d_nn + d_post_blur)
        timings["total_e2e_with_post"].append(d_pre_io + d_bb + d_nn + d_post)

    report = {
        "category": args.category,
        "engine": args.engine,
        "backbone": cfg.backbone,
        "num_images_profiled": len(samples),
        "cpu_threads": engine.num_threads,
        "memory_bank_shape": list(engine.memory_bank.shape),
        "stages": {k: stats_summary(v) for k, v in timings.items()},
    }

    total_mean = report["stages"]["total_e2e_with_post"]["mean_ms"]
    for k, st in report["stages"].items():
        st["share_pct"] = round(100.0 * st["mean_ms"] / total_mean, 2) if total_mean > 0 else 0.0

    out_p = Path(out_file).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 86)
    print(f"[PROFILING BREAKDOWN] Engine: {args.engine.upper()} | Category: {args.category.upper()} | N={len(samples)} images")
    print("=" * 86)
    print(f"{'Stage':<34} {'Mean (ms)':>10} {'P50 (ms)':>10} {'P90 (ms)':>10} {'P95 (ms)':>10} {'Share %':>9}")
    print("-" * 86)
    rows = [
        ("1. Pre-processing (with I/O)", "1_preprocess_with_io"),
        ("   1a. Disk PNG decode", "1a_disk_decode_only"),
        ("   1b. Resize+Crop+Norm (RAM)", "1b_preprocess_in_memory"),
        ("2. Backbone Forward", "2_backbone_total"),
        ("3. Nearest Neighbor (16.4k bank)", "3_nn_search_total"),
        ("   3a. Patch NN search (784x16385)", "3a_nn_patch_search"),
        ("   3b. Eq.7 k-NN re-weighting", "3b_nn_reweight_eq7"),
        ("4. Post-processing (Total)", "4_postprocess_total"),
        ("   4a. Upsample + Gaussian blur", "4a_upsample_and_blur"),
        ("   4b. Hysteresis + Morph + BBox", "4b_hysteresis_morph_bbox"),
        ("   4c. Thermal RGB Overlay", "4c_heatmap_overlay"),
        ("TOTAL (engine.predict only)", "total_predict_only"),
        ("TOTAL End-to-End (with UI post)", "total_e2e_with_post"),
    ]
    for label, key in rows:
        st = report["stages"][key]
        if key.startswith("total"):
            print("-" * 86)
        print(
            f"{label:<34} {st['mean_ms']:>10.2f} {st['p50_ms']:>10.2f} "
            f"{st['p90_ms']:>10.2f} {st['p95_ms']:>10.2f} {st['share_pct']:>8.1f}%"
        )
    print("=" * 86)
    print(f"Saved detailed profile JSON to: {out_p}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
