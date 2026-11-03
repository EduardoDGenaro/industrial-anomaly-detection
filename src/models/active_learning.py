"""Human-in-the-Loop and Active Learning module for PatchCore memory bank updates.

Enables manufacturing line operators to contest false alarms (false positives)
caused by acceptable supplier/batch variations, incorporating representative
patch features directly into the active FAISS index in real time (< 150 ms)
without retraining the model from scratch.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch

from src.models.patchcore_wrapper import (
    PatchCoreInferenceEngine,
    load_rgb,
    preprocess_numpy,
    resize_center_crop,
)
from src.utils.audit_logger import DEFAULT_AUDIT_LOG_PATH, log_contested_sample

DEFAULT_WEIGHTS_DIR = Path("weights")
DEFAULT_FEEDBACK_ROOT = Path("data/feedback")

# Process-local cache for standalone calls to incorporate_normal_sample(image, category)
_STANDALONE_ENGINE_CACHE: Dict[str, Tuple[float, PatchCoreInferenceEngine]] = {}


def select_representative_patches(
    patches_np: np.ndarray,
    faiss_index: Any = None,
    target_threshold: Optional[float] = None,
    subsampling_ratio: float = 0.12,
    min_vectors: int = 32,
    max_vectors: int = 196,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the most representative patch vectors from a contested sample.

    Uses Farthest-Point Novelty Subsampling over the 28x28 patch grid:
    1. Queries the current FAISS index to measure each patch's novelty distance
       relative to the existing normal memory bank.
    2. Iteratively selects the most novel patch (largest distance to the bank
       plus already-selected patches) and updates local neighborhood distances
       until all patches lie strictly below ``0.62 * target_threshold`` and at
       least ``min_vectors`` diverse patches are captured.
    """
    patches = np.ascontiguousarray(patches_np, dtype=np.float32)
    n_total, _ = patches.shape
    if n_total == 0:
        raise ValueError("patches_np cannot be empty.")

    target_k = int(np.clip(round(n_total * subsampling_ratio), min_vectors, min(max_vectors, n_total)))

    if faiss_index is not None and getattr(faiss_index, "ntotal", 0) > 0:
        d2, _ = faiss_index.search(patches, 1)
        min_dists = np.sqrt(np.maximum(d2[:, 0], 0.0)).astype(np.float32)
    else:
        center = np.mean(patches, axis=0, keepdims=True)
        min_dists = np.linalg.norm(patches - center, axis=1).astype(np.float32)

    if target_threshold is not None and target_threshold > 0:
        safe_dist_cap = float(target_threshold) * 0.62
    else:
        safe_dist_cap = float(np.percentile(min_dists, 65.0))

    # Precompute squared norms for fast vectorized L2 distance updates:
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 <a, b>
    sq_norms = np.sum(patches * patches, axis=1)
    selected_indices: List[int] = []
    chosen_mask = np.zeros(n_total, dtype=bool)

    for _ in range(min(max_vectors, n_total)):
        masked_dists = np.where(chosen_mask, -1.0, min_dists)
        best_idx = int(np.argmax(masked_dists))
        best_dist = float(masked_dists[best_idx])

        # Stop once we have at least target_k representative vectors AND all patches
        # are guaranteed to have nearest-neighbor distance < safe_dist_cap
        if len(selected_indices) >= target_k and best_dist < safe_dist_cap:
            break
        if best_dist <= 1e-7 and len(selected_indices) >= min_vectors:
            break

        selected_indices.append(best_idx)
        chosen_mask[best_idx] = True

        # Update min_dists with distance to the newly selected patch
        vec = patches[best_idx]
        d2_new = np.maximum(sq_norms + sq_norms[best_idx] - 2.0 * (patches @ vec), 0.0)
        d_new = np.sqrt(d2_new, dtype=np.float32)
        min_dists = np.minimum(min_dists, d_new)

    idx_arr = np.asarray(selected_indices, dtype=np.int64)
    selected_vectors = np.ascontiguousarray(patches[idx_arr], dtype=np.float32)
    return selected_vectors, idx_arr


def atomic_save_engine_checkpoint(
    engine: PatchCoreInferenceEngine,
    weights_path: Union[str, Path],
    preserve_mtime: bool = False,
) -> Path:
    """Atomically write the updated PatchCore checkpoint via temp file + ``os.replace``."""
    target = Path(weights_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    orig_stat = target.stat() if (preserve_mtime and target.exists()) else None

    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        engine.save(tmp_path)
        os.replace(tmp_path, target)
        if orig_stat is not None:
            os.utime(target, (orig_stat.st_atime, orig_stat.st_mtime))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return target


def save_feedback_image(
    image_path_or_array: Union[str, Path, Image.Image, np.ndarray],
    category: str,
    feedback_root: Union[str, Path] = DEFAULT_FEEDBACK_ROOT,
) -> Path:
    """Save or copy the operator-homologated sample to ``data/feedback/<category>/``."""
    cat_dir = Path(feedback_root) / str(category)
    cat_dir.mkdir(parents=True, exist_ok=True)

    ts_label = time.strftime("%Y%m%d_%H%M%S", time.gmtime())

    if isinstance(image_path_or_array, (str, Path)) and Path(image_path_or_array).is_file():
        src_path = Path(image_path_or_array)
        digest = hashlib.sha1(src_path.read_bytes()).hexdigest()[:8]
        dst_path = cat_dir / f"feedback_{category}_{ts_label}_{digest}_{src_path.name}"
        shutil.copy2(src_path, dst_path)
        return dst_path

    pil_img = load_rgb(image_path_or_array)
    raw_bytes = np.asarray(pil_img, dtype=np.uint8).tobytes()
    digest = hashlib.sha1(raw_bytes).hexdigest()[:8]
    dst_path = cat_dir / f"feedback_{category}_{ts_label}_{digest}.png"
    pil_img.save(dst_path, format="PNG")
    return dst_path


def _resolve_engine_for_category(
    category: str,
    engine: Optional[PatchCoreInferenceEngine] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[PatchCoreInferenceEngine, Path]:
    """Return an active ``PatchCoreInferenceEngine`` (with FAISS + ONNX) and its checkpoint path."""
    w_path = Path(weights_path) if weights_path is not None else (DEFAULT_WEIGHTS_DIR / f"patchcore_{category}.pt")
    if engine is not None:
        if engine.faiss_index is None or engine.ort_session is None:
            engine.setup_onnx_faiss(onnx_path=engine.onnx_path, num_threads=engine.num_threads)
        return engine, w_path

    if not w_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found for category '{category}': {w_path}")

    mtime = w_path.stat().st_mtime
    cache_key = str(w_path.resolve())
    cached = _STANDALONE_ENGINE_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1], w_path

    loaded = PatchCoreInferenceEngine.load(w_path, device="cpu", engine_type="onnx_faiss", num_threads=4)
    _STANDALONE_ENGINE_CACHE[cache_key] = (mtime, loaded)
    return loaded, w_path


def incorporate_normal_sample(
    image_path_or_array: Union[str, Path, Image.Image, np.ndarray],
    category: str,
    *,
    engine: Optional[PatchCoreInferenceEngine] = None,
    weights_path: Optional[Union[str, Path]] = None,
    additional_weights_paths: Optional[List[Union[str, Path]]] = None,
    feedback_root: Union[str, Path] = DEFAULT_FEEDBACK_ROOT,
    audit_log_path: Union[str, Path] = DEFAULT_AUDIT_LOG_PATH,
    reason: str = "Variação normal de acabamento",
    threshold: Optional[float] = None,
    previous_score: Optional[float] = None,
    subsampling_ratio: float = 0.12,
    min_vectors: int = 32,
    max_vectors: int = 196,
    reuse_cached_features: bool = True,
    persist_disk: bool = True,
    preserve_mtime: bool = False,
    log_audit: bool = True,
) -> Dict[str, Any]:
    """Incorporate an operator-approved sample into the PatchCore Memory Bank & FAISS index.

    Steps executed:
    1. Extract patch features using the ONNX backbone extractor (or reuse cached
       features from the current inspection frame when identical).
    2. Select the most representative / novel patches via local subsampling.
    3. Incrementally add the new vectors to the active in-memory FAISS index
       (``faiss_index.add(new_vectors)``) and update ``memory_bank``.
    4. Re-score the sample immediately to verify the anomaly score dropped
       below the operational tolerance threshold.
    5. Atomically update ``weights/patchcore_<category>.pt`` on disk.
    6. Archive the approved image under ``data/feedback/<category>/``.
    7. Append an audit record to ``logs/contested_samples.csv``.
    """
    t_total_start = time.perf_counter()

    active_engine, primary_weights_path = _resolve_engine_for_category(
        category=category, engine=engine, weights_path=weights_path
    )
    cfg = active_engine.config

    # 1. Extract patch features using ONNX backbone
    t_extract_start = time.perf_counter()
    patches_np, rgb_crop = active_engine.extract_patch_vectors_onnx(
        image_path_or_array, reuse_cached=reuse_cached_features
    )
    feature_extraction_ms = (time.perf_counter() - t_extract_start) * 1000.0

    # Compute previous score if not provided
    if previous_score is None:
        _, prev_score_val = active_engine._nearest_and_score_faiss(patches_np)
    else:
        prev_score_val = float(previous_score)

    effective_threshold = float(
        threshold
        if threshold is not None
        else active_engine.calibration.get(
            "f1_threshold",
            active_engine.calibration.get("image_threshold", prev_score_val * 0.75),
        )
    )

    # 2 & 3. Local representative subsampling + Incremental FAISS index update
    t_faiss_start = time.perf_counter()
    bank_size_before = int(active_engine.faiss_index.ntotal)

    new_vectors, selected_idx = select_representative_patches(
        patches_np=patches_np,
        faiss_index=active_engine.faiss_index,
        target_threshold=effective_threshold,
        subsampling_ratio=subsampling_ratio,
        min_vectors=min_vectors,
        max_vectors=max_vectors,
    )

    # Add directly to active FAISS index in memory
    active_engine.faiss_index.add(new_vectors)

    # Keep memory_bank_np and torch memory_bank synchronized
    if active_engine.memory_bank_np is not None:
        active_engine.memory_bank_np = np.concatenate([active_engine.memory_bank_np, new_vectors], axis=0)
    else:
        active_engine.memory_bank_np = new_vectors.copy()

    new_tensor = torch.from_numpy(new_vectors).to(active_engine.device)
    if active_engine.memory_bank is not None:
        active_engine.memory_bank = torch.cat([active_engine.memory_bank, new_tensor], dim=0)
    else:
        active_engine.memory_bank = new_tensor

    bank_size_after = int(active_engine.faiss_index.ntotal)
    added_count = int(new_vectors.shape[0])

    active_engine.metadata["memory_bank_size"] = bank_size_after
    active_engine.metadata["active_learning_updates"] = (
        int(active_engine.metadata.get("active_learning_updates", 0)) + 1
    )
    active_engine.metadata["active_learning_vectors_added"] = (
        int(active_engine.metadata.get("active_learning_vectors_added", 0)) + added_count
    )

    # Re-evaluate score on the updated FAISS index
    patch_scores_after, new_score_val = active_engine._nearest_and_score_faiss(patches_np)
    new_amap = active_engine._anomaly_map(patch_scores_after, (cfg.crop, cfg.crop))

    faiss_update_ms = (time.perf_counter() - t_faiss_start) * 1000.0

    # 4. Archive approved sample in data/feedback/<category>/
    feedback_img_path = save_feedback_image(
        image_path_or_array=image_path_or_array,
        category=category,
        feedback_root=feedback_root,
    )

    # 5. Atomic checkpoint persistence on disk
    saved_paths: List[str] = []
    if persist_disk:
        canonical_path = DEFAULT_WEIGHTS_DIR / f"patchcore_{category}.pt"
        targets = [primary_weights_path]
        if canonical_path.resolve() != primary_weights_path.resolve() and canonical_path.parent.exists():
            # Only mirror to canonical path if its backbone matches the active engine
            if not canonical_path.exists():
                targets.append(canonical_path)
            else:
                try:
                    ckpt_meta = torch.load(canonical_path, map_location="cpu", weights_only=False)
                    if ckpt_meta.get("config", {}).get("backbone") == cfg.backbone:
                        targets.append(canonical_path)
                except Exception:
                    pass

        if additional_weights_paths:
            for extra_p in additional_weights_paths:
                ep = Path(extra_p)
                if all(ep.resolve() != t.resolve() for t in targets):
                    targets.append(ep)

        for t_path in targets:
            atomic_save_engine_checkpoint(active_engine, t_path, preserve_mtime=preserve_mtime)
            saved_paths.append(str(t_path))
            cache_key = str(t_path.resolve())
            if cache_key in _STANDALONE_ENGINE_CACHE and t_path.exists():
                _STANDALONE_ENGINE_CACHE[cache_key] = (t_path.stat().st_mtime, active_engine)

    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    # 6. Append audit log entry to logs/contested_samples.csv
    audit_entry: Optional[Dict[str, Any]] = None
    if log_audit:
        audit_entry = log_contested_sample(
            category=category,
            previous_score=prev_score_val,
            new_score=new_score_val,
            added_vectors_count=added_count,
            memory_bank_size_after=bank_size_after,
            faiss_update_ms=faiss_update_ms,
            reason=reason,
            feedback_image_path=feedback_img_path,
            log_path=audit_log_path,
        )

    return {
        "category": category,
        "previous_score": round(prev_score_val, 4),
        "new_score": round(new_score_val, 4),
        "threshold": round(effective_threshold, 4),
        "is_approved_now": bool(new_score_val < effective_threshold),
        "added_vectors_count": added_count,
        "selected_patch_indices": selected_idx.tolist(),
        "memory_bank_size_before": bank_size_before,
        "memory_bank_size_after": bank_size_after,
        "feature_extraction_ms": round(feature_extraction_ms, 2),
        "faiss_update_ms": round(faiss_update_ms, 2),
        "total_ms": round(total_ms, 2),
        "feedback_image_path": str(feedback_img_path),
        "saved_weights_paths": saved_paths,
        "audit_log_path": str(audit_log_path),
        "audit_entry": audit_entry,
        "reason": reason,
        "updated_anomaly_map": new_amap,
        "image_rgb": rgb_crop,
    }

