"""Unit and integration tests for Human-in-the-Loop Active Learning & Audit Logger."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from src.models.active_learning import (
    incorporate_normal_sample,
    select_representative_patches,
)
from src.models.patchcore_wrapper import PatchCoreInferenceEngine
from src.utils.audit_logger import log_contested_sample, read_contested_samples_log


def test_audit_logger_roundtrip(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "contested_samples.csv"
    entry = log_contested_sample(
        category="metal_nut",
        previous_score=2.4512,
        new_score=1.1823,
        added_vectors_count=48,
        memory_bank_size_after=1772,
        faiss_update_ms=1.84,
        reason="Variação normal de acabamento",
        feedback_image_path="data/feedback/metal_nut/sample.png",
        log_path=log_file,
    )
    assert log_file.is_file()
    assert entry["category"] == "metal_nut"
    assert entry["added_vectors_count"] == 48

    rows = read_contested_samples_log(log_file)
    assert len(rows) == 1
    assert rows[0]["category"] == "metal_nut"
    assert float(rows[0]["previous_score"]) == pytest.approx(2.4512)
    assert float(rows[0]["new_score"]) == pytest.approx(1.1823)
    assert int(rows[0]["added_vectors_count"]) == 48


def test_select_representative_patches_reduces_max_distance() -> None:
    import faiss

    rng = np.random.default_rng(42)
    bank = rng.normal(0.0, 0.2, size=(500, 64)).astype(np.float32)
    index = faiss.IndexFlatL2(64)
    index.add(bank)

    # Simulate a 784-patch image: 744 normal patches near the bank + 40 patches with batch variation
    base_indices = rng.integers(0, 500, size=784)
    patches = (bank[base_indices] + rng.normal(0.0, 0.03, size=(784, 64))).astype(np.float32)
    patches[:40] += 0.60

    d2_before, _ = index.search(patches, 1)
    max_dist_before = float(np.sqrt(np.max(d2_before[:, 0])))
    assert max_dist_before > 1.50

    selected_vecs, selected_idx = select_representative_patches(
        patches_np=patches,
        faiss_index=index,
        target_threshold=1.50,
        subsampling_ratio=0.10,
        min_vectors=32,
        max_vectors=160,
    )
    assert 32 <= selected_vecs.shape[0] <= 160
    assert selected_vecs.shape[1] == 64
    assert len(selected_idx) == selected_vecs.shape[0]

    index.add(selected_vecs)
    d2_after, _ = index.search(patches, 1)
    max_dist_after = float(np.sqrt(np.max(d2_after[:, 0])))
    assert max_dist_after < 1.50 * 0.65


def test_incorporate_normal_sample_e2e_and_sla(tmp_path: Path) -> None:
    src_weights = Path("weights/patchcore_bottle_resnet18_1pct.pt")
    src_onnx = Path("weights/backbone_bottle_resnet18.onnx")
    sample_img = Path("data/mvtec/bottle/test/broken_large/000.png")

    if not (src_weights.is_file() and src_onnx.is_file() and sample_img.is_file()):
        pytest.skip("Bottle weights or sample image not available.")

    tmp_weights = tmp_path / "patchcore_bottle.pt"
    shutil.copy2(src_weights, tmp_weights)
    shutil.copy2(src_onnx, tmp_path / "backbone_bottle_resnet18.onnx")

    engine = PatchCoreInferenceEngine.load(
        tmp_weights,
        device="cpu",
        engine_type="onnx_faiss",
        onnx_path=tmp_path / "backbone_bottle_resnet18.onnx",
        num_threads=4,
    )

    pred_before = engine.predict(sample_img)
    score_before = float(pred_before["anomaly_score"])
    threshold = float(engine.calibration.get("f1_threshold", 1.72))
    assert score_before >= threshold

    feedback_dir = tmp_path / "feedback"
    audit_csv = tmp_path / "logs" / "contested_samples.csv"

    res = incorporate_normal_sample(
        image_path_or_array=sample_img,
        category="bottle",
        engine=engine,
        weights_path=tmp_weights,
        feedback_root=feedback_dir,
        audit_log_path=audit_csv,
        reason="Variação normal de lote do fornecedor",
        threshold=threshold,
        previous_score=score_before,
    )

    assert res["faiss_update_ms"] < 150.0
    assert res["new_score"] < threshold
    assert res["is_approved_now"] is True
    assert res["added_vectors_count"] >= 32
    assert res["memory_bank_size_after"] == res["memory_bank_size_before"] + res["added_vectors_count"]
    assert Path(res["feedback_image_path"]).is_file()
    assert audit_csv.is_file()

    # Verify re-running predict() on the engine returns the updated approved score
    pred_after = engine.predict(sample_img)
    assert float(pred_after["anomaly_score"]) < threshold

