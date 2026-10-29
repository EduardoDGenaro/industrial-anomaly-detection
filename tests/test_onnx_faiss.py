"""Unit tests for Phase 4 ONNX Runtime + FAISS IndexFlatL2 acceleration."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.patchcore_wrapper import PatchCoreInferenceEngine

WEIGHTS_PATH = Path("weights/patchcore_bottle.pt")
ONNX_PATH = Path("weights/backbone_bottle.onnx")


def test_faiss_index_flat_l2_search_parity():
    """Verify FAISS IndexFlatL2 nearest neighbour distances match torch.cdist."""
    import faiss

    rng = np.random.default_rng(42)
    bank = np.ascontiguousarray(rng.standard_normal((512, 128)), dtype=np.float32)
    queries = np.ascontiguousarray(rng.standard_normal((64, 128)), dtype=np.float32)

    index = faiss.IndexFlatL2(128)
    index.add(bank)
    d2_faiss, idx_faiss = index.search(queries, 1)
    d_faiss = np.sqrt(np.maximum(d2_faiss[:, 0], 0.0))

    d_torch = torch.cdist(torch.from_numpy(queries), torch.from_numpy(bank))
    v_torch, idx_torch = d_torch.min(dim=1)

    assert np.allclose(d_faiss, v_torch.numpy(), atol=1e-4)
    assert np.array_equal(idx_faiss[:, 0], idx_torch.numpy())


@pytest.mark.skipif(
    not (WEIGHTS_PATH.exists() and ONNX_PATH.exists()),
    reason="Trained weights or ONNX model not found on disk.",
)
def test_onnx_faiss_vs_pytorch_prediction_parity():
    """Verify onnx_faiss engine produces matching anomaly maps to pytorch engine."""
    eng_pt = PatchCoreInferenceEngine.load(WEIGHTS_PATH, engine_type="pytorch", num_threads=4)
    eng_of = PatchCoreInferenceEngine.load(
        WEIGHTS_PATH, engine_type="onnx_faiss", onnx_path=ONNX_PATH, num_threads=4
    )

    dummy_img = np.full((256, 256, 3), 180, dtype=np.uint8)
    res_pt = eng_pt.predict(dummy_img)
    res_of = eng_of.predict(dummy_img)

    assert res_of["engine_type"] == "onnx_faiss"
    assert res_of["anomaly_map"].shape == (224, 224)
    assert np.allclose(res_pt["anomaly_map"], res_of["anomaly_map"], atol=1e-3)

