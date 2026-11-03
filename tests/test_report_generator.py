"""Unit tests for ISO 9001 Non-Conformance Report (RNC) PDF Generator."""

from __future__ import annotations

import io
from pathlib import Path
import time

import numpy as np

from src.postprocessing.postprocessor import DefectRegion
from src.utils.report_generator import DefectReportGenerator, compute_image_sha256


def _sample_inspection_payload() -> dict:
    rng = np.random.default_rng(7)
    orig = rng.integers(40, 220, size=(224, 224, 3), dtype=np.uint8)
    heat = rng.integers(40, 240, size=(224, 224, 3), dtype=np.uint8)
    demarc = rng.integers(40, 240, size=(224, 224, 3), dtype=np.uint8)

    contour = np.array([[[80, 90]], [[120, 90]], [[120, 130]], [[80, 130]]], dtype=np.int32)
    region = DefectRegion(
        bbox=(80, 90, 40, 40),
        area=1600.0,
        contour=contour,
        centroid=(100, 110),
        max_anomaly_score=0.96,
    )

    return {
        "category": "screw",
        "category_label": "Parafuso (screw)",
        "piece_id": "RNC-SCREW-001",
        "timestamp": "2026-10-30T19:30:00Z",
        "station_id": "EST-OPT-01 (Célula PatchCore)",
        "operator": "OP-402 / Turno A",
        "verdict": "REJEITADA",
        "anomaly_score": 1.9301,
        "threshold": 1.6300,
        "pixel_threshold": 1.4820,
        "defect_count": 1,
        "defect_condition": "Ponta Danificada (manipulated_front)",
        "ai_latency_ms": 46.2,
        "total_latency_ms": 98.4,
        "backbone": "ResNet-18",
        "engine_type": "ONNX Runtime + FAISS IndexFlatL2",
        "original_rgb": orig,
        "heatmap_overlay_rgb": heat,
        "demarcated_rgb": demarc,
        "defect_regions": [region],
        "sha256_hash": compute_image_sha256(orig),
    }


def test_generate_pdf_in_memory_and_latency_under_300ms() -> None:
    gen = DefectReportGenerator()
    payload = _sample_inspection_payload()

    # Warm-up
    _ = gen.generate_pdf(payload)

    t0 = time.perf_counter()
    buf = io.BytesIO()
    pdf_bytes = gen.generate_pdf(payload, output_path=buf)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 300.0, f"PDF compilation took {elapsed_ms:.1f} ms (> 300 ms limit)"
    assert pdf_bytes.startswith(b"%PDF-1.")
    assert len(pdf_bytes) > 3000
    assert buf.getvalue() == pdf_bytes


def test_generate_pdf_writes_to_file_when_path_given(tmp_path: Path) -> None:
    gen = DefectReportGenerator()
    payload = _sample_inspection_payload()
    out_file = tmp_path / "laudo_rnc_screw_001.pdf"

    pdf_bytes = gen.generate_pdf(payload, output_path=out_file)
    assert out_file.is_file()
    assert out_file.read_bytes() == pdf_bytes
    assert pdf_bytes.startswith(b"%PDF-1.")

