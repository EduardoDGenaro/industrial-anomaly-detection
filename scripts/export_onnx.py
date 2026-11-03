"""Export the PatchCore backbone feature extractor to ONNX and benchmark ONNX Runtime.

Creates:
- ``weights/backbone_<category>.onnx`` (FP32 ONNX graph with fused Conv+BN+ReLU and
  exact slice-based 1536->1024 channel pooling)
- Optionally ``weights/backbone_<category>_int8.onnx`` (dynamic/static INT8 quantized)

Validates numerical fidelity against the PyTorch PatchFeatureExtractor and compares
CPU forward-pass latency using ``onnxruntime`` (CPUExecutionProvider).

Usage:
    .venv\\Scripts\\python.exe scripts\\export_onnx.py --weights weights\\patchcore_bottle.pt --out weights\\backbone_bottle.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.patchcore_wrapper import PatchCoreInferenceEngine, preprocess


class OnnxPatchFeatureExtractor(nn.Module):
    """ONNX-friendly wrapper around PatchFeatureExtractor.

    Replaces ``F.adaptive_avg_pool1d`` over channels (1536 -> 1024) with its
    exact algebraic slice identity ``0.5 * (g[:, :, 0:2] + g[:, :, 1:3])``
    (max diff = 0.0) when ``c_in * 2 == c_out * 3``, or a fixed linear buffer
    otherwise, so that standard ONNX opsets export and fuse cleanly.
    """

    def __init__(self, base_extractor: nn.Module) -> None:
        super().__init__()
        self.stem = base_extractor.stem
        self.blocks = base_extractor.blocks
        self.layers = list(base_extractor.layers)
        self.last_layer = base_extractor.last_layer
        self.embedding_dim = int(base_extractor.embedding_dim)
        self.pool = base_extractor.pool
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        h = self.stem(x)
        for name, block in self.blocks.items():
            h = block(h)
            if name in self.layers:
                feats[name] = h
            if name == self.last_layer:
                break

        ref = feats[self.layers[0]]
        target_hw = (int(ref.shape[-2]), int(ref.shape[-1]))
        maps = []
        for name in self.layers:
            f = self.pool(feats[name])
            if (int(f.shape[-2]), int(f.shape[-1])) != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            maps.append(f)
        emb = torch.cat(maps, dim=1)  # (B, 1536, 28, 28)

        b, c, hh, ww = emb.shape
        if c != self.embedding_dim:
            if c == 1536 and self.embedding_dim == 1024:
                # Exact algebraic equivalent to F.adaptive_avg_pool1d(1536 -> 1024)
                g = emb.view(b, 512, 3, hh, ww)
                emb = (0.5 * (g[:, :, 0:2, :, :] + g[:, :, 1:3, :, :])).reshape(
                    b, 1024, hh, ww
                )
            else:
                emb = emb.permute(0, 2, 3, 1).reshape(-1, 1, c)
                emb = F.adaptive_avg_pool1d(emb, self.embedding_dim)
                emb = emb.reshape(b, hh, ww, self.embedding_dim).permute(0, 3, 1, 2)
        return emb.contiguous()


def create_ort_session(onnx_path: Path, num_threads: int = 4) -> ort.InferenceSession:
    """Create an optimized ONNX Runtime CPU session."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_cpu_mem_arena = True
    sess_options.enable_mem_pattern = True
    sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PatchCore backbone to ONNX.")
    parser.add_argument("--weights", default="weights/patchcore_bottle.pt")
    parser.add_argument("--out", default="weights/backbone_bottle.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 4))
    parser.add_argument("--quantize-int8", action="store_true", help="Also generate dynamic INT8 model")
    args = parser.parse_args()

    weights_p = Path(args.weights).resolve()
    print(f"[INFO] Loading PatchCore checkpoint from {weights_p}...")
    engine = PatchCoreInferenceEngine.load(weights_p, device="cpu")
    cfg = engine.config

    onnx_wrapper = OnnxPatchFeatureExtractor(engine.extractor).eval()
    dummy_input = torch.randn(1, 3, cfg.crop, cfg.crop, dtype=torch.float32)

    with torch.no_grad():
        pt_out = engine.extractor(dummy_input)
        wrapper_out = onnx_wrapper(dummy_input)
        diff_wrapper = float((pt_out - wrapper_out).abs().max().item())
        print(f"[INFO] Wrapper vs Original PatchFeatureExtractor max abs diff: {diff_wrapper:.2e}")

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Exporting ONNX model (opset={args.opset}) to {out_path}...")
    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["patch_embeddings"],
        dynamo=False,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] ONNX model verified ({size_mb:.2f} MB): {out_path}")

    # Validate numerical parity and benchmark speed in ONNX Runtime
    session = create_ort_session(out_path, num_threads=args.threads)
    ort_input = {"input": dummy_input.numpy()}
    ort_out = session.run(["patch_embeddings"], ort_input)[0]
    max_diff = float(np.max(np.abs(pt_out.numpy() - ort_out)))
    mean_diff = float(np.mean(np.abs(pt_out.numpy() - ort_out)))
    print(f"[INFO] ONNX Runtime vs PyTorch parity -> max_abs_diff={max_diff:.2e}, mean_abs_diff={mean_diff:.2e}")

    # Quick latency benchmark (warmup + 20 runs)
    for _ in range(5):
        _ = session.run(["patch_embeddings"], ort_input)
    ort_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = session.run(["patch_embeddings"], ort_input)
        ort_times.append((time.perf_counter() - t0) * 1000.0)

    torch.set_num_threads(args.threads)
    with torch.no_grad():
        for _ in range(5):
            _ = engine.extractor(dummy_input)
        pt_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = engine.extractor(dummy_input)
            pt_times.append((time.perf_counter() - t0) * 1000.0)

    pt_mean = float(np.mean(pt_times))
    ort_mean = float(np.mean(ort_times))
    speedup = pt_mean / ort_mean if ort_mean > 0 else 0.0
    print("\n" + "=" * 65)
    print(f"[BACKBONE BENCHMARK] Input: (1, 3, {cfg.crop}, {cfg.crop}) -> Output: {tuple(ort_out.shape)}")
    print("-" * 65)
    print(f"   PyTorch Eager ({args.threads} threads):   Mean={pt_mean:.2f} ms | P50={np.percentile(pt_times, 50):.2f} ms | P95={np.percentile(pt_times, 95):.2f} ms")
    print(f"   ONNX Runtime  ({args.threads} threads):   Mean={ort_mean:.2f} ms | P50={np.percentile(ort_times, 50):.2f} ms | P95={np.percentile(ort_times, 95):.2f} ms")
    print(f"   Backbone Speedup:               {speedup:.2f}x faster ({pt_mean - ort_mean:.1f} ms saved/image)")
    print("=" * 65 + "\n")

    if args.quantize_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8_path = out_path.with_name(out_path.stem + "_int8.onnx")
        print(f"[INFO] Generating dynamic INT8 quantized model: {int8_path}...")
        quantize_dynamic(
            model_input=str(out_path),
            model_output=str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        print(f"[INFO] Saved INT8 model ({int8_path.stat().st_size / (1024 * 1024):.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

