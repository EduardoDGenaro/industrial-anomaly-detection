"""PatchCore anomaly detection (Roth et al., CVPR 2022) - PyTorch & ONNX+FAISS engines.

Supports two execution backends:
- ``engine="pytorch"``: Frozen Torchvision backbone in PyTorch eager mode +
  ``torch.cdist`` nearest-neighbour search (reference baseline).
- ``engine="onnx_faiss"``: ONNX Runtime ``CPUExecutionProvider`` (with fused
  Conv+BN+ReLU and exact slice-based 1536->1024 channel pooling) +
  ``faiss.IndexFlatL2`` SIMD AVX2 nearest-neighbour search and single-vector
  Eq. 7 k-NN density re-weighting.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN_NP = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_INV_STD_NP = (1.0 / np.array(IMAGENET_STD, dtype=np.float32)).reshape(1, 1, 3)

SUPPORTED_BACKBONES = ("wide_resnet50_2", "resnet18")
SUPPORTED_ENGINES = ("pytorch", "onnx_faiss")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    layers: Tuple[str, ...] = ("layer2", "layer3")
    resize: int = 256
    crop: int = 224
    embedding_dim: int = 1024
    coreset_sampling_ratio: float = 0.10
    coreset_projection_dim: int = 128
    n_neighbors: int = 9
    blur_sigma: float = 4.0
    seed: int = 42

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatchCoreConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "layers" in known:
            known["layers"] = tuple(known["layers"])
        return cls(**known)


# -----------------------------------------------------------------------------
# Pre-processing
# -----------------------------------------------------------------------------
def load_rgb(image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Load any supported image input as a PIL RGB image."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            return im.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr[..., :3]).convert("RGB")


def resize_center_crop(img: Image.Image, resize: int, crop: int, resample=Image.BILINEAR) -> Image.Image:
    """Resize to ``resize`` then center-crop ``crop``."""
    img = img.resize((resize, resize), resample)
    off = (resize - crop) // 2
    return img.crop((off, off, off + crop, off + crop))


def preprocess_numpy(
    image: Union[str, Path, Image.Image, np.ndarray], resize: int, crop: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (contiguous float32 array 1x3xHxW, display RGB uint8 HxWx3)."""
    pil = resize_center_crop(load_rgb(image), resize, crop)
    rgb = np.asarray(pil, dtype=np.uint8)
    norm = (rgb.astype(np.float32) * (1.0 / 255.0) - _MEAN_NP) * _INV_STD_NP
    chw = np.ascontiguousarray(norm.transpose(2, 0, 1)[None, ...], dtype=np.float32)
    return chw, rgb


def preprocess(
    image: Union[str, Path, Image.Image, np.ndarray], resize: int, crop: int
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (normalised tensor 1x3xHxW, display RGB uint8 HxWx3) for the cropped view."""
    chw, rgb = preprocess_numpy(image, resize, crop)
    return torch.from_numpy(chw), rgb


def preprocess_mask(mask_path: Optional[Union[str, Path]], resize: int, crop: int) -> np.ndarray:
    """Ground-truth mask aligned with :func:`preprocess` (nearest resize + center crop)."""
    if mask_path is None or not Path(mask_path).exists():
        return np.zeros((crop, crop), dtype=np.uint8)
    with Image.open(mask_path) as m:
        m = resize_center_crop(m.convert("L"), resize, crop, resample=Image.NEAREST)
        return (np.asarray(m) > 127).astype(np.uint8)


# -----------------------------------------------------------------------------
# Feature extractor
# -----------------------------------------------------------------------------
class PatchFeatureExtractor(nn.Module):
    """Frozen backbone returning locally-aware patch embeddings (B, D, h, w)."""

    def __init__(self, backbone: str, layers: Sequence[str], embedding_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        if backbone == "wide_resnet50_2":
            weights = tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.wide_resnet50_2(weights=weights)
        elif backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Options: {SUPPORTED_BACKBONES}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleDict(
            {name: getattr(net, name) for name in ("layer1", "layer2", "layer3", "layer4")}
        )
        self.layers = list(layers)
        self.last_layer = max(self.layers, key=lambda n: int(n[-1]))
        self.embedding_dim = embedding_dim
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: Dict[str, torch.Tensor] = {}
        h = self.stem(x)
        for name, block in self.blocks.items():
            h = block(h)
            if name in self.layers:
                feats[name] = h
            if name == self.last_layer:
                break

        ref = feats[self.layers[0]]
        target_hw = ref.shape[-2:]
        maps = []
        for name in self.layers:
            f = self.pool(feats[name])
            if f.shape[-2:] != target_hw:
                f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            maps.append(f)
        emb = torch.cat(maps, dim=1)  # (B, C_total, h, w)

        b, c, hh, ww = emb.shape
        if c != self.embedding_dim:
            emb = emb.permute(0, 2, 3, 1).reshape(-1, 1, c)
            emb = F.adaptive_avg_pool1d(emb, self.embedding_dim)
            emb = emb.reshape(b, hh, ww, self.embedding_dim).permute(0, 3, 1, 2)
        return emb.contiguous()


# -----------------------------------------------------------------------------
# Coreset
# -----------------------------------------------------------------------------
def greedy_coreset_indices(
    features: torch.Tensor,
    n_select: int,
    projection_dim: int = 128,
    seed: int = 42,
    progress: bool = False,
) -> torch.Tensor:
    """Greedy k-Center (minimax facility location) selection on a random projection."""
    n = features.shape[0]
    if n_select >= n:
        return torch.arange(n)

    gen = torch.Generator().manual_seed(seed)
    if projection_dim and projection_dim < features.shape[1]:
        proj = torch.randn(features.shape[1], projection_dim, generator=gen) / math.sqrt(projection_dim)
        reduced = features @ proj
    else:
        reduced = features

    selected = torch.empty(n_select, dtype=torch.long)
    idx = int(torch.randint(0, n, (1,), generator=gen).item())
    min_d = torch.full((n,), float("inf"))
    step = max(1, n_select // 10)
    t0 = time.perf_counter()
    for i in range(n_select):
        selected[i] = idx
        d = torch.linalg.vector_norm(reduced - reduced[idx], dim=1)
        min_d = torch.minimum(min_d, d)
        idx = int(torch.argmax(min_d).item())
        if progress and (i + 1) % step == 0:
            print(f"   coreset {i + 1}/{n_select} ({time.perf_counter() - t0:.1f}s)")
    return selected


# -----------------------------------------------------------------------------
# Inference engine
# -----------------------------------------------------------------------------
class PatchCoreInferenceEngine:
    """Fit / save / load / predict PatchCore using PyTorch or ONNX+FAISS."""

    def __init__(
        self,
        config: Optional[PatchCoreConfig] = None,
        device: str = "cpu",
        pretrained: bool = True,
        engine_type: str = "pytorch",
        onnx_path: Optional[Union[str, Path]] = None,
        num_threads: int = 4,
    ) -> None:
        if engine_type not in SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported engine_type '{engine_type}'. Options: {SUPPORTED_ENGINES}")
        self.config = config or PatchCoreConfig()
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.engine_type = engine_type
        self.num_threads = num_threads
        self.onnx_path: Optional[Path] = Path(onnx_path) if onnx_path else None
        self.ort_session: Any = None
        self.faiss_index: Any = None
        self.memory_bank_np: Optional[np.ndarray] = None

        # Only instantiate PyTorch backbone when needed (or for training/fallback)
        self.extractor: Optional[PatchFeatureExtractor] = None
        if engine_type == "pytorch" or self.onnx_path is None or not self.onnx_path.exists():
            self.extractor = PatchFeatureExtractor(
                self.config.backbone, self.config.layers, self.config.embedding_dim, pretrained=pretrained
            ).to(self.device)

        self.memory_bank: Optional[torch.Tensor] = None
        self.feature_hw: Optional[Tuple[int, int]] = None
        self.calibration: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.last_latency_ms: float = 0.0

    # ------------------------------------------------- ONNX + FAISS setup
    def setup_onnx_faiss(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        """Initialise ONNX Runtime session and FAISS IndexFlatL2 over the memory bank."""
        import faiss
        import onnxruntime as ort

        assert self.memory_bank is not None, "Cannot build FAISS index: memory_bank is None."
        nt = num_threads or self.num_threads
        self.num_threads = nt

        # 1. Configure FAISS IndexFlatL2 with physical core count and SIMD BLAS threshold
        faiss.omp_set_num_threads(nt)
        if hasattr(faiss, "cvar") and hasattr(faiss.cvar, "distance_compute_blas_threshold"):
            faiss.cvar.distance_compute_blas_threshold = 1
        self.memory_bank_np = np.ascontiguousarray(self.memory_bank.detach().cpu().numpy(), dtype=np.float32)
        index = faiss.IndexFlatL2(int(self.memory_bank_np.shape[1]))
        index.add(self.memory_bank_np)
        self.faiss_index = index

        # 2. Configure ONNX Runtime session
        if onnx_path is not None:
            self.onnx_path = Path(onnx_path)
        if self.onnx_path is None or not self.onnx_path.exists():
            cat = self.calibration.get("category") or self.metadata.get("category") or "bottle"
            default_onnx = Path(__file__).resolve().parent.parent / "weights" / f"backbone_{cat}.onnx"
            if default_onnx.exists():
                self.onnx_path = default_onnx

        if self.onnx_path is not None and self.onnx_path.exists():
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = nt
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.ort_session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        self.engine_type = "onnx_faiss"

    # ---------------------------------------------------------------- features
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        if self.extractor is None:
            self.extractor = PatchFeatureExtractor(
                self.config.backbone, self.config.layers, self.config.embedding_dim, pretrained=True
            ).to(self.device)
        emb = self.extractor(batch.to(self.device))
        self.feature_hw = (int(emb.shape[-2]), int(emb.shape[-1]))
        return emb

    def embed_onnx(self, batch_np: np.ndarray) -> np.ndarray:
        """Run backbone feature extraction in ONNX Runtime returning (B, D, h, w) numpy array."""
        assert self.ort_session is not None, "ONNX Runtime session is not initialized."
        emb = self.ort_session.run(["patch_embeddings"], {"input": batch_np})[0]
        self.feature_hw = (int(emb.shape[-2]), int(emb.shape[-1]))
        return emb

    # --------------------------------------------------------------------- fit
    def fit(self, image_paths: Sequence[Union[str, Path]], batch_size: int = 8, verbose: bool = True) -> Dict[str, Any]:
        """Extract nominal patch features and build the coreset memory bank."""
        cfg = self.config
        torch.manual_seed(cfg.seed)
        t0 = time.perf_counter()
        chunks: List[torch.Tensor] = []
        for start in range(0, len(image_paths), batch_size):
            batch = torch.cat(
                [preprocess(p, cfg.resize, cfg.crop)[0] for p in image_paths[start : start + batch_size]]
            )
            emb = self.embed(batch)  # (B, D, h, w)
            chunks.append(emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1]).cpu())
            if verbose:
                print(f"   features {min(start + batch_size, len(image_paths))}/{len(image_paths)}")
        features = torch.cat(chunks)
        t_feat = time.perf_counter() - t0

        n_select = max(1, int(features.shape[0] * cfg.coreset_sampling_ratio))
        if verbose:
            print(f"[INFO] {features.shape[0]} patches x {features.shape[1]} dims -> coreset {n_select}")
        t1 = time.perf_counter()
        idx = greedy_coreset_indices(features, n_select, cfg.coreset_projection_dim, cfg.seed, progress=verbose)
        self.memory_bank = features[idx].contiguous().to(self.device)
        t_core = time.perf_counter() - t1

        stats = {
            "num_train_images": len(image_paths),
            "num_patches": int(features.shape[0]),
            "memory_bank_size": int(n_select),
            "embedding_dim": int(features.shape[1]),
            "feature_extraction_s": round(t_feat, 2),
            "coreset_selection_s": round(t_core, 2),
        }
        self.metadata.update(stats)
        return stats

    # ----------------------------------------------------------------- scoring
    def _nearest(self, patches: torch.Tensor, chunk: int = 2048) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.memory_bank is not None, "Model not fitted / loaded."
        dists, idxs = [], []
        for s in range(0, patches.shape[0], chunk):
            d = torch.cdist(patches[s : s + chunk], self.memory_bank)
            v, i = d.min(dim=1)
            dists.append(v)
            idxs.append(i)
        return torch.cat(dists), torch.cat(idxs)

    def _image_score(self, patches: torch.Tensor, patch_scores: torch.Tensor, nn_idx: torch.Tensor) -> float:
        """Paper Eq. 7: re-weight the max patch score by the local density of its NN."""
        k = self.config.n_neighbors
        star = int(torch.argmax(patch_scores))
        s_star = patch_scores[star]
        if k <= 1 or self.memory_bank.shape[0] <= k:
            return float(s_star)
        m_star = self.memory_bank[nn_idx[star]].unsqueeze(0)
        _, nb_idx = torch.cdist(m_star, self.memory_bank).topk(k, largest=False)
        nb = self.memory_bank[nb_idx[0]]
        d_nb = torch.cdist(patches[star].unsqueeze(0), nb)[0]  # first element == s_star
        w = 1.0 - torch.softmax(d_nb, dim=0)[0]
        return float(w * s_star)

    def _nearest_and_score_faiss(self, patches_np: np.ndarray) -> Tuple[np.ndarray, float]:
        """Single-pass FAISS IndexFlatL2 nearest neighbour search + Eq. 7 re-weighting."""
        assert self.faiss_index is not None
        k = max(1, int(self.config.n_neighbors))
        d2, _ = self.faiss_index.search(patches_np, k)
        patch_scores = np.sqrt(np.maximum(d2[:, 0], 0.0)).astype(np.float32)

        star = int(np.argmax(patch_scores))
        s_star = float(patch_scores[star])
        if k <= 1:
            return patch_scores, s_star

        d_nb = np.sqrt(np.maximum(d2[star], 0.0))
        exp_d = np.exp(d_nb - np.max(d_nb))
        w = 1.0 - float(exp_d[0] / np.sum(exp_d))
        return patch_scores, float(w * s_star)

    def _anomaly_map(
        self, patch_scores: Union[torch.Tensor, np.ndarray], out_hw: Tuple[int, int]
    ) -> np.ndarray:
        h, w = self.feature_hw
        if isinstance(patch_scores, np.ndarray):
            grid = torch.from_numpy(patch_scores).reshape(1, 1, h, w)
        else:
            grid = patch_scores.reshape(1, 1, h, w)
        up = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        sigma = self.config.blur_sigma
        if sigma > 0:
            import cv2

            ksize = 2 * int(4.0 * sigma + 0.5) + 1
            up = cv2.GaussianBlur(up.astype(np.float32), (ksize, ksize), sigma)
        return up.astype(np.float32)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Full inference on one image (pre-processing included in the latency)."""
        t0 = time.perf_counter()
        cfg = self.config

        if self.engine_type == "onnx_faiss" and self.faiss_index is not None:
            batch_np, rgb = preprocess_numpy(image, cfg.resize, cfg.crop)
            if self.ort_session is not None:
                emb_np = self.embed_onnx(batch_np)
            else:
                emb_np = self.embed(torch.from_numpy(batch_np)).cpu().numpy()
            patches_np = np.ascontiguousarray(
                emb_np[0].transpose(1, 2, 0).reshape(-1, emb_np.shape[1]),
                dtype=np.float32,
            )
            patch_scores_np, score = self._nearest_and_score_faiss(patches_np)
            amap = self._anomaly_map(patch_scores_np, (cfg.crop, cfg.crop))
        else:
            tensor, rgb = preprocess(image, cfg.resize, cfg.crop)
            emb = self.embed(tensor)
            patches = emb[0].permute(1, 2, 0).reshape(-1, emb.shape[1])
            patch_scores, nn_idx = self._nearest(patches)
            score = self._image_score(patches, patch_scores, nn_idx)
            amap = self._anomaly_map(patch_scores, (cfg.crop, cfg.crop))

        latency = (time.perf_counter() - t0) * 1000.0
        self.last_latency_ms = latency

        thr = self.calibration.get("image_threshold")
        return {
            "anomaly_score": score,
            "anomaly_map": amap,
            "image_rgb": rgb,
            "is_anomalous": (score >= thr) if thr is not None else None,
            "latency_ms": latency,
            "engine_type": self.engine_type,
        }

    def normalize_map(self, amap: np.ndarray) -> np.ndarray:
        """Normalise a raw anomaly map to [0, 1] with the calibrated bounds."""
        lo = self.calibration.get("map_min", float(amap.min()))
        hi = self.calibration.get("map_max", float(amap.max()))
        if hi <= lo:
            return np.zeros_like(amap)
        return np.clip((amap - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> Path:
        assert self.memory_bank is not None, "Nothing to save: model not fitted."
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "patchcore-v1",
                "config": asdict(self.config),
                "memory_bank": self.memory_bank.cpu(),
                "feature_hw": self.feature_hw,
                "calibration": self.calibration,
                "metadata": self.metadata,
                "torch_version": torch.__version__,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        engine_type: str = "pytorch",
        onnx_path: Optional[Union[str, Path]] = None,
        num_threads: int = 4,
    ) -> "PatchCoreInferenceEngine":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        if ckpt.get("format") != "patchcore-v1":
            raise ValueError(f"Unsupported checkpoint format in {path}")
        cfg = PatchCoreConfig.from_dict(ckpt["config"])
        calib = dict(ckpt.get("calibration") or {})
        meta = dict(ckpt.get("metadata") or {})

        # Resolve default ONNX path if engine_type == 'onnx_faiss'
        resolved_onnx = Path(onnx_path) if onnx_path else None
        if engine_type == "onnx_faiss" and resolved_onnx is None:
            cat = calib.get("category") or meta.get("category") or "bottle"
            candidate = Path(path).resolve().parent / f"backbone_{cat}.onnx"
            if candidate.exists():
                resolved_onnx = candidate

        engine = cls(
            cfg,
            device=device,
            engine_type=engine_type,
            onnx_path=resolved_onnx,
            num_threads=num_threads,
        )
        engine.memory_bank = ckpt["memory_bank"].to(engine.device)
        engine.feature_hw = tuple(ckpt["feature_hw"]) if ckpt.get("feature_hw") else None
        engine.calibration = calib
        engine.metadata = meta

        if engine_type == "onnx_faiss":
            engine.setup_onnx_faiss(onnx_path=resolved_onnx, num_threads=num_threads)

        return engine


def load_config_yaml(path: Union[str, Path]) -> Tuple[PatchCoreConfig, Dict[str, Any]]:
    """Read ``configs/patchcore_mvtec.yaml`` -> (PatchCoreConfig, raw dict)."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    m = raw.get("model", {})
    d = raw.get("dataset", {})
    p = raw.get("postprocessing", {})
    cfg = PatchCoreConfig(
        backbone=m.get("backbone", "wide_resnet50_2"),
        layers=tuple(m.get("layers", ("layer2", "layer3"))),
        resize=int(d.get("resize", 256)),
        crop=int(d.get("crop", 224)),
        embedding_dim=int(m.get("embedding_dim", 1024)),
        coreset_sampling_ratio=float(m.get("coreset_sampling_ratio", 0.10)),
        coreset_projection_dim=int(m.get("coreset_projection_dim", 128)),
        n_neighbors=int(m.get("n_neighbors", 9)),
        blur_sigma=float(p.get("blur_sigma", 4.0)),
        seed=int(raw.get("project", {}).get("seed", 42)),
    )
    return cfg, raw
