"""PatchCore anomaly detection (Roth et al., CVPR 2022) - self-contained PyTorch implementation.

Pipeline:
1. Frozen ImageNet backbone (WideResNet-50-2 / ResNet-18), taps at ``layer2`` and ``layer3``.
2. Local neighbourhood aggregation (3x3 avg pool), ``layer3`` upsampled to ``layer2`` resolution,
   channel concatenation, then adaptive average pooling over channels to ``embedding_dim``.
3. Greedy k-Center coreset subsampling (on a Johnson-Lindenstrauss random projection) builds
   a compact memory bank of nominal patch embeddings.
4. Inference: nearest-neighbour distance of every test patch to the memory bank gives the
   patch anomaly map (upsampled + Gaussian blur, sigma=4). Image score = re-weighted max
   patch score (paper Eq. 7).

The artefact saved by :meth:`PatchCoreInferenceEngine.save` contains the memory bank, the
model hyper-parameters and (after evaluation) the operational calibration (thresholds and
normalisation bounds) consumed by the Streamlit UI.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_BACKBONES = ("wide_resnet50_2", "resnet18")


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
    """Resize shorter side... (MVTec images are square) to ``resize`` then center-crop ``crop``."""
    img = img.resize((resize, resize), resample)
    off = (resize - crop) // 2
    return img.crop((off, off, off + crop, off + crop))


def preprocess(
    image: Union[str, Path, Image.Image, np.ndarray], resize: int, crop: int
) -> Tuple[torch.Tensor, np.ndarray]:
    """Return (normalised tensor 1x3xHxW, display RGB uint8 HxWx3) for the cropped view."""
    pil = resize_center_crop(load_rgb(image), resize, crop)
    rgb = np.asarray(pil, dtype=np.uint8)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0), rgb


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
    """Fit / save / load / predict PatchCore on CPU or GPU."""

    def __init__(self, config: Optional[PatchCoreConfig] = None, device: str = "cpu", pretrained: bool = True) -> None:
        self.config = config or PatchCoreConfig()
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.extractor = PatchFeatureExtractor(
            self.config.backbone, self.config.layers, self.config.embedding_dim, pretrained=pretrained
        ).to(self.device)
        self.memory_bank: Optional[torch.Tensor] = None
        self.feature_hw: Optional[Tuple[int, int]] = None
        self.calibration: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.last_latency_ms: float = 0.0

    # ---------------------------------------------------------------- features
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        emb = self.extractor(batch.to(self.device))
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

    def _anomaly_map(self, patch_scores: torch.Tensor, out_hw: Tuple[int, int]) -> np.ndarray:
        h, w = self.feature_hw
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
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "PatchCoreInferenceEngine":
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        if ckpt.get("format") != "patchcore-v1":
            raise ValueError(f"Unsupported checkpoint format in {path}")
        engine = cls(PatchCoreConfig.from_dict(ckpt["config"]), device=device)
        engine.memory_bank = ckpt["memory_bank"].to(engine.device)
        engine.feature_hw = tuple(ckpt["feature_hw"]) if ckpt.get("feature_hw") else None
        engine.calibration = dict(ckpt.get("calibration") or {})
        engine.metadata = dict(ckpt.get("metadata") or {})
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
