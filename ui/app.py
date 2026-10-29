"""Industrial Anomaly Detection - Streamlit Inspection & Calibration UI.

Interactive operational dashboard for Quality Assurance engineers to:
1. Load real trained PatchCore models (from weights/patchcore_<category>.pt).
2. Inspect real manufactured parts from MVTec AD test set or upload new images.
3. Perform real-time inference with thermal heatmap overlay and bounding boxes.
4. Calibrate operational decision thresholds based on industrial asymmetric costs.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path for direct invocation
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.dataset.mvtec import IMAGE_EXTENSIONS, MVTEC_CATEGORIES, load_mask
from src.models.patchcore_wrapper import (
    PatchCoreInferenceEngine,
    load_rgb,
    preprocess_mask,
    resize_center_crop,
)
from src.postprocessing.postprocessor import AnomalyPostProcessor
from src.utils.metrics import (
    find_cost_sensitive_threshold,
    find_f1_optimal_threshold,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Anomaly Detection (QA 4.0)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏭 Industrial Anomaly Detection - Quality Assurance Console")
st.caption(
    "Inspeção Visual e Detecção de Falhas Não Supervisionada em Linhas de Produção | MVTec AD Pilot"
)


# -----------------------------------------------------------------------------
# Cached Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(weights_path: str) -> Optional[PatchCoreInferenceEngine]:
    p = Path(weights_path)
    if not p.exists():
        return None
    return PatchCoreInferenceEngine.load(p, device="cpu")


def load_benchmark_report(category: str) -> Optional[dict]:
    p = Path(f"docs/benchmark_{category}.json")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configurações de Inspeção")

category = st.sidebar.selectbox(
    "Classe MVTec AD:",
    options=MVTEC_CATEGORIES,
    index=MVTEC_CATEGORIES.index("bottle") if "bottle" in MVTEC_CATEGORIES else 0,
    help="Selecione uma das 15 classes industriais do benchmark.",
)

weights_file = f"weights/patchcore_{category}.pt"
model = load_trained_model(weights_file)
benchmark = load_benchmark_report(category)

if model is not None:
    calib = model.calibration
    default_th = float(calib.get("image_threshold", 0.45))
    st.sidebar.success(f"✅ Modelo Carregado: `{weights_file}`")
    st.sidebar.caption(f"Backbone: **{model.config.backbone}** | Coreset: **{model.memory_bank.shape[0]} vetores**")
else:
    default_th = 0.45
    st.sidebar.warning(f"⚠️ Modelo `{weights_file}` não encontrado.")
    st.sidebar.caption("Execute `python scripts/train.py` para gerar os pesos.")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Calibração de Threshold")

threshold = st.sidebar.slider(
    "Limiar Operacional (Threshold τ):",
    min_value=0.05,
    max_value=2.50,
    value=min(default_th, 2.50),
    step=0.01,
    help="Valores de anomalia acima deste limiar rejeitam a peça.",
)

use_hysteresis = st.sidebar.checkbox(
    "Ativar Histerese (Dual Threshold)",
    value=True,
    help="Reduz ruído espúrio exigindo que anomalias fracas estejam conectadas a uma anomalia forte.",
)

if use_hysteresis:
    hyst_low = st.sidebar.slider(
        "Histerese Low Threshold:",
        min_value=0.01,
        max_value=threshold,
        value=max(0.01, round(threshold * 0.70, 2)),
        step=0.01,
    )
else:
    hyst_low = threshold

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Matriz de Custo Industrial")
cost_fn = st.sidebar.number_input(
    "Custo de Falso Negativo (C_FN - Escape de peça defeituosa):",
    min_value=10.0,
    max_value=100000.0,
    value=10000.0,
    step=100.0,
)
cost_fp = st.sidebar.number_input(
    "Custo de Falso Positivo (C_FP - Reinspeção/Descarte):",
    min_value=0.1,
    max_value=500.0,
    value=1.0,
    step=0.5,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Filtros Morfológicos")
blur_sigma = st.sidebar.slider("Filtro Gaussiano (Sigma):", 0.0, 10.0, 4.0, 0.5)
opening_k = st.sidebar.slider("Abertura Morfológica (Kernel):", 1, 9, 3, 2)
closing_k = st.sidebar.slider("Fechamento Morfológico (Kernel):", 1, 11, 5, 2)

# -----------------------------------------------------------------------------
# Top Metric KPI Cards (Real measured from benchmark if available)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    if benchmark:
        val = f"{benchmark['metrics']['image_auroc'] * 100:.1f}%"
        delta = "Real Medido"
    else:
        val = "≥ 99.0%"
        delta = "Meta SLA"
    st.metric(label="Image-AUROC", value=val, delta=delta)

with col2:
    if benchmark:
        val = f"{benchmark['metrics']['pixel_auroc'] * 100:.1f}%"
        delta = "Real Medido"
    else:
        val = "≥ 97.5%"
        delta = "Meta SLA"
    st.metric(label="Pixel-AUROC", value=val, delta=delta)

with col3:
    if benchmark:
        val = f"{benchmark['metrics']['pro_metric_fpr_030'] * 100:.1f}%"
        delta = "FPR ≤ 0.30"
    else:
        val = "≥ 93.5%"
        delta = "Meta SLA"
    st.metric(label="PRO Metric", value=val, delta=delta)

with col4:
    if benchmark:
        val = f"{benchmark['latency']['mean_ms']:.1f} ms"
        delta = f"P95: {benchmark['latency']['p95_ms']:.1f} ms"
    else:
        val = "< 50 ms"
        delta = "Meta Edge"
    st.metric(label="Latência Média", value=val, delta=delta)

st.markdown("---")

# -----------------------------------------------------------------------------
# Image Selection (Real MVTec Dataset or Upload)
# -----------------------------------------------------------------------------
st.subheader(f"🔍 Inspeção de Peças: `{category}`")

data_dir = Path("data/mvtec") / category / "test"
tab1, tab2 = st.tabs(["📁 Dataset MVTec AD (Amostras Reais)", "📤 Upload de Nova Imagem"])

selected_img_path: Optional[Path] = None
uploaded_image: Optional[Image.Image] = None
gt_mask_path: Optional[Path] = None
selected_defect_type = "unknown"

with tab1:
    if data_dir.exists():
        defect_folders = sorted(p for p in data_dir.iterdir() if p.is_dir())
        options_map = {}
        for folder in defect_folders:
            imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            for img in imgs:
                label_display = f"[{folder.name.upper()}] - {img.name}"
                options_map[label_display] = (img, folder.name)

        if options_map:
            chosen_key = st.selectbox("Selecione uma peça do lote de teste:", list(options_map.keys()))
            selected_img_path, selected_defect_type = options_map[chosen_key]
            # Check for ground truth mask
            expected_mask = Path("data/mvtec") / category / "ground_truth" / selected_defect_type / f"{selected_img_path.stem}_mask.png"
            if expected_mask.exists():
                gt_mask_path = expected_mask
        else:
            st.info("Nenhuma imagem encontrada na pasta de teste.")
    else:
        st.warning(f"Pasta `{data_dir}` não encontrada. Baixe o dataset usando `python scripts/download_kaggle.py`.")

with tab2:
    file_up = st.file_uploader("Carregue uma imagem de peça para inspecionar:", type=["png", "jpg", "jpeg", "bmp"])
    if file_up is not None:
        uploaded_image = Image.open(file_up).convert("RGB")
        selected_defect_type = "uploaded_inspection"

# -----------------------------------------------------------------------------
# Inference Execution & Visualization
# -----------------------------------------------------------------------------
active_image = uploaded_image if uploaded_image is not None else selected_img_path

if active_image is not None and model is not None:
    # 1. Run real inference
    t_start = time.perf_counter()
    pred_res = model.predict(active_image)
    inference_ms = (time.perf_counter() - t_start) * 1000.0

    raw_score = pred_res["anomaly_score"]
    raw_map = pred_res["anomaly_map"]
    img_rgb = pred_res["image_rgb"]

    # 2. Post-processing
    post_processor = AnomalyPostProcessor(
        blur_sigma=blur_sigma,
        opening_kernel_size=opening_k,
        closing_kernel_size=closing_k,
    )
    norm_map = model.normalize_map(raw_map)

    if use_hysteresis:
        # Scale thresholds relative to norm bounds
        th_norm = norm_map.mean() + (threshold - model.calibration.get("image_threshold", threshold))
        th_norm = float(np.clip(threshold / (model.calibration.get("nominal_score_max", 1.0) * 1.5 + 1e-5), 0.1, 0.95))
        binary_mask = post_processor.apply_hysteresis_threshold(
            norm_map, high_threshold=th_norm, low_threshold=th_norm * 0.70
        )
    else:
        binary_mask = (norm_map >= 0.5).astype(np.uint8) * 255

    filtered_mask = post_processor.filter_morphology(binary_mask)
    defect_regions = post_processor.extract_defect_regions(filtered_mask, norm_map)
    blended_overlay = post_processor.create_heatmap_overlay(img_rgb, norm_map, alpha=0.45)

    # 3. Decision
    is_rejected = (raw_score >= threshold) or (len(defect_regions) > 0)

    # Decision Banner
    dec_col, info_col = st.columns([1, 2])
    with dec_col:
        if is_rejected:
            st.error(f"❌ DECISÃO: REJEITAR (FAIL)\n\nScore: **{raw_score:.4f}** ≥ Limiar: **{threshold:.4f}**")
        else:
            st.success(f"✅ DECISÃO: APROVADO (PASS)\n\nScore: **{raw_score:.4f}** < Limiar: **{threshold:.4f}**")

    with info_col:
        st.info(
            f"**Tipo Real**: `{selected_defect_type}` | **Defeitos Localizados**: `{len(defect_regions)}` | "
            f"**Latência Inferência**: `{inference_ms:.1f} ms`"
        )

    # 4. Visual Panels
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    with vcol1:
        st.image(img_rgb, caption="1. Peça Inspecionada (RGB)", use_container_width=True)

    with vcol2:
        st.image(norm_map, caption="2. Mapa Térmico de Anomalia", clamp=True, use_container_width=True)

    with vcol3:
        mask_annotated = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2RGB)
        for reg in defect_regions:
            x, y, w, h = reg.bbox
            cv2.rectangle(mask_annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
        st.image(mask_annotated, caption=f"3. Máscara Binarizada ({len(defect_regions)} defeitos)", use_container_width=True)

    with vcol4:
        st.image(blended_overlay, caption="4. Overlay Térmico Industrial", use_container_width=True)

    # 5. Defect table
    if defect_regions:
        st.subheader("📋 Regiões com Anomalia Detectada")
        st.dataframe(
            [
                {
                    "Defeito #": idx + 1,
                    "Área (pixels)": reg.area,
                    "Bounding Box (X, Y, W, H)": f"{reg.bbox}",
                    "Centroide": f"{reg.centroid}",
                    "Score Máximo": f"{reg.max_anomaly_score:.3f}",
                }
                for idx, reg in enumerate(defect_regions)
            ],
            use_container_width=True,
        )

elif model is None:
    st.info("ℹ️ Para executar a inspeção real, execute primeiro o treinamento do modelo:")
    st.code("python scripts/train.py --category bottle", language="bash")
