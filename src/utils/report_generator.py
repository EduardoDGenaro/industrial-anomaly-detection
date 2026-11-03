"""ISO 9001 Non-Conformance Technical Report (RNC) PDF Generator.

Compiles single-page executive A4 PDF reports in memory (``io.BytesIO``) or to
disk in < 300 ms, embedding:
- Industrial header (ISO 8601 timestamp, unique piece ID, station, operator).
- Technical parameter table with highlighted REJEITADA verdict.
- Side-by-side 3-panel visual evidence (Original, 50% Heatmap Overlay,
  Exact Contour + Bounding Box demarcation).
- Anti-tampering SHA-256 image digest and signature blocks for Quality
  Inspection and Process Engineering.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fpdf import FPDF
import numpy as np
from PIL import Image


_UNICODE_REPLACEMENTS = {
    "\u2022": "-",
    "\u2192": "->",
    "\u2190": "<-",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u03c4": "tau",
    "\u03c3": "sigma",
    "\u2013": "-",
    "\u2014": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u26a0": "[!]",
    "\ufe0f": "",
}


def _safe_pdf_text(text: Any) -> str:
    """Sanitize text for FPDF core Latin-1 fonts (preserving Portuguese accents)."""
    s = str(text if text is not None else "")
    for src, dst in _UNICODE_REPLACEMENTS.items():
        s = s.replace(src, dst)
    return s.encode("latin-1", errors="replace").decode("latin-1")


def _to_pil_rgb(img_or_arr: Union[np.ndarray, Image.Image, str, Path, None]) -> Image.Image:
    """Convert numpy RGB array, PIL Image, or path into an in-memory 224x224 RGB PIL Image."""
    if img_or_arr is None:
        return Image.new("RGB", (224, 224), color=(30, 41, 59))
    if isinstance(img_or_arr, Image.Image):
        return img_or_arr.convert("RGB")
    if isinstance(img_or_arr, (str, Path)):
        with Image.open(img_or_arr) as im:
            return im.convert("RGB")
    arr = np.asarray(img_or_arr)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr[..., :3]).convert("RGB")


def compute_image_sha256(
    image_source: Union[np.ndarray, Image.Image, str, Path, bytes, None],
) -> str:
    """Compute anti-tampering SHA-256 hex digest over the inspected piece image."""
    if image_source is None:
        return hashlib.sha256(b"empty").hexdigest()
    if isinstance(image_source, (bytes, bytearray)):
        return hashlib.sha256(image_source).hexdigest()
    if isinstance(image_source, (str, Path)) and Path(image_source).is_file():
        return hashlib.sha256(Path(image_source).read_bytes()).hexdigest()
    if isinstance(image_source, Image.Image):
        arr = np.asarray(image_source.convert("RGB"), dtype=np.uint8)
        return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    arr = np.ascontiguousarray(np.asarray(image_source, dtype=np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


class DefectReportGenerator:
    """Executive ISO 9001 Non-Conformance Report (RNC) generator in PDF format."""

    def __init__(
        self,
        station_id: str = "EST-OPT-01 (PatchCore ONNX/FAISS)",
        default_operator: str = "OP-402 / Turno A - Linha Automatizada",
    ) -> None:
        self.station_id = station_id
        self.default_operator = default_operator

    def generate_pdf(
        self,
        inspection_data: Dict[str, Any],
        output_path: Optional[Union[str, Path, io.BytesIO]] = None,
    ) -> bytes:
        """Compile the RNC PDF report in memory (< 300 ms) and optionally write to ``output_path``.

        Parameters
        ----------
        inspection_data : Dict[str, Any]
            Dictionary containing inspection metadata and visual arrays:
            - ``category`` / ``category_label``: piece category identifier and friendly name
            - ``piece_id``: unique sample identifier
            - ``timestamp``: ISO 8601 timestamp (auto-generated if omitted)
            - ``station_id``: optical reading station ID
            - ``operator``: logged operator identifier
            - ``verdict``: verdict text (defaults to ``"REJEITADA"``)
            - ``anomaly_score``: measured irregularity score
            - ``threshold``: operational tolerance limit
            - ``pixel_threshold``: calibrated pixel segmentation threshold
            - ``defect_count``: number of demarcated defect regions
            - ``defect_condition``: friendly defect classification
            - ``ai_latency_ms`` / ``total_latency_ms``: measured inference times
            - ``backbone`` / ``engine_type``: model architecture info
            - ``original_rgb``: (H, W, 3) uint8 original piece image
            - ``heatmap_overlay_rgb``: (H, W, 3) uint8 50% thermal overlay image
            - ``demarcated_rgb``: (H, W, 3) uint8 contour + bounding box image
            - ``defect_regions``: optional list of region dicts or ``DefectRegion`` objects
            - ``sha256_hash``: optional precomputed SHA-256 hex string
        output_path : Optional[Union[str, Path, io.BytesIO]]
            Optional destination path or ``BytesIO`` buffer. When ``None``, compiles
            purely in memory without touching disk.

        Returns
        -------
        bytes
            Compiled PDF document bytes.
        """
        t0 = time.perf_counter()

        category = str(inspection_data.get("category", "industrial_part"))
        category_label = str(inspection_data.get("category_label", category))
        timestamp = str(
            inspection_data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        station_id = str(inspection_data.get("station_id") or self.station_id)
        operator = str(inspection_data.get("operator") or self.default_operator)
        verdict = str(inspection_data.get("verdict") or "REJEITADA")

        score = float(inspection_data.get("anomaly_score", 0.0))
        threshold = float(inspection_data.get("threshold", 1.0))
        pixel_threshold = float(inspection_data.get("pixel_threshold", threshold * 0.85))
        defect_count = int(inspection_data.get("defect_count", 1))
        defect_condition = str(inspection_data.get("defect_condition", "Irregularidade Estrutural"))
        ai_latency_ms = float(inspection_data.get("ai_latency_ms", 0.0))
        total_latency_ms = float(inspection_data.get("total_latency_ms", ai_latency_ms))
        backbone = str(inspection_data.get("backbone", "resnet18"))
        engine_type = str(inspection_data.get("engine_type", "ONNX Runtime + FAISS IndexFlatL2"))

        orig_pil = _to_pil_rgb(inspection_data.get("original_rgb"))
        heat_pil = _to_pil_rgb(inspection_data.get("heatmap_overlay_rgb"))
        demarc_pil = _to_pil_rgb(inspection_data.get("demarcated_rgb"))

        sha256_hex = str(
            inspection_data.get("sha256_hash")
            or compute_image_sha256(inspection_data.get("image_source") or orig_pil)
        )
        piece_id = str(
            inspection_data.get("piece_id")
            or f"RNC-{category.upper()}-{sha256_hex[:8].upper()}"
        )

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=False)
        pdf.set_margins(left=12.0, top=12.0, right=12.0)
        pdf.add_page()

        # ------------------------------------------------------------------
        # 1. Cabeçalho Executivo Industrial (ISO 9001)
        # ------------------------------------------------------------------
        pdf.set_fill_color(15, 23, 42)  # Slate-900
        pdf.rect(12.0, 11.0, 186.0, 24.0, style="F")
        pdf.set_fill_color(220, 38, 38)  # Red accent stripe
        pdf.rect(12.0, 11.0, 4.0, 24.0, style="F")

        pdf.set_xy(19.0, 13.5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(
            130.0,
            6.5,
            _safe_pdf_text("Relatório de Inspeção Visual e Não Conformidade (RNC)"),
            new_x="RIGHT",
            new_y="TOP",
        )

        # Badge ISO 9001 no canto superior direito
        pdf.set_fill_color(185, 28, 28)
        pdf.set_xy(154.0, 13.5)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(40.0, 6.0, _safe_pdf_text("RNC - PADRÃO ISO 9001"), border=0, align="C", fill=True)

        pdf.set_xy(19.0, 20.5)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(203, 213, 225)
        pdf.cell(
            175.0,
            4.5,
            _safe_pdf_text(
                f"ID da Peça: {piece_id}   |   Data/Hora (ISO 8601): {timestamp}"
            ),
            new_x="LEFT",
            new_y="NEXT",
        )
        pdf.set_x(19.0)
        pdf.cell(
            175.0,
            4.5,
            _safe_pdf_text(
                f"Estação de Leitura: {station_id}   |   Operador: {operator}"
            ),
            new_x="LEFT",
            new_y="NEXT",
        )

        # ------------------------------------------------------------------
        # 2. Tabela de Parâmetros Técnicos da Inspeção
        # ------------------------------------------------------------------
        y_cursor = 39.0
        pdf.set_xy(12.0, y_cursor)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(
            186.0,
            6.0,
            _safe_pdf_text("1. PARÂMETROS TÉCNICOS E VEREDITO DE INSPEÇÃO"),
            new_x="LEFT",
            new_y="NEXT",
        )

        y_cursor = pdf.get_y() + 1.0
        col_w_label = 44.0
        col_w_val = 49.0
        row_h = 7.2

        def _draw_param_row(
            y: float,
            lbl1: str,
            val1: str,
            lbl2: str,
            val2: str,
            highlight_val2_red: bool = False,
        ) -> float:
            pdf.set_xy(12.0, y)
            pdf.set_draw_color(203, 213, 225)

            # Cell 1 Label
            pdf.set_fill_color(241, 245, 249)
            pdf.set_text_color(51, 65, 85)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(col_w_label, row_h, _safe_pdf_text(f" {lbl1}"), border=1, fill=True)

            # Cell 1 Value
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(15, 23, 42)
            pdf.set_font("Helvetica", "", 8.2)
            pdf.cell(col_w_val, row_h, _safe_pdf_text(f" {val1}"), border=1, fill=True)

            # Cell 2 Label
            pdf.set_fill_color(241, 245, 249)
            pdf.set_text_color(51, 65, 85)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(col_w_label, row_h, _safe_pdf_text(f" {lbl2}"), border=1, fill=True)

            # Cell 2 Value
            if highlight_val2_red:
                pdf.set_fill_color(220, 38, 38)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 8.5)
            else:
                pdf.set_fill_color(255, 255, 255)
                pdf.set_text_color(15, 23, 42)
                pdf.set_font("Helvetica", "", 8.2)
            pdf.cell(col_w_val, row_h, _safe_pdf_text(f" {val2}"), border=1, fill=True)
            return y + row_h

        y_cursor = _draw_param_row(
            y_cursor,
            "Categoria da Peça",
            category_label,
            "Veredito Final",
            f"PEÇA {verdict.upper()} (NÃO CONFORME)",
            highlight_val2_red=True,
        )
        y_cursor = _draw_param_row(
            y_cursor,
            "Índice Medido (Score)",
            f"{score:.4f} (Acima do Limite)",
            "Limiar de Fábrica (tau)",
            f"{threshold:.4f} (Pixel-Thr: {pixel_threshold:.4f})",
        )
        y_cursor = _draw_param_row(
            y_cursor,
            "Quantidade de Falhas",
            f"{defect_count} região(ões) | {defect_condition}",
            "Tempo de Análise da IA",
            f"{ai_latency_ms:.1f} ms IA | {total_latency_ms:.1f} ms Total",
        )
        y_cursor = _draw_param_row(
            y_cursor,
            "Arquitetura / Backbone",
            backbone,
            "Motor de Inferência",
            engine_type,
        )

        # ------------------------------------------------------------------
        # 3. Painel de Evidências Visuais (3 Imagens Lado a Lado)
        # ------------------------------------------------------------------
        y_cursor += 5.0
        pdf.set_xy(12.0, y_cursor)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(
            186.0,
            6.0,
            _safe_pdf_text("2. PAINEL DE EVIDÊNCIAS VISUAIS E DEMARCAÇÃO ÓPTICA"),
            new_x="LEFT",
            new_y="NEXT",
        )

        y_cards = pdf.get_y() + 1.0
        card_w = 59.0
        card_h = 68.0
        gap_x = 4.5
        img_size = 53.0

        panels = [
            ("01. Foto Original da Peça", "Captura Óptica Calibrada (RGB 1:1)", orig_pil),
            ("02. Mapa Térmico PatchCore", "Overlay Semitransparente (Alpha 50%)", heat_pil),
            ("03. Demarcação de Falhas", "Contorno Exato + Bounding Box", demarc_pil),
        ]

        for idx, (p_title, p_sub, pil_im) in enumerate(panels):
            x0 = 12.0 + idx * (card_w + gap_x)
            pdf.set_draw_color(203, 213, 225)
            pdf.set_fill_color(248, 250, 252)
            pdf.rect(x0, y_cards, card_w, card_h, style="DF")

            # Header of visual card
            pdf.set_fill_color(30, 41, 59)
            pdf.rect(x0, y_cards, card_w, 6.5, style="F")
            pdf.set_xy(x0 + 1.5, y_cards + 1.0)
            pdf.set_font("Helvetica", "B", 7.6)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(card_w - 3.0, 4.5, _safe_pdf_text(p_title), align="C")

            # Embed PIL image directly in memory
            pdf.image(pil_im, x=x0 + (card_w - img_size) / 2.0, y=y_cards + 8.0, w=img_size, h=img_size)

            # Footer caption of visual card
            pdf.set_xy(x0 + 1.0, y_cards + card_h - 5.5)
            pdf.set_font("Helvetica", "", 6.8)
            pdf.set_text_color(71, 85, 105)
            pdf.cell(card_w - 2.0, 4.0, _safe_pdf_text(p_sub), align="C")

        y_cursor = y_cards + card_h + 4.5

        # ------------------------------------------------------------------
        # 4. Detalhamento Geométrico das Regiões Demarcadas
        # ------------------------------------------------------------------
        pdf.set_xy(12.0, y_cursor)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(
            186.0,
            6.0,
            _safe_pdf_text("3. GEOMETRIA DAS REGIÕES DE NÃO CONFORMIDADE DETECTADAS"),
            new_x="LEFT",
            new_y="NEXT",
        )

        y_table = pdf.get_y() + 0.8
        regions_raw: List[Any] = list(inspection_data.get("defect_regions") or [])

        headers = [
            ("Ocorrência", 24.0),
            ("Bounding Box (X, Y, L, A)", 48.0),
            ("Área Afetada (px²)", 38.0),
            ("Centroide Óptico (X, Y)", 40.0),
            ("Pico Térmico (%)", 36.0),
        ]

        pdf.set_xy(12.0, y_table)
        pdf.set_fill_color(30, 41, 59)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7.8)
        for h_text, h_w in headers:
            pdf.cell(h_w, 6.2, _safe_pdf_text(h_text), border=1, align="C", fill=True)
        y_table += 6.2

        pdf.set_font("Helvetica", "", 7.8)
        pdf.set_text_color(15, 23, 42)
        if regions_raw:
            for r_idx, reg in enumerate(regions_raw[:4], start=1):
                if isinstance(reg, dict):
                    bbox = reg.get("bbox", (0, 0, 0, 0))
                    area = float(reg.get("area", 0.0))
                    centroid = reg.get("centroid", (0, 0))
                    max_sc = float(reg.get("max_anomaly_score", 1.0)) * 100.0
                else:
                    bbox = getattr(reg, "bbox", (0, 0, 0, 0))
                    area = float(getattr(reg, "area", 0.0))
                    centroid = getattr(reg, "centroid", (0, 0))
                    max_sc = float(getattr(reg, "max_anomaly_score", 1.0)) * 100.0

                pdf.set_xy(12.0, y_table)
                pdf.set_fill_color(255, 255, 255)
                pdf.cell(24.0, 5.8, _safe_pdf_text(f"#{r_idx}"), border=1, align="C", fill=True)
                pdf.cell(
                    48.0,
                    5.8,
                    _safe_pdf_text(f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"),
                    border=1,
                    align="C",
                    fill=True,
                )
                pdf.cell(38.0, 5.8, _safe_pdf_text(f"{int(round(area))} px²"), border=1, align="C", fill=True)
                pdf.cell(
                    40.0,
                    5.8,
                    _safe_pdf_text(f"({centroid[0]}, {centroid[1]})"),
                    border=1,
                    align="C",
                    fill=True,
                )
                pdf.cell(36.0, 5.8, _safe_pdf_text(f"{max_sc:.1f}%"), border=1, align="C", fill=True)
                y_table += 5.8
        else:
            pdf.set_xy(12.0, y_table)
            pdf.set_fill_color(255, 255, 255)
            pdf.cell(
                186.0,
                6.0,
                _safe_pdf_text(
                    f"Anomalia global detectada (Score {score:.4f} >= Limite {threshold:.4f}) - Condição: {defect_condition}"
                ),
                border=1,
                align="C",
                fill=True,
            )
            y_table += 6.0

        # ------------------------------------------------------------------
        # 5. Rodapé de Integridade Criptográfica (SHA-256) e Assinaturas
        # ------------------------------------------------------------------
        y_footer = 244.0
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(12.0, y_footer, 186.0, 14.0, style="DF")

        pdf.set_xy(14.5, y_footer + 2.0)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(
            180.0,
            4.5,
            _safe_pdf_text("GARANTIA DE INTEGRIDADE ANTI-ADULTERAÇÃO (HASH SHA-256 DA IMAGEM INSPECIONADA):"),
            new_x="LEFT",
            new_y="NEXT",
        )
        pdf.set_x(14.5)
        pdf.set_font("Courier", "B", 8.0)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(180.0, 5.0, _safe_pdf_text(sha256_hex), new_x="LEFT", new_y="NEXT")

        # Bloco de Assinaturas Técnicas
        y_sig = 269.0
        pdf.set_draw_color(100, 116, 139)
        pdf.line(18.0, y_sig, 96.0, y_sig)
        pdf.line(114.0, y_sig, 192.0, y_sig)

        pdf.set_xy(18.0, y_sig + 1.5)
        pdf.set_font("Helvetica", "B", 8.0)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(78.0, 4.2, _safe_pdf_text("Inspetor de Qualidade / Operador"), align="C")
        pdf.set_xy(114.0, y_sig + 1.5)
        pdf.cell(78.0, 4.2, _safe_pdf_text("Engenharia de Processo / Qualidade (ISO 9001)"), align="C")

        pdf.set_xy(18.0, y_sig + 5.8)
        pdf.set_font("Helvetica", "", 7.0)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(78.0, 3.8, _safe_pdf_text(f"Responsável: {operator}"), align="C")
        pdf.set_xy(114.0, y_sig + 5.8)
        compile_ms = (time.perf_counter() - t0) * 1000.0
        pdf.cell(
            78.0,
            3.8,
            _safe_pdf_text(f"Laudo compilado em memória em {compile_ms:.1f} ms"),
            align="C",
        )

        # Emit bytes entirely in RAM via io.BytesIO
        mem_buffer = io.BytesIO()
        pdf.output(mem_buffer)
        pdf_bytes = mem_buffer.getvalue()

        if output_path is not None:
            if isinstance(output_path, io.BytesIO):
                output_path.write(pdf_bytes)
                output_path.seek(0)
            else:
                out_file = Path(output_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_bytes(pdf_bytes)

        return pdf_bytes

