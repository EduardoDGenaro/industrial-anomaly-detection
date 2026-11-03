# Catálogo de Visuais Profissionais e Referências de UI

Este diretório reúne referências visuais de alta qualidade de sistemas de **Inspeção Visual por Computador (Machine Vision)**, **Detecção de Anomalias Industriais** e **Dashboards de Qualidade (HMI / SCADA ISA-101)**. Estas referências servem de base direta para as evoluções estéticas, estruturais e ergonômicas da interface do projeto (`ui/app.py`).

---

## Galeria de Referências Baixadas

| Arquivo | Dimensões | Fonte / Plataforma | Foco / Aplicação na UI |
|---|---|---|---|
| [`01_anomalib_studio_hmi.png`](./01_anomalib_studio_hmi.png) | 1620 × 1022 | **Intel Anomalib Studio** | Layout completo de estação HMI para detecção de anomalias industriais, painel lateral de projetos e navegação por lotes. |
| [`02_anomalib_inspection_pipeline.png`](./02_anomalib_inspection_pipeline.png) | 861 × 167 | **Anomalib Official Docs** | Fluxo visual padrão de inspeção: *Imagem Bruta → Ground Truth → Heatmap Térmico → Máscara de Segmentação*. |
| [`03_landinglens_snapshot_dashboard.png`](./03_landinglens_snapshot_dashboard.png) | 2244 × 994 | **Landing AI (LandingLens)** | Dashboard comercial de visão computacional: contadores de peças aprovadas/rejeitadas, telemetria de produção e cards de confiança. |
| [`04_vision_ui_dark_mode.png`](./04_vision_ui_dark_mode.png) | 1600 × 1200 | **Vision UI (Dark Theme)** | Estética moderna *Dark Slate / Glassmorphism* com cards translúcidos, métricas destacadas e badges de alto contraste. |
| [`05_vision_ui_chakra_preview.png`](./05_vision_ui_chakra_preview.png) | 800 × 600 | **Vision UI Chakra** | Componentização modular para painéis analíticos, tabelas de dados compactas e cards de telemetria. |
| [`06_modern_cv_dashboard.png`](./06_modern_cv_dashboard.png) | 2443 × 424 | **Modern Computer Vision UI** | Barra de telemetria em tempo real: FPS, latência de inferência, status operacional da linha e alertas. |
| [`07_deye_industrial_inspection_gui.png`](./07_deye_industrial_inspection_gui.png) | 720 × 488 | **DEye Industrial Inspection** | Interface de software de inspeção em chão de fábrica: controles do operador, botões rápidos de ação e visualização de câmera. |
| [`08_deye_defect_tracking_view.png`](./08_deye_defect_tracking_view.png) | 720 × 488 | **DEye Tracking & Bounding Box** | Demarcação simultânea de Regiões de Interesse (ROI), contornos de defeitos e caixas delimitadoras coloridas. |
| [`09_deye_classification_statistics.png`](./09_deye_classification_statistics.png) | 720 × 488 | **DEye Defect Analytics** | Gráficos de distribuição de falhas (Pareto / Pizza), histórico de defeitos por classe e indicadores de conformidade. |
| [`10_ultralytics_app_ui.png`](./10_ultralytics_app_ui.png) | 1920 × 540 | **Ultralytics Vision Platform** | Layout moderno de navegação e inspeção de modelos de visão com design responsivo e minimalista. |

---

## Princípios de Design Integrados para o Projeto (`ui/app.py`)

### 1. Ergonomia Industrial (Padrão ISA-101)
- **Cores Semânticas Estritas:**
  - **Azul Industrial Operacional (`#1d4ed8` / `#2563eb`):** Ações normais, botões de comando, abas e seleção de parâmetros.
  - **Verde de Conformidade (`#10b981` / `#05cd99`):** Peça Aprovada (Pass), dentro dos limites de tolerância.
  - **Vermelho Exclusivo de Alarme/Rejeição (`#ef4444` / `#dc2626`):** Peça Reprovada (Fail), falha estrutural, acima do limiar crítico.
  - **Âmbar de Atenção (`#f59e0b`):** Peça marginal próxima ao threshold ou aviso de calibração.
- **Fundo Neutro / Baixa Fadiga Visual:** Reduz o cansaço dos operadores durante turnos contínuos de produção.

### 2. Layout em 3 Níveis de Informação
- **Nível 1 (Cabeçalho de Telemetria Operacional):**
  - Status instantâneo do lote (Taxa de Aprovação %, Total Inspecionado, Peças Conformes vs Defeituosas).
  - Telemetria de execução: Latência E2E (ms), FPS equivalente, Engine ativa (ONNX+FAISS vs PyTorch).
- **Nível 2 (Palco de Inspeção Visual Triplo):**
  - Coluna 1: Imagem da Peça com Região de Interesse.
  - Coluna 2: Mapa Térmico de Anomalias PatchCore (fusão semitransparente sobre a base física).
  - Coluna 3: Demarcação Vetorial (Contorno exato amarelo + Bounding box vermelho + Identificador numérico `#1`, `#2`).
- **Nível 3 (Tabela de Inteligência de Defeitos & Ações):**
  - Lista de anomalias encontradas com severidade, área em pixels e coordenadas.
  - Ações operacionais: Reinspecionar, Registrar no Lote, Ajustar Sensibilidade.

---

## Snippets de Estilo e Customização Prontos para Uso

### Paleta de Cores (CSS Tokens)
```css
:root {
  --bg-primary: #0f172a;        /* Fundo escuro industrial */
  --card-bg: rgba(30, 41, 59, 0.85); /* Superfície de card */
  --card-border: rgba(148, 163, 184, 0.15);
  --accent-blue: #1d4ed8;      /* Primário operacional */
  --pass-green: #10b981;       /* Aprovado */
  --fail-red: #ef4444;         /* Rejeitado */
  --warn-amber: #f59e0b;       /* Alerta / Marginal */
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
}
```

### Card de Métrica de Alta Performance
```css
.kpi-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 10px;
  padding: 16px;
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.kpi-label {
  font-size: 0.85rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.kpi-value {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text-main);
  font-family: 'JetBrains Mono', monospace;
}
```

