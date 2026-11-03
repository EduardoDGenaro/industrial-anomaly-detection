# Industrial Anomaly Detection (IAD)
### Inspeção Visual e Detecção de Falhas Não Supervisionada em Linhas de Produção

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Anomalib%20%7C%20PyTorch-orange.svg)](https://github.com/openvinotoolkit/anomalib)
[![Dataset](https://img.shields.io/badge/Dataset-MVTec%20AD%20(15%20classes)-green.svg)](https://www.mvtec.com/company/research/datasets/mvtec-ad)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

---

## Visão Geral do Projeto

O **Industrial Anomaly Detection (IAD)** é uma plataforma completa de visão computacional e MLOps para inspeção de qualidade contínua em manufatura (Quality Assurance 4.0).

Diferente de classificadores supervisionados convencionais, este sistema opera no paradigma **Unsupervised / Few-Shot Anomaly Detection**:
- **Treinamento 100% em Peças Boas**: O modelo é alimentado exclusivamente com imagens normais (`good`).
- **Detecção de Falhas Inéditas**: Qualquer desvio estrutural (arranhão, furo, trinca, deformação ou contaminação) é detectado sem necessidade de exemplos prévios de falha.
- **Localização em Nível de Pixel**: Produz mapas térmicos de anomalia espacial (*segmentation heatmaps*) com contornos e caixas delimitadoras (*bounding boxes*).

---

## Metas de Produção e KPIs

| Métrica / KPI | Meta Alvo (SLA de Produção) | Descrição e Impacto |
| :--- | :--- | :--- |
| **Image-level AUROC** | $\ge 99.0\%$ | Classificação binária da peça na linha (PASS / REJECT). |
| **Pixel-level AUROC** | $\ge 97.5\%$ | Qualidade da segmentação espacial do defeito. |
| **PRO Metric (Per-Region Overlap)** | $\ge 93.5\%$ (FPR $\le 0.30$) | Avaliação de overlap ponderada por cada defeito individual. |
| **Latência por Peça** | **$< 50\text{ ms}$ (P95)** | Compatível com esteiras industriais a $\ge 20\text{ peças/segundo}$. |
| **Taxa de Falso Negativo (FNR)** | $< 0.1\%$ | Quase zero escape de peças defeituosas para expedição. |

Consulte o documento completo de especificações em [`PROJECT_SPEC.md`](PROJECT_SPEC.md).

---

## Estrutura do Repositório

```text
├── configs/
│   ├── patchcore_mvtec.yaml      # Configurações de treino/inferência PatchCore (Backbone, Coreset, Histerese)
│   └── autoencoder_mvtec.yaml    # Configurações do baseline comparativo de reconstrução
├── data/
│   ├── .gitkeep                  # Pasta ignorada no Git para armazenamento do MVTec AD
├── docs/
│   ├── architecture.md           # Diagrama de sequência, pipeline e orçamento de latência
│   └── mvtec_classes.md          # Guia detalhado das 15 classes do MVTec AD (5 texturas + 10 objetos)
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── mvtec.py              # Parser, loader e downloader automático do MVTec AD
│   ├── models/
│   │   ├── __init__.py
│   │   └── patchcore_wrapper.py  # Wrapper de inferência PatchCore, Coreset e benchmark de latência
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── postprocessor.py      # Filtro gaussiano, histerese, abertura morfológica e heatmap overlay
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # AUROC, PRO metric e calibração de threshold por matriz de custo
│       └── visualization.py      # Painel de inspeção visual (5 painéis com bounding boxes)
├── ui/
│   ├── __init__.py
│   └── app.py                    # Aplicação interativa Streamlit para ajuste de threshold e inspeção
├── tests/
│   ├── test_dataset.py           # Testes do parser e validação de classes
│   ├── test_metrics.py           # Testes de AUROC, PRO metric e custos
│   └── test_postprocessing.py    # Testes de histerese, morfologia e contornos
├── .gitignore                    # Regras de exclusão para pesos, caches e datasets
├── requirements.txt              # Dependências pinadas do projeto
├── PROJECT_SPEC.md               # Especificação técnica detalhada de arquitetura e escopo
└── README.md                     # Este guia rápido
```

---

## Setup do Ambiente

### 1. Pré-requisitos
- Python 3.10, 3.11 ou 3.12
- GPU NVIDIA recomendada (CUDA 11.8+ ou 12.x) para inferência $< 50\text{ ms}$; CPU compatível com AVX2/AVX-512.

### 2. Criação do Ambiente Virtual

Utilizando **uv** (rápido):
```bash
uv venv .venv --python 3.12
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS
uv pip install -r requirements.txt
```

Ou utilizando **venv** padrão:
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Ingestão do Dataset MVTec AD

Para baixar e preparar automaticamente uma categoria específica (ex: `bottle`):

```python
from src.dataset.mvtec import download_and_extract_mvtec

# Baixa a categoria oficial para data/mvtec_anomaly_detection/bottle
download_and_extract_mvtec("bottle", root_dir="./data/mvtec_anomaly_detection")
```

As 15 classes suportadas incluem:
- **Texturas**: `carpet`, `grid`, `leather`, `tile`, `wood`.
- **Objetos**: `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`.

---

## Aplicação Streamlit de Inspeção e Calibração

Inicie o console visual de controle de qualidade para operadores e engenheiros de automação:

```bash
streamlit run ui/app.py
```

### Funcionalidades do Dashboard:
1. **Inspeção Visual 4-Painéis**: Imagem bruta da câmera, mapa de calor térmico, máscara binária com *bounding boxes* e overlay de auditoria.
2. **Ajuste Fino de Threshold**: Slider interativo de threshold com visualização instantânea do veredito `PASS` vs `REJECT`.
3. **Filtro de Histerese**: Redução drástica de falsos alarmes causados por ruído de sensor ou poeira industrial.
4. **Otimizador de Custo Assimétrico**: Calibração com base no custo de falso negativo ($C_{FN}$) vs falso positivo ($C_{FP}$).

---

## Execução dos Testes Automatizados

Execute a suíte de testes unitários:

```bash
pytest tests/ -v
```

Com relatório de cobertura de código:
```bash
pytest --cov=src tests/
```

---

## Modelagem: PatchCore vs. Autoencoder

| Característica | PatchCore (Recomendado) | Autoencoder Convolucional |
| :--- | :--- | :--- |
| **Abordagem** | Coreset de Features de Redes Pré-treinadas (WideResNet / DINOv2) | Reconstrução de Pixel via Bottleneck Latente |
| **Amostras de Treino** | Poucas amostras (10 a 50 imagens são suficientes) | Centenas a milhares de amostras necessárias |
| **Defeitos Sutis** | Alta sensibilidade a micro-riscos e bordas | Tende a desfocar e ignorar pequenas alterações |
| **Desempenho** | **$\ge 99.1\%$ AUROC** | $\approx 88 - 92\%$ AUROC |

Para mais detalhes sobre as decisões arquiteturais, veja [`PROJECT_SPEC.md`](PROJECT_SPEC.md) e [`docs/architecture.md`](docs/architecture.md).

