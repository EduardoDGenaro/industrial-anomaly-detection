# Guia das 15 Classes Industriais do MVTec AD

O benchmark MVTec Anomaly Detection (MVTec AD) é composto por 5 texturas contínuas e 10 objetos discretos manufaturados. Abaixo estão as especificidades de inspeção para cada classe.

---

## 1. Texturas Industriais (5 Classes)

| Classe | Descrição | Principais Tipos de Defeito | Desafio de Inspeção |
| :--- | :--- | :--- | :--- |
| `carpet` | Fibras têxteis de carpete | Furos, cortes, manchas de óleo, nós metálicos | Variação natural do entrelaçamento de fios; requer invariância a pequenas rotações. |
| `grid` | Malha metálica em grade | Fios partidos, dobras na tela, cola, resíduos | Defeitos pequenos em pontos de cruzamento; alta exigência na métrica PRO. |
| `leather` | Superfície de couro | Dobras, marcas de corte, furos, manchas de pigmento | Textura natural e rugosidade orgânica não devem ser confundidas com micro-fissuras. |
| `tile` | Revestimento cerâmico | Trincas, rachaduras no esmalte, manchas, óleo | Alta refletância especular; trincas ultrafinas requerem sensibilidade em alta resolução. |
| `wood` | Lâmina de madeira tratada | Furos, nós estranhos, arranhões, manchas | Veios naturais da madeira; exige backbone com rica representação semântica (DINOv2). |

---

## 2. Objetos Manufaturados (10 Classes)

| Classe | Descrição | Principais Tipos de Defeito | Desafio de Inspeção |
| :--- | :--- | :--- | :--- |
| `bottle` | Garrafa translúcida de vidro/plástico | Trinca no gargalo, contaminação interna, quebra | Transparência e refração de luz; defeitos no líquido vs. vidro. |
| `cable` | Cabo elétrico multi-fios | Fio desencapado, corte de isolamento, dobra, fio solto | Geometria não-rígida e curvatura variável. |
| `capsule` | Cápsula farmacêutica | Amassado, arranhão na casca, contaminação, furo | Defeitos com pouco contraste tonal; crítico para saúde. |
| `hazelnut` | Avelã / alimento descascado | Trinca na casca, furo de praga, fungo, corte | Variação biológica de formato e cor. |
| `metal_nut` | Porca metálica sextavada | Furação torta, rebarba, arranhão na rosca, oxidação | Reflexo metálico forte; defeitos na rosca interna. |
| `pill` | Comprimido farmacêutico | Fratura, cor inadequada, contaminação de pó, risco | Requisito de zero escape ($FN=0$ absoluto). |
| `screw` | Parafuso de precisão | Rosca amassada, cabeça deformada, risco, ponta torta | Dimensões milimétricas; orientação e alinhamento mecânico estrito. |
| `toothbrush` | Cabeça de escova de dentes | Cerdas desalinhadas, falha no tufo, contaminação | Resolução de cerdas individuais. |
| `transistor` | Transistor eletrônico montado | Perna torta, quebra no encapsulamento, solda fria | Orientação precisa; posicionamento dos pinos condutores. |
| `zipper` | Zíper metálico | Dente quebrado, dente torto, tecido desfiado, mancha | Periodicidade mecânica dos dentes metálicos. |

