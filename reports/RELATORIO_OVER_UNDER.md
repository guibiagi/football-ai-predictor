# Relatório — Backtest Over/Under 2,5 vs Pinnacle

**Aviso:** simulação histórica apenas; sem apostas reais.

- Ligas com Pinnacle O/U: **16**
- Partidas com predição + odds abertura Pinnacle: **12717**
- Pré-registro: `reports/HIPOTESES_OVER_UNDER.md`

## Métricas globais (modelo vs Pinnacle fechamento, sem vig)

| Métrica | Modelo | Pinnacle close |
|---------|--------|----------------|
| Log-loss | 0.6869 | 0.6718 |
| Brier | 0.2468 | 0.2396 |
| n jogos | 12717 | 12717 |

White reality check (melhor ROI vs permutação): **p = 0.5067**

## Resultados por hipótese (agregado em todas as ligas; treino vs holdout)

| Hipótese | Variante | Split | n | ROI | IC95% | CLV médio | %CLV+ | p raw | p Bonf | q BH | Sobrevive |
|----------|----------|-------|---|-----|-------|-----------|-------|-------|--------|------|-----------|
| H1_model | H1a_edge10_multiplicative | holdout | 238 | -8.40% | [-22.16%, +5.82%] | +0.0000 | 49.6% | 0.124 | 1.000 | 0.206 | não |
| H1_model | H1a_edge10_multiplicative | train | 1651 | -7.62% | [-13.21%, -2.05%] | -0.0023 | 46.1% | 0.004 | 0.144 | 0.012 | não |
| H1_model | H1a_edge3_multiplicative | holdout | 1324 | -4.91% | [-10.53%, +0.69%] | +0.0009 | 48.2% | 0.046 | 1.000 | 0.097 | não |
| H1_model | H1a_edge3_multiplicative | train | 6980 | -5.90% | [-8.40%, -3.37%] | -0.0008 | 48.1% | 0.000 | 0.000 | 0.000 | não |
| H1_model | H1a_edge5_multiplicative | holdout | 860 | -2.89% | [-9.94%, +4.28%] | +0.0016 | 48.5% | 0.213 | 1.000 | 0.275 | não |
| H1_model | H1a_edge5_multiplicative | train | 4912 | -7.04% | [-10.09%, -4.01%] | -0.0013 | 47.2% | 0.000 | 0.000 | 0.000 | não |
| H1_model | H1a_edge7_multiplicative | holdout | 527 | +0.32% | [-8.91%, +9.60%] | +0.0016 | 47.6% | 0.474 | 1.000 | 0.482 | não |
| H1_model | H1a_edge7_multiplicative | train | 3242 | -6.67% | [-10.49%, -2.89%] | -0.0015 | 46.9% | 0.000 | 0.012 | 0.001 | não |
| H1_model | H1b_edge5_power | holdout | 897 | -2.92% | [-10.01%, +4.04%] | +0.0043 | 57.7% | 0.217 | 1.000 | 0.275 | não |
| H1_model | H1b_edge5_power | train | 5050 | -6.76% | [-9.74%, -3.71%] | +0.0024 | 53.0% | 0.000 | 0.000 | 0.000 | não |
| H1_blend_xg | H1c_blend_edge10_multiplicative | holdout | 87 | +3.32% | [-21.45%, +28.55%] | -0.0013 | 50.6% | 0.405 | 1.000 | 0.450 | não |
| H1_blend_xg | H1c_blend_edge10_multiplicative | train | 410 | +0.93% | [-11.47%, +13.14%] | -0.0005 | 44.9% | 0.441 | 1.000 | 0.476 | não |
| H1_blend_xg | H1c_blend_edge5_multiplicative | holdout | 341 | +4.92% | [-7.24%, +16.92%] | -0.0033 | 43.1% | 0.214 | 1.000 | 0.275 | não |
| H1_blend_xg | H1c_blend_edge5_multiplicative | train | 1390 | -6.50% | [-12.60%, -0.36%] | -0.0018 | 45.0% | 0.018 | 0.732 | 0.046 | não |
| H1_close_sanity | H1d_edge5_multiplicative_close | holdout | 930 | -2.69% | [-9.76%, +4.51%] | -0.0107 | 35.9% | 0.220 | 1.000 | 0.275 | não |
| H1_close_sanity | H1d_edge5_multiplicative_close | train | 5298 | -6.01% | [-9.01%, -3.05%] | -0.0120 | 34.1% | 0.000 | 0.000 | 0.000 | não |
| H2_shrink | H2_w0.25 | holdout | 9 | +92.78% | [-4.89%, +177.22%] | +0.0215 | 88.9% | 0.035 | 1.000 | 0.082 | sim |
| H2_shrink | H2_w0.25 | train | 181 | -14.45% | [-32.59%, +4.37%] | +0.0068 | 59.1% | 0.066 | 1.000 | 0.133 | não |
| H2_shrink | H2_w0.50 | holdout | 258 | -7.42% | [-21.06%, +6.65%] | +0.0061 | 62.0% | 0.150 | 1.000 | 0.222 | não |
| H2_shrink | H2_w0.50 | train | 1780 | -7.73% | [-13.09%, -2.22%] | +0.0030 | 54.1% | 0.003 | 0.104 | 0.009 | não |
| H2_shrink | H2_w0.75 | holdout | 604 | -2.59% | [-11.27%, +6.30%] | +0.0056 | 59.3% | 0.285 | 1.000 | 0.336 | não |
| H2_shrink | H2_w0.75 | train | 3676 | -6.08% | [-9.62%, -2.43%] | +0.0025 | 53.2% | 0.001 | 0.020 | 0.002 | não |
| H3_soft | H3_B365_e2 | holdout | 14 | -9.57% | [-57.64%, +38.64%] | -0.0122 | 42.9% | 0.352 | 1.000 | 0.402 | não |
| H3_soft | H3_B365_e2 | train | 14 | -42.50% | [-100.00%, +28.93%] | -0.0232 | 57.1% | 0.132 | 1.000 | 0.210 | não |
| H3_soft | H3_B365_e5 | holdout | 6 | -44.50% | [-100.00%, +15.50%] | -0.0014 | 50.0% | 0.104 | 1.000 | 0.184 | não |
| H3_soft | H3_B365_e5 | train | 7 | -100.00% | [-100.00%, -100.00%] | -0.0064 | 85.7% | 0.000 | 0.000 | 0.000 | não |
| H3_soft | H3_Max_e2 | holdout | 41 | -25.17% | [-52.10%, +2.88%] | -0.0119 | 36.6% | 0.040 | 1.000 | 0.089 | não |
| H3_soft | H3_Max_e2 | train | 279 | +6.07% | [-7.02%, +19.11%] | -0.0068 | 45.2% | 0.180 | 1.000 | 0.258 | não |
| H3_soft | H3_Max_e5 | holdout | 11 | -53.55% | [-100.00%, -6.91%] | -0.0009 | 54.5% | 0.012 | 0.488 | 0.033 | não |
| H3_soft | H3_Max_e5 | train | 42 | +22.95% | [-11.19%, +57.67%] | -0.0048 | 52.4% | 0.093 | 1.000 | 0.177 | não |
| H4_drift | H4a_follow | holdout | 2131 | -5.45% | [-9.74%, -1.17%] | +nan | nan% | 0.005 | 0.212 | 0.016 | não |
| H4_drift | H4a_follow | train | 10570 | -7.05% | [-8.92%, -5.09%] | +nan | nan% | 0.000 | 0.000 | 0.000 | não |
| H4_drift | H4b_fade | holdout | 2131 | -2.37% | [-6.58%, +1.73%] | +nan | nan% | 0.137 | 1.000 | 0.210 | não |
| H4_drift | H4b_fade | train | 10570 | -2.13% | [-4.03%, -0.24%] | +nan | nan% | 0.011 | 0.420 | 0.030 | não |
| H5_segment | H5_early60d | holdout | 402 | -6.82% | [-17.37%, +3.82%] | +0.0080 | 61.9% | 0.106 | 1.000 | 0.184 | não |
| H5_segment | H5_early60d | train | 987 | -2.39% | [-9.04%, +4.40%] | +0.0046 | 55.3% | 0.247 | 1.000 | 0.299 | não |
| H5_segment | H5_low_goals | holdout | 462 | -0.45% | [-10.25%, +9.74%] | +0.0058 | 60.4% | 0.469 | 1.000 | 0.482 | não |
| H5_segment | H5_low_goals | train | 2923 | -6.69% | [-10.59%, -2.78%] | +0.0004 | 50.0% | 0.000 | 0.012 | 0.001 | não |
| H5_segment | H5_not_early | holdout | 495 | +0.25% | [-9.32%, +9.93%] | +0.0014 | 54.3% | 0.482 | 1.000 | 0.482 | não |
| H5_segment | H5_not_early | train | 4063 | -7.82% | [-11.22%, -4.49%] | +0.0019 | 52.5% | 0.000 | 0.000 | 0.000 | não |

## Veredito

Variantes que passam os critérios pré-registrados (ver nota de amostra):
- **H2_w0.25** (holdout): n=9, ROI +92.78%, p=0.035, q_BH=0.082 ⚠️ amostra pequena

**Nota:** sobrevivência estatística com `n` muito baixo (ex.: holdout) não implica edge real — interpretar com extrema cautela.

## Calibração (modelo, decílicos)

| bin | n | pred | actual |
|-----|---|------|--------|
| 0.20-0.30 | 177 | 0.279 | 0.412 |
| 0.30-0.40 | 1752 | 0.363 | 0.447 |
| 0.40-0.50 | 4869 | 0.451 | 0.477 |
| 0.50-0.60 | 4229 | 0.547 | 0.555 |
| 0.60-0.70 | 1541 | 0.634 | 0.640 |
| 0.70-0.80 | 146 | 0.730 | 0.705 |
| 0.80-0.90 | 3 | 0.817 | 0.667 |

## Limitações

- Holdout = última temporada por liga; amostra pode ser pequena.
- Ligas sem `P>2.5` excluídas.
- xG só em subconjunto (H1c).
- Múltiplos testes; Bonferroni é conservador.
- Não modela execução, limites ou comissão.
