# Relatório — Backtest Over/Under 2,5 vs Pinnacle

**Aviso:** simulação histórica apenas; sem apostas reais.

- Ligas com Pinnacle O/U: **16**
- Partidas OOS com odds abertura+fechamento Pinnacle: **12717**
- Holdout: código season holdout = última temporada completa por liga (16 ligas com season em curso excluída do treino/holdout); jogos com Pinnacle abertura no holdout: 2131 partidas, 2025-07-25 a 2026-01-14 (cobertura parcial do calendário 2025/26 — odds abertura só em parte da temporada)
- Variantes distintas: **24**; linhas testadas (× treino/holdout): **80**
- Pré-registro: `reports/HIPOTESES_OVER_UNDER.md`
- Changelog vs relatório anterior: `reports/OU_BACKTEST_CHANGELOG.md`

## Métricas globais (modelo vs Pinnacle fechamento, devig multiplicativo)

| Métrica | Modelo | Pinnacle close |
|---------|--------|----------------|
| Log-loss | 0.6869 | 0.6718 |
| Brier | 0.2468 | 0.2396 |
| n jogos | 12717 | 12717 |

Teste nulo bootstrap do melhor ROI (não é White/Hansen SPA): **p = 0.8276**. Com dezenas de estratégias de treino e muitas com `n` pequeno (H3), o teste tem **baixo poder** para separar um único vencedor de ruído — interpretar junto com ROI, `n` e q-values.

H2 (shrinkage) **não** é estimado à parte: equivale a H1 com limiar `edge/w` (ver changelog).

## Resultados agregados (todas as ligas)

| Hipótese | Variante | Split | n | ROI | IC95% | CLV médio | %CLV+ | p raw | p Bonf | q BH | Sobrevive |
|----------|----------|-------|---|-----|-------|-----------|-------|-------|--------|------|-----------|
| H1_blend_xg | H1c_blend_edge10_multiplicative | holdout | 87 | +3.32% | [-21.49%, +28.67%] | -0.0163 | 24.1% | 0.394 | 1.000 | 1.000 | não |
| H1_blend_xg | H1c_blend_edge10_multiplicative | train | 410 | +0.93% | [-11.34%, +13.53%] | -0.0148 | 28.0% | 0.440 | 1.000 | 1.000 | não |
| H1_blend_xg | H1c_blend_edge5_multiplicative | holdout | 341 | +4.92% | [-6.93%, +16.69%] | -0.0189 | 21.4% | 0.211 | 1.000 | 1.000 | não |
| H1_blend_xg | H1c_blend_edge5_multiplicative | train | 1390 | -6.50% | [-12.63%, -0.27%] | -0.0167 | 25.3% | 0.979 | 1.000 | 1.000 | não |
| H1_close_sanity | H1d_edge5_multiplicative_close | holdout | 930 | -2.69% | [-9.63%, +4.32%] | -0.0156 | 0.1% | 0.773 | 1.000 | 1.000 | não |
| H1_close_sanity | H1d_edge5_multiplicative_close | train | 5298 | -6.01% | [-8.91%, -3.07%] | -0.0153 | 0.0% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge10_shin | holdout | 275 | -6.51% | [-20.07%, +7.12%] | -0.0219 | 21.1% | 0.827 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge10_shin | train | 1849 | -7.80% | [-13.22%, -2.54%] | -0.0270 | 20.5% | 0.998 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge5_shin | holdout | 916 | -3.64% | [-10.71%, +3.44%] | -0.0212 | 20.4% | 0.852 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge5_shin | train | 5105 | -6.47% | [-9.46%, -3.30%] | -0.0236 | 21.7% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1a_edge10_multiplicative | holdout | 238 | -8.40% | [-22.19%, +5.83%] | -0.0182 | 22.3% | 0.874 | 1.000 | 1.000 | não |
| H1_model | H1a_edge10_multiplicative | train | 1651 | -7.62% | [-13.17%, -2.21%] | -0.0200 | 24.2% | 0.997 | 1.000 | 1.000 | não |
| H1_model | H1a_edge3_multiplicative | holdout | 1324 | -4.91% | [-10.57%, +0.67%] | -0.0184 | 21.5% | 0.958 | 1.000 | 1.000 | não |
| H1_model | H1a_edge3_multiplicative | train | 6980 | -5.90% | [-8.43%, -3.42%] | -0.0192 | 23.5% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1a_edge5_multiplicative | holdout | 860 | -2.89% | [-10.20%, +4.04%] | -0.0174 | 22.8% | 0.796 | 1.000 | 1.000 | não |
| H1_model | H1a_edge5_multiplicative | train | 4912 | -7.04% | [-10.07%, -4.08%] | -0.0194 | 23.3% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1a_edge7_multiplicative | holdout | 527 | +0.32% | [-9.08%, +9.71%] | -0.0169 | 23.5% | 0.475 | 1.000 | 1.000 | não |
| H1_model | H1a_edge7_multiplicative | train | 3242 | -6.67% | [-10.46%, -2.95%] | -0.0195 | 23.4% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1b_edge10_power | holdout | 258 | -7.42% | [-21.71%, +6.67%] | -0.0206 | 22.5% | 0.850 | 1.000 | 1.000 | não |
| H1_model | H1b_edge10_power | train | 1780 | -7.73% | [-13.10%, -2.32%] | -0.0253 | 20.6% | 0.997 | 1.000 | 1.000 | não |
| H1_model | H1b_edge3_power | holdout | 1330 | -4.68% | [-10.47%, +0.95%] | -0.0203 | 21.0% | 0.943 | 1.000 | 1.000 | não |
| H1_model | H1b_edge3_power | train | 7088 | -6.12% | [-8.74%, -3.59%] | -0.0217 | 22.5% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1b_edge5_power | holdout | 897 | -2.92% | [-9.99%, +4.11%] | -0.0201 | 21.7% | 0.781 | 1.000 | 1.000 | não |
| H1_model | H1b_edge5_power | train | 5050 | -6.76% | [-9.81%, -3.68%] | -0.0223 | 22.1% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1b_edge7_power | holdout | 548 | +0.37% | [-8.70%, +9.62%] | -0.0199 | 22.3% | 0.471 | 1.000 | 1.000 | não |
| H1_model | H1b_edge7_power | train | 3409 | -6.65% | [-10.32%, -2.94%] | -0.0233 | 21.3% | 1.000 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e2 | holdout | 6 | +25.50% | [-100.00%, +154.50%] | +0.0239 | 100.0% | 0.345 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e2 | train | 16 | +16.75% | [-55.19%, +109.00%] | +0.0166 | 87.5% | 0.400 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e5 | holdout | 3 | +29.00% | [-100.00%, +287.00%] | +0.0344 | 100.0% | 0.294 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e5 | train | 11 | +4.36% | [-100.00%, +109.27%] | +0.0332 | 100.0% | 0.388 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e2 | holdout | 18 | -4.28% | [-62.24%, +59.33%] | +0.0202 | 88.9% | 0.577 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e2 | train | 31 | +7.42% | [-46.45%, +66.29%] | +0.0113 | 67.7% | 0.413 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e5 | holdout | 9 | +29.78% | [-66.00%, +139.78%] | +0.0300 | 88.9% | 0.287 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e5 | train | 15 | +1.67% | [-75.00%, +96.67%] | +0.0243 | 86.7% | 0.492 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e2 | holdout | 50 | -8.00% | [-41.04%, +27.42%] | +0.0180 | 76.0% | 0.673 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e2 | train | 414 | +0.11% | [-12.06%, +12.19%] | +0.0109 | 65.2% | 0.494 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e5 | holdout | 16 | -15.25% | [-75.94%, +58.63%] | +0.0348 | 93.8% | 0.692 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e5 | train | 80 | +19.88% | [-11.15%, +51.95%] | +0.0253 | 80.0% | 0.105 | 1.000 | 1.000 | não |
| H4_drift | H4a_follow | holdout | 2131 | -5.45% | [-9.68%, -1.25%] | — | — | 0.995 | 1.000 | 1.000 | não |
| H4_drift | H4a_follow | train | 10570 | -6.91% | [-8.80%, -4.99%] | — | — | 1.000 | 1.000 | 1.000 | não |
| H4_drift | H4b_fade | holdout | 2131 | -1.73% | [-6.01%, +2.57%] | — | — | 0.788 | 1.000 | 1.000 | não |
| H4_drift | H4b_fade | train | 10570 | -1.33% | [-3.20%, +0.54%] | — | — | 0.919 | 1.000 | 1.000 | não |
| H5_segment | H5_early60d | holdout | 391 | -5.60% | [-16.02%, +4.63%] | -0.0143 | 26.3% | 0.854 | 1.000 | 1.000 | não |
| H5_segment | H5_early60d | train | 977 | -2.52% | [-9.26%, +4.25%] | -0.0165 | 25.6% | 0.766 | 1.000 | 1.000 | não |
| H5_segment | H5_low_goals | holdout | 444 | -0.59% | [-10.50%, +9.33%] | -0.0160 | 23.9% | 0.547 | 1.000 | 1.000 | não |
| H5_segment | H5_low_goals | train | 2869 | -7.63% | [-11.50%, -3.66%] | -0.0211 | 21.2% | 1.000 | 1.000 | 1.000 | não |
| H5_segment | H5_not_early | holdout | 469 | -0.62% | [-10.32%, +9.23%] | -0.0200 | 19.8% | 0.548 | 1.000 | 1.000 | não |
| H5_segment | H5_not_early | train | 3935 | -8.17% | [-11.58%, -4.77%] | -0.0201 | 22.7% | 1.000 | 1.000 | 1.000 | não |

## Veredito

**Nenhuma variante sobrevive** (agregado nem por liga) aos critérios pré-registrados (ROI>0, p bootstrap <0,05 e Bonferroni ou BH-FDR).

## Por liga (holdout, H1a edge 5% multiplicativo)

| Liga | n | ROI | p raw | q BH | Sobrevive |
|------|---|-----|-------|------|-----------|
| 2. Bundesliga | 38 | -27.82% | 0.969 | 1.000 | não |
| Belgium Pro League | 35 | -4.91% | 0.618 | 1.000 | não |
| Bundesliga | 38 | +11.95% | 0.213 | 1.000 | não |
| Championship | 92 | +0.18% | 0.489 | 1.000 | não |
| Eredivisie | 55 | -24.65% | 0.939 | 1.000 | não |
| La Liga | 73 | -1.82% | 0.558 | 1.000 | não |
| LaLiga2 | 79 | -6.05% | 0.702 | 1.000 | não |
| Ligue 1 | 62 | -2.16% | 0.563 | 1.000 | não |
| Ligue 2 | 40 | -11.33% | 0.761 | 1.000 | não |
| Premier League | 89 | +18.89% | 0.036 | 1.000 | não |
| Primeira Liga | 46 | -4.35% | 0.625 | 1.000 | não |
| Scottish Premiership | 12 | +25.92% | 0.210 | 1.000 | não |
| Serie A | 90 | -13.50% | 0.876 | 1.000 | não |
| Serie B | 32 | +17.09% | 0.214 | 1.000 | não |
| Super League Greece | 20 | +28.65% | 0.121 | 1.000 | não |
| Super Lig | 59 | -14.71% | 0.859 | 1.000 | não |

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

- Temporada em andamento excluída do treino e do holdout.
- H4 ainda usa fechamentos de jogos anteriores (viés de timing fino).
- `Max`/`Avg` não são totalmente executáveis.
- Família de testes inclui variantes agregadas e por liga.
