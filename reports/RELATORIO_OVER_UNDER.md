# Relatório — Backtest Over/Under 2,5 vs Pinnacle

**Aviso:** simulação histórica apenas; sem apostas reais.

- Ligas com Pinnacle O/U: **16**
- Partidas OOS com odds abertura+fechamento Pinnacle: **12717**
- Holdout: **última temporada completa por liga; em curso excluída (16 ligas com season em andamento)**
- Variantes distintas: **24**; linhas testadas (× treino/holdout): **80**
- Pré-registro: `reports/HIPOTESES_OVER_UNDER.md`
- Changelog vs relatório anterior: `reports/OU_BACKTEST_CHANGELOG.md`

## Métricas globais (modelo vs Pinnacle fechamento, devig multiplicativo)

| Métrica | Modelo | Pinnacle close |
|---------|--------|----------------|
| Log-loss | 0.6869 | 0.6718 |
| Brier | 0.2468 | 0.2396 |
| n jogos | 12717 | 12717 |

Teste nulo bootstrap do melhor ROI (não é White/Hansen SPA): **p = 0.9220**

H2 (shrinkage) **não** é estimado à parte: equivale a H1 com limiar `edge/w` (ver changelog).

## Resultados agregados (todas as ligas)

| Hipótese | Variante | Split | n | ROI | IC95% | CLV médio | %CLV+ | p raw | p Bonf | q BH | Sobrevive |
|----------|----------|-------|---|-----|-------|-----------|-------|-------|--------|------|-----------|
| H1_blend_xg | H1c_blend_edge10_multiplicative | holdout | 87 | +3.32% | [-21.17%, +28.70%] | -0.0013 | 50.6% | 0.405 | 1.000 | 1.000 | não |
| H1_blend_xg | H1c_blend_edge10_multiplicative | train | 410 | +0.93% | [-11.19%, +13.22%] | -0.0005 | 44.9% | 0.445 | 1.000 | 1.000 | não |
| H1_blend_xg | H1c_blend_edge5_multiplicative | holdout | 341 | +4.92% | [-7.27%, +17.09%] | -0.0033 | 43.1% | 0.214 | 1.000 | 1.000 | não |
| H1_blend_xg | H1c_blend_edge5_multiplicative | train | 1390 | -6.50% | [-12.62%, -0.29%] | -0.0018 | 45.0% | 0.983 | 1.000 | 1.000 | não |
| H1_close_sanity | H1d_edge5_multiplicative_close | holdout | 930 | -2.69% | [-9.73%, +4.34%] | -0.0128 | 28.1% | 0.774 | 1.000 | 1.000 | não |
| H1_close_sanity | H1d_edge5_multiplicative_close | train | 5298 | -6.01% | [-9.05%, -3.01%] | -0.0150 | 28.8% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge10_shin | holdout | 264 | -9.18% | [-22.99%, +5.01%] | +0.0031 | 51.9% | 0.897 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge10_shin | train | 2058 | -8.74% | [-13.73%, -3.78%] | -0.0009 | 46.3% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge5_shin | holdout | 880 | -4.03% | [-11.04%, +3.21%] | +0.0023 | 49.1% | 0.865 | 1.000 | 1.000 | não |
| H1_model | H1_shin_edge5_shin | train | 5275 | -7.00% | [-9.93%, -4.03%] | -0.0000 | 48.5% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1a_edge10_multiplicative | holdout | 238 | -8.40% | [-22.49%, +5.70%] | +0.0000 | 49.6% | 0.876 | 1.000 | 1.000 | não |
| H1_model | H1a_edge10_multiplicative | train | 1651 | -7.62% | [-13.15%, -2.08%] | -0.0023 | 46.1% | 0.997 | 1.000 | 1.000 | não |
| H1_model | H1a_edge3_multiplicative | holdout | 1324 | -4.91% | [-10.36%, +0.61%] | +0.0009 | 48.2% | 0.957 | 1.000 | 1.000 | não |
| H1_model | H1a_edge3_multiplicative | train | 6980 | -5.90% | [-8.37%, -3.40%] | -0.0008 | 48.1% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1a_edge5_multiplicative | holdout | 860 | -2.89% | [-10.05%, +4.22%] | +0.0016 | 48.5% | 0.779 | 1.000 | 1.000 | não |
| H1_model | H1a_edge5_multiplicative | train | 4912 | -7.04% | [-10.11%, -4.02%] | -0.0013 | 47.2% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1a_edge7_multiplicative | holdout | 527 | +0.32% | [-8.82%, +9.44%] | +0.0016 | 47.6% | 0.480 | 1.000 | 1.000 | não |
| H1_model | H1a_edge7_multiplicative | train | 3242 | -6.67% | [-10.46%, -2.90%] | -0.0015 | 46.9% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1b_edge10_power | holdout | 258 | -7.42% | [-21.18%, +6.47%] | +0.0020 | 50.8% | 0.850 | 1.000 | 1.000 | não |
| H1_model | H1b_edge10_power | train | 1780 | -7.73% | [-13.16%, -2.32%] | -0.0022 | 45.9% | 0.998 | 1.000 | 1.000 | não |
| H1_model | H1b_edge3_power | holdout | 1330 | -4.68% | [-10.33%, +1.21%] | +0.0011 | 48.3% | 0.943 | 1.000 | 1.000 | não |
| H1_model | H1b_edge3_power | train | 7088 | -6.12% | [-8.68%, -3.59%] | -0.0006 | 48.3% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1b_edge5_power | holdout | 897 | -2.92% | [-9.92%, +4.38%] | +0.0017 | 48.9% | 0.789 | 1.000 | 1.000 | não |
| H1_model | H1b_edge5_power | train | 5050 | -6.76% | [-9.90%, -3.74%] | -0.0009 | 47.7% | 1.000 | 1.000 | 1.000 | não |
| H1_model | H1b_edge7_power | holdout | 548 | +0.37% | [-9.20%, +9.71%] | +0.0021 | 47.6% | 0.471 | 1.000 | 1.000 | não |
| H1_model | H1b_edge7_power | train | 3409 | -6.65% | [-10.40%, -2.90%] | -0.0012 | 47.2% | 1.000 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e2 | holdout | 6 | +25.50% | [-100.00%, +158.00%] | +0.0031 | 33.3% | 0.351 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e2 | train | 16 | +16.75% | [-55.19%, +109.56%] | -0.0078 | 6.2% | 0.402 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e5 | holdout | 3 | +29.00% | [-100.00%, +287.00%] | +0.0006 | 33.3% | 0.300 | 1.000 | 1.000 | não |
| H3_soft | H3_Avg_e5 | train | 11 | +4.36% | [-100.00%, +109.64%] | +0.0030 | 9.1% | 0.382 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e2 | holdout | 18 | -4.28% | [-60.39%, +62.22%] | -0.0044 | 16.7% | 0.564 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e2 | train | 31 | +7.42% | [-45.81%, +65.32%] | -0.0100 | 19.4% | 0.414 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e5 | holdout | 9 | +29.78% | [-73.33%, +140.56%] | -0.0030 | 33.3% | 0.298 | 1.000 | 1.000 | não |
| H3_soft | H3_B365_e5 | train | 15 | +1.67% | [-75.00%, +98.33%] | -0.0078 | 13.3% | 0.488 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e2 | holdout | 50 | -8.00% | [-42.06%, +28.12%] | -0.0027 | 34.0% | 0.680 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e2 | train | 414 | +0.11% | [-11.88%, +12.44%] | -0.0057 | 40.8% | 0.489 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e5 | holdout | 16 | -15.25% | [-75.94%, +58.81%] | -0.0020 | 31.2% | 0.687 | 1.000 | 1.000 | não |
| H3_soft | H3_Max_e5 | train | 80 | +19.88% | [-11.19%, +51.71%] | -0.0093 | 35.0% | 0.105 | 1.000 | 1.000 | não |
| H4_drift | H4a_follow | holdout | 2131 | -5.45% | [-9.70%, -1.25%] | — | — | 0.994 | 1.000 | 1.000 | não |
| H4_drift | H4a_follow | train | 10570 | -6.91% | [-8.85%, -4.99%] | — | — | 1.000 | 1.000 | 1.000 | não |
| H4_drift | H4b_fade | holdout | 2131 | -1.73% | [-5.84%, +2.50%] | — | — | 0.788 | 1.000 | 1.000 | não |
| H4_drift | H4b_fade | train | 10570 | -1.33% | [-3.19%, +0.52%] | — | — | 0.914 | 1.000 | 1.000 | não |
| H5_segment | H5_early60d | holdout | 391 | -5.60% | [-15.93%, +4.82%] | +0.0056 | 55.5% | 0.854 | 1.000 | 1.000 | não |
| H5_segment | H5_early60d | train | 977 | -2.52% | [-9.32%, +4.31%] | +0.0015 | 50.3% | 0.765 | 1.000 | 1.000 | não |
| H5_segment | H5_low_goals | holdout | 444 | -0.59% | [-10.70%, +9.46%] | +0.0032 | 51.1% | 0.546 | 1.000 | 1.000 | não |
| H5_segment | H5_low_goals | train | 2869 | -7.63% | [-11.44%, -3.76%] | -0.0023 | 45.4% | 1.000 | 1.000 | 1.000 | não |
| H5_segment | H5_not_early | holdout | 469 | -0.62% | [-10.59%, +9.17%] | -0.0017 | 42.6% | 0.542 | 1.000 | 1.000 | não |
| H5_segment | H5_not_early | train | 3935 | -8.17% | [-11.56%, -4.77%] | -0.0019 | 46.5% | 1.000 | 1.000 | 1.000 | não |

## Veredito

**Nenhuma variante sobrevive** aos critérios pré-registrados (ROI>0, p bootstrap <0,05 e Bonferroni ou BH-FDR) em treino ou holdout agregado.

## Por liga (holdout, H1a edge 5% multiplicativo)

| Liga | n | ROI | p raw |
|------|---|-----|-------|
| 2. Bundesliga | 38 | -27.82% | 0.966 |
| Belgium Pro League | 35 | -4.91% | 0.614 |
| Bundesliga | 38 | +11.95% | 0.212 |
| Championship | 92 | +0.18% | 0.502 |
| Eredivisie | 55 | -24.65% | 0.942 |
| La Liga | 73 | -1.82% | 0.559 |
| LaLiga2 | 79 | -6.05% | 0.702 |
| Ligue 1 | 62 | -2.16% | 0.557 |
| Ligue 2 | 40 | -11.33% | 0.770 |
| Premier League | 89 | +18.89% | 0.036 |
| Primeira Liga | 46 | -4.35% | 0.615 |
| Scottish Premiership | 12 | +25.92% | 0.212 |
| Serie A | 90 | -13.50% | 0.885 |
| Serie B | 32 | +17.09% | 0.212 |
| Super League Greece | 20 | +28.65% | 0.123 |
| Super Lig | 59 | -14.71% | 0.855 |

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
