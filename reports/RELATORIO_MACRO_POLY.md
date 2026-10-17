# Relatório — Backtest Polymarket macro (Fed / inflação)

**Modo:** somente simulação (sem ordens reais).  
**Mercados no manifest:** 469  
**Eventos no holdout (~30%):** 25  
**Liquidez por trades:** sim (cobertura treino=0.9649805447470817)  
**Testes ROI na família múltipla:** 17  
**Tercis (volume trades cumulativo, treino):** q33=983.3655055077162, q66=7323.004700053785  
**Stake simulado:** US$ 25.0  

Pré-registro: [HIPOTESES_MACRO_POLY_PREREG.md](HIPOTESES_MACRO_POLY_PREREG.md)

## Changelog vs v2 (abb273a)

| Correção | Efeito |
|----------|--------|
| FOMC exige Fed/FOMC; exclui ECB/BoJ/derivativos | Menos contaminação em H2/H4/H5 |
| Dedupe um evento FOMC por reunião (ex. set/2026) | Um outcome por meeting |
| Volume = trades data-api cumulativos até `decision_ts` | Tercis alinhados ao instante da entrada |
| H3 janela desde `start_ts` | Cobertura H3 reportada (`h3_coverage_stats`) |
| H2b estado resetado por limiar; H2a deduplicado | Família de testes menor |
| Células com n&lt;3 fora da família múltipla | `in_test_family` no JSON |

## Changelog vs relatório anterior (PR #4 v1)

| Correção | Efeito esperado nos números |
|----------|----------------------------|
| Removido teto `min(0.99, …)` na entrada | ROI H1 deixa de colapsar em ~0,99% artificial |
| H1 excluído de sobreviventes / família de testes ROI | Nenhum "survivor" tradeável vem de H1 |
| Timestamps FOMC oficiais (14:00 ET, calendário Fed) + filtro de mercados | H1c/H2 FOMC perdem eventos de menção e derivativos |
| Sem fallback de preço stale em `price_at_or_after` | Menos trades; entradas só com candle dentro da tolerância |
| Holdout determinístico `(release_ts, event_id)` | `holdout_n` / ROI holdout estáveis entre runs |
| Tercis: volume interpolado + cortes só no treino | Segmentos T1/T2/T3 realocados |
| H4a/H4b direção corrigida (fade vs continuação) | ROI H4 alterado |
| Cache de preços por janela `(token, start, end)` | H3 usa abertura real do mercado |
| H5 calibração no holdout | Nova seção de log-loss/Brier agregado |

### H5 — Calibração (holdout, preço 1h antes do release)

n=212, log-loss=0.2323938040529587, Brier=0.06460687853773585

### H3 — cobertura (mercados elegíveis vs trades simulados)
- **H3a_fav**: elegíveis=369, trades=5, sem histórico=0, sem preço na abertura=19
- **H3a_long**: elegíveis=369, trades=1, sem histórico=0, sem preço na abertura=19
- **H3b_fav**: elegíveis=58, trades=1, sem histórico=0, sem preço na abertura=0

## H1 — Diagnóstico (não tradeável; não entra em sobreviventes)

| ID | tercil | n | ROI médio | IC 95% | CLV médio | log-loss | Brier |
|----|--------|---|-----------|--------|-----------|----------|-------|
| H1a_60m__T1_low | T1_low | 186 | 0.0006 | [0.000, 0.001] | — | 0.04270000612449434 | 0.00897925 |
| H1a_60m__T2_mid | T2_mid | 104 | 0.0481 | [0.013, 0.098] | -0.042159999999999996 | 0.4038758043280595 | 0.11650802884615384 |
| H1a_60m__T3_high | T3_high | 56 | 0.0427 | [0.000, 0.109] | -0.00285156757015107 | 0.6552441564736158 | 0.22321235267857142 |
| H1b_60m__T1_low | T1_low | 31 | 0.0001 | [0.000, 0.000] | — | 0.10981549765614847 | 0.037610629032258064 |
| H1b_60m__T2_mid | T2_mid | 9 | 0.0001 | [0.000, 0.000] | — | 0.43188842148453455 | 0.14464233333333334 |
| H1b_60m__T3_high | T3_high | 2 | 0.0001 | [0.000, 0.000] | — | 1.8486574664675188 | 0.500861125 |
| H1c_60m__T1_low | T1_low | 3 | 0.0001 | [0.000, 0.000] | — | 0.004345804648199146 | 2.483333333333333e-05 |
| H1c_60m__T2_mid | T2_mid | 4 | 0.0001 | [0.000, 0.000] | — | 0.002127472872737846 | 4.9375e-06 |
| H1c_60m__T3_high | T3_high | 48 | 0.0001 | [0.000, 0.000] | -0.000630769230769187 | 0.02723797623201597 | 0.0036587239583333337 |

## H2–H4 — Testes tradeáveis por tercil

| ID | tercil | n | ROI médio | IC 95% bootstrap | hit rate | p raw | p Bonf. | p BH | sobrevive | log-loss | Brier | n holdout | ROI holdout |
|----|--------|---|-----------|------------------|----------|-------|---------|------|-----------|----------|-------|-----------|-------------|
| H2a_10pp_24h__T1_low | T1_low | 4 | -1.0000 | [-1.000, -1.000] | 0.000 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H2a_10pp_24h__T2_mid | T2_mid | 5 | -0.6000 | [-1.000, -0.200] | 0.400 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H2a_10pp_24h__T3_high | T3_high | 35 | -0.6180 | [-0.771, -0.450] | 0.371 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 12 | -0.4690973326265481 |
| H2b_0.1pp__T1_low | T1_low | 1 | 0.9229 | [0.923, 0.923] | 1.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H2b_0.1pp__T2_mid | T2_mid | 3 | -1.0000 | [-1.000, -1.000] | 0.000 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H2b_0.1pp__T3_high | T3_high | 2 | -0.3775 | [-1.000, 0.245] | 0.500 | 0.7497 | nan | nan | não | — | — | 0 | — |
| H2b_0.2pp__T1_low | T1_low | 1 | 0.9229 | [0.923, 0.923] | 1.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H2b_0.2pp__T2_mid | T2_mid | 2 | -1.0000 | [-1.000, -1.000] | 0.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H2b_0.2pp__T3_high | T3_high | 1 | 0.2450 | [0.245, 0.245] | 1.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_fav__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_fav__T2_mid | T2_mid | 3 | -0.0522 | [-1.000, 0.613] | 0.667 | 0.6305 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H3a_fav__T3_high | T3_high | 2 | 0.5818 | [0.576, 0.588] | 1.000 | 0.0001 | nan | nan | não | — | — | 0 | — |
| H3a_long__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_long__T2_mid | T2_mid | 1 | -1.0000 | [-1.000, -1.000] | 0.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_long__T3_high | T3_high | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3b_fav__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3b_fav__T2_mid | T2_mid | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3b_fav__T3_high | T3_high | 1 | 0.0001 | [0.000, 0.000] | 1.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H4a_3pp__T1_low | T1_low | 149 | -0.1816 | [-0.315, -0.030] | 0.624 | 0.9963 | 1.0000 | 1.0000 | não | — | — | 91 | -0.16331636722667872 |
| H4a_3pp__T2_mid | T2_mid | 81 | -0.2086 | [-0.418, 0.020] | 0.494 | 0.9744 | 1.0000 | 1.0000 | não | — | — | 21 | -0.3925886123507485 |
| H4a_3pp__T3_high | T3_high | 23 | 0.1296 | [-0.264, 0.531] | 0.652 | 0.2565 | 1.0000 | 1.0000 | não | — | — | 4 | 0.462279642693363 |
| H4a_5pp__T1_low | T1_low | 114 | -0.0289 | [-0.173, 0.116] | 0.693 | 0.6476 | 1.0000 | 1.0000 | não | — | — | 75 | -0.09010598259551869 |
| H4a_5pp__T2_mid | T2_mid | 53 | -0.2672 | [-0.553, 0.054] | 0.396 | 0.9644 | 1.0000 | 1.0000 | não | — | — | 16 | -0.7311764705882353 |
| H4a_5pp__T3_high | T3_high | 24 | 0.3244 | [-0.075, 0.731] | 0.708 | 0.0594 | 1.0000 | 1.0000 | não | — | — | 2 | 0.885869271417585 |
| H4b_3pp__T1_low | T1_low | 50 | -0.4862 | [-0.775, -0.085] | 0.280 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 35 | -0.5441586140947202 |
| H4b_3pp__T2_mid | T2_mid | 73 | -0.2532 | [-0.445, -0.049] | 0.507 | 0.9955 | 1.0000 | 1.0000 | não | — | — | 25 | -0.310996477870803 |
| H4b_3pp__T3_high | T3_high | 45 | -0.1672 | [-0.421, 0.105] | 0.533 | 0.8949 | 1.0000 | 1.0000 | não | — | — | 10 | 0.008527792260047799 |
| H4b_5pp__T1_low | T1_low | 29 | -0.5293 | [-0.752, -0.277] | 0.379 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 20 | -0.6775327042237709 |
| H4b_5pp__T2_mid | T2_mid | 40 | -0.3160 | [-0.545, -0.085] | 0.500 | 0.9970 | 1.0000 | 1.0000 | não | — | — | 18 | -0.3917082484832722 |
| H4b_5pp__T3_high | T3_high | 33 | -0.2709 | [-0.537, 0.011] | 0.485 | 0.9775 | 1.0000 | 1.0000 | não | — | — | 5 | -0.25558620782022096 |

## Veredito

**H1** reportado apenas como diagnóstico de mispricing residual pós-release (compra do vencedor já conhecido). **Não** entra na família de testes ROI nem em sobreviventes.

**H2–H4:** nenhuma variante tradeável passou holdout (n≥3, ROI>0) com p Bonferroni/BH < 0,05. Não há evidência de edge operável pré-release nesta amostra e proxies.

## Limitações

- Volume interpolado entre start/end (proxy; não é book histórico).
- Fidelidade CLOB 60 min → variantes H1 1/5/15 min frequentemente idênticas.
- H2 usa proxies fracos (não FedWatch/consenso).
- Adendo de liquidez no pré-registro é **pós-hoc** (ver arquivo).

*Gerado por `scripts/run_macro_backtest.py` — números medidos, não inventados.*