# Relatório — Backtest Polymarket macro (Fed / inflação)

**Modo:** somente simulação (sem ordens reais).  
**Mercados no manifest:** 469  
**Eventos no holdout (~30%):** 25  
**Liquidez por trades:** sim (cobertura treino=1.0; histórico incompleto={'registered': 469, 'fetch_incomplete': 0, 'gamma_ratio_below_min': 346})  
**Testes ROI na família múltipla:** 16  
**Tercis (volume trades cumulativo, treino):** q33=749.8546642027497, q66=2523.1042916989613  
**Stake simulado:** US$ 25.0  

Pré-registro: [HIPOTESES_MACRO_POLY_PREREG.md](HIPOTESES_MACRO_POLY_PREREG.md)

## Changelog vs v3 (re-review Opus 5559976794)

| Correção | Efeito |
|----------|--------|
| Trades data-api: paginação `before`, YES+NO, meta vs Gamma | Volume completo; mercados incompletos sinalizados |
| H3: volume zero → T1_low (não drop silencioso) | Cobertura H3 em `h3_coverage_stats` |
| Tercis H2b/H4 no instante de entrada (treino) | Segmentos alinhados à regra de trade |
| H3b favorito pré-reg **0,60** (FOMC) | Restaura limiar documentado |
| Calendário FOMC 2023 | Releases 2023 quando existirem no manifest |

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
- **H3a_fav**: elegíveis=369, trades=12, sem histórico=0, sem preço na abertura=19, trades API incompleto=0, volume zero→T1=0
- **H3a_long**: elegíveis=369, trades=42, sem histórico=0, sem preço na abertura=19, trades API incompleto=0, volume zero→T1=0
- **H3b_fav**: elegíveis=58, trades=6, sem histórico=0, sem preço na abertura=0, trades API incompleto=0, volume zero→T1=0

## H1 — Diagnóstico (não tradeável; não entra em sobreviventes)

| ID | tercil | n | ROI médio | IC 95% | CLV médio | log-loss | Brier |
|----|--------|---|-----------|--------|-----------|----------|-------|
| H1a_60m__T1_low | T1_low | 199 | 0.0373 | [0.013, 0.069] | -0.026218181818181804 | 0.15340632636995946 | 0.0463238040201005 |
| H1a_60m__T2_mid | T2_mid | 100 | 0.0001 | [0.000, 0.000] | -0.0008374999999999633 | 0.32284219350311166 | 0.0895822725 |
| H1a_60m__T3_high | T3_high | 47 | 0.0001 | [0.000, 0.000] | -0.00039999999999995595 | 0.5069548777417768 | 0.17255751063829788 |
| H1b_60m__T1_low | T1_low | 41 | 0.0001 | [0.000, 0.000] | — | 0.18518003203622768 | 0.06183684146341464 |
| H1b_60m__T2_mid | T2_mid | 0 | nan | — | — | — | — |
| H1b_60m__T3_high | T3_high | 1 | 0.0001 | [0.000, 0.000] | — | 3.396209840151116 | 0.93412225 |
| H1c_60m__T1_low | T1_low | 14 | 0.0001 | [0.000, 0.000] | -0.00039999999999995595 | 0.0392674141100219 | 0.005721482142857144 |
| H1c_60m__T2_mid | T2_mid | 3 | 0.0001 | [0.000, 0.000] | — | 0.00284283883471867 | 1.8916666666666665e-05 |
| H1c_60m__T3_high | T3_high | 38 | 0.0001 | [0.000, 0.000] | -0.0008285714285713849 | 0.02028152238231191 | 0.0025146184210526317 |

## H2–H4 — Testes tradeáveis por tercil

| ID | tercil | n | ROI médio | IC 95% bootstrap | hit rate | p raw | p Bonf. | p BH | sobrevive | log-loss | Brier | n holdout | ROI holdout |
|----|--------|---|-----------|------------------|----------|-------|---------|------|-----------|----------|-------|-----------|-------------|
| H2a_10pp_24h__T1_low | T1_low | 36 | -0.6322 | [-0.796, -0.463] | 0.361 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 11 | -0.5235520923091795 |
| H2a_10pp_24h__T2_mid | T2_mid | 2 | -0.5000 | [-1.000, 0.000] | 0.500 | 0.7425 | nan | nan | não | — | — | 0 | — |
| H2a_10pp_24h__T3_high | T3_high | 6 | -0.8333 | [-1.000, -0.500] | 0.167 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 1 | 9.800980098008722e-05 |
| H2b_0.1pp__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H2b_0.1pp__T2_mid | T2_mid | 2 | -0.0385 | [-1.000, 0.923] | 0.500 | 0.7497 | nan | nan | não | — | — | 0 | — |
| H2b_0.1pp__T3_high | T3_high | 4 | -0.6887 | [-1.000, -0.066] | 0.250 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H2b_0.2pp__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H2b_0.2pp__T2_mid | T2_mid | 1 | 0.9229 | [0.923, 0.923] | 1.000 | 1.0000 | nan | nan | não | — | — | 0 | — |
| H2b_0.2pp__T3_high | T3_high | 3 | -0.5850 | [-1.000, 0.245] | 0.333 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H3a_fav__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_fav__T2_mid | T2_mid | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_fav__T3_high | T3_high | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_long__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_long__T2_mid | T2_mid | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3a_long__T3_high | T3_high | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3b_fav__T1_low | T1_low | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3b_fav__T2_mid | T2_mid | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H3b_fav__T3_high | T3_high | 0 | nan | — | nan | 1.0000 | nan | nan | não | — | — | 0 | — |
| H4a_3pp__T1_low | T1_low | 285 | -0.1787 | [-0.309, -0.028] | 0.530 | 0.9975 | 1.0000 | 1.0000 | não | — | — | 134 | -0.17296379539988832 |
| H4a_3pp__T2_mid | T2_mid | 10 | -0.0117 | [-0.383, 0.347] | 0.800 | 0.5586 | 1.0000 | 1.0000 | não | — | — | 5 | 0.15444902922552745 |
| H4a_3pp__T3_high | T3_high | 7 | -0.2499 | [-0.857, 0.674] | 0.429 | 0.6839 | 1.0000 | 1.0000 | não | — | — | 1 | 0.015934959349593398 |
| H4a_5pp__T1_low | T1_low | 221 | -0.1212 | [-0.243, 0.007] | 0.561 | 0.9749 | 1.0000 | 1.0000 | não | — | — | 111 | -0.2851598298862171 |
| H4a_5pp__T2_mid | T2_mid | 5 | 0.1366 | [-0.667, 0.940] | 0.600 | 0.3366 | 1.0000 | 1.0000 | não | — | — | 3 | 0.8942516191057166 |
| H4a_5pp__T3_high | T3_high | 10 | -0.0899 | [-0.677, 0.584] | 0.500 | 0.5824 | 1.0000 | 1.0000 | não | — | — | 1 | -1.0 |
| H4b_3pp__T1_low | T1_low | 101 | -0.2434 | [-0.401, -0.075] | 0.515 | 0.9989 | 1.0000 | 1.0000 | não | — | — | 26 | -0.2521053684352179 |
| H4b_3pp__T2_mid | T2_mid | 38 | -0.5121 | [-0.757, -0.235] | 0.316 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 24 | -0.6228199025234838 |
| H4b_3pp__T3_high | T3_high | 34 | -0.3071 | [-0.701, 0.260] | 0.382 | 0.9149 | 1.0000 | 1.0000 | não | — | — | 20 | -0.27334672518485553 |
| H4b_5pp__T1_low | T1_low | 54 | -0.2878 | [-0.500, -0.068] | 0.481 | 0.9969 | 1.0000 | 1.0000 | não | — | — | 13 | -0.537233577539743 |
| H4b_5pp__T2_mid | T2_mid | 19 | -0.5458 | [-0.811, -0.260] | 0.368 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 14 | -0.6516045518015167 |
| H4b_5pp__T3_high | T3_high | 34 | -0.4354 | [-0.645, -0.218] | 0.471 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 16 | -0.3685564121618333 |

## Veredito

**H1** reportado apenas como diagnóstico de mispricing residual pós-release (compra do vencedor já conhecido). **Não** entra na família de testes ROI nem em sobreviventes.

**H2–H4:** nenhuma variante tradeável passou holdout (n≥3, ROI>0) com p Bonferroni/BH < 0,05. Não há evidência de edge operável pré-release nesta amostra e proxies.

## Limitações

- Volume = soma size×price (tokens YES+NO) até decision_ts; mercados com fetch incompleto excluídos dos tercis.
- H3: ver cobertura acima — maioria dos mercados não gera trade (filtro fav/longshot).
- Fidelidade CLOB 60 min → variantes H1 1/5/15 min frequentemente idênticas.
- H2 usa proxies fracos (não FedWatch/consenso).
- Adendo de liquidez no pré-registro é **pós-hoc** (ver arquivo).

*Gerado por `scripts/run_macro_backtest.py` — números medidos, não inventados.*