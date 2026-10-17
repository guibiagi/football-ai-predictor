# Relatório — Backtest Polymarket macro (Fed / inflação)

**Modo:** somente simulação (sem ordens reais).  
**Mercados no manifest:** 469  
**Eventos no holdout (~30%):** 25  
**Liquidez por trades:** não (cobertura treino=0.7276264591439688; proxy=gamma_volume; trade_seg_ok=False)  
**Testes ROI na família múltipla:** 10  
**Tercis (volume trades cumulativo, treino):** q33=34398.645536909884, q66=120424.03598295011  
**Stake simulado:** US$ 25.0  

Pré-registro: [HIPOTESES_MACRO_POLY_PREREG.md](HIPOTESES_MACRO_POLY_PREREG.md)

## Changelog vs v4 (re-review Opus 5560067731)

| Correção | Efeito |
|----------|--------|
| Paginação real: `start`/`end` + `offset`, `takerOnly=false` | `before` descartado; teste de integração no repo |
| Tercis por trades desligados se histórico incompleto | Família **ALL**; slippage via volume Gamma |
| Dedupe `(txHash, asset, side, size)` | Contagem de fills correta |
| H3 na família múltipla (agregado) | Sem células T1 fictícias com n=0 |
| Pré-reg: 75 testes condicionais | Só com segmentação validada |

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
| H1a_60m__ALL | ALL | 346 | 0.0261 | [0.010, 0.047] | -0.0042357378233019456 | 0.25040161142137746 | 0.07597360476878613 |
| H1b_60m__ALL | ALL | 42 | 0.0001 | [0.000, 0.000] | — | 0.2616331227056298 | 0.08260554166666667 |
| H1c_60m__ALL | ALL | 55 | 0.0001 | [0.000, 0.000] | -0.000630769230769187 | 0.024163093901314825 | 0.003194781818181819 |

## H2–H4 — Testes tradeáveis por tercil

| ID | tercil | n | ROI médio | IC 95% bootstrap | hit rate | p raw | p Bonf. | p BH | sobrevive | log-loss | Brier | n holdout | ROI holdout |
|----|--------|---|-----------|------------------|----------|-------|---------|------|-----------|----------|-------|-----------|-------------|
| H2a_10pp_24h__ALL | ALL | 44 | -0.6505 | [-0.792, -0.503] | 0.341 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 12 | -0.4686304626422683 |
| H2b_0.1pp__ALL | ALL | 6 | -0.4697 | [-1.000, 0.171] | 0.333 | 0.9122 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H2b_0.2pp__ALL | ALL | 4 | -0.2045 | [-1.000, 0.591] | 0.500 | 0.6892 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H3a_fav__ALL | ALL | 12 | -0.2261 | [-0.632, 0.208] | 0.500 | 0.8115 | 1.0000 | 1.0000 | não | — | — | 0 | — |
| H3a_long__ALL | ALL | 42 | -1.0000 | [-1.000, -1.000] | 0.000 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 9 | -1.0 |
| H3b_fav__ALL | ALL | 6 | 0.3074 | [0.117, 0.495] | 1.000 | 0.0001 | 0.0010 | 0.0010 | não | — | — | 1 | 0.45062470199919535 |
| H4a_3pp__ALL | ALL | 302 | -0.1526 | [-0.284, -0.002] | 0.536 | 0.9912 | 1.0000 | 1.0000 | não | — | — | 140 | -0.1541835345531918 |
| H4a_5pp__ALL | ALL | 236 | -0.0851 | [-0.208, 0.046] | 0.559 | 0.9064 | 1.0000 | 1.0000 | não | — | — | 115 | -0.2534412267705102 |
| H4b_3pp__ALL | ALL | 173 | -0.2865 | [-0.436, -0.123] | 0.445 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 70 | -0.36914316763062394 |
| H4b_5pp__ALL | ALL | 107 | -0.3564 | [-0.501, -0.204] | 0.458 | 1.0000 | 1.0000 | 1.0000 | não | — | — | 43 | -0.5011026870485518 |

### Lead exploratório (paper-only, não sobrevivente)

**H3b_fav** (FOMC favorito ≥0,60 na abertura+1h): n=6, hit=100%, ROI médio=0.307, holdout n=1 (1 evento). Não passa critério de sobrevivência (holdout n&lt;3); tratar como hipótese para acompanhamento, não edge confirmado.


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