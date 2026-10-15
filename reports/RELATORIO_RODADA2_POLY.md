# Rodada 2 — Polymarket (pesquisa e sinais)

**Gerado:** 2026-10-15T13:06:38.007133+00:00  
**Eventos no pool:** 17074  
**Oportunidades RN1 com histórico:** 135

Critério PnL viável: últimos **365 dias** (holdout), **n ≥ 300**, ROI médio > 0, **IC 95% inferior > 0**.

| Hipótese | n holdout | ROI líquido | IC 95% ROI | PnL viável? | Detalhe |
|----------|-----------|-------------|------------|-------------|---------|
| H-RN1_expanded_rewards | 10 | 0.1061 | [-0.3629, 0.5561] | não | offset 0.02; fill_rate 0.2273; soccer opps 133; outros 2; rewards sim |
| H-RN1_expanded_rewards | 9 | 0.2388 | [-0.1777, 0.6227] | não | offset 0.025; fill_rate 0.2045; soccer opps 133; outros 2; rewards sim |
| H-RN1_expanded_rewards | 5 | 0.2813 | [-0.3803, 0.6873] | não | offset 0.03; fill_rate 0.1136; soccer opps 133; outros 2; rewards sim |
| H-COH_paired | 2458 | -0.0841 | [-0.1645, -0.0075] | não | 67 |
| W2_weather_taker_trades | 1 | -1.038 | [-1.038, -1.038] | não | W2_spread_4c |
| W2_weather_taker_trades | 1 | -1.0365 | [-1.0365, -1.0365] | não | W2_spread_7c |
| W2_weather_taker_trades | 1 | -1.035 | [-1.035, -1.035] | não | W2_spread_10c |

## Forward test em papel (RN1)

- Estratégia: favorito **0,60–0,80** (fair Pinnacle `ps`), bid **2,5¢** abaixo do fair.
- `python monitor.py --paper-rn1` — registra sinais em `data/paper_rn1_forward.csv`.
- `python monitor.py --paper-rn1-report` — liquida jogos encerrados (fill simulado se preço ≤ bid antes do KO) e resume PnL.

## Geoblock Brasil

Trading restrito no BR (código `BR`): [documentação Polymarket](https://docs.polymarket.com/api-reference/geoblock).

## Notas metodológicas

- RN1: fair Pin (futebol) ou preço CLOB ~24h antes (NBA/NFL/tênis/MMA). PnL maker inclui clobRewards estimados.
- Meta n≥300 fills no holdout; amostra limitada pelo cache de prices-history e matching Pin.

## Conclusão

Nenhuma hipótese de **execução taker/maker na Poly** atingiu viabilidade PnL no holdout com n≥300 nesta rodada. O forward test em papel serve para acumular evidência out-of-sample da RN1 em futebol com Pin.

