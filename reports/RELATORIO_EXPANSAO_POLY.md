# Expansão Polymarket (além de futebol)

Gerado: 2026-10-15T11:53:30.285016+00:00

## Geobloqueio Brasil

- **Status:** close-only (frontend e API) (código `BR`)
- **Documentação oficial:** [https://docs.polymarket.com/api-reference/geoblock](https://docs.polymarket.com/api-reference/geoblock)
- Usuários no Brasil podem fechar posições existentes, mas não abrir novas.

## Taxas
Referência: [https://docs.polymarket.com/polymarket-learn/trading/fees](https://docs.polymarket.com/polymarket-learn/trading/fees) — retornos **líquidos** usam fórmula taker por categoria.

## Tabela por hipótese (holdout)

| ID | n hold | ROI líquido | IC 95% ROI | Viável PnL (n≥300, IC>0)? |
|----|--------|-------------|------------|---------------------------|
| W1_weather_buy_no_favorite | 1 | -1.0375 | [-1.0375, -1.0375] | não |
| C1_favorite_taker_yes_90 | 9 | 0.0483 | [0.0269, 0.0707] | não |
| C1_favorite_maker_bid_90 | 70 | -0.7592 | [-0.8555, -0.6469] | não |
| P1_favorite_taker_yes_90 | 12 | 0.0238 | [0.0156, 0.0328] | não |
| P1_favorite_maker_bid_90 | 66 | -0.8298 | [-0.9149, -0.7276] | não |
| B1_btc_low_prob_vs_iv | 0 | — | — | não |

## Maker futebol — rewards na API (hoje)

- Mercados abertos escaneados: 19419
- Com `clobRewards` ativo: 887
- Amostra fechada: 0.0% com rewards (mediana daily rate USDC: None)

## Conclusão honesta

Ver JSON completo para gross vs net e variantes de horizonte (clima). Critério operacional: **PnL líquido** no holdout com IC 95% acima de zero e n≥300 quando possível.
