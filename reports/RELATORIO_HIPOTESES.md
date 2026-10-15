# Relatório de hipóteses (CLV vs fechamento Pinnacle)

Gerado em: 2026-10-15T11:34:31.279865+00:00
Holdout: últimos **365** dias (4773 jogos no painel).
Entrada: **Bet365 open** | CLV ref: **Pinnacle close (devig)**

Critério de viabilidade: n≥300 no holdout, CLV médio > 0, IC 95% (bootstrap) inteiro acima de 0.
Bonferroni (referência): α ajustado ≈ 0.0038 para 13 testes.

| Hipótese | n (dev) | CLV dev | IC CLV dev | ROI dev | n (hold) | CLV hold | IC CLV hold | ROI hold | Viável? |
|----------|---------|---------|------------|---------|----------|----------|-------------|----------|---------|
| H1_poisson_1x2_edge8 | 2735 | -0.0032 | [-0.0043, -0.0022] | -0.132 | 1672 | -0.0029 | [-0.0042, -0.0015] | -0.180 | não |
| H2_blend_isotonic_1x2_edge8 | 5355 | -0.0035 | [-0.0043, -0.0028] | -0.172 | 3171 | -0.0035 | [-0.0044, -0.0025] | -0.119 | não |
| H3_poisson_ou_edge8 | 1926 | -0.0034 | [-0.0048, -0.0020] | -0.075 | 1034 | 0.0000 | [-0.0020, 0.0018] | -0.084 | não |
| H4_pin_open_vs_b365_1x2 | 97 | 0.0267 | [0.0218, 0.0317] | 0.204 | 64 | 0.0246 | [0.0178, 0.0312] | -0.288 | não |
| H5_poly_proxy_pin_open_b365 | 11 | 0.0388 | [0.0278, 0.0507] | 0.140 | 22 | 0.0182 | [0.0016, 0.0333] | -0.480 | não |
| H6_pin_open_vs_b365_edge5 | 4 | 0.0426 | [0.0294, 0.0572] | 0.362 | 6 | 0.0378 | [0.0129, 0.0654] | -0.500 | não |
| H7_lower_tier_poisson_ou_edge8 | 298 | -0.0056 | [-0.0089, -0.0027] | -0.121 | 127 | 0.0041 | [-0.0005, 0.0085] | -0.113 | não |
| H8_brasileirao_poisson_1x2_edge8 | 0 | nan | [nan, nan] | nan | 0 | nan | [nan, nan] | nan | não |
| H9_odds_band_18_35_poisson | 1102 | -0.0038 | [-0.0057, -0.0018] | -0.130 | 678 | -0.0016 | [-0.0040, 0.0007] | -0.166 | não |
| H10_poisson_ou_edge5 | 3645 | -0.0023 | [-0.0033, -0.0014] | -0.092 | 2029 | -0.0004 | [-0.0017, 0.0010] | -0.059 | não |
| H11_movement_gbm_1x2 | 0 | nan | [nan, nan] | nan | 0 | nan | [nan, nan] | nan | não |
| H12_xg_boost_ou | 0 | nan | [nan, nan] | nan | 0 | nan | [nan, nan] | nan | não |
| H4b_pin_open_vs_b365_1pp | 3377 | 0.0154 | [0.0145, 0.0163] | 0.013 | 1707 | 0.0128 | [0.0116, 0.0140] | -0.001 | **sim** |

## Descrições

- **H1_poisson_1x2_edge8**: Poisson 1X2 vs B365 abertura, edge≥8%
- **H2_blend_isotonic_1x2_edge8**: Blend isotônico + mercado (1X2), edge≥8%
- **H3_poisson_ou_edge8**: Poisson O/U 2.5 vs B365 abertura, edge≥8%
- **H4_pin_open_vs_b365_1x2**: Pinnacle abertura > B365 abertura ≥3pp (1X2), execução B365
- **H5_poly_proxy_pin_open_b365**: Proxy Polymarket: fair Pinnacle abertura vs exec B365 ≥4pp
- **H6_pin_open_vs_b365_edge5**: Pinnacle abertura > B365 ≥5pp (1X2)
- **H7_lower_tier_poisson_ou_edge8**: O/U Poisson, edge≥8%, só divisões inferiores / ligas menores
- **H8_brasileirao_poisson_1x2_edge8**: Brasileirão A: Poisson 1X2 edge≥8%
- **H9_odds_band_18_35_poisson**: Poisson 1X2, edge≥8%, odds decimais 1.8–3.5 no pick
- **H10_poisson_ou_edge5**: Poisson O/U edge≥5% (filtro mais solto)
- **H11_movement_gbm_1x2**: GBM prevê CLV (lado pin−b365), mensal walk-forward
- **H12_xg_boost_ou**: GBM O/U (λ Poisson + mercado abertura) prevê CLV
- **H4b_pin_open_vs_b365_1pp**: Pinnacle abertura > B365 abertura ≥1pp (1X2) — timing cross-book

## Conclusão

Estratégia(s) passaram o critério de holdout: H4b_pin_open_vs_b365_1pp.
**Atenção:** H4b executa na **Bet365 abertura**, não na Polymarket. É sinal de *cross-book* (Pinnacle abertura como fair), alinhado com a literatura de que edge direcional vs Pinnacle na Poly é fraco; não ativa o monitor Polymarket automaticamente.

## Recomendação de stake

Para **H4b** (se operar manualmente na B365): stake fixa 0,5–1% da banca por aposta, máximo 3 apostas simultâneas; edge mínimo 1pp Pinnacle abertura vs B365 abertura no mesmo resultado; parar se CLV rolling 100 apostas ≤ 0. **Não** escalar sem forward test.
## Hipóteses maker Polymarket (CLOB prices-history)

Fonte: `data/polymarket_closed_matches.json` | Holdout desde 2025-10-09

| Cenário | Fair | Spread | n hold | CLV hold | IC CLV | ROI hold | IC ROI | CLV ok | PnL ok |
|---------|------|--------|--------|----------|--------|----------|--------|--------|--------|
| H13_maker_pin_open_spread2c | Pinnacle open | 0.02 | 507 | 0.01241 | [0.01032, 0.01452] | -0.0621 | [-0.159, 0.0382] | sim | não |
| H14_maker_pin_open_spread1c | Pinnacle open | 0.01 | 807 | 0.00791 | [0.00639, 0.00951] | -0.1049 | [-0.1814, -0.0332] | sim | não |
| H15_maker_pin_open_spread3c | Pinnacle open | 0.03 | 289 | 0.01725 | [0.01421, 0.02026] | -0.0572 | [-0.1738, 0.0551] | não | não |
| H16_maker_pin_close_oracle_spread2c | Pinnacle close (oracle — só sensibilidade) | 0.02 | 566 | 0.02 | [0.02, 0.02] | -0.0588 | [-0.1462, 0.0355] | sim | não |

**Nota metodológica:** fair por abertura Pinnacle (`ps_*`) é o cenário honesto para cotação pré-jogo; fair por fechamento (`psc_*`) é cenário-oráculo (sensibilidade). Fill: primeiro candle onde preço cruza bid/ask. Rewards: fração conservadora do `rewardsDailyRate` quando spread ≤ `rewardsMaxSpread`.

**Leitura vs pesquisa:** confirma o consenso — taker direcional vs Pinnacle na Poly não passou; maker tem **CLV positivo** nas cotações preenchidas, mas **PnL negativo** no holdout (IC do ROI abaixo de zero em spreads apertados), típico de *adverse selection*. Sem modelo de rewards observado no histórico, **não há estratégia Polymarket operável** pelo critério PnL; continuar só com forward test / coleta de rebates reais.

### Síntese final

| Abordagem | CLV holdout | PnL holdout | Operável? |
|-----------|-------------|-------------|-----------|
| Poisson / blend vs Poly (proxy B365) | ≤ 0 | negativo | não |
| H4b cross-book Pin→B365 (taker) | +1,3pp | ~0 | só B365, fora do monitor |
| Maker Poly vs fair Pin (passivo) | +0,8–1,2pp | negativo | não (sem rewards) |
