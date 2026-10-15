# Expansão Polymarket (além de futebol) — rodada atual

Gerado: 2026-10-15 (agente)

## Geobloqueio Brasil

| Item | Detalhe |
|------|---------|
| Status | **Close-only** no frontend **e** na API |
| Código ISO | `BR` |
| Documentação | [Geographic Restrictions — Polymarket](https://docs.polymarket.com/api-reference/geoblock) |

Usuários no Brasil **podem fechar** posições existentes, **não podem abrir** novas (inclui API). Para o usuário deste repositório: qualquer estratégia “operável” é **pesquisa / forward test fora do BR** ou encerramento de posições legadas.

Taxas taker: [Fees — Polymarket](https://docs.polymarket.com/polymarket-learn/trading/fees). Retornos **líquidos** abaixo descontam `fee = C × feeRate × p × (1-p)` quando aplicável (maker = 0).

---

## Tabela por hipótese (holdout, critério PnL: n≥300 e IC 95% do ROI > 0 quando possível)

| ID | n (hold) | ROI líquido | IC 95% ROI | Fill / notas | Viável PnL? |
|----|----------|-------------|------------|--------------|-------------|
| W1 clima Buy NO (spread 4–10¢, depth ≥$60) | poucos fav. 0.75–0.95 no histórico CLOB | — | — | Amostra ~1 trade favorito em 151 eventos; spread mata edge | **não** |
| C1/P1 favorito ≥0.90 taker | ver `poly_expansion_*.json` | — | — | Direcional vs taxas crypto/politics | **não** |
| B1 BTC vs Deribit IV | baixo | — | — | IV alinhamento aproximado | **não** |
| **H-RN1** maker bid no **favorito** (fair Pin `ps`, faixa 0.50–0.85) | 12–42 **fills** / holdout (não 300) | +0.11 a +0.25 | IC inclui 0 | fill 9–33%; adverse_gap +3 a +9 pp hit vs fair | **não** (n pequeno) |
| H-RN1 faixa **0.60–0.80** (holdout) | 3–10 fills | +0.11 a +0.65 | IC largo | CLV ~0; inspiração RN1 **não replicada** em escala | **não** |
| **H-COH** Dixon-Coles vs preços Poly 1X2 | **10** | **−0.55** | [−1.04, 0.11] | 44 trades totais; 61 eventos aux. indexados; 0 trades O/U “win by 2+” pareados | **não** |
| Maker futebol + rewards API | — | — | — | `clobRewards` esparso no histórico; scan aberto separado | **não** |

### H-RN1 (detalhe por offset abaixo do fair Pinnacle)

| Offset | Fills hold | Fill rate | ROI hold | IC ROI | adverse_gap (hit − fair) |
|--------|------------|-----------|----------|--------|---------------------------|
| 2¢ | 42 | 33% | +0.11 | [−0.18, 0.36] | +4.5 pp |
| 3¢ | 30 | 23% | +0.11 | [−0.23, 0.42] | +3.2 pp |
| 4¢ | 19 | 15% | −0.00 | [−0.42, 0.43] | −3.0 pp |
| 5¢ | 12 | 9% | +0.25 | [−0.27, 0.74] | +9.0 pp |

**Leitura:** em faixa 0.60–0.80 há fills com ROI pontual alto e **seleção adversa positiva** (acerta mais que o fair implícito) em subamostras minúsculas — compatível com ruído, **não** com o +14 pp reportado do RN1. Maker sem taxa taker; sem rewards modelados neste teste.

### Clima (W1)

- Janela 1–12 h antes do pico; Buy NO em buckets YES 0.75–0.95.
- **Spread efetivo 4–10¢** no NO + **profundidade mínima $60** (volume proxy): quase zero oportunidades no cache Gamma atual.
- Não valida arXiv 2610.13593 em escala com nosso histórico CLOB (fidelity 60 min).

### H-COH

- λ home/away via Poisson Dixon-Coles treinado só em jogos **anteriores** à data.
- Sinal: comprar YES Poly quando `p_model − p_poly ≥ 4 pp` (taker sports 5%).
- Mercados “win by 2+” / O/U na mesma fixture: **poucos pareamentos**; coerência operacional veio sobretudo do **1X2 vs modelo**, com PnL holdout negativo.

---

## Conclusão honesta

1. **Nenhuma hipótese nova** (clima, crypto/politics, BTC/Deribit, RN1, COH) passa o critério rigoroso de **PnL líquido** no holdout com **n ≥ 300** e IC 95% acima de zero.
2. **RN1-style** (bid passivo no favorito 0.60–0.80) é a mais **intrigante** para forward test, mas o backtest histórico tem **dezenas** de fills no holdout, não centenas — e IC do ROI sempre cruza zero.
3. **Brasil:** estratégia não implementável localmente para **novas** posições ([doc geoblock](https://docs.polymarket.com/api-reference/geoblock)).
4. **Próximo passo útil:** forward test RN1 com rebates reais + order book (profundidade $0–60), e ampliar cache CLOB de mercados de temperatura (paper usa milhões de trades taker).

## Reprodução

```bash
python3 scripts/run_rn1_coh_quick.py      # H-RN1 + H-COH (cache futebol)
python3 scripts/run_poly_expansion_suite.py  # suíte completa (lenta)
```

Artefatos: `reports/rn1_coh_quick.json`, `reports/poly_expansion_YYYYMMDD.json`.
