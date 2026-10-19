# Rodada 5 — Validação H4-NEG (neg-risk)

**Gerado:** 2026-10-15T14:04:52.373697+00:00

## Pergunta

O ROI proxy **+52%** (n=863) da Rodada 4 sobrevive com **ASK/BID reais**, **profundidade simultânea** em todas as pernas, **taxas taker** e **sincronização**?

## Resumo

| Métrica | Valor |
|---------|-------|
| Eventos negRisk analisados (trades) | 80 |
| Eventos com ≥1 snapshot executável | 0 |
| Snapshots executáveis (todas pernas) | 0 |
| Eventos com desvio proxy (preço resolução) | 0 |
| Taxa falso positivo proxy→real | 0 |
| Lucro líquido total (snapshots) | 0 USDC |
| ROI médio realista | None |
| IC 95% ROI | None |
| Oportunidades/semana (estimativa) | 0.0 |
| Oportunidades **agora** (live book) | 0 |

## Conclusão

**Não executável de forma confiável.** O proxy da Rodada 4 mistura preços assíncronos e slippage fixo; com trades/order book sincronizados, **quase nenhuma** oportunidade passa nos limiares de lucro e tamanho. Varredura ao vivo atual não sustenta arb recorrente.

## Por evento (top lucro)

| Slug | Pernas | Snapshots | Max lucro USDC | Max tamanho USDC | UMA |
|------|--------|-----------|----------------|------------------|-----|

## Live agora

_Nenhuma oportunidade líquida positiva na varredura atual._

## Mecânica e riscos

- **Capital travado** até resolução UMA; ROI anualizado depende de `days_to_resolution`.
- **Resolução ambígua/disputa** não modelada quantitativamente; ver flag UMA por evento.
- **Neg-risk**: arb comprando YES em todas as pernas exige execução simultânea; conversão NO→YES pode reduzir margem de colateral mas não foi simulada atomically.

```bash
python scripts/run_round5_negrisk.py
python monitor.py --paper-negrisk  # se habilitado
```

