# Changelog — correções pós-review PR #5

## Bugs corrigidos

1. **p-value bootstrap**: ROI ≤ 0 agora dá p ≈ 1 (antes invertia e gerava falsos sobreviventes).
2. **Devig vazado**: H1d/H3–H5 usam multiplicativo; H1b usa power; H1_shin usa Shin; CLV com o mesmo devig na abertura e fechamento.
3. **H4 NaN**: `nanmean` no drift; jogos sem histórico finito não apostam.
4. **Holdout**: última temporada **completa** por liga; temporada em andamento fora do treino/holdout.
5. **H5b**: ligas baixo gols definidas só com dados de treino (sem holdout/em curso).
6. **H2**: removido como família separada (equivalente a H1 com `edge/w`).
7. **Variantes**: Shin, power com todos os limiares, H3 `Avg`, testes por liga; família de testes múltiplos ampliada.
8. **White RC**: renomeado para bootstrap do melhor ROI (não é Hansen SPA).

## Números vs relatório anterior (commit 378b26b)

| Métrica | Antes | Depois |
|---------|-------|--------|
| n jogos OOS | 12717 | 12717 |
| Log-loss modelo | 0.6869 | 0.6869 |
| Log-loss Pinnacle | 0.6718 | 0.6718 |
| Sobreviventes | 1 (H2_w0.25 holdout, artefato) | 0 |
| Teste nulo melhor ROI | p=0.5067 | p=0.9220 |

O sobrevivente anterior (H2_w0.25 holdout, n=9, q_BH=0,082) desaparece com o p-value corrigido (q→1,0).
