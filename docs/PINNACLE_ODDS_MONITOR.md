# Pinnacle no monitor / paper RN1

## Fontes

1. **`data/matches_xg.csv`** — histórico com `ps_*` / `psc_*` (atualize com `python update_data.py`).
2. **football-data.co.uk (automático)** — ao rodar `monitor.py`, `load_monitor_dataset()`:
   - anexa jogos **já disputados** após a última data do CSV (temporada corrente, odds `PSH` ou `PPH`);
   - anexa **fixtures futuros** de `fixtures.csv` (global football-data) com `PSH`/`PPH`/`AvgCH`, mais CSVs de temporada quando houver linha sem placar (cache `data/pinnacle_fixtures_live.json`, TTL 6h).

## Limitações

- O CSV da temporada pode **atrasar alguns dias** em relação ao calendário real; jogos de **hoje** só têm fair Pin se a football-data já publicou a linha com odds.
- Ligas fora do mapa `COMP_TO_CODE` em `src/football_data_fixtures.py` não recebem refresh.
- Ligas fora de `fixtures.csv` (ex.: MLS, Brasileirão) continuam só nos CSVs `new/{code}.csv` quando publicados.
- Sem API paga (Pinnacle direto): **PPH** é proxy quando **PSH** não existe no arquivo.

## Comandos

```bash
python scripts/refresh_pinnacle_fixtures.py   # força cache de fixtures
python monitor.py --paper-rn1
python monitor.py --paper-rn1-report
```

## Fill simulado do paper RN1

Nenhuma ordem é enviada. O relatório mostra três simulações, no preço descido à grade `tick_size` do livro (`sim_price`):

- **touch** (otimista) — fluxo taker que alcança esse preço enche a ordem inteira.
- **queue** (central) — esse fluxo só nos alcança depois da fila que já estava no nível. Um print estritamente abaixo do preço esvazia o nível, então o fill é cheio. Sem isso, o fill é parcial.
- **through** (pessimista) — só enche se algum print ficar estritamente abaixo do preço simulado.

Quando os três estão medidos, touch ≥ queue ≥ through.

O fluxo que consome um bid YES é lido com `takerOnly=true`: venda do YES a preço ≤ `sim_price`, mais compra do NO a preço ≥ 1 − `sim_price` (mint; o tamanho entra 1:1 em shares do YES). Sem `tick_size` no livro, `queue_ahead` fica vazio — não gravamos 0 inventado.

A regra antiga (`fill_simulado`: último preço do CLOB até o kickoff ≤ bid) continua no CSV. O retorno dela usa o bid cru; o retorno dos três modelos usa `sim_price`.

O monitor grava o livro em `data/paper_rn1_tapes/` quando o sinal nasce. Um slug que entrou só na faixa (`sinal` falso, sem `placed_at`) e depois vira sinal ganha esse snapshot; um slug que já tem `placed_at` não é regravado. Se o menor ask do snapshot já está ≤ `sim_price`, a limite cruzaria na hora (`entry_cross`) e os três modelos enchem por inteiro.

A fita fica `partial` até o kickoff, `complete` quando uma leitura sem erro termina depois do kickoff, e congela uma hora depois. Fita congelada não é consultada de novo e um 429 posterior não rebaixa a cobertura. Janela longa é fatiada; 429 espera e tenta de novo; página vazia no meio da paginação ou offset no teto deixam a fita incompleta. Buraco não vira zero. O tamanho da ordem é suposição de US$ 25 de stake (shares = 25 / `sim_price`), não é tamanho medido.
