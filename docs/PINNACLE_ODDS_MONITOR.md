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

Nenhuma ordem é enviada. O relatório mostra três simulações:

- **touch** (otimista) — uma venda no bid, ou melhor, enche a ordem inteira.
- **queue** (central) — só conta o volume que passa da fila já descansando nesse preço no instante do sinal. Fill parcial entra na conta.
- **through** (pessimista) — só enche se o preço negociar estritamente abaixo do bid.

A regra antiga (`fill_simulado`: último preço do CLOB até o kickoff ≤ bid) continua no CSV.

O monitor grava o livro em `data/paper_rn1_tapes/` quando o sinal nasce e, nas rodadas seguintes, acumula trades públicos da data-api até o kickoff. Se essa janela vier cortada (erro ou teto de páginas), o fill de fila fica **não medido**. Não substituímos buraco por zero. O tamanho da ordem é suposição de US$ 25 de stake (shares = 25 / bid), não é tamanho medido.
