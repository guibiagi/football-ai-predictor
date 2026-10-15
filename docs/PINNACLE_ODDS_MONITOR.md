# Pinnacle no monitor / paper RN1

## Fontes

1. **`data/matches_xg.csv`** — histórico com `ps_*` / `psc_*` (atualize com `python update_data.py`).
2. **football-data.co.uk (automático)** — ao rodar `monitor.py`, `load_monitor_dataset()`:
   - anexa jogos **já disputados** após a última data do CSV (temporada corrente, odds `PSH` ou `PPH`);
   - tenta **fixtures futuros** sem placar com `PSH`/`PPH`/`AvgCH` (cache `data/pinnacle_fixtures_live.json`, TTL 6h).

## Limitações

- O CSV da temporada pode **atrasar alguns dias** em relação ao calendário real; jogos de **hoje** só têm fair Pin se a football-data já publicou a linha com odds.
- Ligas fora do mapa `COMP_TO_CODE` em `src/football_data_fixtures.py` não recebem refresh.
- `fixtures.csv` global da football-data costuma estar vazio; dependemos dos CSVs por liga/temporada.
- Sem API paga (Pinnacle direto): **PPH** é proxy quando **PSH** não existe no arquivo.

## Comandos

```bash
python scripts/refresh_pinnacle_fixtures.py   # força cache de fixtures
python monitor.py --paper-rn1
```
