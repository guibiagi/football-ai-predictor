# Football AI Predictor (Poisson Dixon-Coles + monitor Polymarket)

Modelo probabilístico de futebol com backtests walk-forward, avaliação contra linhas de fechamento (Pinnacle) e monitor de forward test na Polymarket.

## Como funciona

```
Jogos históricos → forças ataque/defesa (Poisson + Dixon-Coles ρ)
                 → matriz de placares → 1X2, O/U, AH
                 → (opcional) shrink para o mercado + isotônica por liga
```

O **monitor** (`monitor.py`) compara o modelo com preços da Polymarket, grava `data/forward_test.csv` e só alerta com edge executável (preço ask/bid, volume, gate por liga).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Previsão pontual
python3 run_prediction.py "Brazil" "France"

# Backtest de skill vs mercado
python3 backtest_skill.py

# Métricas reproduzíveis (JSON em reports/)
python3 backtest_write_metrics.py

# Simulação acelerada do monitor (últimas 2 temporadas)
python3 backtest_monitor_sim.py

# Monitor Polymarket (não aposta)
python3 monitor.py
python3 monitor.py --report

# Testes
PYTHONPATH=. python3 -m pytest tests/ -v
```

## Estrutura

| Caminho | Função |
|---------|--------|
| `src/poisson_model.py` | Poisson + Dixon-Coles, decay temporal |
| `src/blend_model.py` | Mistura gols / xG / SOT no treino |
| `src/market_blend.py` | Shrink para fechamento sem margem + isotônica |
| `src/team_matching.py` | `rapidfuzz` + `data/team_aliases.json` |
| `src/reporting.py` | JSON de métricas, bootstrap, calibração |
| `monitor.py` | Forward test + CLV (Polymarket + Pinnacle quando há match) |
| `data/monitor_gates.json` | Ligas/mercados elegíveis para alerta |
| `reports/` | `metrics_YYYYMMDD.json`, `monitor_sim_2seasons.json` |

## Forward test e CLV

Colunas novas em `data/forward_test.csv` (linhas antigas continuam válidas):

- `preco_exec` — preço de execução (best ask/bid quando a API informa)
- `preco_poly_fechamento` — último preço CLOB antes do kickoff
- `preco_pinnacle_fech` — probabilidade devig do fechamento Pinnacle (match no dataset)
- `clv_poly`, `clv_pinnacle` — CLV em pontos de probabilidade na pick

`settle()` preenche resultado e CLV ao resolver jogos.

## Gates de alerta

Gerados sem vazamento a partir de backtests:

```bash
python3 scripts/build_monitor_gates.py
```

Só alerta onde `log_loss_gain_vs_close >= -0.005` no histórico walk-forward (por liga e mercado 1X2 / O/U).

## Métricas e comparação

```bash
python3 backtest_write_metrics.py
python3 scripts/compare_reports.py reports/metrics_A.json reports/metrics_B.json
```

## Auditoria de vazamento

- Treino e ρ Dixon-Coles usam apenas jogos **anteriores** à data prevista (`expanding_window_backtest`).
- Shrink/isotônica: calibrados na **primeira metade** do OOS por liga; avaliados na segunda (simulação e `backtest_write_metrics`).
- `reference_date = max(date)` no `fit()` é seguro dentro de cada janela de treino (sem futuro).
- Recalibração O/U em `backtest_totals.py` ainda usa metade inicial do OOS — tratar como otimismo leve ao interpretar.

## Avisos

- Probabilidades, não certezas. ROI de curto prazo é ruído; priorize log-loss e CLV.
- Sem Kelly no monitor — stake fixa nas simulações.
- Não é recomendação de aposta.

## Dados

Principal: `data/matches_xg.csv` (football-data.co.uk + xG/SOT). Internacionais: `data/matches.csv`.
