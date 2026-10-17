# Pré-registro — backtest Over/Under 2,5 vs Pinnacle

**Data de registro:** 2026-10-17 (antes de qualquer leitura dos resultados deste estudo).

**Objetivo:** testar se o modelo Poisson/Dixon-Coles (e variantes) gera valor apostável ou
calibrado melhor que a Pinnacle no mercado de total de gols 2,5, em dados históricos
football-data.co.uk. **Somente simulação** — sem apostas reais.

## Dados e mercado

- Fonte: CSVs football-data.co.uk (ligas principais + extras quando houver Pinnacle).
- Mercado: **Over/Under 2,5 gols**.
- Pinnacle: abertura `P>2.5` / `P<2.5`; fechamento `PC>2.5` / `PC<2.5`.
- Comparadores (H3): Bet365 abertura `B365>2.5`, máximo `Max>2.5`, média `Avg>2.5` quando existir.
- Remoção de vig: **multiplicativa** (proporcional) como padrão; variantes **power** e **Shin** em sub-testes.
- Holdout: **última temporada completa por liga** (código `season` no dataset), intocada na escolha de limiares.
- Treino / avaliação in-sample: todas as temporadas anteriores ao holdout.
- Predições do modelo: **janela expansiva** (4 anos iniciais, passo 12 meses), sem lookahead — mesma regra de `backtest_markets.py`.
- Dixon-Coles ρ estimado só em dados anteriores à primeira data prevista de cada liga.

## Hipóteses e variantes (todas serão logadas)

### H1 — Modelo vs Pinnacle abertura

Apostar quando a probabilidade do modelo (P(over 2,5) da matriz Dixon-Coles) excede a
probabilidade justa implícita da Pinnacle **abertura** (sem vig) por um limiar de edge.

| ID | Variante |
|----|----------|
| H1a | Poisson gols puro; edge ∈ {3%, 5%, 7%, 10%}; devig multiplicativa |
| H1b | H1a com devig **power** |
| H1c | Blend xG (`BlendModel` 50% gols / 50% xG) onde xG existir; edge 5% e 10% |
| H1d | H1a com limiar 5% mas odds de **fechamento** Pinnacle (sanity: não é o foco, só contraste) |

Aposta: um lado por jogo (over ou under) com maior edge positivo; stake flat 1u.

### H2 — Modelo encolhido para a Pinnacle (shrinkage)

`p* = w·p_modelo + (1−w)·p_pin_open` (renormalizado no mercado binário); apostar vs abertura Pinnacle com edge > 5%.

| ID | w |
|----|---|
| H2a | 0,75 |
| H2b | 0,50 |
| H2c | 0,25 |

`w` fixo pré-registrado (não otimizado no holdout).

### H3 — Pinnacle justa vs casa “mole”

Usar `p_pin_open` (sem vig) como referência; apostar na **Bet365 abertura** ou **Max abertura** quando
`p_pin_fair × odds_soft − 1 > limiar` (edge ∈ {2%, 5%}).

| ID | Livro |
|----|-------|
| H3a | B365 abertura |
| H3b | Max abertura |

### H4 — Direção do drift Pinnacle (abertura → fechamento)

Sem usar o fechamento do jogo atual: para cada partida, a média **expansiva** (só jogos anteriores na mesma liga)
do sinal `implied_close_over − implied_open_over` define se a liga “steam” para over ou under;
apostar no lado alinhado a esse sinal na abertura Pinnacle (flat 1u, todos os jogos com odds válidas).

| ID | Descrição |
|----|-----------|
| H4a | Seguir média histórica de drift (over se média > 0) |
| H4b | **Fade** (apostar o lado oposto à média) |

### H5 — Segmentos liga / temporada

Subconjuntos definidos **antes** do holdout:

| ID | Segmento |
|----|----------|
| H5a | Início de temporada: primeiros 60 dias do calendário de cada `season` |
| H5b | Ligas “baixo gols”: média de gols no treino abaixo da mediana global |
| H5c | Restante (complemento de H5a) |

Aplicar apenas a estratégia H1a edge 5% dentro de cada segmento.

## Métricas (por hipótese × treino × holdout)

- `n` apostas, ROI simulado (stake 1), **IC 95% bootstrap** (10 000 réplicas, seed 42).
- **CLV** vs Pinnacle fechamento: média de `p_close − p_entry` no lado apostado; % CLV > 0.
- **Log-loss** e **Brier** binários: modelo vs probabilidade Pinnacle **fechamento** sem vig (todas as partidas com odds, não só apostas).
- Calibração: bins decílicos de `p_modelo` vs frequência de over.
- Teste H0: ROI ≤ 0; p-value bootstrap unilateral; ajuste **Bonferroni** e **BH-FDR** sobre **todas** as variantes listadas.
- **White reality check** (bootstrap): distribuição do melhor ROI entre variantes nulas (permutação de resultados dentro da estratégia) — reportar se o melhor observado excede o percentil 95% nulo.

## Critério de “sobrevivência”

Uma variante **sobrevive** no conjunto (treino ou holdout) se:

1. ROI pontual > 0 **e**
2. p-value bootstrap (ROI ≤ 0) < 0,05 **e**
3. p-value ajustado Bonferroni < 0,05 **ou** q-value BH-FDR < 0,10.

Declaração explícita se nenhuma variante sobreviver.

## Limitações pré-declaradas

- Pinnacle abertura ausente em ligas `/new/` — excluídas de H1–H4 que exigem `P>2.5`.
- xG só em 6 ligas; H1c não é global.
- Power/Shin são aproximações no mercado binário; sensibilidade reportada.
- Múltiplos testes reduzem poder; Bonferroni é conservador.
- Não modela limites de stake, recusa de aposta ou comissão.

---

*Este arquivo não será alterado após o início da execução do script `scripts/run_backtest_over_under.py`.*
