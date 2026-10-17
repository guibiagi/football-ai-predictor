# Pré-registro — Backtest Polymarket macro (Fed / inflação)

**Data de registro:** 2026-10-17 (UTC)  
**Objetivo:** testar hipóteses de reação lenta e desalinhamento entre Polymarket e informação externa.  
**Regra:** nenhum resultado numérico foi consultado antes deste arquivo.

## Dados e janelas

- **Polymarket:** Gamma API (descoberta), CLOB `prices-history` (fidelidade mínima disponível), data-api trades quando útil para spread.
- **Externos:** timestamps de releases CPI (BLS), PCE (BEA via FRED), comunicados FOMC; séries FRED/ALFRED para valores realizados; proxy de expectativa via mudança em fed funds futures (FRED DFF / mercado) quando aplicável.
- **Execução simulada:** compra taker no ask (ou último trade + meio spread estimado); taxa taker Polymarket **2%** sobre lucro em contratos (modelo simplificado: custo efetivo ≈ `fee_rate * max(0, 1 - p_entry)` por unidade de payout, mais spread).
- **Walk-forward:** ordenação temporal por evento; **holdout** = últimos ~30% dos eventos (não usados na escolha de parâmetros).
- **Métricas por hipótese:** n trades, ROI simulado, IC 95% bootstrap (10k réplicas), hit rate, log-loss e Brier (preço Polymarket vs resultado binário ou bucket), CLV médio (entrada vs preço t+Δ).
- **Múltiplos testes:** Bonferroni e Benjamini–Hochberg (FDR) sobre **todas** as variantes listadas abaixo; p-valor unilateral bootstrap H0: ROI ≤ 0.

## Hipóteses e variantes (lista completa para correção)

| ID | Descrição | Parâmetros (variantes) |
|----|-----------|------------------------|
| **H1a** | Drift pós-release CPI: comprar YES no outcome que o print implica, logo após release BLS | Δ entrada após release: 1, 5, 15, 60 min |
| **H1b** | Drift pós-release PCE: mesma lógica para PCE (BEA) | Δ: 1, 5, 15, 60 min |
| **H1c** | Drift pós-comunicado FOMC (decisão de taxa / statement window) | Δ: 1, 5, 15, 60 min |
| **H2a** | Divergência pré-evento: quando proxy de probabilidade de hike/cut (via preço pré-release em mercado de taxa) difere de Polymarket em >5pp, apostar no lado “externo” | limiar 5pp, 10pp; janela entrada: 24h e 4h antes do evento |
| **H2b** | Divergência pré-CPI: surpresa esperada (consenso FRED/placeholder vs último print) vs preço em mercado bucket CPI | limiar surpresa implícita 0.1pp, 0.2pp MoM |
| **H3a** | Favorite-longshot em buckets CPI: comprar favorito (preço >0.55) vs longshot (<0.25) após abertura do mercado | favorito vs longshot |
| **H3b** | Favorite-longshot em mercados FOMC binários (hike/hold/cut quando existirem) | favorito >0.60 |
| **H4a** | Mean-reversion intradiária: após movimento \|Δp\|>3pp em 30 min sem release, fade | limiar 3pp, 5pp |
| **H4b** | Continuação intradiária pós-movimento grande pré-release (48h antes) | limiar 3pp |
| **H5** | Calibração pura (sem trade rule): Brier/log-loss de preço 1h antes do settle vs outcome | n/a (métrica apenas no holdout) |

**Total de testes de ROI (variantes com regra de trade):**  
H1: 3 event types × 4 Δ = 12; H2a: 2 limiares × 2 janelas = 4; H2b: 2 = 2; H3: 3 = 3; H4: 4 = 4 → **25 testes base** (+ H5 apenas calibração).

### Adendo (2026-10-17) — **PÓS-HOC** (não estava no commit inicial de pré-registro)

> Este bloco foi adicionado após o primeiro pré-registro e antes da reexecução corrigida do pipeline. Conta como exploração; correção múltipla inclui estes segmentos nas versões que os aplicam.

### Segmentação por liquidez (adendo)

- **Dimensão extra:** cada variante de ROI tradeável em **3 tercis** de liquidez (T1/T2/T3), com volume USD **cumulativo de trades (data-api, tokens YES e NO)** até `decision_ts` da regra (cortes **q33/q66** estimados só no **treino**, **por hipótese**, no instante de entrada relevante: H2a 24h antes do release, H2b 4h, H3 abertura+1h, H4 no candle do sinal).
- **Tercis:** quantis em volume USD bruto (não `log1p`); mercados com histórico de trades incompleto (API truncada ou ratio trade/Gamma &lt; limiar) ficam **fora** da segmentação por tercil; volume zero no instante → **T1_low**.
- **Família de testes (células n≥3):** após deduplicação v2, a lista efetiva de segmentos T1/T2/T3 entrou na correção múltipla **após** a revisão v2 (não estava fixada no pré-registro inicial de 25×3).
- **Execução:** stake nominal **US$ 25** (faixa 10–50); spread + impacto linear `0.5 × stake / max(250, 2% volume)` em pontos de probabilidade, mais meio-spread 0,5 pp e taxa taker 2%.
- **Hipótese exploratória do usuário:** reação lenta concentrada em **T1** (baixa liquidez).
- **Correção múltipla atualizada:** 25 × 3 = **75 testes de ROI** entram em Bonferroni e BH (H5 calibração fora da família de ROI).

## Critério de “sobrevivência”

- ROI médio > 0 no **holdout** e p ajustado (Bonferroni ou BH) < 0.05 para aquele teste.
- Relatar ambos os ajustes; sobrevivência exige passar em **pelo menos um** ajuste conservador (Bonferroni) para claim forte; BH para exploração.

## Limitações antecipadas (a confirmar nos dados)

- Amostra de mercados macro fechados pode ser pequena.
- Histórico CLOB pode ser grosso (60 min) em janelas curtas (1 min).
- FedWatch histórico gratuito limitado — proxy via FRED e preços Polymarket apenas.

---

*Fim do pré-registro. Resultados medidos aparecem apenas em `RELATORIO_MACRO_POLY.md` após execução do pipeline.*
