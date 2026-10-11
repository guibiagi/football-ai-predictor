"""blend_model.py — Poisson treinado numa mistura de sinais, não no placar puro.

O placar é uma amostra ruidosa da força de um time: um 1x0 com 3,0 de xG e 8
chutes a gol não é um time pior que o 3x0 com 1,0 de xG e 3 chutes a gol. Cada
sinal mede uma coisa diferente e erra de um jeito diferente:

  gols          — o que aconteceu (ruidoso, mas é o alvo e carrega informação)
  xG            — a qualidade da chance (o mercado já usa; cobre 6 ligas, 2014+)
  chutes a gol  — proxy de chance, disponível em toda a série histórica

Nenhum é estritamente melhor que o outro — mediu-se que xG puro perde para gols
puros em algumas ligas e ganha em outras. Daí a mistura.

Escala: os sinais têm médias diferentes (chute a gol ~4,5/jogo, gol ~1,5). Como
o Poisson só usa RAZÕES (média do time / média da liga), cada sinal é primeiro
posto na escala de gols multiplicando pela razão das médias da liga. Sem isso a
mistura seria dominada pelo sinal de maior magnitude, não pelo mais informativo.
"""

from __future__ import annotations

import pandas as pd

from src.poisson_model import PoissonModel

MIN_JOGOS = 100

# nome do sinal -> (coluna casa, coluna fora)
SINAIS = {
    "gols": ("home_goals", "away_goals"),
    "xg": ("xg_h", "xg_a"),
    "sot": ("sot_h", "sot_a"),
}


class BlendModel(PoissonModel):
    """Poisson cujo sinal de treino é a mistura dos sinais; prevê gols.

    Args:
        pesos: peso de cada sinal, ex. {"gols": 0.25, "xg": 0.75}. Normalizado
            internamente. Sinais com peso 0 são ignorados (e não exigidos no
            dado). Jogos sem todos os sinais usados são descartados do treino —
            misturar um jogo com xG e outro sem deixaria dois níveis de ruído
            na mesma estatística.
    """

    def __init__(self, *args, pesos: dict[str, float] | None = None,
                 blend_xg: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if pesos is None:
            # Compatibilidade: blend_xg=0.75 -> 25% gols, 75% xG.
            b = 0.5 if blend_xg is None else blend_xg
            pesos = {"gols": 1.0 - b, "xg": b}
        self.pesos = pesos

    def fit(self, df: pd.DataFrame) -> BlendModel:
        usar = {k: v for k, v in self.pesos.items() if v > 0}
        if not usar:
            raise ValueError("BlendModel sem nenhum sinal com peso > 0")
        # O placar é sempre necessário como REFERÊNCIA DE ESCALA (é a média dos
        # gols que define o nível), mesmo quando não entra na mistura.
        for c in ("home_goals", "away_goals"):
            if c not in df.columns:
                raise ValueError(f"BlendModel precisa da coluna {c!r} para escala")

        for nome in usar:
            if nome not in SINAIS:
                raise ValueError(f"sinal desconhecido: {nome!r} (use {list(SINAIS)})")
            ch, _ = SINAIS[nome]
            if ch not in df.columns:
                raise ValueError(
                    f"o sinal {nome!r} precisa da coluna {ch!r} — rode update_data.py"
                    " e update_xg.py"
                )

        d = df.copy()
        mascara = pd.Series(True, index=d.index)
        for nome in usar:
            if nome == "gols":
                continue
            ch, ca = SINAIS[nome]
            mascara &= d[ch].notna() & d[ca].notna()
        d = d[mascara]
        if len(d) < MIN_JOGOS:
            raise ValueError(
                f"só {len(d)} jogos com todos os sinais da mistura "
                f"({' + '.join(sorted(usar))}) — mínimo {MIN_JOGOS}; algum sinal "
                f"não cobre esse período"
            )

        total = sum(usar.values())
        h = pd.Series(0.0, index=d.index)
        a = pd.Series(0.0, index=d.index)
        for nome, peso in usar.items():
            ch, ca = SINAIS[nome]
            # põe o sinal na escala de gols: mesma média de liga que o placar
            fh = d["home_goals"].mean() / d[ch].mean()
            fa = d["away_goals"].mean() / d[ca].mean()
            h += (peso / total) * d[ch] * fh
            a += (peso / total) * d[ca] * fa

        d["home_goals"] = h
        d["away_goals"] = a
        return super().fit(d)
