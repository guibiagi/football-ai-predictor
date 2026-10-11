"""xg_model.py — Poisson treinado em xG em vez de placar.

Gols são uma amostra ruidosa da força de um time: um 1x0 com 3,0 de xG não é um
time pior que o 3x0 com 1,0. O placar mede o que aconteceu; o xG mede a qualidade
das chances criadas — é a informação que o mercado já usa e que o modelo de gols
puro não tinha.

Como funciona sem mudar a matemática do Poisson: o xG é calibrado nos gols no
nível da liga (média de xG ≈ média de gols — verificado em update_xg.py), então
o λ estimado a partir de xG já sai em unidade de gols. Nada de fator de
conversão; só troca-se a coluna que alimenta o fit.

Jogos sem xG são descartados do treino (não preenchidos com o placar, que
misturaria duas escalas de ruído diferentes).
"""

from __future__ import annotations

import pandas as pd

from src.poisson_model import PoissonModel

MIN_JOGOS_XG = 100


class XGModel(PoissonModel):
    """Poisson treinado numa mistura de xG e gols; a previsão continua sendo gols.

    Args:
        blend_xg: peso do xG no sinal de treino. 1.0 = só xG, 0.0 = só gols.
            Puro xG descarta informação real (o placar também diz algo sobre o
            time) e mediu-se levemente pior que gols; puro gols ignora a
            qualidade da chance. O meio é onde a literatura costuma apontar o
            melhor resultado, e é o que se mede aqui em vez de assumir.
    """

    def __init__(self, *args, blend_xg: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.blend_xg = blend_xg

    def fit(self, df: pd.DataFrame) -> XGModel:
        if "xg_h" not in df.columns or "xg_a" not in df.columns:
            raise ValueError(
                "XGModel precisa das colunas xg_h/xg_a — rode update_xg.py e use "
                "data/matches_xg.csv"
            )
        d = df.copy()
        d = d[d["xg_h"].notna() & d["xg_a"].notna()]
        if len(d) < MIN_JOGOS_XG:
            raise ValueError(
                f"só {len(d)} jogos com xG na janela de treino (mínimo "
                f"{MIN_JOGOS_XG}) — o xG não cobre esse período"
            )
        # O fit lê home_goals/away_goals; é aqui que a mistura entra no lugar do
        # placar puro.
        w = self.blend_xg
        d["home_goals"] = w * d["xg_h"] + (1.0 - w) * d["home_goals"]
        d["away_goals"] = w * d["xg_a"] + (1.0 - w) * d["away_goals"]
        return super().fit(d)
