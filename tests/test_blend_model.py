"""Testes do BlendModel — o que precisa falhar se a mistura de xG quebrar."""

import numpy as np
import pandas as pd
import pytest

from src.poisson_model import PoissonModel
from src.blend_model import BlendModel


def _jogos(n: int = 400, com_xg: bool = True) -> pd.DataFrame:
    """Jogos sintéticos: 'Forte' cria mais e sofre menos que 'Fraco'."""
    rng = np.random.default_rng(7)
    times = ["Forte", "Medio", "Fraco", "Fraquinho"]
    datas = pd.date_range("2023-01-01", periods=n // 4, freq="7D")
    linhas = []
    for i, d in enumerate(datas):
        for h, a in [("Forte", "Fraco"), ("Medio", "Fraquinho"),
                     ("Fraco", "Forte"), ("Fraquinho", "Medio")]:
            forca = {"Forte": 2.0, "Medio": 1.3, "Fraco": 0.8, "Fraquinho": 0.5}
            lam_h = forca[h] * 0.9
            lam_a = forca[a] * 0.6
            linhas.append({
                "date": d, "home_team": h, "away_team": a, "neutral": False,
                "home_goals": int(rng.poisson(lam_h)),
                "away_goals": int(rng.poisson(lam_a)),
                "xg_h": float(lam_h + rng.normal(0, 0.3)),
                "xg_a": float(lam_a + rng.normal(0, 0.3)),
            })
    df = pd.DataFrame(linhas)
    if not com_xg:
        df = df.drop(columns=["xg_h", "xg_a"])
    return df


def test_sem_coluna_xg_falha_alto():
    """Sem xg_h/xg_a o modelo tem que gritar, não silenciosamente usar gols."""
    with pytest.raises(ValueError, match="xg_h"):
        BlendModel().fit(_jogos(com_xg=False))


def test_xg_insuficiente_falha_alto():
    """Poucos jogos com xG = janela sem cobertura; melhor falhar que treinar torto."""
    df = _jogos()
    df.loc[df.index[:380], ["xg_h", "xg_a"]] = np.nan
    with pytest.raises(ValueError, match="não cobre esse período"):
        BlendModel(pesos={"gols": 0.25, "xg": 0.75}).fit(df)


def test_peso_zero_e_igual_ao_modelo_de_gols():
    """blend_xg=0.0 tem que reproduzir o Poisson de gols exatamente.

    É a âncora: se isto divergir, a mistura está mexendo em algo que não devia.
    """
    df = _jogos()
    a = BlendModel(blend_xg=0.0).fit(df)
    b = PoissonModel().fit(df)
    p_a = a.predict_match("Forte", "Fraco", neutral=False)
    p_b = b.predict_match("Forte", "Fraco", neutral=False)
    assert p_a["home_win_probability"] == pytest.approx(p_b["home_win_probability"])


def _jogos_com_xg_exagerado(n: int = 400) -> pd.DataFrame:
    """xG que separa MUITO os times — 'Forte' cria 3,0 e os fracos 0,2.

    O xG tem que variar POR TIME. xG constante em todos os jogos apaga a
    informação de força e a previsão vira a média da liga.
    """
    df = _jogos(n)
    forte = {"Forte", "Medio"}
    df["xg_h"] = np.where(df["home_team"].isin(forte), 3.0, 0.2)
    df["xg_a"] = np.where(df["away_team"].isin(forte), 3.0, 0.2)
    return df


def test_sinal_em_outra_escala_nao_domina():
    """Sinal com magnitude 3x maior não pode dominar a mistura.

    Chute a gol tem média ~4,5/jogo e gol ~1,5. Sem pôr na mesma escala, um
    peso de 0,5 em chute a gol engoliria o de gols. Com 'sot' proporcional a
    'gols', a mistura tem que dar o MESMO resultado que gols puros.
    """
    df = _jogos()
    df["sot_h"] = df["home_goals"] * 3.0
    df["sot_a"] = df["away_goals"] * 3.0
    so_gols = PoissonModel().fit(df).predict_match("Forte", "Fraco")
    misto = BlendModel(pesos={"gols": 0.5, "sot": 0.5}).fit(df).predict_match("Forte", "Fraco")
    assert misto["expected_goals_home"] == pytest.approx(
        so_gols["expected_goals_home"], rel=0.02)


def test_sinal_desconhecido_falha_alto():
    with pytest.raises(ValueError, match="sinal desconhecido"):
        BlendModel(pesos={"gols": 0.5, "chute_de_canto": 0.5}).fit(_jogos())


def test_peso_um_usa_so_xg():
    """blend_xg=1.0 tem que seguir o xG quando ele contradiz o placar."""
    df = _jogos_com_xg_exagerado()
    so_xg = BlendModel(blend_xg=1.0).fit(df)
    so_gols = PoissonModel().fit(df)
    assert (so_xg.predict_match("Forte", "Fraco")["expected_goals_home"]
            > so_gols.predict_match("Forte", "Fraco")["expected_goals_home"])


def test_mistura_fica_entre_os_dois_extremos():
    """w=0.5 tem que cair entre só-gols e só-xG. Fora disso é bug de peso."""
    df = _jogos_com_xg_exagerado()
    gh = PoissonModel().fit(df).predict_match("Forte", "Fraco")["expected_goals_home"]
    gx = BlendModel(blend_xg=1.0).fit(df).predict_match("Forte", "Fraco")["expected_goals_home"]
    meio = BlendModel(blend_xg=0.5).fit(df).predict_match("Forte", "Fraco")["expected_goals_home"]
    assert min(gh, gx) <= meio <= max(gh, gx)
