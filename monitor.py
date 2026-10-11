#!/usr/bin/env python3
"""monitor.py — forward test do modelo contra o mercado. NÃO APOSTA.

Compara as probabilidades do modelo Poisson com os preços da Polymarket para os
jogos de Série A do dia, registra tudo em data/forward_test.csv e imprime um
alerta APENAS quando o modelo vê vantagem acima do limite.

Por que só monitora: o backtest do modelo deu ROI negativo, e um edge achado em
backtest é overfitting até prova em contrário. O único edge confiável é o que
aparece em teste ao vivo, jogo a jogo. Este script acumula essa evidência.

Uso:
    python monitor.py            # roda, registra e alerta se houver edge
    python monitor.py --report   # só mostra o histórico acumulado
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data_loader import load_matches  # noqa: E402
from src.poisson_model import PoissonModel  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
DATA = Path(__file__).resolve().parent / "data" / "matches_brasileirao.csv"
LOG = Path(__file__).resolve().parent / "data" / "forward_test.csv"
# Limite alto de propósito: o modelo erra mais que o mercado, então exigir muito
# evita alerta por ruído. Ajustar só com evidência do próprio forward test.
EDGE_MIN = 0.08
MODEL_PARAMS = dict(min_games=5, decay_lambda=2.0, regularization_k=15.0)

CLUB_PREFIXES = ("ca ", "se ", "cr ", "sc ", "ec ", "fc ", "fbpa", "ac ", "aa ")
# Nomes que a Polymarket usa e que não casam por texto com o dataset.
ALIASES = {
    "mineiro": "atletico-mg",
    "paranaense": "athletico-pr",
    "botafogo": "botafogo rj",
    "flamengo": "flamengo rj",
    "corinthians": "corinthians",
    "gremio": "gremio",
    "sao": "sao paulo",
}


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def normalize(name: str) -> str:
    """Nome do time → chave comparável ('SE Palmeiras' → 'palmeiras')."""
    import unicodedata

    n = unicodedata.normalize("NFKD", name.strip().lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    for p in CLUB_PREFIXES:
        if n.startswith(p):
            n = n[len(p):]
    n = n.replace("paulista", "").strip()
    first = n.split()[0] if n.split() else n
    return ALIASES.get(first, first)


def build_map(teams: list[str]) -> dict[str, str]:
    """Mapeia nome da Polymarket → nome do dataset.

    ponytail: casamento por primeira palavra (com apelidos); se dois times
    colidirem, o mapeamento devolve o errado e o jogo é ignorado. Trocar por
    fuzzy matching (rapidfuzz) se aparecer colisão real no log.
    """
    return {normalize(t): t for t in teams}


def today_fixtures(day: date) -> list[dict]:
    """Eventos de Série A da Polymarket com jogo perto da data (qualquer fuso)."""
    out = []
    events = _get(f"{GAMMA}/events?tag_slug=brazil-serie-a&closed=false&limit=100")
    for e in events:
        slug = e.get("slug", "")
        if not slug.startswith("bra-") or slug.count("-") != 5:
            continue  # só mercados de jogo (bra-xxx-yyy-YYYY-MM-DD)
        end = e.get("endDate") or ""
        if end[:10] not in {str(day), str(day + timedelta(days=1)), str(day - timedelta(days=1))}:
            continue
        ms = e.get("markets", [])
        if len(ms) != 3:
            continue
        rows = []
        for m in ms:
            try:
                price = float(json.loads(m["outcomePrices"])[0])
            except Exception:
                continue
            rows.append((m.get("groupItemTitle") or m["question"], price))
        if len(rows) == 3:
            out.append({"slug": slug, "end": end, "markets": rows,
                        "closed": all(m.get("closed") for m in ms)})
    return out


def model_probs(home: str, away: str) -> dict[str, float] | None:
    df = load_matches(DATA)
    m = PoissonModel(**MODEL_PARAMS).fit(df)
    try:
        p = m.predict_match(home, away, neutral=False)
    except Exception:
        return None
    return {
        "home": p["home_win_probability"],
        "draw": p["draw_probability"],
        "away": p["away_win_probability"],
    }


def settle(log: pd.DataFrame) -> pd.DataFrame:
    """Marca o resultado dos jogos já encerrados na Polymarket.

    Só resolve quando um dos três mercados fechou em ~1.0 — se estiver ambíguo,
    deixa pendente em vez de chutar.
    """
    pend = log[log["resultado"].isna()]
    for i, r in pend.iterrows():
        try:
            ev = _get(f"{GAMMA}/events?slug={r['slug']}")
        except Exception:
            continue
        if not ev:
            continue
        ms = ev[0].get("markets", [])
        if len(ms) != 3 or not all(m.get("closed") for m in ms):
            continue
        prices = []
        for m in ms:
            try:
                prices.append(float(json.loads(m["outcomePrices"])[0]))
            except Exception:
                prices.append(0.0)
        if max(prices) < 0.9:
            continue
        winner = ["casa", "empate", "fora"][prices.index(max(prices))]
        acertou = winner == r["pick"]
        preco = float(r["preco"])
        log.at[i, "resultado"] = "acerto" if acertou else "erro"
        log.at[i, "vencedor"] = winner
        log.at[i, "retorno"] = round((1.0 - preco) / preco if acertou else -1.0, 4)
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="só o histórico acumulado")
    a = ap.parse_args()

    # Liquida o que já terminou antes de qualquer coisa (mantém o log em dia).
    if LOG.exists():
        log = settle(pd.read_csv(LOG))
        log.to_csv(LOG, index=False)

    if a.report:
        if not LOG.exists():
            print("Sem histórico ainda.")
            return 0
        df = pd.read_csv(LOG)
        done = df[df["resultado"].notna()]
        print(f"registros: {len(df)} | resolvidos: {len(done)}")
        for _, r in done.tail(20).iterrows():
            print(f"  {r['data']} {r['casa']} x {r['fora']}: modelo {r['pick']} "
                  f"({r['p_modelo']:.0%}) mercado {r['p_mercado']:.0%} → {r['resultado']}")
        if len(done):
            print(f"\nAcerto: {(done['resultado'] == 'acerto').mean():.0%} "
                  f"| retorno acumulado (1 unidade fixa): {done['retorno'].sum():+.2f}u "
                  f"em {len(done)} apostas simuladas")
        return 0

    df = load_matches(DATA)
    model = PoissonModel(**MODEL_PARAMS).fit(df)
    teams = sorted(set(df.home_team) | set(df.away_team))
    idx = build_map(teams)

    today = datetime.now(timezone.utc).date()
    fixtures = today_fixtures(today)
    if not fixtures:
        print("Nenhum jogo de Série A com mercado aberto hoje.")
        return 0

    rows = []
    alerts = []
    for f in fixtures:
        names = [t for t, _ in f["markets"]]
        # o mercado de empate vem como "Draw (A vs. B)"
        home_name = names[0]
        away_name = names[2]
        home = idx.get(normalize(home_name))
        away = idx.get(normalize(away_name))
        if not home or not away or home == away:
            print(f"sem mapeamento: {home_name} / {away_name} — ignorado")
            continue
        p = model_probs(home, away)
        if not p:
            continue
        probs = [p["home"], p["draw"], p["away"]]
        prices = [f["markets"][0][1], f["markets"][1][1], f["markets"][2][1]]
        k = max(range(3), key=lambda i: probs[i] - prices[i])
        label = ["casa", "empate", "fora"][k]
        edge = probs[k] - prices[k]
        rows.append({
            "data": str(today), "slug": f["slug"], "casa": home, "fora": away,
            "pick": label, "p_modelo": round(probs[k], 4), "p_mercado": round(prices[k], 4),
            "edge": round(edge, 4), "preco": prices[k], "inicio": f["end"], "resultado": None,
        })
        if edge >= EDGE_MIN:
            alerts.append(f"{home} x {away}: modelo {probs[k]:.0%} vs mercado {prices[k]:.0%} "
                          f"em {label} (edge {edge:+.0%}, preço ${prices[k]:.2f})")

    if rows:
        new = pd.DataFrame(rows)
        if LOG.exists():
            old = pd.read_csv(LOG)
            new = pd.concat([old, new]).drop_duplicates(subset=["slug"], keep="last")
        new.to_csv(LOG, index=False)
        print(f"{len(rows)} jogos registrados em {LOG.name}")

    if alerts:
        print("\nALERTA — vantagem acima de %.0f%% (não é recomendação de aposta):" % (EDGE_MIN * 100))
        for x in alerts:
            print("  " + x)
    else:
        print(f"Nenhum edge acima de {EDGE_MIN:.0%}. Nada a fazer.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # self-check: normalização de nomes reais da Polymarket
        assert normalize("SE Palmeiras") == "palmeiras"
        assert normalize("SC Corinthians Paulista") == "corinthians"
        assert normalize("CA Mineiro") == "atletico-mg"
        assert normalize("Atletico-Mg") == "atletico-mg"
        assert normalize("Grêmio FBPA") == "gremio"
        assert normalize("Gremio") == "gremio"
        assert normalize("CR Flamengo") == "flamengo rj"
        assert normalize("Flamengo Rj") == "flamengo rj"
    sys.exit(main())
