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
# Volume mínimo negociado no mercado (US$). Abaixo disso o preço é enfeite: sem
# negociação real, o ask fica desatualizado e o "edge" é ruído. Jogo recém-aberto
# fica de fora até ganhar mercado.
MIN_VOLUME = 1000.0
MODEL_PARAMS = dict(min_games=5, decay_lambda=2.0, regularization_k=15.0)

CLUB_PREFIXES = ("ca ", "se ", "cr ", "sc ", "ec ", "fc ", "fbpa", "ac ", "aa ")
# Só o que o nome da Polymarket não entrega por texto: o prefixo do clube é
# removido e sobra um apelido que não existe no dataset.
#
# NUNCA colocar apelido genérico aqui (tipo 'sao' -> 'sao paulo'): ele sequestra
# qualquer time que comece igual. 'São Bento', 'São Caetano' e 'São José' são
# clubes reais — todos virariam São Paulo, e o monitor compararia o jogo errado
# sem avisar. O alvo do apelido ainda precisa existir no dataset (ver lookup).
ALIASES = {
    "mineiro": "atletico-mg",
    "paranaense": "athletico-pr",
    "flamengo": "flamengo rj",
    "botafogo": "botafogo rj",
}


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def _parse_iso(s: str | None):
    """ISO da Polymarket ('2026-10-11T20:30:00Z') → datetime UTC, ou None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _clean(name: str) -> str:
    """Minúsculas, sem acento, sem prefixo de clube, sem 'Paulista'."""
    import unicodedata

    n = unicodedata.normalize("NFKD", name.strip().lower())
    n = "".join(c for c in n if not unicodedata.combining(c))
    for p in CLUB_PREFIXES:
        if n.startswith(p):
            n = n[len(p):]
    return n.replace("paulista", "").strip()


def normalize(name: str) -> str:
    """Primeira palavra limpa do nome ('SE Palmeiras' → 'palmeiras').

    Sem apelido de propósito: resolver apelido exige saber o que existe no
    dataset, e isso é papel do lookup. Aqui é só limpeza.
    """
    words = _clean(name).split()
    return words[0] if words else _clean(name)


def build_map(teams: list[str]) -> dict[str, str]:
    """Mapeia nome da Polymarket → nome do dataset.

    A chave exata é o nome completo limpo ('gremio prudente'), não a primeira
    palavra: senão 'Gremio' e 'Gremio Prudente' colidem e o Grêmio casaria com
    o time errado. Nome completo sempre vence; a primeira palavra entra só como
    atalho, para nomes compridos tipo 'Associação Chapecoense de Futebol'.

    ponytail: casamento por palavra; colisão residual devolve o time errado.
    Trocar por fuzzy matching (rapidfuzz) se aparecer erro real no log.
    """
    idx: dict[str, str] = {}
    for t in teams:
        idx[_clean(t)] = t  # chave exata, autoritativa
    for t in teams:
        for w in _clean(t).replace("-", " ").split():
            if len(w) > 3:
                idx.setdefault(w, t)
    return idx


def lookup(idx: dict[str, str], name: str) -> str | None:
    """Resolve o nome da Polymarket: nome completo, apelido, depois palavra.

    O apelido só vale se o alvo existir no dataset. É isso que impede um time
    desconhecido de ser silenciosamente tratado como outro — melhor não achar
    (o jogo é ignorado) do que achar errado (compara o jogo errado).
    """
    key = _clean(name)
    if key in idx:
        return idx[key]
    first = key.split()[0] if key.split() else key
    alias = ALIASES.get(first)
    if alias and alias in idx:
        return idx[alias]
    for w in key.replace("-", " ").split():
        if len(w) > 3 and w in idx:
            return idx[w]
    return None


def _open_events(page: int = 100) -> list[dict]:
    """Todos os eventos abertos da tag, não só a primeira página.

    A API corta em 100 por resposta independente do 'limit' pedido. Sem paginar,
    a tag (183 eventos abertos hoje) esconderia jogos que caíssem da página 2 —
    o monitor simplesmente não os veria.
    """
    out: list[dict] = []
    offset = 0
    while True:
        chunk = _get(f"{GAMMA}/events?tag_slug=brazil-serie-a&closed=false"
                     f"&limit={page}&offset={offset}")
        out.extend(chunk)
        if len(chunk) < page or offset > 2000:  # fim, ou trava de segurança
            break
        offset += page
    return out


def today_fixtures(day: date) -> list[dict]:
    """Eventos de Série A da Polymarket com jogo perto da data (qualquer fuso).

    Ignora jogos já iniciados: durante a partida o preço é ao vivo e despenca
    conforme o placar, o que produziria "vantagens" falsas gigantes contra as
    probabilidades pré-jogo do modelo.
    """
    out = []
    now = datetime.now(timezone.utc)
    events = _open_events()
    for e in events:
        slug = e.get("slug", "")
        if not slug.startswith("bra-") or slug.count("-") != 5:
            continue  # só mercados de jogo (bra-xxx-yyy-YYYY-MM-DD)
        if e.get("live") or e.get("ended"):
            continue  # jogo em andamento ou encerrado → preço ao vivo
        kickoff = _parse_iso(e.get("startTime") or e.get("endDate"))
        if kickoff is None or kickoff <= now:
            continue
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
            rows.append({
                "title": m.get("groupItemTitle") or m["question"],
                "price": price,
                "volume": float(m.get("volume") or 0),
                "tradable": bool(m.get("acceptingOrders")),
            })
        if len(rows) == 3:
            out.append({"slug": slug, "end": end, "markets": rows,
                        "closed": all(m.get("closed") for m in ms)})
    return out


def model_probs(model: PoissonModel, home: str, away: str) -> dict[str, float] | None:
    """Probabilidades do modelo já ajustado (não reajustar por jogo)."""
    try:
        p = model.predict_match(home, away, neutral=False)
    except Exception:
        return None
    return {
        "home": p["home_win_probability"],
        "draw": p["draw_probability"],
        "away": p["away_win_probability"],
    }


def outcome_roles(ms: list[dict], casa: str, fora: str) -> list[str] | None:
    """Diz qual mercado é casa/empate/fora, pelo nome — não por posição.

    Devolve os rótulos literais 'casa'/'empate'/'fora' (é o que o log usa como
    'pick'; devolver o nome do time faria o acerto nunca bater). Devolve None se
    não identificar exatamente um de cada, e aí o jogo não é liquidado: melhor
    deixar pendente do que gravar o vencedor errado.
    """
    idx = build_map([casa, fora])
    roles = []
    for m in ms:
        title = m.get("groupItemTitle") or m.get("question") or ""
        if "draw" in title.lower():
            roles.append("empate")
            continue
        who = lookup(idx, title)
        roles.append("casa" if who == casa else "fora" if who == fora else None)
    if None in roles or sorted(roles) != ["casa", "empate", "fora"]:
        return None
    return roles


def settle(log: pd.DataFrame) -> pd.DataFrame:
    """Marca o resultado dos jogos já encerrados na Polymarket.

    Só resolve quando um dos três mercados fechou em ~1.0 e quando a ordem dos
    mercados foi identificada com certeza — se estiver ambíguo, deixa pendente
    em vez de chutar.
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
        roles = outcome_roles(ms, r["casa"], r["fora"])
        if roles is None:
            continue
        prices = []
        for m in ms:
            try:
                prices.append(float(json.loads(m["outcomePrices"])[0]))
            except Exception:
                prices.append(0.0)
        if max(prices) < 0.9:
            continue
        winner = roles[prices.index(max(prices))]
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
        # O placar que importa é o das jogadas que o monitor de fato sinalizaria
        # (edge >= EDGE_MIN). O resto é só registro e dilui o resultado.
        sinal = done[done["edge"] >= EDGE_MIN]
        for _, r in sinal.tail(20).iterrows():
            print(f"  {r['data']} {r['casa']} x {r['fora']}: modelo {r['pick']} "
                  f"({r['p_modelo']:.0%}) mercado {r['p_mercado']:.0%} → {r['resultado']}")
        if len(sinal):
            print(f"\nSó as sinalizadas (edge >= {EDGE_MIN:.0%}): "
                  f"{(sinal['resultado'] == 'acerto').mean():.0%} de acerto "
                  f"| retorno {sinal['retorno'].sum():+.2f}u em {len(sinal)} apostas")
        else:
            print(f"\nNenhuma jogada sinalizada (edge >= {EDGE_MIN:.0%}) resolvida ainda.")
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
    skipped_illiquid = []
    for f in fixtures:
        # o mercado de empate vem como "Draw (A vs. B)"; ordem casa/empate/fora
        home_name = f["markets"][0]["title"]
        away_name = f["markets"][2]["title"]
        home = lookup(idx, home_name)
        away = lookup(idx, away_name)
        if not home or not away or home == away:
            # Nenhum dos dois no dataset → outro campeonato (a tag pega Série B
            # também): ignora calado. Um só mapeado → mapeamento quebrou: avisa.
            if bool(home) != bool(away):
                print(f"sem mapeamento: {home_name} / {away_name} — ignorado")
            continue
        p = model_probs(model, home, away)
        if not p:
            continue
        probs = [p["home"], p["draw"], p["away"]]
        prices = [m["price"] for m in f["markets"]]
        k = max(range(3), key=lambda i: probs[i] - prices[i])
        label = ["casa", "empate", "fora"][k]
        edge = probs[k] - prices[k]
        vol = f["markets"][k]["volume"]
        rows.append({
            "data": str(today), "slug": f["slug"], "casa": home, "fora": away,
            "pick": label, "p_modelo": round(probs[k], 4), "p_mercado": round(prices[k], 4),
            "edge": round(edge, 4), "preco": prices[k], "volume": round(vol),
            "inicio": f["end"], "resultado": None,
        })
        if edge < EDGE_MIN:
            continue
        # Edge grande em mercado vazio é preço desatualizado, não oportunidade.
        if not f["markets"][k]["tradable"] or vol < MIN_VOLUME:
            skipped_illiquid.append(
                f"{home} x {away}: edge {edge:+.0%} mas o mercado "
                f"{'não aceita ordem' if not f['markets'][k]['tradable'] else f'só negociou ${vol:,.0f}'}")
            continue
        alerts.append(f"{home} x {away}: modelo {probs[k]:.0%} vs mercado {prices[k]:.0%} "
                      f"em {label} (edge {edge:+.0%}, preço ${prices[k]:.2f}, "
                      f"volume ${vol:,.0f})")

    if rows:
        new = pd.DataFrame(rows)
        if LOG.exists():
            old = pd.read_csv(LOG)
            # keep="first": o registro pré-jogo é o válido; nunca deixar um
            # snapshot ao vivo sobrescrever o preço de antes da bola rolar.
            new = pd.concat([old, new]).drop_duplicates(subset=["slug"], keep="first")
        new.to_csv(LOG, index=False)
        print(f"{len(rows)} jogos registrados em {LOG.name}")

    if alerts:
        print("\nALERTA — vantagem acima de %.0f%% (não é recomendação de aposta):" % (EDGE_MIN * 100))
        for x in alerts:
            print("  " + x)
    else:
        print(f"Nenhum edge acima de {EDGE_MIN:.0%}. Nada a fazer.")

    if skipped_illiquid:
        print(f"\nDescartados por mercado vazio (edge não é executável):")
        for x in skipped_illiquid:
            print("  " + x)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # self-check: limpeza de nome
        assert normalize("SE Palmeiras") == "palmeiras"
        assert normalize("SC Corinthians Paulista") == "corinthians"
        assert normalize("CA Mineiro") == "mineiro"  # apelido é do lookup, não daqui
        # self-check: jogo já iniciado NUNCA entra na comparação
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        assert _parse_iso(past.isoformat()) < datetime.now(timezone.utc)
        assert _parse_iso("2026-10-11T20:30:00Z") is not None
        assert _parse_iso(None) is None
        # self-check: nome comprido da Polymarket cai no time certo do dataset
        teams = ["Chapecoense-Sc", "Athletico-PR", "Gremio", "Gremio Prudente",
                 "Flamengo Rj", "Botafogo Rj", "Atletico-Mg", "Sao Paulo", "Sao Bento"]
        idx = build_map(teams)
        assert lookup(idx, "Associação Chapecoense de Futebol") == "Chapecoense-Sc"
        assert lookup(idx, "CA Paranaense") == "Athletico-PR"
        assert lookup(idx, "Grêmio FBPA") == "Gremio"
        assert lookup(idx, "CA Mineiro") == "Atletico-Mg"
        assert lookup(idx, "CR Flamengo") == "Flamengo Rj"
        assert lookup(idx, "Botafogo FR") == "Botafogo Rj"
        assert lookup(idx, "Gremio Prudente") == "Gremio Prudente"  # exato vence o atalho
        assert lookup(idx, "Time Inventado FC") is None
        # self-check: apelido genérico não pode sequestrar time que começa igual.
        # Sem o guarda, 'São Bento' viraria 'Sao Paulo' e o monitor compararia o
        # jogo errado em silêncio.
        idx2 = build_map(["Sao Paulo", "Santos"])  # São Bento NÃO está no dataset
        assert lookup(idx2, "São Bento") is None, "apelido sequestrou time desconhecido"
        assert lookup(idx2, "São Paulo FC") == "Sao Paulo"
        # self-check: rótulos casa/empate/fora pelo NOME do mercado, não pela
        # posição. Títulos reais da Polymarket, um empate e duas vitórias.
        def _mk(*titles):
            return [{"groupItemTitle": t} for t in titles]
        ms = _mk("CR Vasco da Gama", "Draw (CR Vasco da Gama vs. Clube do Remo)", "Clube do Remo")
        assert outcome_roles(ms, "Vasco", "Remo") == ["casa", "empate", "fora"]
        ms = _mk("Cruzeiro EC", "Draw (Cruzeiro EC vs. São Paulo FC)", "São Paulo FC")
        assert outcome_roles(ms, "Cruzeiro", "Sao Paulo") == ["casa", "empate", "fora"]
        # ordem trocada: tem que acompanhar, não chutar pela posição
        ms = _mk("Clube do Remo", "Draw (CR Vasco da Gama vs. Clube do Remo)", "CR Vasco da Gama")
        assert outcome_roles(ms, "Vasco", "Remo") == ["fora", "empate", "casa"]
        # time que não bate com o log: recusa liquidar em vez de gravar errado
        ms = _mk("Coritiba", "Draw (Vasco vs. Coritiba)", "Vasco")
        assert outcome_roles(ms, "Vasco", "Remo") is None
    sys.exit(main())
