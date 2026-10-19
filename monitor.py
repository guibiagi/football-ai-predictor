#!/usr/bin/env python3
"""monitor.py — forward test do modelo contra o mercado, em várias ligas. NÃO APOSTA.

Para cada liga com mercado aberto na Polymarket, compara as probabilidades do
modelo Poisson com os preços, registra tudo em data/forward_test.csv e imprime
um alerta APENAS quando o modelo vê vantagem executável.

Por que só monitora: o backtest do modelo deu ROI negativo, e um edge achado em
backtest é overfitting até prova em contrário. O único edge confiável é o que
aparece em teste ao vivo, jogo a jogo. Este script acumula essa evidência.

Uso:
    python monitor.py            # roda, registra e alerta se houver edge
    python monitor.py --report   # histórico acumulado
    python monitor.py --ligas    # o que está coberto e o que não está
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
from src.clv import clv_probability  # noqa: E402
from src.data_loader import load_matches  # noqa: E402
from src.football_data_fixtures import (  # noqa: E402
    fetch_season_rows_since,
    load_upcoming_pinnacle_cached,
    lookup_pin_open_row,
    merge_hist_with_fixtures,
)
from src.monitor_gates import load_gates, market_allowed  # noqa: E402
from src.monitor_strategies import load_strategy_config, strategy_enabled  # noqa: E402
from src.odds_utils import market_probs  # noqa: E402
from src.pinnacle_lookup import lookup_1x2_close  # noqa: E402
from src.poisson_model import PoissonModel  # noqa: E402
from src.polymarket_prices import (  # noqa: E402
    closing_price_before_kickoff,
    execution_price_yes,
    parse_token_ids,
)
from src.paper_rn1 import (  # noqa: E402
    PAPER_LOG,
    ensure_log,
    scan_fixture,
    settle_paper,
    summarize_paper,
)
from src.paper_taker_pin import (  # noqa: E402
    PAPER_LOG as PAPER_PIN_LOG,
    ensure_log as ensure_pin_log,
    scan_pin_gap_fixture,
)
from src.paper_negrisk import PAPER_LOG as PAPER_NEG_LOG, ensure_log as ensure_neg_log, scan_live_signals  # noqa: E402
from src.team_matching import _clean, build_map, lookup, normalize  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
# O dataset com xG é o preferido: é superconjunto do de gols (mesmas linhas, mais
# as colunas de xG/chutes), então carregar ele nunca perde informação.
DATA_XG = Path(__file__).resolve().parent / "data" / "matches_xg.csv"
DATA = Path(__file__).resolve().parent / "data" / "matches_football.csv"
LOG = Path(__file__).resolve().parent / "data" / "forward_test.csv"

# Limite alto de propósito: o modelo erra mais que o mercado, então exigir muito
# evita alerta por ruído. Ajustar só com evidência do próprio forward test.
EDGE_MIN = 0.08
# Volume mínimo negociado no mercado (US$). Abaixo disso o preço é enfeite: sem
# negociação real, o ask fica desatualizado e o "edge" é ruído.
MIN_VOLUME = 1000.0
MODEL_PARAMS = dict(min_games=5, decay_lambda=2.0, regularization_k=15.0)

# Liga na Polymarket (tag) → nome da competição no dataset.
# Só entram ligas com histórico no football-data.co.uk, e só tags TESTADAS —
# tag inventada devolve zero em silêncio e a liga some sem aviso.
LEAGUE_TAGS = {
    "premier-league": "Premier League",
    "la-liga": "La Liga",
    "bundesliga": "Bundesliga",
    "ligue-1": "Ligue 1",
    "primeira-liga": "Primeira Liga",
    "brazil-serie-a": "Brasileirao Serie A",
    "mls": "MLS",
    "efl-championship": "Championship",
    "scottish-premiership": "Scottish Premiership",
    "austria-bundesliga": "Austria Bundesliga",
    "denmark-superliga": "Danish Superliga",
    "norway-eliteserien": "Eliteserien",
    "sweden-allsvenskan": "Allsvenskan",
    "poland-ekstraklasa": "Ekstraklasa",
    "chinese-super-league": "Chinese Super League",
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


def _open_events(tag: str, page: int = 100) -> list[dict]:
    """Todos os eventos abertos da tag, não só a primeira página.

    A API corta em 100 por resposta independente do 'limit' pedido, e devolve
    422 quando o offset passa do fim. Sem paginar, jogos ficariam invisíveis.
    """
    out: list[dict] = []
    offset = 0
    while offset <= 1500:
        try:
            chunk = _get(f"{GAMMA}/events?tag_slug={tag}&closed=false"
                         f"&limit={page}&offset={offset}")
        except Exception:
            break  # 422 = passou do fim
        out.extend(chunk)
        if len(chunk) < page:
            break
        offset += page
    return out


def _is_match_event(e: dict) -> bool:
    """Evento é um jogo (1X2) e não um mercado derivado dele?

    Os derivados (resultado do 1º tempo, placar exato, escanteios) vivem em
    slugs com sufixo, que acrescentam hífens. O jogo é o slug de 5 hífens
    (liga-mandante-visitante-AAAA-MM-DD), com o empate no meio.
    """
    slug = e.get("slug") or ""
    if slug.count("-") != 5:
        return False
    ms = e.get("markets") or []
    if len(ms) != 3:
        return False
    return "draw" in (ms[1].get("groupItemTitle") or "").lower()


def _comp_index(df: pd.DataFrame) -> dict[str, str]:
    """Competição do dataset → chave normalizada, para casar o nome da liga."""
    return {_clean(c): c for c in df["competition"].unique()}


def match_competition(names: list[str | None], idx: dict[str, str]) -> str | None:
    """Nome de liga da Polymarket → competição do dataset, ou None.

    Primeiro nome exato; depois conjunto de palavras ('EFL Championship' casa
    com 'Championship'). Exige resultado único — empate devolve None, porque
    liga errada é pior que liga nenhuma.
    """
    for n in names:
        if n and _clean(n) in idx:
            return idx[_clean(n)]
    for n in names:
        if not n:
            continue
        toks = {w for w in _clean(n).split() if len(w) > 3}
        if not toks:
            continue
        hits = set()
        for k, comp in idx.items():
            kt = {w for w in k.split() if len(w) > 3}
            if toks <= kt or kt <= toks:
                hits.add(comp)
        if len(hits) == 1:
            return hits.pop()
    return None


def _fixture_from_event(e: dict, comp: str, day: date, now: datetime) -> dict | None:
    """Evento da Polymarket → jogo comparável, ou None se não servir.

    Ignora jogos já iniciados: durante a partida o preço é ao vivo e despenca
    conforme o placar, o que produziria "vantagens" falsas gigantes contra as
    probabilidades pré-jogo do modelo.
    """
    if not _is_match_event(e):
        return None
    if e.get("live") or e.get("ended"):
        return None
    kickoff = _parse_iso(e.get("startTime") or e.get("endDate"))
    if kickoff is None or kickoff <= now:
        return None
    end = e.get("endDate") or ""
    if end[:10] not in {str(day), str(day + timedelta(days=1)), str(day - timedelta(days=1))}:
        return None
    rows = []
    for m in e["markets"]:
        price = execution_price_yes(m)
        token_yes, _ = parse_token_ids(m)
        rows.append({
            "title": m.get("groupItemTitle") or m["question"],
            "price": price,
            "preco_exec": price,
            "token_yes": token_yes,
            "volume": float(m.get("volume") or 0),
            "tradable": bool(m.get("acceptingOrders")),
            # paper RN1 lê estes campos; o mercado cru não chega no scan.
            "gameStartTime": m.get("gameStartTime") or e.get("startTime") or "",
            "endDate": m.get("endDate") or end,
        })
    if any(r["price"] is None for r in rows):
        return None
    def _iso_z(raw) -> str:
        parsed = _parse_iso(raw if isinstance(raw, str) else None)
        if parsed is None:
            return ""
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Início de verdade em qualquer mercado, antes de cair no endDate.
    kickoff_iso = ""
    start_raws = []
    for raw_m in e.get("markets") or []:
        start_raws.append(raw_m.get("gameStartTime"))
        start_raws.append(raw_m.get("startTime"))
    start_raws.append(e.get("startTime"))
    start_raws.append(e.get("gameStartTime"))
    for raw in start_raws:
        kickoff_iso = _iso_z(raw)
        if kickoff_iso:
            break
    if not kickoff_iso:
        kickoff_iso = _iso_z(e.get("endDate"))
    return {
        "slug": e["slug"], "liga": comp, "end": end, "kickoff": kickoff_iso, "markets": rows,
    }


def collect_fixtures(df: pd.DataFrame, day: date, now: datetime) -> tuple[list[dict], list[str], dict]:
    """Jogos de hoje em todas as ligas cobertas, sem repetir slug.

    Duas fontes: as tags de liga verificadas (garantem as grandes) e a varredura
    da tag 'soccer', que pega o resto e descobre a liga pelo nome do evento.

    Devolve (jogos, tags sospeitosas, quantos jogos abertos cada tag tem). A
    contagem por tag existe para distinguir "a tag está errada" de "hoje não tem
    jogo nessa liga" — sem isso, uma tag quebrada some em silêncio.
    """
    comp_idx = _comp_index(df)
    vistos: set[str] = set()
    out: list[dict] = []
    suspeitas: list[str] = []
    abertos: dict[str, int] = {}

    def add(f: dict | None):
        if f and f["slug"] not in vistos:
            vistos.add(f["slug"])
            out.append(f)

    for tag, comp in LEAGUE_TAGS.items():
        if comp not in set(df["competition"]):
            continue
        total = 0
        for e in _open_events(tag):
            if not _is_match_event(e):
                continue
            total += 1
            add(_fixture_from_event(e, comp, day, now))
        abertos[tag] = total
        if total == 0:
            suspeitas.append(f"{tag} ({comp}): nenhum jogo aberto — tag suspeita")

    for e in _open_events("soccer"):
        names = [(e.get("sport") or {}).get("name")]
        names += [t.get("label") for t in (e.get("tags") or [])]
        comp = match_competition(names, comp_idx)
        if comp:
            add(_fixture_from_event(e, comp, day, now))
    return out, suspeitas, abertos


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
        title = m.get("groupItemTitle") or m.get("question") or m.get("title") or ""
        if "draw" in title.lower():
            roles.append("empate")
            continue
        who = lookup(idx, title)
        roles.append("casa" if who == casa else "fora" if who == fora else None)
    if None in roles or sorted(roles) != ["casa", "empate", "fora"]:
        return None
    return roles


FORWARD_COLS = [
    "data", "liga", "slug", "casa", "fora", "pick", "p_modelo", "p_mercado", "edge",
    "preco", "preco_exec", "preco_poly_fechamento", "preco_pinnacle_fech", "clv_poly",
    "clv_pinnacle", "volume", "inicio", "resultado", "vencedor", "retorno",
]


def _ensure_forward_schema(log: pd.DataFrame) -> pd.DataFrame:
    for c in FORWARD_COLS:
        if c not in log.columns:
            log[c] = None
    return log


def _pin_dataset_row(
    df: pd.DataFrame,
    comp: str,
    home: str,
    away: str,
    today: date,
) -> pd.Series | None:
    """Odds Pinnacle (ps_*) para jogo de hoje: fixtures live ou histórico."""
    row = lookup_pin_open_row(df, comp, home, away, pd.Timestamp(today, tz="UTC"))
    if row is not None:
        return row
    pin_row = df[
        (df.competition == comp)
        & (df.home_team == home)
        & (df.away_team == away)
        & (df.date.dt.date >= today - timedelta(days=3))
        & (df.date.dt.date <= today + timedelta(days=1))
    ].sort_values("date")
    pin_row = pin_row[pin_row[["ps_h", "ps_d", "ps_a"]].notna().all(axis=1)]
    return pin_row.iloc[-1] if not pin_row.empty else None


def load_monitor_dataset() -> pd.DataFrame:
    """Histórico + fixtures futuros com Pinnacle (football-data.co.uk)."""
    path = DATA_XG if DATA_XG.exists() else DATA
    hist = load_matches(path)
    try:
        hist_max = hist["date"].max()
        if hist_max.tzinfo is None:
            hist_max = hist_max.tz_localize("UTC")
        extra = fetch_season_rows_since(hist_max)
        fx = load_upcoming_pinnacle_cached()
        hist = merge_hist_with_fixtures(hist, fx, extra_played=extra)
    except Exception as exc:
        print(f"Aviso: fixtures Pinnacle não carregados ({exc}). Só histórico até {hist['date'].max()}.")
    return hist


def settle(log: pd.DataFrame, hist_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Marca o resultado dos jogos já encerrados na Polymarket.

    Só resolve quando um dos três mercados fechou em ~1.0 e quando a ordem dos
    mercados foi identificada com certeza — se estiver ambíguo, deixa pendente
    em vez de chutar.
    """
    log = _ensure_forward_schema(log)
    log["resultado"] = log["resultado"].astype(object)
    pick_idx = {"casa": 0, "empate": 1, "fora": 2}
    for i, r in log[log["resultado"].isna()].iterrows():
        try:
            ev = _get(f"{GAMMA}/events?slug={r['slug']}")
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
            preco = float(r["preco_exec"] if pd.notna(r.get("preco_exec")) else r["preco"])
            log.at[i, "resultado"] = "acerto" if acertou else "erro"
            log.at[i, "vencedor"] = winner
            log.at[i, "retorno"] = round((1.0 - preco) / preco if acertou else -1.0, 4)

            kickoff = _parse_iso(str(r.get("inicio") or ""))
            pi = pick_idx.get(r["pick"], 0)
            token = None
            if len(ms) > pi:
                token, _ = parse_token_ids(ms[pi])
            if token and kickoff and pd.isna(r.get("preco_poly_fechamento")):
                close_p = closing_price_before_kickoff(token, kickoff)
                if close_p is not None:
                    log.at[i, "preco_poly_fechamento"] = round(close_p, 4)
                    log.at[i, "clv_poly"] = round(clv_probability(preco, close_p), 4)
            if hist_df is not None and pd.isna(r.get("preco_pinnacle_fech")):
                pin = lookup_1x2_close(
                    hist_df, str(r.get("liga", "")), r["casa"], r["fora"], kickoff or r["inicio"],
                )
                if pin:
                    import numpy as np
                    from src.odds_utils import market_probs

                    p_devig = market_probs(np.array([pin], dtype=float))[0, pi]
                    log.at[i, "preco_pinnacle_fech"] = round(float(p_devig), 4)
                    log.at[i, "clv_pinnacle"] = round(clv_probability(preco, float(p_devig)), 4)
        except Exception:
            continue
    return log


def _models(df: pd.DataFrame):
    """Devolve um resolvedor preguiçoso: (modelo, mapa de times) por competição.

    Treina só a liga que tem jogo hoje — treinar as 28 na largada seria lento à
    toa, e o cron roda todo dia.

    Usa gols puros. O BlendModel (xG / chute a gol) existe e está testado, mas
    NÃO é o padrão: a melhora do xG depende de qual amostra se olha (ver
    tune_blend.py) e o chute a gol piora em toda combinação. Efeito que muda de
    sinal conforme a amostra não é vantagem, é ruído — e foi assim que o grid
    search rendeu +2,9% num ano e -11,4% no seguinte. Trocar o padrão só com
    evidência que não dependa da amostra.
    """
    models: dict[str, PoissonModel] = {}
    maps: dict[str, dict] = {}

    def get(comp: str) -> tuple[PoissonModel, dict]:
        if comp not in models:
            sub = df[df["competition"] == comp]
            models[comp] = PoissonModel(**MODEL_PARAMS).fit(sub)
            maps[comp] = build_map(sorted(set(sub.home_team) | set(sub.away_team)))
        return models[comp], maps[comp]

    return get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="histórico acumulado")
    ap.add_argument("--ligas", action="store_true", help="cobertura de ligas")
    ap.add_argument(
        "--maker",
        action="store_true",
        help="cotação passiva (fair Pinnacle ps do dataset) — não aposta",
    )
    ap.add_argument(
        "--paper-rn1",
        action="store_true",
        help="forward test em papel RN1 (favorito 0.60–0.80, bid Pin−2.5¢)",
    )
    ap.add_argument(
        "--paper-rn1-report",
        action="store_true",
        help="relatório do log paper_rn1_forward.csv",
    )
    ap.add_argument(
        "--paper-taker-pin",
        action="store_true",
        help="forward test papel taker Pin−Poly (gap 2¢, só se estratégia habilitada)",
    )
    ap.add_argument(
        "--paper-negrisk",
        action="store_true",
        help="varredura leitura neg-risk (soma YES); grava paper_negrisk_forward.csv",
    )
    a = ap.parse_args()

    df = load_monitor_dataset()

    if a.ligas:
        print("ligas no dataset:")
        for comp, n in df["competition"].value_counts().items():
            print(f"   {n:6d} jogos  {comp}")
        return 0

    if a.paper_negrisk:
        if not strategy_enabled("paper_negrisk", load_strategy_config()):
            print("paper_negrisk não habilitado em data/monitor_strategy.json (Rodada 5).")
            return 0
        sigs = scan_live_signals()
        log = ensure_neg_log()
        if sigs:
            log = pd.concat([log, pd.DataFrame(sigs)]).drop_duplicates(
                subset=["slug", "direction", "data"], keep="first",
            )
        log.to_csv(PAPER_NEG_LOG, index=False)
        print(f"PAPER NEG-RISK: {len(sigs)} sinais → {PAPER_NEG_LOG.name}")
        for s in sigs[:10]:
            print(f"  {s['slug']}: {s['direction']} sum={s['sum_prices']} "
                  f"lucro≈${s['net_profit_usdc']} size≈${s['size_usdc']}")
        return 0

    if a.paper_rn1_report:
        plog = settle_paper(ensure_log(), df)
        plog.to_csv(PAPER_LOG, index=False)
        c = summarize_paper(plog)
        print(
            f"paper RN1: {c['registros']} registros | resolvidos: {c['resolvidos']} "
            f"(fills simulados: {c['fills']}, sem fill: {c['sem_fill']}, pendentes: {c['pendentes']})"
        )
        if c["fills"]:
            hit = "n/a" if c["hit"] != c["hit"] else f"{c['hit']:.0%}"
            print(f"  retorno simulado: {c['retorno']:+.2f}u | hit {hit}")
        return 0

    # Liquida o que já terminou antes de qualquer coisa (mantém o log em dia).
    gates = load_gates()
    strat_cfg = load_strategy_config()
    maker_mode = a.maker or strategy_enabled("maker_pin_fair", strat_cfg)
    if LOG.exists():
        settle(pd.read_csv(LOG), hist_df=df).to_csv(LOG, index=False)
    if a.paper_rn1 or a.paper_rn1_report:
        plog = settle_paper(ensure_log(), df)
        plog.to_csv(PAPER_LOG, index=False)

    paper_pin_enabled = a.paper_taker_pin or any(
        s.get("enabled") and s.get("paper_only")
        and str(s.get("hypothesis_id", "")).startswith("H4-PIN")
        for s in strat_cfg.get("strategies", {}).values()
    )

    if a.report:
        if not LOG.exists():
            print("Sem histórico ainda.")
            return 0
        d = pd.read_csv(LOG)
        done = d[d["resultado"].notna()]
        sinal = done[done["edge"] >= EDGE_MIN]
        print(f"registros: {len(d)} | resolvidos: {len(done)} | sinalizados: {len(sinal)}")
        for _, r in sinal.tail(20).iterrows():
            print(f"  {r['data']} [{r.get('liga','?')}] {r['casa']} x {r['fora']}: "
                  f"{r['pick']} ({r['p_modelo']:.0%} vs {r['p_mercado']:.0%}) → {r['resultado']}")
        if len(sinal):
            print(f"\nSó as sinalizadas (edge >= {EDGE_MIN:.0%}): "
                  f"{(sinal['resultado'] == 'acerto').mean():.0%} de acerto | "
                  f"retorno {sinal['retorno'].sum():+.2f}u em {len(sinal)} apostas")
        else:
            print(f"\nNenhuma jogada sinalizada (edge >= {EDGE_MIN:.0%}) resolvida ainda.")
        return 0

    now = datetime.now(timezone.utc)
    today = now.date()
    get_model = _models(df)

    fixtures, suspeitas, abertos = collect_fixtures(df, today, now)
    rows, alerts, illiquid, sem_time = [], [], [], []
    maker_quotes: list[str] = []
    paper_rows: list[dict] = []
    paper_pin_rows: list[dict] = []
    rn1_with_pin = 0
    rn1_in_band = 0
    rn1_no_pin = 0
    for f in fixtures:
        comp = f["liga"]
        model, idx = get_model(comp)
        home = lookup(idx, f["markets"][0]["title"])
        away = lookup(idx, f["markets"][2]["title"])
        if maker_mode:
            r0 = _pin_dataset_row(df, comp, home, away, today)
            if r0 is not None and r0[["ps_h", "ps_d", "ps_a"]].notna().all():
                fair = market_probs(r0[["ps_h", "ps_d", "ps_a"]].to_numpy(dtype=float).reshape(1, -1))[0]
                hs = float(strat_cfg.get("strategies", {}).get("maker_pin_fair", {}).get("half_spread", 0.02))
                labels = ["casa", "empate", "fora"]
                parts = []
                for i, lab in enumerate(labels):
                    bid = max(0.01, fair[i] - hs)
                    ask = min(0.99, fair[i] + hs)
                    parts.append(f"{lab} bid={bid:.3f} ask={ask:.3f} (fair={fair[i]:.3f})")
                maker_quotes.append(f"[{comp}] {home} x {away}: " + " | ".join(parts))

        if a.paper_rn1 and home and away and home != away:
            roles = outcome_roles(f["markets"], home, away)
            if roles:
                ds = _pin_dataset_row(df, comp, home, away, today)
                if ds is not None and ds[["ps_h", "ps_d", "ps_a"]].notna().all():
                    from src.paper_rn1 import pin_fair_for_fixture

                    rn1_with_pin += 1
                    fav = max(
                        pin_fair_for_fixture(ds, i) or 0.0 for i in range(3)
                    )
                    if 0.60 <= fav <= 0.80:
                        rn1_in_band += 1
                elif a.paper_rn1:
                    rn1_no_pin += 1
                pr = scan_fixture(
                    comp, f["slug"], home, away, f["markets"], roles, ds, str(today),
                    kickoff=f.get("kickoff") or "",
                )
                if pr:
                    paper_rows.append(pr)

        if paper_pin_enabled and home and away and home != away:
            roles = outcome_roles(f["markets"], home, away)
            if roles:
                ds = _pin_dataset_row(df, comp, home, away, today)
                ppr = scan_pin_gap_fixture(
                    comp, f["slug"], home, away, f["markets"], roles, ds, str(today),
                )
                if ppr:
                    paper_pin_rows.append(ppr)

        if not home or not away or home == away:
            # Dizer QUAL lado falhou: listar o par esconde o time problemático.
            faltando = [n for n, t in ((f["markets"][0]["title"], home),
                                       (f["markets"][2]["title"], away)) if not t]
            sem_time.append(f"[{comp}] {f['markets'][0]['title']} x {f['markets'][2]['title']}"
                            f"  ← sem mapeamento: {', '.join(faltando)}")
            continue
        p = model.predict_match(home, away, neutral=False)
        probs = [p["home_win_probability"], p["draw_probability"], p["away_win_probability"]]
        prices = [m["preco_exec"] or m["price"] for m in f["markets"]]
        k = max(range(3), key=lambda i: probs[i] - prices[i])
        label = ["casa", "empate", "fora"][k]
        edge = probs[k] - prices[k]
        vol = f["markets"][k]["volume"]
        preco_exec = prices[k]
        rows.append({
            "data": str(today), "liga": comp, "slug": f["slug"], "casa": home,
            "fora": away, "pick": label, "p_modelo": round(probs[k], 4),
            "p_mercado": round(prices[k], 4), "edge": round(edge, 4),
            "preco": preco_exec, "preco_exec": preco_exec, "volume": round(vol),
            "inicio": f["end"], "resultado": None,
        })
        if not market_allowed(gates, comp, "1X2"):
            continue
        if edge < EDGE_MIN:
            continue
        # Edge grande em mercado vazio é preço desatualizado, não oportunidade.
        if not f["markets"][k]["tradable"] or vol < MIN_VOLUME:
            illiquid.append(f"[{comp}] {home} x {away}: edge {edge:+.0%} mas "
                            f"só negociou ${vol:,.0f}")
            continue
        alerts.append(f"[{comp}] {home} x {away}: modelo {probs[k]:.0%} vs "
                      f"mercado {prices[k]:.0%} em {label} "
                      f"(edge {edge:+.0%}, ${prices[k]:.2f}, volume ${vol:,.0f})")

    if rows:
        new = pd.DataFrame(rows)
        if LOG.exists():
            old = pd.read_csv(LOG)
            # keep="first": o registro pré-jogo é o válido; nunca deixar um
            # snapshot ao vivo sobrescrever o preço de antes da bola rolar.
            new = pd.concat([old, new]).drop_duplicates(subset=["slug"], keep="first")
        new.to_csv(LOG, index=False)
        print(f"{len(rows)} jogos registrados em {LOG.name} "
              f"({len({r['liga'] for r in rows})} ligas)")

    if a.paper_rn1:
        print(
            f"\nPAPER RN1 diagnóstico: {len(fixtures)} jogos Poly hoje | "
            f"{rn1_with_pin} com fair Pin (ps_*/PPH) | {rn1_in_band} na faixa 0.60–0.80 | "
            f"{rn1_no_pin} sem linha Pin no dataset"
        )

    if paper_rows:
        plog = ensure_log()
        newp = pd.DataFrame(paper_rows)
        plog = pd.concat([plog, newp]).drop_duplicates(subset=["slug"], keep="first")
        plog.to_csv(PAPER_LOG, index=False)
        n_sinal = sum(1 for pr in paper_rows if pr.get("sinal") is True)
        print(
            f"PAPER RN1: {len(paper_rows)} jogos na faixa 0.60–0.80 → {PAPER_LOG.name} "
            f"({n_sinal} com sinal: preço ≤ bid+1¢; sinal=False é só a faixa)"
        )
        for pr in paper_rows:
            tok = str(pr.get("token") or "")
            tok_s = (tok[:8] + "…") if len(tok) > 8 else (tok or "—")
            print(
                f"  [{pr['liga']}] {pr['casa']} x {pr['fora']}: fair={pr['fair_pin']:.2f} "
                f"bid={pr['bid']:.2f} poly={pr['poly_exec']:.2f} pick={pr['pick']} "
                f"sinal={bool(pr['sinal'])} kickoff={pr.get('kickoff') or '—'} token={tok_s}"
            )

    if a.paper_rn1 and not paper_rows:
        print("PAPER RN1: nenhum sinal executável na faixa 0.60–0.80 hoje "
              "(volume mínimo / preço Poly).")

    if paper_pin_rows:
        plogp = ensure_pin_log()
        newpp = pd.DataFrame(paper_pin_rows)
        plogp = pd.concat([plogp, newpp]).drop_duplicates(subset=["slug", "pick"], keep="first")
        plogp.to_csv(PAPER_PIN_LOG, index=False)
        print(f"\nPAPER TAKER PIN: {len(paper_pin_rows)} sinais → {PAPER_PIN_LOG.name}")
    elif a.paper_taker_pin:
        print("\nPAPER TAKER PIN: nenhum gap Pin−Poly ≥ 2¢ hoje.")

    if maker_quotes:
        print("\nMAKER (fair Pinnacle ps no dataset, spread ± configurado):")
        for x in maker_quotes:
            print("  " + x)

    if alerts:
        print("\nALERTA — vantagem acima de %.0f%% (não é recomendação de aposta):" % (EDGE_MIN * 100))
        for x in alerts:
            print("  " + x)
    else:
        print(f"Nenhum edge executável acima de {EDGE_MIN:.0%}. Nada a fazer.")

    if illiquid:
        print("\nDescartados por mercado vazio (edge não é executável):")
        for x in illiquid:
            print("  " + x)
    if sem_time:
        print(f"\n{len(sem_time)} jogos sem mapeamento de time (ignorados):")
        for x in sem_time[:10]:
            print("  " + x)
    if suspeitas:
        print("\nLiga coberta mas sem nenhum jogo aberto (tag pode ter mudado):")
        for x in suspeitas:
            print("  " + x)
    com_jogo = [t for t, n in abertos.items() if n]
    if com_jogo:
        print(f"\n{len(com_jogo)} ligas com mercado aberto: " + ", ".join(sorted(com_jogo)))
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
                 "Flamengo Rj", "Botafogo Rj", "Atletico-Mg", "Sao Paulo"]
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
        idx2 = build_map(["Sao Paulo", "Santos"])  # São Bento NÃO está no dataset
        assert lookup(idx2, "São Bento") is None, "apelido sequestrou time desconhecido"
        assert lookup(idx2, "São Paulo FC") == "Sao Paulo"
        # self-check: letras que o NFKD não decompõe (ø, æ, ß, ł). Sem isso,
        # 'Sønderjyske' nunca casa com 'Sonderjyske' e o jogo some calado.
        assert _clean("Sønderjyske Fodbold") == "sonderjyske fodbold"
        assert _clean("Brøndby") == "brondby"
        assert _clean("Malmö FF") == "malmo ff"
        assert _clean("Łódź") == "lodz"
        assert _clean("Preußen") == "preussen"
        idx3 = build_map(["Sonderjyske", "Halmstad", "Elfsborg"])
        assert lookup(idx3, "Sønderjyske Fodbold") == "Sonderjyske"
        # self-check: variante de grafia ('halmstads' vs 'halmstad'), só quando
        # a palavra é longa e o acerto é único
        assert lookup(idx3, "Halmstads BK") == "Halmstad"
        assert lookup(idx3, "IF Elfsborg") == "Elfsborg"
        assert lookup(idx3, "Time Que Nao Existe") is None
        # self-check: Inglaterra — nome por extenso e palavra genérica/ambígua
        idx4 = build_map(["Wolves", "Bolton", "Sheffield United", "Sheffield Weds",
                          "Bristol City", "Stoke", "West Brom", "QPR", "Hull"])
        assert lookup(idx4, "Wolverhampton Wanderers FC") == "Wolves"
        assert lookup(idx4, "Bolton Wanderers FC") == "Bolton"
        assert lookup(idx4, "Sheffield Wednesday FC") == "Sheffield Weds"
        assert lookup(idx4, "Sheffield United FC") == "Sheffield United"
        assert lookup(idx4, "Manchester City FC") is None, "City sequestrou Bristol City"
        assert lookup(idx4, "Stoke City FC") == "Stoke"
        assert lookup(idx4, "Queens Park Rangers FC") == "QPR"
        assert lookup(idx4, "Hull City AFC") == "Hull"
        assert lookup(build_map(["Man City", "Man United"]), "Stoke City FC") is None
        idx5 = build_map(["Real Madrid", "Betis", "Sociedad", "Ath Madrid"])
        assert lookup(idx5, "Real Betis Balompié") == "Betis", "Real sequestrou Betis"
        assert lookup(idx5, "Real Madrid CF") == "Real Madrid"
        assert lookup(idx5, "Club Atlético de Madrid") == "Ath Madrid"
        # self-check: liga da Polymarket → competição do dataset
        ci = _comp_index(pd.DataFrame({"competition": ["Championship", "Serie A",
                                                      "Brasileirao Serie A", "Eliteserien"]}))
        assert match_competition(["EFL Championship"], ci) == "Championship"
        assert match_competition(["Serie A"], ci) == "Serie A"
        assert match_competition(["Brasileirão Série A"], ci) == "Brasileirao Serie A"
        assert match_competition(["Eliteserien"], ci) == "Eliteserien"
        assert match_competition(["Liga De La Cosa"], ci) is None
        # self-check: jogo vs mercado derivado do mesmo jogo
        def _ev(slug, meio):
            return {"slug": slug, "markets": [{"groupItemTitle": "A"},
                                              {"groupItemTitle": meio},
                                              {"groupItemTitle": "B"}]}
        assert _is_match_event(_ev("epl-cov-new-2026-10-12", "Draw (A vs. B)"))
        assert _is_match_event(_ev("lal-lev-bil-2026-09-16", "Draw (A vs. B)"))
        assert not _is_match_event(_ev("epl-cov-new-2026-10-12-halftime-result", "Draw (A vs. B)"))
        assert not _is_match_event(_ev("epl-cov-new-2026-10-12-exact-score", "1-0"))
        # self-check: rótulos casa/empate/fora pelo NOME do mercado, não pela posição
        def _mk(*titles):
            return [{"groupItemTitle": t} for t in titles]
        ms = _mk("CR Vasco da Gama", "Draw (CR Vasco da Gama vs. Clube do Remo)", "Clube do Remo")
        assert outcome_roles(ms, "Vasco", "Remo") == ["casa", "empate", "fora"]
        ms = _mk("Clube do Remo", "Draw (CR Vasco da Gama vs. Clube do Remo)", "CR Vasco da Gama")
        assert outcome_roles(ms, "Vasco", "Remo") == ["fora", "empate", "casa"]
        ms = _mk("Coritiba", "Draw (Vasco vs. Coritiba)", "Vasco")
        assert outcome_roles(ms, "Vasco", "Remo") is None
    sys.exit(main())
