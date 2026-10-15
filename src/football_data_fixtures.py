"""Fixtures futuros + odds Pinnacle (PSH) via football-data.co.uk (gratuito)."""

from __future__ import annotations

import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.team_matching import build_map, load_aliases, lookup

BASE = "https://www.football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0"}
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "pinnacle_fixtures_live.json"
CACHE_TTL_SEC = 6 * 3600

# Competição do monitor → código football-data (temporada mmz4281)
COMP_TO_CODE = {
    "Premier League": "E0",
    "Championship": "E1",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "Primeira Liga": "P1",
    "Brasileirao Serie A": "BRA",
    "MLS": "USA",
    "Scottish Premiership": "SC0",
    "Austria Bundesliga": "AUT",
    "Danish Superliga": "DNK",
    "Eliteserien": "NOR",
    "Allsvenskan": "SWE",
    "Ekstraklasa": "POL",
    "Chinese Super League": "CHN",
}


def _season_code_for_now() -> str:
    now = datetime.now(timezone.utc)
    y = now.year
    # Temporada europeia: julho+ → y/(y+1)
    if now.month >= 7:
        return f"{y % 100:02d}{(y + 1) % 100:02d}"
    return f"{(y - 1) % 100:02d}{y % 100:02d}"


def _fetch_csv(url: str) -> pd.DataFrame | None:
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45).read()
    except Exception:
        return None
    if not raw:
        return None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return None


def _parse_seasonal_fixtures(raw: pd.DataFrame, competition: str) -> pd.DataFrame:
    if "HomeTeam" in raw.columns:
        home_col, away_col = "HomeTeam", "AwayTeam"
        goal_col = "FTHG"
    elif "Home" in raw.columns:
        home_col, away_col = "Home", "Away"
        goal_col = "HG"
    else:
        return pd.DataFrame()
    upcoming = raw[raw[goal_col].isna()] if goal_col in raw.columns else raw.iloc[0:0]
    if upcoming.empty:
        return pd.DataFrame()
    rows = []
    for _, r in upcoming.iterrows():
        psh, psd, psa = r.get("PSH"), r.get("PSD"), r.get("PSA")
        if pd.isna(psh):
            psh, psd, psa = r.get("PPH"), r.get("PPD"), r.get("PPA")
        if pd.isna(psh):
            psh, psd, psa = r.get("AvgCH"), r.get("AvgCD"), r.get("AvgCA")
        if pd.isna(psh) or pd.isna(psd) or pd.isna(psa):
            continue
        dt = pd.to_datetime(r.get("Date"), dayfirst=True, errors="coerce")
        if pd.isna(dt):
            continue
        home = str(r.get(home_col) or "").strip()
        away = str(r.get(away_col) or "").strip()
        if not home or not away:
            continue
        rows.append({
            "date": dt,
            "competition": competition,
            "season": str(r.get("Season", "")),
            "stage": "fixture",
            "home_team": home,
            "away_team": away,
            "home_goals": pd.NA,
            "away_goals": pd.NA,
            "neutral": False,
            "ps_h": float(psh),
            "ps_d": float(psd),
            "ps_a": float(psa),
            "psc_h": float(r.get("PSCH")) if pd.notna(r.get("PSCH")) else float("nan"),
            "psc_d": float(r.get("PSCD")) if pd.notna(r.get("PSCD")) else float("nan"),
            "psc_a": float(r.get("PSCA")) if pd.notna(r.get("PSCA")) else float("nan"),
        })
    return pd.DataFrame(rows)


def fetch_season_rows_since(
    hist_max: pd.Timestamp,
    competitions: list[str] | None = None,
) -> pd.DataFrame:
    """Jogos já disputados no CSV da temporada corrente após `hist_max` (com gols + odds)."""
    competitions = competitions or list(COMP_TO_CODE.keys())
    seasons = {_season_code_for_now()}
    y = datetime.now(timezone.utc).year
    if datetime.now(timezone.utc).month < 7:
        seasons.add(f"{(y - 1) % 100:02d}{y % 100:02d}")
    parts: list[pd.DataFrame] = []
    for comp in competitions:
        code = COMP_TO_CODE.get(comp)
        if not code:
            continue
        for season in seasons:
            if code in ("BRA", "USA", "AUT", "DNK", "NOR", "SWE", "POL", "CHN"):
                url = f"{BASE}/new/{code}.csv"
            else:
                url = f"{BASE}/mmz4281/{season}/{code}.csv"
            raw = _fetch_csv(url)
            if raw is None or raw.empty:
                continue
            goal_col = "FTHG" if "FTHG" in raw.columns else "HG"
            if goal_col not in raw.columns:
                continue
            played = raw[raw[goal_col].notna()]
            if played.empty:
                continue
            rows = []
            home_col = "HomeTeam" if "HomeTeam" in played.columns else "Home"
            away_col = "AwayTeam" if "AwayTeam" in played.columns else "Away"
            for _, r in played.iterrows():
                psh, psd, psa = r.get("PSH"), r.get("PSD"), r.get("PSA")
                if pd.isna(psh):
                    psh, psd, psa = r.get("PPH"), r.get("PPD"), r.get("PPA")
                dt = pd.to_datetime(r.get("Date"), dayfirst=True, errors="coerce", utc=True)
                if pd.isna(dt) or dt <= hist_max:
                    continue
                hg = r.get(goal_col)
                ag = r.get("FTAG" if goal_col == "FTHG" else "AG")
                rows.append({
                    "date": dt,
                    "competition": comp,
                    "season": season,
                    "stage": "league",
                    "home_team": str(r.get(home_col)).strip(),
                    "away_team": str(r.get(away_col)).strip(),
                    "home_goals": int(hg),
                    "away_goals": int(ag),
                    "neutral": False,
                    "ps_h": float(psh) if pd.notna(psh) else float("nan"),
                    "ps_d": float(psd) if pd.notna(psd) else float("nan"),
                    "ps_a": float(psa) if pd.notna(psa) else float("nan"),
                    "psc_h": float(r.get("PSCH")) if pd.notna(r.get("PSCH")) else float("nan"),
                    "psc_d": float(r.get("PSCD")) if pd.notna(r.get("PSCD")) else float("nan"),
                    "psc_a": float(r.get("PSCA")) if pd.notna(r.get("PSCA")) else float("nan"),
                })
            if rows:
                parts.append(pd.DataFrame(rows))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def fetch_upcoming_pinnacle(competitions: list[str] | None = None) -> pd.DataFrame:
    """Baixa jogos futuros com Pinnacle (PSH) das ligas do monitor."""
    competitions = competitions or list(COMP_TO_CODE.keys())
    season = _season_code_for_now()
    parts: list[pd.DataFrame] = []
    for comp in competitions:
        code = COMP_TO_CODE.get(comp)
        if not code:
            continue
        if code in ("BRA", "USA", "AUT", "DNK", "NOR", "SWE", "POL", "CHN"):
            url = f"{BASE}/new/{code}.csv"
        else:
            url = f"{BASE}/mmz4281/{season}/{code}.csv"
        raw = _fetch_csv(url)
        if raw is None or raw.empty:
            continue
        part = _parse_seasonal_fixtures(raw, comp)
        if len(part):
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _cache_valid() -> bool:
    if not CACHE_PATH.exists():
        return False
    try:
        meta = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        ts = meta.get("fetched_at", 0)
        return (datetime.now(timezone.utc).timestamp() - ts) < CACHE_TTL_SEC
    except Exception:
        return False


def load_upcoming_pinnacle_cached(
    competitions: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if not force_refresh and _cache_valid():
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return pd.DataFrame(data.get("rows", []))
    df = fetch_upcoming_pinnacle(competitions)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps({
            "fetched_at": datetime.now(timezone.utc).timestamp(),
            "rows": json.loads(df.to_json(orient="records", date_format="iso")),
        }),
        encoding="utf-8",
    )
    return df


def merge_hist_with_fixtures(
    hist: pd.DataFrame,
    fixtures: pd.DataFrame,
    extra_played: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Anexa fixtures futuros; deduplica por jogo (fica o registro mais recente)."""
    if extra_played is not None and not extra_played.empty:
        fixtures = pd.concat([fixtures, extra_played], ignore_index=True) if fixtures is not None and not fixtures.empty else extra_played
    if fixtures is None or fixtures.empty:
        return hist
    teams = sorted(set(hist["home_team"].astype(str)) | set(hist["away_team"].astype(str)))
    idx = build_map(teams, load_aliases())
    fx = fixtures.copy()
    fx["home_team"] = fx["home_team"].map(lambda t: lookup(idx, t) or str(t).strip().title())
    fx["away_team"] = fx["away_team"].map(lambda t: lookup(idx, t) or str(t).strip().title())
    fx["date"] = pd.to_datetime(fx["date"], utc=True)
    hist = hist.copy()
    if hist["date"].dt.tz is None:
        hist["date"] = pd.to_datetime(hist["date"], utc=True)
    else:
        hist["date"] = hist["date"].dt.tz_convert("UTC")
    combined = pd.concat([hist, fx], ignore_index=True)
    combined["_d"] = combined["date"].dt.normalize()
    combined = combined.sort_values("date").drop_duplicates(
        subset=["competition", "home_team", "away_team", "_d"],
        keep="last",
    )
    combined = combined.drop(columns=["_d"])
    return combined.sort_values("date").reset_index(drop=True)


def lookup_pin_open_row(
    df: pd.DataFrame,
    liga: str,
    casa: str,
    fora: str,
    day: datetime | pd.Timestamp,
    tol_days: int = 2,
) -> pd.Series | None:
    """Linha com ps_* para fixture (pré-jogo)."""
    from src.pinnacle_lookup import as_utc

    day = as_utc(day)
    sub = df[
        (df.competition == liga)
        & (df.home_team == casa)
        & (df.away_team == fora)
    ].copy()
    if sub.empty:
        return None
    sub["delta"] = (sub["date"].map(as_utc) - day).abs()
    row = sub.loc[sub["delta"] <= pd.Timedelta(days=tol_days)].sort_values("delta").head(1)
    if row.empty:
        return None
    r = row.iloc[0]
    if pd.isna(r.get("ps_h")) or pd.isna(r.get("ps_d")) or pd.isna(r.get("ps_a")):
        return None
    return r
