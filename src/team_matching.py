"""Mapeamento Polymarket → dataset com rapidfuzz e tabela versionada de aliases."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz, process

ALIASES_PATH = Path(__file__).resolve().parent.parent / "data" / "team_aliases.json"

CLUB_PREFIXES = ("ca ", "se ", "cr ", "sc ", "ec ", "fc ", "fbpa", "ac ", "aa ")
GENERIC = {
    "united", "city", "town", "county", "rovers", "wanderers", "athletic",
    "albion", "west", "north", "south", "east", "real", "sporting", "club",
}

SPECIAL = str.maketrans({
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "ß": "ss", "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "þ": "th", "Þ": "th", "ı": "i", "ħ": "h", "ŋ": "n", "ĸ": "k",
})

_FUZZ_THRESHOLD = 88


def load_aliases(path: Path | None = None) -> dict[str, str]:
    p = path or ALIASES_PATH
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k).lower(): str(v).lower() for k, v in data.get("aliases", {}).items()}


def _clean(name: str) -> str:
    n = name.strip().lower().translate(SPECIAL)
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    for prefix in CLUB_PREFIXES:
        if n.startswith(prefix):
            n = n[len(prefix):]
    n = n.replace("paulista", "").strip()
    for suffix in (" afc", " fc", " cf", " sc"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip()


def normalize(name: str) -> str:
    words = _clean(name).split()
    return words[0] if words else _clean(name)


def build_map(teams: list[str], aliases: dict[str, str] | None = None) -> dict[str, str]:
    aliases = aliases if aliases is not None else load_aliases()
    idx: dict[str, str] = {}
    for t in teams:
        idx[_clean(t)] = t
    owners: dict[str, set[str]] = {}
    for t in teams:
        for w in _clean(t).replace("-", " ").split():
            if len(w) > 3 and w not in GENERIC:
                owners.setdefault(w, set()).add(t)
    for w, ts in owners.items():
        if len(ts) == 1:
            idx.setdefault(w, next(iter(ts)))
    for alias, target in aliases.items():
        if target in idx:
            idx.setdefault(alias, idx[target])
    return idx


def lookup(idx: dict[str, str], name: str, aliases: dict[str, str] | None = None) -> str | None:
    aliases = aliases if aliases is not None else load_aliases()
    key = _clean(name)
    if key in idx:
        return idx[key]
    first = key.split()[0] if key.split() else key
    alias = aliases.get(key) or aliases.get(first)
    if alias and alias in idx:
        return idx[alias]
    words = key.replace("-", " ").split()
    for w in words:
        if len(w) > 3 and w in idx:
            return idx[w]
    for w in words:
        if len(w) < 6:
            continue
        hits = {
            v for k, v in idx.items()
            if (k.startswith(w) and len(k) >= 6) or (w.startswith(k) and len(k) >= 6)
        }
        if len(hits) == 1:
            return hits.pop()
    choices = list({v: _clean(v) for v in idx.values()}.items())
    if not choices:
        return None
    labels = [c[1] for c in choices]
    match = process.extractOne(key, labels, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= _FUZZ_THRESHOLD:
        label = match[0]
        for team, cleaned in choices:
            if cleaned == label:
                return team
    return None
