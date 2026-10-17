"""outcome_roles usa títulos normalizados do monitor."""

from monitor import outcome_roles


def test_outcome_roles_reads_title_key():
    ms = [
        {"title": "Newcastle United FC"},
        {"title": "Draw (Newcastle United FC vs. Aston Villa FC)"},
        {"title": "Aston Villa FC"},
    ]
    assert outcome_roles(ms, "Newcastle", "Aston Villa") == ["casa", "empate", "fora"]
