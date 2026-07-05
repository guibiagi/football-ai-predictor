"""Tests for poisson_model.py"""

import numpy as np
import pytest

from src.data_loader import load_matches
from src.poisson_model import (
    PoissonModel,
    poisson_prob,
    build_score_matrix,
    extract_probabilities,
    most_likely_scores,
)


def test_poisson_prob_sums_to_one():
    """Poisson PMF should sum to (approximately) 1."""
    for lam in [0.5, 1.0, 2.0, 3.0]:
        probs = poisson_prob(lam)
        assert abs(probs.sum() - 1.0) < 0.001
        assert len(probs) == 11  # MAX_GOALS + 1


def test_build_score_matrix_sums_to_one():
    """Score matrix probabilities should sum to ~1."""
    matrix = build_score_matrix(1.5, 1.2)
    assert abs(matrix.sum() - 1.0) < 0.001
    assert matrix.shape == (11, 11)


def test_extract_probabilities_consistency():
    """Home + Draw + Away should equal 1."""
    matrix = build_score_matrix(1.5, 1.2)
    probs = extract_probabilities(matrix)
    total = (
        probs["home_win_probability"]
        + probs["draw_probability"]
        + probs["away_win_probability"]
    )
    assert abs(total - 1.0) < 0.001


def test_home_advantage():
    """When home team is stronger, they should have higher win probability."""
    matrix = build_score_matrix(2.5, 0.8)
    probs = extract_probabilities(matrix)
    assert probs["home_win_probability"] > probs["away_win_probability"]


def test_dixon_coles_increases_draws():
    """Dixon-Coles with rho>0 should increase 0-0 and 1-1 probability."""
    m_ind = build_score_matrix(1.0, 1.0, rho=0.0)
    m_dc = build_score_matrix(1.0, 1.0, rho=0.1)
    # 0-0 should be MORE likely with DC
    assert m_dc[0, 0] > m_ind[0, 0], f"DC 0-0: {m_dc[0,0]:.4f} vs ind: {m_ind[0,0]:.4f}"
    # 1-1 should be MORE likely with DC
    assert m_dc[1, 1] > m_ind[1, 1], f"DC 1-1: {m_dc[1,1]:.4f} vs ind: {m_ind[1,1]:.4f}"


def test_dixon_coles_decreases_unbalanced():
    """Dixon-Coles should decrease 1-0 probability (fewer narrow wins)."""
    m_ind = build_score_matrix(1.0, 1.0, rho=0.0)
    m_dc = build_score_matrix(1.0, 1.0, rho=0.1)
    assert m_dc[1, 0] < m_ind[1, 0]


def test_dixon_coles_matrix_sums_to_one():
    """Even with rho adjustment, matrix must sum to ~1."""
    for rho in [0.0, 0.05, 0.10, 0.15]:
        matrix = build_score_matrix(1.5, 1.2, rho=rho)
        assert abs(matrix.sum() - 1.0) < 0.001, f"Sum={matrix.sum():.4f} for rho={rho}"


def test_most_likely_scores():
    matrix = build_score_matrix(1.5, 1.2)
    scores = most_likely_scores(matrix, top_n=3)
    assert len(scores) == 3
    # Probabilities should be in descending order
    assert scores[0]["probability"] >= scores[1]["probability"] >= scores[2]["probability"]


def test_fit_and_predict():
    """End-to-end: fit model on real data, predict a match."""
    df = load_matches("data/matches.csv")
    model = PoissonModel(min_games=3)
    model.fit(df)

    pred = model.predict_match("Brazil", "France", neutral=True)
    assert pred["home_team"] == "Brazil"
    assert pred["neutral"] is True
    assert 0 < pred["expected_goals_home"] < 5
    assert 0 < pred["expected_goals_away"] < 5
    assert "home_win_probability" in pred
    assert "draw_probability" in pred
    assert "most_likely_scores" in pred
    assert 0 <= pred["home_win_probability"] <= 1
    assert 0 <= pred["draw_probability"] <= 1


def test_fallback_for_new_team():
    """Unknown teams should get default strengths (no crash)."""
    df = load_matches("data/matches.csv")
    model = PoissonModel(min_games=3)
    model.fit(df)

    pred = model.predict_match("Uzbekistan", "Thailand", neutral=True)
    assert pred["expected_goals_home"] > 0
    assert pred["expected_goals_away"] > 0


def test_unfitted_model_raises():
    """Calling predict before fit should raise an error."""
    model = PoissonModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_match("Brazil", "France")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])