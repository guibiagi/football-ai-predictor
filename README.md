# ⚽ Football AI Predictor

**Probabilistic football match prediction — educational project.**

Predicts match outcomes using a **Poisson distribution model** trained on historical match data. Not a betting tool — a statistics and ML learning project.

## 🧠 How it works

```
Historical matches → Team attack/defense strengths → Expected goals (λ) → Poisson distribution → Score probability matrix → All predictions
```

1. **Estimate expected goals** for each team based on historical attacking and defensive strength
2. **Apply Poisson distribution** to get P(0 goals), P(1 goal), P(2 goals), etc.
3. **Build a score matrix** by multiplying probabilities (assuming independence)
4. **Extract all probabilities** from the matrix: win/draw/loss, both teams score, over/under, most likely scores

## 🚀 Quick start

```bash
# Install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run a prediction
python run_prediction.py "Brazil" "France"

# Run all tests
PYTHONPATH=. python -m pytest tests/ -v
```

## 📁 Project structure

```
football-ai-predictor/
├── README.md              # You are here
├── requirements.txt       # Dependencies
├── run_prediction.py      # Quick CLI prediction script
├── data/
│   └── matches.csv        # Historical match data
├── notebooks/
│   ├── 01_exploracao.ipynb     # (coming soon) Data exploration
│   └── 02_modelo_poisson.ipynb # (coming soon) Model walkthrough
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Load, validate, clean match data
│   ├── poisson_model.py   # Poisson prediction model
│   ├── features.py        # (coming soon) Feature engineering
│   ├── metrics.py         # (coming soon) Model evaluation
│   ├── simulator.py       # (coming soon) Match simulator
│   └── app.py             # (coming soon) Streamlit dashboard
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    └── test_poisson_model.py
```

## 📊 Data

The project uses the **International Football Results** dataset by [Mart Jürisoo](https://github.com/martj42/international_results), containing **25,433 matches from 2000 to 2026** across 127 competitions and 314 national teams.

- Source: [github.com/martj42/international_results](https://github.com/martj42/international_results)
- Format: CSV (1.7 MB), ready to use — no Kaggle login required
- Includes: FIFA World Cup, continental championships, qualifiers, friendlies, Nations League
- Neutral venue flag included for tournament matches

| Column | Type | Description |
|--------|------|-------------|
| date | YYYY-MM-DD | Match date |
| competition | string | e.g. "World Cup", "Brasileirão" |
| season | int | Year of the season |
| stage | string | "Group", "Round of 16", "Quarterfinal", etc. |
| home_team | string | Home team name |
| away_team | string | Away team name |
| home_goals | int | Goals scored by home team |
| away_goals | int | Goals scored by away team |
| neutral | bool | True if match is on neutral ground |

## 🎯 MVP roadmap

- [x] **MVP 1**: Simple Poisson model with global + team averages
- [ ] **MVP 2**: Add offensive/defensive strength adjustments (regression-based)
- [ ] **MVP 3**: Add Elo ratings or custom team rating
- [ ] **MVP 4**: Add recent form, rest days, competition stage
- [ ] **MVP 5**: Adapt from World Cup to Brasileirão
- [ ] **MVP 6**: Streamlit dashboard for interactive predictions

## ⚠️ Important caveats

- This is a **probabilistic** model — it estimates likelihoods, not certainties
- **Small sample sizes** (e.g., World Cup with 64 games) lead to noisy estimates
- The model assumes **goal independence** between teams (Poisson assumption)
- It does NOT account for: injuries, red cards, weather, tactical changes, motivation
- **NEVER use this for real betting** — it's an educational tool

## 📚 Why Poisson?

Football goals are:
- **Rare** (~2.5 per match on average)
- **Discrete** (0, 1, 2, 3... — not 1.7)
- **Roughly independent** within short time intervals

The Poisson distribution is the simplest model that captures these properties. It's the right starting point before moving to more complex models (Dixon-Coles, XGBoost, etc.).

## 🧪 Testing philosophy

- **Temporal splits only** — never shuffle football data (future information leak)
- **Log loss > accuracy** — we care about calibrated probabilities, not just right/wrong
- **Fallback for small data** — teams with fewer than N games use global averages

---

*Built for learning. AI that's good doesn't give certainty — it shows uncertainty honestly.*