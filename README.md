# match-engine

A football match prediction system built as a learning project for AI engineering. The system combines Elo ratings, logistic regression, and an LLM context layer to produce calibrated win probabilities for upcoming Premier League fixtures.

---

## What it does

Given a set of upcoming fixtures, the system:

1. Looks up current Elo ratings for both teams
2. Computes a base win probability using a trained logistic regression model
3. Fetches recent team news and asks an LLM to score the match context
4. Adjusts the base probability by the context score
5. Returns a ranked prediction list with confidence tiers

---

## Project structure

```
match-engine/
├── data/
│   └── raw/                  # downloaded CSVs from football-data.co.uk
├── db/
│   └── matches.db            # SQLite database
├── models/
│   └── model.pkl             # trained logistic regression model
├── pipeline/
│   ├── fetch.py              # download CSVs
│   ├── parse.py              # normalize and clean CSVs
│   ├── store.py              # write to SQLite
│   ├── elo.py                # Elo rating computation
│   ├── features.py           # feature engineering
│   ├── model.py              # walk-forward validation and training
│   ├── backtest.py           # betting simulation against bookmaker odds
│   ├── fixtures.py           # fetch upcoming fixtures from football-data.org
│   ├── news.py               # fetch team news from GNews
│   ├── llm.py                # LLM context scoring via Anthropic API
│   └── predict.py            # prediction pipeline
├── main.py                   # rebuild data pipeline and retrain model
├── predict.py                # run live predictions
├── TRADEOFFS.md              # design decisions and honest evaluation
├── .env                      # API keys (never commit this)
├── .gitignore
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install requests pandas scikit-learn matplotlib anthropic python-dotenv numpy
```

### 2. Set up API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
GNEWS_API_KEY=your_key_here
FOOTBALL_DATA_API_KEY=your_key_here
```

| Key | Where to get it | Cost |
|-----|----------------|------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Pay per use |
| `GNEWS_API_KEY` | [gnews.io](https://gnews.io) | Free (100 req/day) |
| `FOOTBALL_DATA_API_KEY` | [football-data.org](https://www.football-data.org/client/register) | Free |

### 3. Build the pipeline

Downloads historical data, computes Elo ratings, trains the model, and runs the backtest:

```bash
python main.py
```

### 4. Run predictions

Fetches upcoming Premier League fixtures and produces ranked predictions:

```bash
python predict.py
```

---

## How it works

### Data pipeline

Historical match results are downloaded from [football-data.co.uk](https://www.football-data.co.uk/data.php) as free CSV files. The pipeline is resumable — if it fails mid-download it picks up where it left off. Data is stored in a local SQLite database.

Seasons covered: 1993/94 through 2025/26  
Leagues: Premier League (extendable to others)

### Elo ratings

Every team starts at 1500. After each match, ratings update based on the result and how surprising it was. A 100-point home advantage offset is applied when computing expected scores. Ratings are processed in strict chronological order — no future data ever touches past ratings.

### Logistic regression

Features used:

| Feature | Description |
|---------|-------------|
| `elo_diff` | Rating gap between home and away team |
| `home_advantage` | Constant 1.0 — model learns the magnitude |
| `home_avg_goals_scored` | Attack strength over last 5 matches |
| `home_avg_goals_conceded` | Defensive weakness over last 5 matches |
| `away_avg_goals_scored` | Same for away team |
| `away_avg_goals_conceded` | Same for away team |
| `home_days_rest` | Days since home team last played (capped at 30) |
| `away_days_rest` | Days since away team last played (capped at 30) |

Validation uses walk-forward methodology — train on all seasons up to N, test on season N+1. This prevents any future data from leaking into training. The model is wrapped with Platt scaling for calibration.

### LLM context layer

For each upcoming fixture, recent news is fetched for both teams via GNews. The news is passed to Claude with a structured prompt that returns a JSON context score:

```json
{
  "home_context_score": 1,
  "away_context_score": -2,
  "home_reasoning": "Arsenal fully fit ahead of the fixture.",
  "away_reasoning": "Chelsea missing two key defenders through suspension.",
  "confidence": "HIGH",
  "data_quality": "GOOD"
}
```

Scores range from -3 to +3. The net score shifts the base probability by up to ±8 percentage points. The statistical model stays in control — the LLM is a signal, not an override.

### Confidence tiers

| Tier | Probability range | Meaning |
|------|------------------|---------|
| HIGH | 0.65 – 1.00 | Model strongly favours home team |
| MEDIUM | 0.55 – 0.65 | Meaningful edge |
| LOW | 0.00 – 0.55 | Close match, treat with caution |

---

## Backtest results

Evaluated on Premier League data from 2000/01 to 2025/26 using historical Bet365 odds (William Hill for pre-2003 seasons).

```
Bets placed:    5,793
ROI:            -4.10%
Hit rate:       41.50%
Avg odds:       3.06
Bookmaker margin: 5.97%
```

The model has genuine predictive signal — a 41.5% hit rate against a 32.7% breakeven requirement at these odds. The gap between signal and profitability is the bookmaker margin. The model recovers approximately 1.9 percentage points of the 5.97% margin.

Profitable in 15 of 26 seasons tested. The weakest period is post-2020, where home advantage has declined and the model — trained on 25 years of stronger home advantage — systematically overestimates home win probability.

---

## Known limitations

- **No player data.** Injuries, suspensions, and squad rotations are invisible to the statistical model. The LLM layer partially addresses this for live predictions.
- **Home advantage decline.** Post-COVID football shows weaker home advantage than the historical training data assumes.
- **Single league.** The model is trained and validated on Premier League data only. Performance on other leagues is untested.
- **News quota.** GNews free tier allows 100 requests per day. With 19 fixtures requiring 2 news calls each, quota can be exhausted mid-run. A local daily cache avoids this.
- **Days rest at season boundaries.** The gap between the last match of one season and the first of the next is capped at 30 days to prevent the model extrapolating outside its trained range.

---

## Requirements

```
requests
pandas
scikit-learn
matplotlib
anthropic
python-dotenv
numpy
```

---

## Data sources

- Match results and odds: [football-data.co.uk](https://www.football-data.co.uk/data.php) — free, no API key
- Upcoming fixtures: [football-data.org](https://www.football-data.org) — free tier, API key required
- Team news: [GNews](https://gnews.io) — free tier (100 req/day), API key required
- LLM context scoring: [Anthropic Claude](https://console.anthropic.com) — pay per use

---

## License

MIT