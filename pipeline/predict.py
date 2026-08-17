import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from pipeline.elo import get_current_ratings
from pipeline.llm import score_match_context, context_score_to_prob_adjustment
from pipeline.store import DB_PATH

load_dotenv()

MODEL_PATH = "models/model.pkl"

# Confidence tiers — define BEFORE running, never adjust after seeing output
CONFIDENCE_TIERS = {
    "HIGH":   (0.65, 1.00),  # model is strongly confident
    "MEDIUM": (0.55, 0.65),  # meaningful edge
    "LOW":    (0.00, 0.55),  # close match, treat with caution
}

HOME_ADVANTAGE_ELO = 100  # must match what you used in elo.py


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle["features"]


def get_team_form(team_name: str, n: int = 5) -> dict:
    """
    Pull the last n matches for a team from the DB to compute
    avg goals scored/conceded and days since last match.
    Returns dict of form features, or None values if insufficient history.
    """
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT date, home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE home_team = ? OR away_team = ?
        ORDER BY date DESC
        LIMIT ?
    """, (team_name, team_name, n)).fetchall()
    con.close()

    if not rows:
        return {
            "avg_goals_scored": None,
            "avg_goals_conceded": None,
            "days_rest": None,
        }

    goals_scored, goals_conceded = [], []
    for date, home_team, away_team, home_goals, away_goals in rows:
        if home_team == team_name:
            goals_scored.append(home_goals)
            goals_conceded.append(away_goals)
        else:
            goals_scored.append(away_goals)
            goals_conceded.append(home_goals)

    last_date = datetime.strptime(rows[0][0], "%Y-%m-%d")
    # days_rest = (datetime.utcnow() - last_date).days
    days_rest = min((datetime.utcnow() - last_date).days, 30)

    return {
        "avg_goals_scored":   round(np.mean(goals_scored), 4),
        "avg_goals_conceded": round(np.mean(goals_conceded), 4),
        "days_rest":          days_rest,
    }


def assign_confidence_tier(prob: float) -> str:
    for tier, (low, high) in CONFIDENCE_TIERS.items():
        if low <= prob < high:
            return tier
    return "LOW"


def predict_fixture(
    home_team: str,
    away_team: str,
    match_date: str,
    model,
    scaler,
    feature_cols: list,
    ratings: dict,
    use_llm: bool = True,
) -> dict:
    """
    Produce a full prediction for one fixture.
    Returns a dict with all prediction components.
    """
    result = {
        "home_team": home_team,
        "away_team": away_team,
        "match_date": match_date,
        "home_elo": None,
        "away_elo": None,
        "base_prob": None,
        "adjusted_prob": None,
        "confidence": None,
        "home_context_score": 0,
        "away_context_score": 0,
        "home_reasoning": "",
        "away_reasoning": "",
        "llm_confidence": "LOW",
        "data_quality": "NONE",
        "error": None,
    }

    # --- Step 1: Elo ratings ---
    home_elo = ratings.get(home_team)
    away_elo = ratings.get(away_team)

    if home_elo is None or away_elo is None:
        missing = home_team if home_elo is None else away_team
        result["error"] = f"No Elo rating found for {missing}"
        print(f"[PREDICT] Skipping {home_team} vs {away_team}: {result['error']}")
        return result

    result["home_elo"] = round(home_elo, 1)
    result["away_elo"] = round(away_elo, 1)

    # --- Step 2: Form features ---
    home_form = get_team_form(home_team)
    away_form = get_team_form(away_team)

    # Build feature vector — must match feature_cols order exactly
    feature_values = {
        "elo_diff":                float(home_elo - away_elo),
        "home_advantage":          1.0,
        "home_avg_goals_scored":   float(home_form["avg_goals_scored"]   or 1.3),
        "home_avg_goals_conceded": float(home_form["avg_goals_conceded"] or 1.2),
        "away_avg_goals_scored":   float(away_form["avg_goals_scored"]   or 1.1),
        "away_avg_goals_conceded": float(away_form["avg_goals_conceded"] or 1.3),
        "home_days_rest":          float(home_form["days_rest"]          or 7),
        "away_days_rest":          float(away_form["days_rest"]          or 7),
    }

    # Fallback values (or None) use league averages — document this in TRADEOFFS.md
    X = np.array([[feature_values[col] for col in feature_cols]])

    if np.isnan(X).any():
        result["error"] = f"NaN in feature vector: {feature_values}"
        return result

    X_scaled = scaler.transform(X)

    # --- Step 3: Base probability ---
    base_prob = float(model.predict_proba(X_scaled)[0][1])
    result["base_prob"] = round(base_prob, 4)

    # --- Step 4: LLM context ---
    context = {"home_context_score": 0, "away_context_score": 0,
                "home_reasoning": "LLM skipped.", "away_reasoning": "LLM skipped.",
                "confidence": "LOW", "data_quality": "NONE", "error": None}

    if use_llm:
        context = score_match_context(home_team, away_team, match_date)

    result["home_context_score"] = context["home_context_score"]
    result["away_context_score"] = context["away_context_score"]
    result["home_reasoning"]     = context["home_reasoning"]
    result["away_reasoning"]     = context["away_reasoning"]
    result["llm_confidence"]     = context["confidence"]
    result["data_quality"]       = context["data_quality"]

    # --- Step 5: Adjusted probability ---
    adjusted_prob = context_score_to_prob_adjustment(
        context["home_context_score"],
        context["away_context_score"],
        base_prob,
    )
    result["adjusted_prob"] = round(adjusted_prob, 4)

    # --- Step 6: Confidence tier ---
    result["confidence"] = assign_confidence_tier(adjusted_prob)

    return result


def run_predictions(fixtures: list[dict], use_llm: bool = True) -> pd.DataFrame:
    """
    Run predictions for a list of fixtures.

    fixtures format:
    [
        {"home_team": "Arsenal", "away_team": "Chelsea", "date": "2024-04-20"},
        ...
    ]

    Returns a DataFrame ranked by adjusted_prob descending.
    """
    print("[PIPELINE] Loading model...")
    model, scaler, feature_cols = load_model()

    print("[PIPELINE] Loading current Elo ratings...")
    ratings = get_current_ratings()
    print(f"[PIPELINE] {len(ratings)} teams rated")

    results = []
    for fixture in fixtures:
        home  = fixture["home_team"]
        away  = fixture["away_team"]
        date  = fixture.get("date", "")
        print(f"\n[PIPELINE] Predicting: {home} vs {away}")

        pred = predict_fixture(
            home, away, date,
            model, scaler, feature_cols,
            ratings, use_llm=use_llm,
        )
        results.append(pred)

    df = pd.DataFrame(results)

    # Rank by adjusted probability, errors at the bottom
    df = df.sort_values("adjusted_prob", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    df.index += 1  # rank starts at 1

    return df


def print_predictions(df: pd.DataFrame):
    """Pretty-print the prediction table to terminal."""
    print("\n" + "=" * 80)
    print(f"{'RANK':<5} {'FIXTURE':<35} {'BASE':>6} {'ADJ':>6} {'TIER':<8} {'ELO DIFF':>9}")
    print("=" * 80)

    for rank, row in df.iterrows():
        if row["error"]:
            print(f"{rank:<5} {row['home_team']} vs {row['away_team']:<20} ERROR: {row['error']}")
            continue

        fixture_str = f"{row['home_team']} vs {row['away_team']}"
        elo_diff    = row["home_elo"] - row["away_elo"] if row["home_elo"] else 0

        print(
            f"{rank:<5} "
            f"{fixture_str:<35} "
            f"{row['base_prob']:>6.3f} "
            f"{row['adjusted_prob']:>6.3f} "
            f"{row['confidence']:<8} "
            f"{elo_diff:>+9.0f}"
        )

        # Context detail for HIGH confidence predictions
        if row["confidence"] == "HIGH" and row["data_quality"] != "NONE":
            print(f"       Home context ({row['home_context_score']:+d}): {row['home_reasoning']}")
            print(f"       Away context ({row['away_context_score']:+d}): {row['away_reasoning']}")

    print("=" * 80)
    print(f"\nSummary: {len(df)} fixtures | "
        f"HIGH: {(df['confidence']=='HIGH').sum()} | "
        f"MEDIUM: {(df['confidence']=='MEDIUM').sum()} | "
        f"LOW: {(df['confidence']=='LOW').sum()}")
    
    