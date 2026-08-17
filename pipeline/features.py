import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "db/matches.db"


def load_matches_with_elo() -> pd.DataFrame:
    """Load all matches joined with their pre-match Elo ratings."""
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            m.id,
            m.date,
            m.home_team,
            m.away_team,
            m.home_goals,
            m.away_goals,
            m.result,
            m.league,
            m.season,
            m.odds_home,
            m.odds_draw,
            m.odds_away,
            e.home_elo_before,
            e.away_elo_before
        FROM matches m
        JOIN elo_ratings e ON m.id = e.match_id
        ORDER BY m.date ASC, m.id ASC
    """, con, parse_dates=["date"])
    con.close()
    return df


def compute_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each match, compute each team's avg goals scored/conceded
    over their last `window` matches — using only past data.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Build a per-team match history as we go
    team_history = {}  # team -> list of (date, scored, conceded)

    home_scored, home_conceded = [], []
    away_scored, away_conceded = [], []
    home_rest, away_rest = [], []

    last_played = {}  # team -> last match date

    for _, row in df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        date = row["date"]

        def get_avg(team, col_idx, n=window):
            history = team_history.get(team, [])
            recent = history[-n:]
            if not recent:
                return None
            return np.mean([h[col_idx] for h in recent])

        # Compute features from history BEFORE this match
        home_scored.append(get_avg(ht, 0))
        home_conceded.append(get_avg(ht, 1))
        away_scored.append(get_avg(at, 0))
        away_conceded.append(get_avg(at, 1))

        # # Days rest
        # home_rest.append(
        #     (date - last_played[ht]).days if ht in last_played else None
        # )
        # away_rest.append(
        #     (date - last_played[at]).days if at in last_played else None
        # )

        # Days rest — capped at 30 to avoid season-boundary extrapolation
        if ht in last_played:
            home_days = (date - last_played[ht]).days
            home_rest.append(min(home_days, 30))
        else:
            home_rest.append(None)

        if at in last_played:
            away_days = (date - last_played[at]).days
            away_rest.append(min(away_days, 30))
        else:
            away_rest.append(None)
            
        # Now update history with this match's result
        team_history.setdefault(ht, []).append((row["home_goals"], row["away_goals"]))
        team_history.setdefault(at, []).append((row["away_goals"], row["home_goals"]))
        last_played[ht] = date
        last_played[at] = date

    df["home_avg_goals_scored"]   = home_scored
    df["home_avg_goals_conceded"] = home_conceded
    df["away_avg_goals_scored"]   = away_scored
    df["away_avg_goals_conceded"] = away_conceded
    df["home_days_rest"]          = home_rest
    df["away_days_rest"]          = away_rest

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Elo and rolling form into a single feature matrix.
    Drops rows where features aren't available yet (first few matches
    of each team's history).
    """
    df = df.copy()

    df["elo_diff"]       = df["home_elo_before"] - df["away_elo_before"]
    df["home_advantage"] = 1.0

    feature_cols = [
        "elo_diff",
        "home_advantage",
        "home_avg_goals_scored",
        "home_avg_goals_conceded",
        "away_avg_goals_scored",
        "away_avg_goals_conceded",
        "home_days_rest",
        "away_days_rest",
    ]

    # Target: 1 if home win, 0 otherwise (we model home win probability)
    df["target"] = (df["result"] == "H").astype(int)

    # Drop rows with any missing features
    df = df.dropna(subset=feature_cols)

    return df, feature_cols