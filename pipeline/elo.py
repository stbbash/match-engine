import sqlite3
import math
from collections import defaultdict

DB_PATH = "db/matches.db"

INITIAL_RATING = 1500.0
K_FACTOR = 32
HOME_ADVANTAGE = 100  # add to home team's effective rating when computing expected score


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that team A wins given ratings."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_rating(rating: float, expected: float, actual: float) -> float:
    """Return new rating after one match."""
    return rating + K_FACTOR * (actual - expected)


def run_elo():
    con = sqlite3.connect(DB_PATH)

    # Load every match in strict chronological order
    # Where dates are equal, order by id (insertion order) as tiebreaker
    matches = con.execute("""
        SELECT id, date, home_team, away_team, result
        FROM matches
        ORDER BY date ASC, id ASC
    """).fetchall()

    print(f"[ELO] Processing {len(matches)} matches...")

    # ratings dict: team name -> current rating
    ratings = defaultdict(lambda: INITIAL_RATING)

    rows_to_insert = []

    for match_id, date, home_team, away_team, result in matches:

        home_r = ratings[home_team]
        away_r = ratings[away_team]

        # Apply home advantage to expected score only, not to stored rating
        home_expected = expected_score(home_r + HOME_ADVANTAGE, away_r)
        away_expected = 1 - home_expected

        # Actual scores
        if result == "H":
            home_actual, away_actual = 1.0, 0.0
        elif result == "A":
            home_actual, away_actual = 0.0, 1.0
        else:  # Draw
            home_actual, away_actual = 0.5, 0.5

        # Compute new ratings
        home_r_new = update_rating(home_r, home_expected, home_actual)
        away_r_new = update_rating(away_r, away_expected, away_actual)

        rows_to_insert.append((
            match_id,
            round(home_r, 4),
            round(away_r, 4),
            round(home_r_new, 4),
            round(away_r_new, 4),
        ))

        # Update ratings for next match
        ratings[home_team] = home_r_new
        ratings[away_team] = away_r_new

    # Write to DB
    con.execute("DELETE FROM elo_ratings")  # clean slate on re-run
    con.executemany("""
        INSERT OR REPLACE INTO elo_ratings
        (match_id, home_elo_before, away_elo_before, home_elo_after, away_elo_after)
        VALUES (?, ?, ?, ?, ?)
    """, rows_to_insert)
    con.commit()
    con.close()

    print(f"[ELO] Done. Stored ratings for {len(rows_to_insert)} matches.")
    return ratings  # return final state for inspection


def get_current_ratings() -> dict:
    """Return the latest Elo rating for every team."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT m.home_team, e.home_elo_after, m.away_team, e.away_elo_after
        FROM elo_ratings e
        JOIN matches m ON e.match_id = m.id
        ORDER BY m.date ASC, m.id ASC
    """).fetchall()
    con.close()

    ratings = {}
    for home_team, home_after, away_team, away_after in rows:
        ratings[home_team] = home_after
        ratings[away_team] = away_after

    return ratings