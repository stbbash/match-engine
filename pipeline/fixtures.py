import requests
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = "db/matches.db"

# Free, no API key needed
FOOTBALL_DATA_URL = "https://api.football-data.org/v4/competitions/PL/matches"
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_API_KEY")


def get_known_teams() -> set:
    """Pull all team names we have in the DB."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT DISTINCT home_team FROM matches
        UNION
        SELECT DISTINCT away_team FROM matches
    """).fetchall()
    con.close()
    return {row[0] for row in rows}


def fetch_upcoming_fixtures(days_ahead: int = 7) -> list[dict]:
    """
    Fetch upcoming Premier League fixtures from football-data.org.
    Free tier allows 10 calls/minute, no payment needed.
    Get your free key at: https://www.football-data.org/client/register
    """
    if not FOOTBALL_DATA_KEY:
        print("[FIXTURES] No FOOTBALL_DATA_API_KEY in .env — using manual fixtures")
        return []

    date_from = datetime.utcnow().strftime("%Y-%m-%d")
    date_to   = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    try:
        response = requests.get(
            FOOTBALL_DATA_URL,
            headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
            params={"dateFrom": date_from, "dateTo": date_to},
            timeout=60,
        )
        response.raise_for_status()
        matches = response.json().get("matches", [])

        fixtures = []
        for m in matches:
            fixtures.append({
                "home_team": m["homeTeam"]["shortName"],
                "away_team": m["awayTeam"]["shortName"],
                "date":      m["utcDate"][:10],  # "2026-04-22T14:00:00Z" -> "2026-04-22"
            })
        return fixtures

    except Exception as e:
        print(f"[FIXTURES] Failed to fetch fixtures: {e}")
        return []


def normalize_team_name(api_name: str, known_teams: set) -> str | None:
    """
    Map API team names to DB team names.
    API returns names like "Manchester City", DB has "Man City".
    Returns None if no match found.
    """
    # Direct match first
    if api_name in known_teams:
        return api_name

    # Manual mapping for known mismatches
    NAME_MAP = {
        "Manchester City":          "Man City",
        "Manchester United":        "Man United",
        "Tottenham Hotspur":        "Tottenham",
        "Nottingham Forest":        "Nott'm Forest",
        "Newcastle United":         "Newcastle",
        "West Ham United":          "West Ham",
        "Wolverhampton Wanderers":  "Wolves",
        "Sheffield United":         "Sheffield United",
        "Brighton & Hove Albion":   "Brighton",
        "Leicester City":           "Leicester",
        "Leeds United":             "Leeds",
        "Aston Villa":              "Aston Villa",
        "Crystal Palace":           "Crystal Palace",
        "AFC Bournemouth":          "Bournemouth",
        "Brentford":                "Brentford",
        "Fulham":                   "Fulham",
        "Everton":                  "Everton",
        "Arsenal":                  "Arsenal",
        "Chelsea":                  "Chelsea",
        "Liverpool":                "Liverpool",
        "Ipswich Town":             "Ipswich",
        "Southampton":              "Southampton",
        "Luton Town":               "Luton",
        "Burnley":                  "Burnley",
        "Nottingham Forest":        "Nott'm Forest",
        "Nottingham":               "Nott'm Forest",
        "Wolverhampton Wanderers":  "Wolves",
        "Wolverhampton":            "Wolves",
        "Brighton & Hove Albion":   "Brighton",
        "Brighton Hove":            "Brighton",
    }

    return NAME_MAP.get(api_name)


def get_predictable_fixtures(days_ahead: int = 7) -> list[dict]:
    """
    Fetch upcoming fixtures, normalize names, and filter to only
    teams we have Elo ratings for. Returns ready-to-predict list.
    """
    known_teams = get_known_teams()
    raw_fixtures = fetch_upcoming_fixtures(days_ahead)

    if not raw_fixtures:
        print("[FIXTURES] No fixtures fetched — check your API key or use manual fixtures")
        return []

    predictable = []
    skipped = []

    for fixture in raw_fixtures:
        home_db = normalize_team_name(fixture["home_team"], known_teams)
        away_db = normalize_team_name(fixture["away_team"], known_teams)

        if home_db and away_db:
            predictable.append({
                "home_team": home_db,
                "away_team": away_db,
                "date":      fixture["date"],
            })
            print(f"[FIXTURES] ✓  {home_db} vs {away_db} on {fixture['date']}")
        else:
            missing = fixture["home_team"] if not home_db else fixture["away_team"]
            skipped.append(missing)
            print(f"[FIXTURES] ✗  Skipped — '{missing}' not in DB")

    print(f"\n[FIXTURES] {len(predictable)} predictable, {len(skipped)} skipped")
    return predictable