import sqlite3
import pandas as pd

DB_PATH = "db/matches.db"

def init_db():
    """Create the matches table if it doesn't exist."""
    import os
    os.makedirs("db", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            home_goals INTEGER NOT NULL,
            away_goals INTEGER NOT NULL,
            result TEXT NOT NULL,
            league TEXT NOT NULL,
            season TEXT NOT NULL,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            UNIQUE(date, home_team, away_team)  -- prevent duplicates on re-run
        )
    """)
    con.commit()
    con.close()

def store_matches(df: pd.DataFrame):
    """
    Insert a DataFrame of matches into the DB.
    Uses INSERT OR IGNORE so re-running is safe.
    """
    con = sqlite3.connect(DB_PATH)
    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row["Date"].date()),
            row["HomeTeam"],
            row["AwayTeam"],
            int(row["FTHG"]),
            int(row["FTAG"]),
            row["FTR"],
            row["league"],
            row["season"],
            row.get("odds_home"),   
            row.get("odds_draw"),   
            row.get("odds_away"), 
        ))

    con.executemany("""
        INSERT OR IGNORE INTO matches
        (date, home_team, away_team, home_goals, away_goals,
        result, league, season, odds_home, odds_draw, odds_away)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()
    inserted = con.total_changes
    con.close()
    print(f"[DB] Inserted {inserted} new rows")
    
    
    
def init_elo_table():
    """Store pre-match Elo ratings for every fixture."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            match_id INTEGER PRIMARY KEY,
            home_elo_before REAL NOT NULL,
            away_elo_before REAL NOT NULL,
            home_elo_after REAL NOT NULL,
            away_elo_after REAL NOT NULL,
            FOREIGN KEY (match_id) REFERENCES matches(id)
        )
    """)
    con.commit()
    con.close()
    
    
def init_context_table():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS match_context (
            match_id INTEGER PRIMARY KEY,
            home_context_score INTEGER,
            away_context_score INTEGER,
            home_reasoning TEXT,
            away_reasoning TEXT,
            confidence TEXT,
            data_quality TEXT,
            error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (match_id) REFERENCES matches(id)
        )
    """)
    con.commit()
    con.close()


def store_context(match_id: int, context: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO match_context
        (match_id, home_context_score, away_context_score,
        home_reasoning, away_reasoning, confidence, data_quality, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        match_id,
        context["home_context_score"],
        context["away_context_score"],
        context["home_reasoning"],
        context["away_reasoning"],
        context["confidence"],
        context["data_quality"],
        context["error"],
    ))
    con.commit()
    con.close()