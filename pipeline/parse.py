import pandas as pd
import os

REQUIRED_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

ODDS_COLS_PRIORITY = [
    ("B365H", "B365D", "B365A"),   # preferred — available from ~2005
    ("WHH",   "WHD",   "WHA"),     # William Hill — older seasons
    ("LBH",   "LBD",   "LBA"),     # Ladbrokes — fallback
]

LEAGUE_MAP = {
    "E0": "premier_league",
    "E1": "championship",
    "SP1": "la_liga",
    "D1": "bundesliga",
}

def pick_odds_columns(df_columns):
    for h, d, a in ODDS_COLS_PRIORITY:
        if all(c in df_columns for c in (h, d, a)):
            return h, d, a
    return None, None, None


def parse_csv(filepath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        print(f"[PARSE ERROR] {filepath}: {e}")
        return None

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[SKIP] {filepath} missing columns: {missing}")
        return None

    # Pick the best available odds columns for THIS file
    odds_h, odds_d, odds_a = pick_odds_columns(df.columns)
    odds_cols = [odds_h, odds_d, odds_a] if odds_h else []

    keep_cols = REQUIRED_COLS + odds_cols
    df = df[keep_cols].copy()

    # Rename odds columns to standard names regardless of source
    if odds_h:
        df = df.rename(columns={
            odds_h: "odds_home",
            odds_d: "odds_draw",
            odds_a: "odds_away",
        })
        print(f"[PARSE] {os.path.basename(filepath)} — using odds: {odds_h}/{odds_d}/{odds_a}")
    else:
        print(f"[PARSE] {os.path.basename(filepath)} — no odds columns found")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce").astype("Int64")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["FTHG", "FTAG"])

    basename = os.path.basename(filepath)
    parts = basename.replace(".csv", "").split("_")
    league_code = parts[0]
    season = parts[1]

    df["league"] = LEAGUE_MAP.get(league_code, league_code)
    df["season"] = season

    return df.sort_values("Date").reset_index(drop=True)