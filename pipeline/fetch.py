import requests
import os
from datetime import datetime


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = {
    "E0": "premier_league",
    # "E1": "championship",
    # "SP1": "la_liga",
    # "D1": "bundesliga",
}

RAW_DIR = "data/raw"

def generate_seasons(start_year=1993, end_year=None):
    if end_year is None:
        end_year = datetime.now().year

    seasons = []

    for year in range(start_year, end_year):
        next_year = year + 1

        # Special case (site format)
        if year == 2020:
            seasons.append("2021")
        else:
            season_code = str(year)[-2:] + str(next_year)[-2:]
            seasons.append(season_code)

    return seasons



SEASONS = generate_seasons()

def is_valid(url):
    try:
        r = requests.head(url, timeout=5)
        return r.status_code == 200
    except:
        return False




def download_csv(league_code: str, season: str) -> str | None:
    """
    Download one CSV. Returns local filepath on success, None if already exists.
    Resumable: skips download if file already exists.
    """
    filename = f"{league_code}_{season}.csv"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        print(f"[SKIP] {filename} already exists")
        return filepath

    url = BASE_URL.format(season=season, league=league_code)
    try:
        is_url_valid = is_valid(url)
        if is_url_valid:
            print("URL is valid downloading CSV")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            os.makedirs(RAW_DIR, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"[OK]   Downloaded {filename}")
            return filepath
        else:
            print("URL is Invalid")
    except requests.RequestException as e:
        print(f"[FAIL] {url} — {e}")
        return None

def fetch_all():
    paths = []
    for league_code in LEAGUES:
        for season in SEASONS:
            path = download_csv(league_code, season)
            if path:
                paths.append(path)
    return paths