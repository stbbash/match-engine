import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"


def fetch_team_news(team_name: str, days_back: int = 5) -> list[dict]:
    """
    Fetch recent news headlines for a team.
    Returns list of {title, description, published_at} dicts.
    Returns empty list on any failure — caller must handle gracefully.
    """
    if not GNEWS_API_KEY:
        print(f"[NEWS] No GNews API key — skipping news for {team_name}")
        return []

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "q": f"{team_name} football",
        "lang": "en",
        "from": from_date,
        "max": 5,
        "apikey": GNEWS_API_KEY,
    }

    try:
        response = requests.get(GNEWS_URL, params=params, timeout=60)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "published_at": a.get("publishedAt", ""),
            }
            for a in articles
        ]
    except Exception as e:
        print(f"[NEWS] Failed to fetch news for {team_name}: {e}")
        return []


def format_news_for_prompt(team_name: str, articles: list[dict]) -> str:
    """Format news articles into a compact string for the LLM prompt."""
    if not articles:
        return f"{team_name}: No recent news available."

    lines = [f"{team_name} recent news:"]
    for a in articles:
        title = a["title"] or ""
        desc = a["description"] or ""
        lines.append(f"- {title}. {desc}".strip())

    return "\n".join(lines)