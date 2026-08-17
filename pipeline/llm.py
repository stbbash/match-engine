import os
import json
import anthropic
from dotenv import load_dotenv
from pipeline.news import fetch_team_news, format_news_for_prompt

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# The JSON schema we expect back — define this BEFORE writing the prompt
EXPECTED_SCHEMA = {
    "home_context_score": "integer from -3 to +3",
    "away_context_score": "integer from -3 to +3",
    "home_reasoning": "one sentence",
    "away_reasoning": "one sentence",
    "confidence": "LOW | MEDIUM | HIGH",
    "data_quality": "GOOD | PARTIAL | NONE"
}

SYSTEM_PROMPT = """You are a football match context analyst. Your job is to assess
how recent news and team circumstances might affect an upcoming match, relative
to what a statistical model already knows from historical results and Elo ratings.

You must respond with ONLY a valid JSON object — no preamble, no explanation,
no markdown code fences. The JSON must follow this exact schema:

{
  "home_context_score": <integer from -3 to +3>,
  "away_context_score": <integer from -3 to +3>,
  "home_reasoning": "<one sentence explaining the home score>",
  "away_reasoning": "<one sentence explaining the away score>",
  "confidence": "<LOW | MEDIUM | HIGH>",
  "data_quality": "<GOOD | PARTIAL | NONE>"
}

Scoring guide:
+3: Major positive context (e.g. key player returns from injury, strong recent form)
+2: Moderate positive (e.g. good run of results, opposition missing key players)
+1: Slight positive (e.g. minor favourable news)
 0: Neutral or no meaningful context
-1: Slight negative (e.g. one player doubtful)
-2: Moderate negative (e.g. key player suspended, poor recent form)
-3: Major negative (e.g. multiple injuries, managerial crisis, 3-day turnaround)

confidence reflects how much you trust the news quality and completeness.
data_quality reflects whether the news contained match-relevant information.

If no news is available for either team, set both scores to 0 and
data_quality to NONE."""


def score_match_context(
    home_team: str,
    away_team: str,
    match_date: str = None,
) -> dict:
    """
    Fetch news for both teams and ask the LLM to score the match context.

    Returns a dict with context scores, or a safe default on any failure.
    The pipeline must never crash because this function failed.
    """
    safe_default = {
        "home_context_score": 0,
        "away_context_score": 0,
        "home_reasoning": "LLM layer unavailable.",
        "away_reasoning": "LLM layer unavailable.",
        "confidence": "LOW",
        "data_quality": "NONE",
        "error": None,
    }

    try:
        # Fetch news for both teams
        home_articles = fetch_team_news(home_team)
        away_articles = fetch_team_news(away_team)

        home_news_str = format_news_for_prompt(home_team, home_articles)
        away_news_str = format_news_for_prompt(away_team, away_articles)

        date_str = f"Match date: {match_date}\n" if match_date else ""

        user_prompt = f"""{date_str}
Fixture: {home_team} (home) vs {away_team} (away)

{home_news_str}

{away_news_str}

Analyse the news above and return your context assessment as JSON."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text.strip()
        result = parse_llm_response(raw)
        result["error"] = None
        return result

    except Exception as e:
        print(f"[LLM] Failed for {home_team} vs {away_team}: {e}")
        safe_default["error"] = str(e)
        return safe_default


def parse_llm_response(raw: str) -> dict:
    """
    Safely parse the LLM's JSON response.
    Handles common failure modes: extra whitespace, partial JSON,
    markdown fences accidentally included.
    """
    # Strip markdown fences if the model included them despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])  # strip first and last line

    parsed = json.loads(raw)  # let this raise — caller handles it

    # Validate and clamp scores to expected range
    parsed["home_context_score"] = max(-3, min(3, int(parsed["home_context_score"])))
    parsed["away_context_score"] = max(-3, min(3, int(parsed["away_context_score"])))

    # Validate enum fields
    if parsed.get("confidence") not in ("LOW", "MEDIUM", "HIGH"):
        parsed["confidence"] = "LOW"
    if parsed.get("data_quality") not in ("GOOD", "PARTIAL", "NONE"):
        parsed["data_quality"] = "PARTIAL"

    return parsed


def context_score_to_prob_adjustment(
    home_score: int,
    away_score: int,
    base_prob: float,
    max_adjustment: float = 0.08,
) -> float:
    """
    Convert context scores into a probability adjustment.

    The net score ranges from -6 to +6.
    We map this linearly to [-max_adjustment, +max_adjustment].

    max_adjustment=0.08 means the LLM can shift the probability
    by at most 8 percentage points in either direction.
    This keeps the statistical model in control.
    """
    net_score = home_score - away_score  # positive = home favoured by context
    adjustment = (net_score / 6) * max_adjustment
    adjusted_prob = base_prob + adjustment

    # Clamp to valid probability range
    return max(0.05, min(0.95, adjusted_prob))