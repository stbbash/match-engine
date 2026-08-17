import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

DB_PATH = "db/matches.db"
STAKE = 1.0  # fixed stake per bet — keeps ROI comparable across experiments


def load_backtest_data() -> pd.DataFrame:
    """
    Load all matches that have:
    - Elo ratings (computed)
    - Model predictions (from walk-forward validation)
    - Bookmaker odds
    """
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            m.id,
            m.date,
            m.home_team,
            m.away_team,
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
        WHERE m.odds_home IS NOT NULL
        AND m.odds_draw IS NOT NULL
        AND m.odds_away IS NOT NULL
        AND m.odds_home > 1.0
        AND m.odds_draw > 1.0
        AND m.odds_away > 1.0
        ORDER BY m.date ASC, m.id ASC
    """, con, parse_dates=["date"])
    con.close()
    return df


def compute_margin(row) -> float:
    return (1/row["odds_home"] + 1/row["odds_draw"] + 1/row["odds_away"]) - 1


def fair_probability(odds_home, odds_draw, odds_away) -> tuple:
    implied = [1/odds_home, 1/odds_draw, 1/odds_away]
    total = sum(implied)
    return tuple(i / total for i in implied)


def simulate_bets(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge model predictions with historical odds and simulate bets.
    Strategy: bet on home win when model probability > fair bookmaker probability.
    """
    df = df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    merged = df.merge(
        predictions[["home_team", "away_team", "match_date", "adjusted_prob"]],
        left_on=["home_team", "away_team", "date_str"],
        right_on=["home_team", "away_team", "match_date"],
        how="inner"
    )

    if merged.empty:
        print("[BACKTEST] No matching predictions found.")
        print("  → Run walk_forward_validate and save predictions first.")
        return pd.DataFrame()

    results = []

    for _, row in merged.iterrows():
        fair_home, fair_draw, fair_away = fair_probability(
            row["odds_home"], row["odds_draw"], row["odds_away"]
        )
        margin = compute_margin(row)
        model_prob = row["adjusted_prob"]

        has_edge = model_prob > fair_home
        actual_result = row["result"]
        bet_won = actual_result == "H"

        if has_edge:
            profit = (row["odds_home"] - 1) * STAKE if bet_won else -STAKE
        else:
            profit = 0.0

        results.append({
            "date":       row["date"],
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "league":     row["league"],
            "season":     row["season"],
            "result":     actual_result,
            "model_prob": round(model_prob, 4),
            "fair_prob":  round(fair_home, 4),
            "book_odds":  row["odds_home"],
            "margin":     round(margin, 4),
            "has_edge":   has_edge,
            "bet_won":    bet_won if has_edge else None,
            "profit":     round(profit, 4),
        })

    return pd.DataFrame(results)


def compute_roi(bets: pd.DataFrame) -> float:
    """ROI on bets placed only (ignoring no-bet rows)."""
    placed = bets[bets["has_edge"]]
    if placed.empty:
        return 0.0
    total_staked = len(placed) * STAKE
    total_profit = placed["profit"].sum()
    return total_profit / total_staked


def evaluate(bets: pd.DataFrame):
    """Full evaluation: overall, per league, per confidence tier."""

    placed = bets[bets["has_edge"]].copy()

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    if placed.empty:
        print("No bets placed — model never exceeded fair bookmaker probability.")
        print("This means you have no edge. That's the correct result for v1.")
        return

    # --- Overall ---
    total_bets   = len(placed)
    total_staked = total_bets * STAKE
    total_profit = placed["profit"].sum()
    roi          = total_profit / total_staked
    hit_rate     = placed["bet_won"].mean()
    avg_odds     = placed["book_odds"].mean()
    avg_margin   = bets["margin"].mean()

    print(f"\nOVERALL")
    print(f"  Bets placed:    {total_bets}")
    print(f"  Total staked:   £{total_staked:.2f}")
    print(f"  Total profit:   £{total_profit:.2f}")
    print(f"  ROI:            {roi:.2%}")
    print(f"  Hit rate:       {hit_rate:.2%}")
    print(f"  Avg odds:       {avg_odds:.2f}")
    print(f"  Avg bk margin:  {avg_margin:.2%}")

    ev_per_bet = placed["profit"].mean()
    print(f"  EV per bet:     £{ev_per_bet:.4f}")

    if roi > 0:
        print(f"\n  ⚠ Positive ROI detected. Before celebrating:")
        print(f"    1. Is this consistent across all seasons or driven by one fold?")
        print(f"    2. Does it survive if you reduce max_adjustment to 0.04?")
        print(f"    3. What's the Sharpe ratio? High variance ROI is not real edge.")
    else:
        print(f"\n  Expected result for v1: no edge after bookmaker margin.")
        print(f"  Document why in TRADEOFFS.md.")

    # --- Per league ---
    print(f"\nPER LEAGUE")
    league_stats = placed.groupby("league").apply(lambda g: pd.Series({
        "bets":    len(g),
        "roi":     g["profit"].sum() / (len(g) * STAKE),
        "hit_rate":g["bet_won"].mean(),
        "avg_odds":g["book_odds"].mean(),
    })).round(4)
    print(league_stats.to_string())

    # --- Per season ---
    print(f"\nPER SEASON")
    season_stats = placed.groupby("season").apply(lambda g: pd.Series({
        "bets":    len(g),
        "roi":     g["profit"].sum() / (len(g) * STAKE),
        "hit_rate":g["bet_won"].mean(),
    })).round(4)
    print(season_stats.to_string())

    print("=" * 70)


def plot_cumulative_pnl(bets: pd.DataFrame):
    """Plot cumulative profit/loss over time for placed bets."""
    placed = bets[bets["has_edge"]].copy().sort_values("date")

    if placed.empty:
        return

    placed["cumulative_pnl"] = placed["profit"].cumsum()

    plt.figure(figsize=(12, 5))
    plt.plot(placed["date"], placed["cumulative_pnl"],
            color="steelblue", linewidth=1.5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.fill_between(
        placed["date"], placed["cumulative_pnl"], 0,
        where=placed["cumulative_pnl"] >= 0,
        alpha=0.2, color="green", label="Profit"
    )
    plt.fill_between(
        placed["date"], placed["cumulative_pnl"], 0,
        where=placed["cumulative_pnl"] < 0,
        alpha=0.2, color="red", label="Loss"
    )
    plt.xlabel("Date")
    plt.ylabel("Cumulative P&L (£)")
    plt.title("Cumulative P&L — Fixed Stake Betting Strategy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cumulative_pnl.png", dpi=150)
    plt.show()
    print("[PLOT] Saved cumulative_pnl.png")


def plot_roi_by_league(bets: pd.DataFrame):
    """Bar chart of ROI per league."""
    placed = bets[bets["has_edge"]].copy()

    if placed.empty:
        return

    league_roi = placed.groupby("league").apply(
        lambda g: g["profit"].sum() / (len(g) * STAKE)
    ).sort_values()

    colors = ["green" if r > 0 else "red" for r in league_roi]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(league_roi.index, league_roi.values, color=colors, alpha=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("ROI")
    plt.title("ROI by League (after bookmaker margin)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("roi_by_league.png", dpi=150)
    plt.show()
    print("[PLOT] Saved roi_by_league.png")


def run_backtest(predictions: pd.DataFrame):
    """Main entry point. Pass in your walk-forward predictions DataFrame."""
    print("[BACKTEST] Loading historical odds data...")
    df = load_backtest_data()
    print(f"[BACKTEST] {len(df)} matches with odds available")

    print("[BACKTEST] Simulating bets...")
    bets = simulate_bets(df, predictions)

    if bets.empty:
        return

    evaluate(bets)
    plot_cumulative_pnl(bets)
    plot_roi_by_league(bets)

    bets.to_csv("backtest_results.csv", index=False)
    print("\n[BACKTEST] Full results saved to backtest_results.csv")

    return bets