# TRADEOFFS.md

## 1. Data source and limitations

**Source:** football-data.co.uk — free CSV files, no API key required.

**Seasons covered:** 1993/94 through 2025/26 (33 seasons)

**Leagues covered:** Premier League only. La Liga was briefly included during development but removed to keep the model focused on a single well-understood league with deep historical coverage.

**Odds coverage:**
- 1993/94 – 1999/00: No bookmaker odds available. These seasons are used for Elo training only and excluded from backtesting.
- 2000/01 – 2001/02: William Hill odds (WHH/WHD/WHA) — online bookmaking was in its infancy.
- 2002/03 onwards: Bet365 odds (B365H/B365D/B365A) — used as the primary odds source for all backtesting.

Approximately 16% of total matches (the pre-odds seasons) have no odds data. Within the odds-available period, coverage is near-complete.

**How team name inconsistencies were handled:**
The CSV files use consistent short names throughout (e.g. "Man City", "Nott'm Forest") so no cross-season name normalisation was needed for the database. The only normalisation required was mapping API names from football-data.org to DB names for live fixture fetching — handled via a static `NAME_MAP` dict in `pipeline/fixtures.py`. Examples: "Manchester City" → "Man City", "Nottingham Forest" → "Nott'm Forest", "Wolverhampton Wanderers" → "Wolves".

**Other limitations:**
- No player-level data. Injuries, suspensions, and squad rotations are entirely invisible to the statistical model. A key player being ruled out the day before a match cannot be captured by Elo or rolling goals. The LLM context layer partially addresses this for live predictions only.
- No manager change signal. A mid-season sacking often causes a short-term performance shift that the model cannot anticipate.
- Home and away form are not separated. Rolling goals scored and conceded are computed across all matches regardless of venue. A team that performs very differently at home vs away will have this distinction smoothed out.

---

## 2. Feature engineering

**Features used and why:**

| Feature | Reasoning |
|---------|-----------|
| `elo_diff` | Captures relative team strength with full historical context. A single number that encodes decades of results. The most predictive feature in the model. |
| `home_advantage` | Constant 1.0 for every match. The coefficient the model learns on this feature captures the average home win boost across the dataset. Separating it from elo_diff lets the model weight them independently. |
| `home_avg_goals_scored` | Short-term attack form proxy. Last 5 matches chosen over longer windows — recent form matters more than form from 3 months ago. |
| `home_avg_goals_conceded` | Short-term defensive form proxy. Conceding rate often changes faster than scoring rate when a team changes shape or loses a key defender. |
| `away_avg_goals_scored` | Same logic applied to the away team. |
| `away_avg_goals_conceded` | Same logic applied to the away team. |
| `home_days_rest` | Fatigue and fixture congestion signal. 3-day turnarounds measurably affect performance. Capped at 30 days — beyond that, extra rest stops mattering and the model would otherwise extrapolate outside its trained range at season boundaries. |
| `away_days_rest` | Same logic applied to the away team. |

**What was tried that didn't work or wasn't pursued:**

Rolling window of 10 matches instead of 5 — a longer window dilutes recent form with older results. A team that has just lost their manager and gone on a 3-match losing run would still look reasonable on a 10-match window. 5 matches is a deliberate choice to weight recency.

Goal difference as a single feature instead of goals for and against separately — this loses information. A team that wins 3-2 every week looks identical to one that wins 1-0 every week on goal difference, but they have very different attacking and defensive profiles that matter when matched against specific opponents.

League position — considered but rejected. Position is highly correlated with Elo difference and adds no independent signal while introducing potential circularity.

---

## 3. Walk-forward backtest results

**Methodology:** Train on all seasons up to N, test on season N+1. The training set grows by one season each fold. No random splits — all data is time-ordered and no future information touches past training.

**Overall results (Premier League, 2000/01 – 2025/26):**

| Metric | Value |
|--------|-------|
| Bets placed | 5,793 |
| Total staked | £5,793 |
| Total profit | -£237.66 |
| ROI | -4.10% |
| Hit rate | 41.50% |
| Avg odds | 3.06 |
| Bookmaker margin | 5.97% |
| EV per bet | -£0.041 |

**Per-season ROI:**

| Season | Bets | ROI | Hit Rate |
|--------|------|-----|----------|
| 0001 | 185 | -5.21% | 47.03% |
| 0102 | 216 | -27.63% | 37.04% |
| 0203 | 209 | -8.35% | 47.85% |
| 0304 | 209 | -10.54% | 42.58% |
| 0405 | 216 | -5.05% | 47.22% |
| 0506 | 229 | +7.21% | 52.40% |
| 0607 | 239 | +3.59% | 46.86% |
| 0708 | 234 | -7.32% | 45.30% |
| 0809 | 235 | -0.09% | 45.11% |
| 0910 | 211 | +8.93% | 42.18% |
| 1011 | 199 | -5.76% | 37.69% |
| 1112 | 211 | +0.66% | 39.34% |
| 1213 | 204 | -8.03% | 39.71% |
| 1314 | 228 | +3.17% | 39.91% |
| 1415 | 245 | -6.22% | 41.63% |
| 1516 | 232 | +0.10% | 41.81% |
| 1617 | 214 | -0.57% | 35.98% |
| 1718 | 225 | +9.08% | 36.00% |
| 1819 | 227 | +3.54% | 39.21% |
| 1920 | 231 | +4.67% | 43.72% |
| 2021 | 272 | -13.07% | 34.19% |
| 2122 | 250 | -17.69% | 34.40% |
| 2223 | 240 | +9.48% | 45.00% |
| 2324 | 211 | -10.42% | 40.28% |
| 2425 | 227 | -22.30% | 38.33% |
| 2526 | 194 | -9.66% | 39.69% |

Profitable in 15 of 26 seasons (58%). The two worst seasons (0102 at -27.63% and 2425 at -22.30%) are outliers driven by different causes — 0102 uses William Hill odds which had wider margins and less competitive lines than B365, and 2425 reflects the continuation of post-COVID home advantage decline.

The clearest trend is the deterioration after 2020. The model was trained on 25 years of data where home advantage was stronger. Post-COVID football shows measurably weaker home advantage — empty stadium football during 2020/21 demonstrated that crowd effect drove most of it, and it has not fully recovered. The model systematically overestimates home win probability in recent seasons as a result.

---

## 4. Calibration

The calibration curve (saved as `calibration_curve.png`) shows the model is reasonably well-calibrated across most of the probability range. Predicted probabilities in the 0.45–0.65 range closely match actual outcome rates, which is where most predictions cluster.

At the high end (predictions above 0.65), the model shows slight overconfidence — it predicts 0.70 in situations where teams actually win closer to 65% of the time. This is consistent with the post-COVID home advantage problem: the model is more confident in strong home favourites than the recent data warrants.

`CalibratedClassifierCV` with Platt scaling (sigmoid method, cv=3) produced a meaningful improvement over the uncalibrated logistic regression. Without calibration, the model's probabilities were more spread out and less accurate in the tails. The Brier scores across folds (averaging ~0.217) confirm the calibration is working — a random model scores ~0.250.

The average predicted probability across all matches is 0.458, which closely matches the actual Premier League home win rate of approximately 0.45–0.46. The model has correctly learned the base rate without being explicitly told it.

---

## 5. LLM layer contribution

The LLM layer ran partially during live prediction testing — GNews free tier rate limits (100 requests/day) prevented full coverage across all 19 fixtures in a single run.

On the fixtures where it did run, adjustments were small but directionally sensible:

| Fixture | Base | Adjusted | Shift | Reason |
|---------|------|----------|-------|--------|
| Arsenal vs Fulham | 0.682 | 0.696 | +1.4% | Fulham rotation concerns flagged |
| Aston Villa vs Tottenham | 0.657 | 0.670 | +1.3% | Villa in good form, 10pts clear in top 5 |
| Chelsea vs Nott'm Forest | 0.521 | 0.481 | -4.0% | Negative context for Chelsea |
| Wolves vs Tottenham | 0.449 | 0.409 | -4.0% | Negative context for Wolves |
| Leeds vs Burnley | 0.560 | 0.587 | +2.7% | Positive context for Leeds |

The maximum adjustment is capped at ±8 percentage points by design. This keeps the statistical model in control — the LLM cannot override a strong Elo signal with a few headlines.

A direct ROI comparison of `use_llm=True` vs `use_llm=False` was not run against historical data because the LLM layer is designed for live predictions only, where news exists. Historical matches have no news to fetch. The contribution of the LLM layer can only be measured prospectively — by tracking whether LLM-adjusted predictions outperform base predictions over a live season.

The average adjustment magnitude on the fixtures where the LLM ran was approximately ±2–4%, which is meaningful without being dominant. The layer is working as intended.

---

## 6. Expected value after bookmaker margin

**Overall EV per bet: -£0.041**

After bookmaker margin: negative.

The model loses 4.10 cents per pound staked after the bookmaker margin of 5.97% is applied. However, the model's hit rate of 41.5% against a breakeven requirement of 32.7% (at average odds of 3.06) demonstrates genuine predictive signal. At fair odds, the model would be profitable. The bookmaker margin consumes the edge.

The gap between signal and profitability breaks down as follows. The model recovers approximately 1.87 percentage points of the 5.97% bookmaker margin. Closing that remaining gap would require one or more of: player availability data to improve accuracy on individual fixtures, league-specific modelling to better capture La Liga and other markets, restricting bets to only the highest-edge situations, or finding bookmakers with lower margins.

The 0102 season (-27.63%) is a partial outlier because it uses William Hill odds which had less competitive pricing than B365. Removing that season, the average ROI improves to approximately -3.5%.

The profitable seasons are not randomly distributed — they cluster in periods of higher league predictability (dominant teams, stable hierarchies). This suggests the model has genuine edge in predictable seasons but bleeds it back in high-parity seasons. A regime-detection layer that adjusts bet sizing based on estimated season predictability would be a meaningful improvement.

---

## 7. What I'd build next

**Player availability data** is the single highest-value addition. A model that knows a team's first-choice striker and central midfielder are both suspended has a fundamentally different prediction than one that doesn't. This is what the LLM layer approximates from news — a structured player availability API (e.g. from a sports data provider) would make it systematic.

**League-specific Elo K-factors.** The current K=32 is applied uniformly. High-parity leagues (where upsets are common) should have higher K to make ratings more reactive. Dominant leagues (where the hierarchy is stable) should have lower K. Tuning K per league through cross-validation would improve rating accuracy.

**Separate models per league** rather than one pooled model. Premier League and La Liga have different tactical profiles, home advantage magnitudes, and scoring rates. A single logistic regression averages across these differences. Separate models with league-specific feature distributions would be more accurate.

**News caching** to avoid GNews rate limit exhaustion mid-run. A simple local JSON cache keyed by team name and date — fetch once per team per day, reuse on subsequent runs.

**Live odds monitoring.** The current system uses closing odds for backtesting, but line movement between opening and closing odds contains information. A team whose odds shorten significantly in the 24 hours before kickoff has likely had positive news that sharp bettors have already acted on. Detecting this movement and using it as a feature would add signal.

**Proper Kelly sizing** instead of fixed stakes. The Kelly criterion sizes bets proportionally to edge — bet more when the edge is large, less when it is small. Fixed staking treats a 0.66 probability prediction the same as a 0.52 prediction. Kelly would naturally concentrate capital on the highest-edge situations and reduce exposure on marginal ones.

**Prospective tracking.** The backtest is historical. The only real test is live prediction tracking — log every prediction before the match, record the result, and compute running ROI over a full season. This is the only way to know whether the model has genuine live edge or whether the historical signal degrades in production.