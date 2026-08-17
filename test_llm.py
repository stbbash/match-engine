from pipeline.llm import score_match_context, context_score_to_prob_adjustment

# Test with a real upcoming fixture
home = "Chelsea"
away = "Manchester United"
base_prob = 0.52  # what our logistic model predicts

print(f"Testing LLM context scorer: {home} vs {away}")
print("-" * 50)

result = score_match_context(home, away, match_date="2024-04-20")

print(f"Home score:      {result['home_context_score']}")
print(f"Away score:      {result['away_context_score']}")
print(f"Home reasoning:  {result['home_reasoning']}")
print(f"Away reasoning:  {result['away_reasoning']}")
print(f"Confidence:      {result['confidence']}")
print(f"Data quality:    {result['data_quality']}")
print(f"Error:           {result['error']}")
print()

adjusted = context_score_to_prob_adjustment(
    result["home_context_score"],
    result["away_context_score"],
    base_prob,
)
print(f"Base prob:       {base_prob:.3f}")
print(f"Adjusted prob:   {adjusted:.3f}")
print(f"Shift:           {adjusted - base_prob:+.3f}")