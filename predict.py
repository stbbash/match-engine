# predict.py (root file — this is what you run)
from pipeline.predict import run_predictions, print_predictions
from pipeline.fixtures import get_predictable_fixtures
import sys
sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("=== Fetching upcoming fixtures ===")
    fixtures = get_predictable_fixtures(days_ahead=15)

    if not fixtures:
        print("No predictable fixtures found.")
    else:
        print(f"\n=== Running predictions for {len(fixtures)} fixtures ===")
        predictions = run_predictions(fixtures, use_llm=False)
        print_predictions(predictions)
        predictions.to_csv("predictions_latest.csv", index=False)
        print("\n[DONE] Saved to predictions_latest.csv")