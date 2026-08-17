from pipeline.fetch import fetch_all
from pipeline.parse import parse_csv
from pipeline.store import init_db, init_elo_table, store_matches
from pipeline.elo import run_elo
from pipeline.features import load_matches_with_elo, compute_rolling_form, build_feature_matrix
from pipeline.model import walk_forward_validate, plot_calibration_curve, train_final_model, print_summary
import pickle, os
from pipeline.store import init_context_table
from pipeline.backtest import run_backtest


def run():
    print("=== Initialising database ===")
    init_db()
    init_elo_table()

    print("\n=== Fetching CSVs ===")
    paths = fetch_all()

    print("\n=== Parsing and storing ===")
    for path in paths:
        df = parse_csv(path)
        if df is not None:
            store_matches(df)

    print("\n=== Computing Elo ratings ===")
    run_elo()

    print("\n=== Building features ===")
    df = load_matches_with_elo()
    df = compute_rolling_form(df)
    df, feature_cols = build_feature_matrix(df)
    print(f"[FEATURES] {len(df)} matches with full feature set")

    print("\n=== Walk-forward validation ===")
    predictions = walk_forward_validate(df, feature_cols)
    print_summary(predictions)

    print("\n=== Calibration curve ===")
    plot_calibration_curve(predictions)

    print("\n=== Training final model ===")
    model, scaler = train_final_model(df, feature_cols)
    # Save model and scaler for Phase 5
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": feature_cols}, f)
    print("[MODEL] Saved to models/model.pkl")
    
    # after training final model, add:
    print("\n=== Running backtest ===")
    predictions["match_date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    predictions["adjusted_prob"] = predictions["predicted_prob"]
    run_backtest(predictions)

    

if __name__ == "__main__":
    run()
    