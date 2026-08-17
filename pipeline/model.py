import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def season_sort_key(s):
    # "9394" -> 1993, "0102" -> 2001, "2324" -> 2023
    start = int(s[:2])
    return start + 2000 if start < 50 else start + 1900

def walk_forward_validate(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Train on all seasons up to N, test on season N+1.
    Returns a DataFrame of out-of-sample predictions.
    """
    # seasons = sorted(df["season"].unique())
    seasons = sorted(df["season"].unique(), key=season_sort_key)
    print(f"[MODEL] Seasons available: {seasons}")

    if len(seasons) < 2:
        raise ValueError("Need at least 2 seasons for walk-forward validation.")

    all_predictions = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season   = seasons[i]

        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        if len(train) < 50 or len(test) < 10:
            print(f"[MODEL] Skipping fold {test_season} — not enough data")
            continue

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_test  = test[feature_cols].values
        y_test  = test["target"].values

        # Scale features — logistic regression needs this
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # Train base model
        base_model = LogisticRegression(max_iter=1000, C=1.0)

        # Wrap with Platt scaling for calibration
        # cv=3 means it uses 3-fold CV internally to fit the calibration layer
        model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        model.fit(X_train_scaled, y_train)

        # Predict probabilities on test set
        probs = model.predict_proba(X_test_scaled)[:, 1]  # P(home win)

        fold_df = test.copy()
        fold_df["predicted_prob"] = probs
        fold_df["train_seasons"]  = str(train_seasons)
        fold_df["test_season"]    = test_season
        fold_df["correct"]        = (
            (probs >= 0.5) == (y_test == 1)
        ).astype(int)

        # Metrics for this fold
        acc    = fold_df["correct"].mean()
        brier  = brier_score_loss(y_test, probs)
        logloss = log_loss(y_test, probs)

        print(f"[FOLD] Test: {test_season} | "
            f"Train size: {len(train):>5} | "
            f"Test size: {len(test):>4} | "
            f"Accuracy: {acc:.3f} | "
            f"Brier: {brier:.3f} | "
            f"LogLoss: {logloss:.3f}")

        all_predictions.append(fold_df)

    return pd.concat(all_predictions, ignore_index=True)


def plot_calibration_curve(predictions: pd.DataFrame):
    """
    Reliability diagram: if the model is honest, predicted
    probabilities should match actual outcome rates per bucket.
    """
    y_true = predictions["target"].values
    y_prob = predictions["predicted_prob"].values

    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_predicted, fraction_of_positives,
        "s-", label="Model", color="steelblue"
    )
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives (actual win rate)")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=150)
    plt.show()
    print("[PLOT] Saved calibration_curve.png")


def train_final_model(df: pd.DataFrame, feature_cols: list):
    """
    Train on ALL available data. This is the model used for live predictions.
    Returns (model, scaler) tuple.
    """
    X = df[feature_cols].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_model = LogisticRegression(max_iter=1000, C=1.0)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_scaled, y)

    print(f"[MODEL] Final model trained on {len(df)} matches.")
    return model, scaler


def print_summary(predictions: pd.DataFrame):
    """Print per-season and per-league accuracy."""
    print("\n=== Per-season accuracy ===")
    summary = predictions.groupby("test_season").agg(
        matches=("correct", "count"),
        accuracy=("correct", "mean"),
        avg_pred_prob=("predicted_prob", "mean"),
        brier=("target", lambda x: brier_score_loss(
            x, predictions.loc[x.index, "predicted_prob"]
        ))
    ).round(3)
    print(summary.to_string())

    print("\n=== Per-league accuracy ===")
    league_summary = predictions.groupby("league").agg(
        matches=("correct", "count"),
        accuracy=("correct", "mean"),
        avg_pred_prob=("predicted_prob", "mean"),
    ).round(3)
    print(league_summary.to_string())