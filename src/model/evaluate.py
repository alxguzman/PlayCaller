"""
evaluate.py
-----------
Evaluates the trained model on the held-out 2025 season:
  - accuracy, ROC-AUC, log-loss
  - confusion matrix
  - SHAP values: which features actually drive the predictions?

Run with:  python -m src.model.evaluate
(Charts are saved to the model/ folder when run as a script; the
functions also return figures so notebook 04 can display them inline.)
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from src.model.train import MODEL_DIR, MODEL_PATH, load_processed, prepare_data, split_by_season


def load_artifact():
    """Load the saved model artifact (model + encoders + lookups)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found - run `python -m src.model.train` first."
        )
    return joblib.load(MODEL_PATH)


def get_test_set():
    """Rebuild the held-out 2025 test set exactly as train.py defined it."""
    df = load_processed()
    X, y, meta = prepare_data(df)
    *_, X_test, y_test = split_by_season(X, y, meta)
    return X_test, y_test


def evaluate(artifact=None, show=False):
    """
    Print headline metrics and build a confusion-matrix figure.

    A note on expectations: play success is genuinely hard to predict -
    even Vegas can't tell you if a specific play will work. An AUC around
    0.60-0.65 is normal here. The model's value is in the RELATIVE
    ranking of play calls, not in predicting single plays.
    """
    artifact = artifact or load_artifact()
    model = artifact["model"]
    X_test, y_test = get_test_set()

    proba = model.predict_proba(X_test)[:, 1]   # P(success) per play
    pred = (proba >= 0.5).astype(int)

    print("=" * 50)
    print("EVALUATION - held-out 2025 season")
    print("=" * 50)
    print(f"Plays      : {len(y_test):,}")
    print(f"Accuracy   : {accuracy_score(y_test, pred):.3f}")
    print(f"ROC-AUC    : {roc_auc_score(y_test, proba):.3f}")
    print(f"Log-loss   : {log_loss(y_test, proba):.3f}")
    print(f"Base rate  : {y_test.mean():.3f} (always-predict-fail accuracy: {1 - y_test.mean():.3f})")

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["fail", "success"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix - 2025 test season")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def evaluate_epa_model(artifact=None):
    """
    Headline metrics for the EPA regressor on the 2025 test season.

    Expectations check: R-squared will look tiny (~0.05). That's normal -
    play-level EPA is dominated by randomness no pre-snap model can see.
    What matters is beating the naive baseline (always predict the mean)
    and, for the recommender, whether the AVERAGE predicted EPA per play
    call separates the good calls from the bad ones.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    artifact = artifact or load_artifact()
    epa_model = artifact.get("epa_model")
    if epa_model is None:
        print("No EPA regressor in this artifact - re-run training.")
        return

    df = load_processed()
    X, y, meta = prepare_data(df)
    is_test = meta["season"] == 2025
    X_test = X[is_test]
    epa_test = meta.loc[is_test, "epa"]

    pred = epa_model.predict(X_test)
    rmse = mean_squared_error(epa_test, pred) ** 0.5
    baseline_rmse = mean_squared_error(
        epa_test, np.full(len(epa_test), epa_test.mean())
    ) ** 0.5

    print("=" * 50)
    print("EPA REGRESSOR - held-out 2025 season")
    print("=" * 50)
    print(f"RMSE          : {rmse:.4f}")
    print(f"Baseline RMSE : {baseline_rmse:.4f} (always predict the mean)")
    print(f"R-squared     : {r2_score(epa_test, pred):.4f}")


def calibration(artifact=None, n_bins=10, show=False):
    """
    Reliability diagram: when the model says "60% success", do those plays
    actually succeed ~60% of the time?

    A recommender can rank plays correctly while quoting bogus
    probabilities. Good calibration means the numbers we show users
    ("74% success chance") are trustworthy, not just the ordering.
    Points on the diagonal = perfectly calibrated.
    """
    from sklearn.calibration import calibration_curve

    artifact = artifact or load_artifact()
    model = artifact["model"]
    X_test, y_test = get_test_set()
    proba = model.predict_proba(X_test)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color="steelblue", label="model")
    ax.set_xlabel("Predicted success probability")
    ax.set_ylabel("Actual success rate")
    ax.set_title("Calibration - 2025 test season")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def recommendation_agreement(artifact=None, show=False):
    """
    Grade the model AS A RECOMMENDER, not a predictor.

    For every 2025 play we compute what the model would have called in
    that exact situation (same teams, same field position, same
    formation). Then we split plays into "coach agreed with the model"
    vs "coach called something else" and compare actual outcomes.

    If agreed-plays show higher EPA and success rate, the model's
    preferences carry real signal. Caveat: agreement isn't random
    (selection bias again), so this is evidence, not proof.

    Uses the same ranking policy as recommend_play - the floor-and-ceiling
    blend: pick the highest expected EPA (regressor) among plays whose
    success probability (classifier) is within SUCCESS_FLOOR_GAP of the
    safest option. Falls back to pure success_prob if no regressor.
    """
    from src.model.recommend import SUCCESS_FLOOR_GAP

    artifact = artifact or load_artifact()
    model = artifact["model"]
    epa_model = artifact.get("epa_model")

    df = load_processed()
    X, y, meta = prepare_data(df)
    is_test = (meta["season"] == 2025).to_numpy()
    X_test = X[is_test]
    actual_concept = meta.loc[is_test, "play_concept"].to_numpy()
    actual_epa = meta.loc[is_test, "epa"].to_numpy()
    actual_success = y[is_test].to_numpy()

    concepts = artifact["concept_classes"]
    concept_cols = [f"play_concept_{c}" for c in concepts]

    # Score every play once per candidate concept: zero out the play_concept
    # one-hot columns, switch on one concept at a time, predict with BOTH
    # models (success prob for the floor, expected EPA for the ranking).
    probs = np.zeros((len(X_test), len(concepts)))
    epas = np.zeros((len(X_test), len(concepts)))
    for j, concept in enumerate(concepts):
        X_c = X_test.copy()
        X_c[concept_cols] = False
        X_c[f"play_concept_{concept}"] = True
        probs[:, j] = model.predict_proba(X_c)[:, 1]
        if epa_model is not None:
            epas[:, j] = epa_model.predict(X_c)

    # Same guardrails as recommend_play: no sneaks beyond 2 yards to go,
    # and trick plays are never the recommendation.
    ydstogo = X_test["ydstogo"].to_numpy()
    if "qb_sneak" in concepts:
        probs[ydstogo > 2, concepts.index("qb_sneak")] = -np.inf
    if "trick_play" in concepts:
        probs[:, concepts.index("trick_play")] = -np.inf

    if epa_model is not None:
        # Floor-and-ceiling: eligible = within the gap of the safest play;
        # recommendation = highest expected EPA among eligible.
        floor = probs.max(axis=1, keepdims=True) - SUCCESS_FLOOR_GAP
        score = np.where(probs >= floor, epas, -np.inf)
    else:
        score = probs

    recommended = np.array(concepts)[score.argmax(axis=1)]
    agreed = recommended == actual_concept

    summary = pd.DataFrame({
        "plays": [agreed.sum(), (~agreed).sum()],
        "actual_success_rate": [
            actual_success[agreed].mean(), actual_success[~agreed].mean(),
        ],
        "actual_epa_per_play": [
            actual_epa[agreed].mean(), actual_epa[~agreed].mean(),
        ],
    }, index=["coach agreed with model", "coach called something else"])

    print("=" * 60)
    print("RECOMMENDER EVALUATION - 2025 season")
    print("=" * 60)
    print(f"Agreement rate: {agreed.mean():.1%} of plays\n")
    print(summary.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, label in zip(
        axes,
        ["actual_success_rate", "actual_epa_per_play"],
        ["Actual success rate", "Actual EPA per play"],
    ):
        ax.bar(["agreed", "disagreed"], summary[col],
               color=["seagreen", "lightgray"])
        ax.set_title(label)
        ax.axhline(0, color="gray", linewidth=1)
    fig.suptitle("Outcomes when the coach's call matched the model's recommendation")
    fig.tight_layout()

    if show:
        plt.show()
    return summary, fig


def shap_summary(artifact=None, sample_size=2000, show=False):
    """
    SHAP summary chart: ranks features by how much they move predictions.

    We explain a random sample (default 2,000 plays) instead of all 30k+
    test plays - SHAP is slow and the picture doesn't change.
    """
    import shap  # imported here: it's slow to import and optional

    artifact = artifact or load_artifact()
    model = artifact["model"]
    X_test, _ = get_test_set()

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig = plt.figure()
    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    fig = plt.gcf()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    fig_cm = evaluate()
    fig_cm.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)

    evaluate_epa_model()

    fig_cal = calibration()
    fig_cal.savefig(os.path.join(MODEL_DIR, "calibration.png"), dpi=150)

    _, fig_agree = recommendation_agreement()
    fig_agree.savefig(os.path.join(MODEL_DIR, "recommendation_agreement.png"), dpi=150)

    fig_shap = shap_summary()
    fig_shap.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), dpi=150)
    print(f"\nCharts saved to {MODEL_DIR}")
