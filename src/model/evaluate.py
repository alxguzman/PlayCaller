"""
evaluate.py
-----------
Evaluates the trained model on the held-out season saved in its artifact:
  - accuracy, ROC-AUC, log-loss
  - confusion matrix
  - SHAP values: which features actually drive the predictions?

Run with:  python -m src.model.evaluate
(Charts are saved to the model/ folder when run as a script; the
functions also return figures so notebook 04 can display them inline.)
"""

import os

import joblib
import numpy as np

# matplotlib is only needed for the plotting/eval functions below, not for
# load_artifact() which the API imports at startup. Keep it optional so the
# slim production image (see requirements-deploy.txt) can import this module
# without it. plt stays defined (None) so the plot functions still resolve.
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from src.model.train import (
    MODEL_DIR,
    MODEL_PATH,
    TEST_SEASON,
    load_processed,
    prepare_data,
    split_by_season,
)


def load_artifact():
    """Load the saved model artifact (model + encoders + lookups)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found - run `python -m src.model.train` first."
        )
    return joblib.load(MODEL_PATH)


def get_test_set(artifact=None):
    """Rebuild the held-out test set exactly as train.py defined it."""
    df = load_processed()
    X, y, meta = prepare_data(df)
    artifact = artifact or load_artifact()
    *_, X_test, y_test = split_by_season(
        X, y, meta, artifact.get("test_season", TEST_SEASON),
        artifact.get("val_season", TEST_SEASON - 1),
    )
    X_test = X_test.reindex(columns=artifact["feature_columns"], fill_value=0)
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
    test_season = artifact.get("test_season", TEST_SEASON)
    model = artifact["model"]
    X_test, y_test = get_test_set(artifact)

    if len(y_test) == 0:
        print(f"No {test_season} plays in the dataset yet - nothing to evaluate.")
        return None

    proba = model.predict_proba(X_test)[:, 1]   # P(success) per play
    pred = (proba >= 0.5).astype(int)

    print("=" * 50)
    print(f"EVALUATION - held-out {test_season} season")
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
    ax.set_title(f"Confusion matrix - {test_season} test season")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def evaluate_epa_model(artifact=None):
    """
    Headline metrics for the EPA regressor on the held-out test season.

    Expectations check: R-squared will look tiny (~0.05). That's normal -
    play-level EPA is dominated by randomness no pre-snap model can see.
    What matters is beating the naive baseline (always predict the mean)
    and, for the recommender, whether the AVERAGE predicted EPA per play
    call separates the good calls from the bad ones.
    """
    from sklearn.metrics import mean_squared_error, r2_score

    artifact = artifact or load_artifact()
    test_season = artifact.get("test_season", TEST_SEASON)
    epa_model = artifact.get("epa_model")
    if epa_model is None:
        print("No EPA regressor in this artifact - re-run training.")
        return

    df = load_processed()
    X, y, meta = prepare_data(df)
    is_test = meta["season"] == test_season
    X_test = X[is_test].reindex(columns=artifact["feature_columns"], fill_value=0)
    epa_test = meta.loc[is_test, "epa"]

    if len(epa_test) == 0:
        print(f"No {test_season} plays in the dataset yet - nothing to evaluate.")
        return

    pred = epa_model.predict(X_test)
    rmse = mean_squared_error(epa_test, pred) ** 0.5
    baseline = artifact.get("epa_baseline")
    if baseline is None:
        # Compatibility for old artifacts: reconstruct their training split.
        X_train, *_ = split_by_season(X, y, meta, test_season,
                                     artifact.get("val_season", test_season - 1))
        baseline = float(meta.loc[X_train.index, "epa"].mean())
    baseline_rmse = mean_squared_error(epa_test, np.full(len(epa_test), baseline)) ** 0.5

    print("=" * 50)
    print(f"EPA REGRESSOR - held-out {test_season} season")
    print("=" * 50)
    print(f"RMSE          : {rmse:.4f}")
    print(f"Baseline RMSE : {baseline_rmse:.4f} (training-set mean)")
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
    test_season = artifact.get("test_season", TEST_SEASON)
    model = artifact["model"]
    X_test, y_test = get_test_set(artifact)

    if len(y_test) == 0:
        print(f"No {test_season} plays in the dataset yet - nothing to evaluate.")
        return None

    proba = model.predict_proba(X_test)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", color="steelblue", label="model")
    ax.set_xlabel("Predicted success probability")
    ax.set_ylabel("Actual success rate")
    ax.set_title(f"Calibration - {test_season} test season")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def model_recommendations(X_situations, artifact, call_credibility=None):
    """
    What the model would have called for every row of X_situations
    (one-hot encoded plays, same columns as training).

    Vectorized version of recommend_play's ranking policy - the
    floor-and-ceiling blend plus the run/pass balance and call-credibility
    credits and the guardrails (no sneaks beyond 2 yards, trick plays
    never top-ranked) - so it grades the exact policy the API ships.
    Falls back to pure success_prob if the artifact has no EPA regressor.

    Returns a numpy array of concept names, one per row.
    """
    from src.model.recommend import (
        CALL_CREDIBILITY,
        RUN_PASS_BALANCE,
        SUCCESS_FLOOR_GAP,
        _MIN_CALL_PROB,
        run_credibility,
    )
    from src.model.train import RUN_CONCEPTS

    if call_credibility is None:
        call_credibility = CALL_CREDIBILITY

    model = artifact["model"]
    epa_model = artifact.get("epa_model")
    concepts = artifact["concept_classes"]
    concept_cols = [f"play_concept_{c}" for c in concepts]

    # Score every play once per candidate concept: zero out the play_concept
    # one-hot columns, switch on one concept at a time, predict with BOTH
    # models (success prob for the floor, expected EPA for the ranking).
    probs = np.zeros((len(X_situations), len(concepts)))
    epas = np.zeros((len(X_situations), len(concepts)))
    for j, concept in enumerate(concepts):
        X_c = X_situations.copy()
        X_c[concept_cols] = False
        X_c[f"play_concept_{concept}"] = True
        # Recompute call-dependent inputs for the hypothetical call. Keeping
        # the coach's original box interaction gives a contradictory row.
        if "defenders_in_box" in X_c:
            is_run = concept in RUN_CONCEPTS
            if "box_x_run" in X_c:
                X_c["box_x_run"] = X_c["defenders_in_box"] * is_run
            if "box_x_pass" in X_c:
                X_c["box_x_pass"] = X_c["defenders_in_box"] * (not is_run)
        probs[:, j] = model.predict_proba(X_c)[:, 1]
        if epa_model is not None:
            epas[:, j] = epa_model.predict(X_c)

    ydstogo = X_situations["ydstogo"].to_numpy()
    if "qb_sneak" in concepts:
        probs[ydstogo > 2, concepts.index("qb_sneak")] = -np.inf
    if "trick_play" in concepts:
        probs[:, concepts.index("trick_play")] = -np.inf

    if epa_model is not None:
        # Floor-and-ceiling: eligible = within the gap of the safest play;
        # recommendation = highest balance-adjusted EPA among eligible.
        is_run = np.array([c in RUN_CONCEPTS for c in concepts])
        down = X_situations["down"].to_numpy()
        cred = np.array([run_credibility(d, t) for d, t in zip(down, ydstogo)])
        credit = RUN_PASS_BALANCE * cred[:, None] * is_run[None, :]

        # Call-credibility credit (see recommend.py): rarely-called plays
        # pay a penalty proportional to log P(coach calls this here).
        call_model = artifact.get("call_model")
        if call_model is not None and call_credibility > 0:
            call_p = call_model.predict_proba(X_situations[artifact["call_features"]])
            credit = credit + call_credibility * np.log(
                np.clip(call_p, _MIN_CALL_PROB, None)
            )

        floor = probs.max(axis=1, keepdims=True) - SUCCESS_FLOOR_GAP
        score = np.where(probs >= floor, epas + credit, -np.inf)
    else:
        score = probs

    return np.array(concepts)[score.argmax(axis=1)]


def recommendation_agreement(artifact=None, show=False):
    """
    Grade the model AS A RECOMMENDER, not a predictor.

    For every test-season play we compute what the model would have called in
    that exact situation (same teams, same field position, same
    formation). Then we split plays into "coach agreed with the model"
    vs "coach called something else" and compare actual outcomes.

    If agreed-plays show higher EPA and success rate, the model's
    preferences carry real signal. Caveat: agreement isn't random
    (selection bias again), so this is evidence, not proof. And note what
    this does NOT claim: when the coach called something else, we never
    observe how the model's play would have gone (the counterfactual),
    so disagreement is never counted as the model being wrong.
    """
    artifact = artifact or load_artifact()
    test_season = artifact.get("test_season", TEST_SEASON)

    df = load_processed()
    X, y, meta = prepare_data(df)
    is_test = (meta["season"] == test_season).to_numpy()
    if is_test.sum() == 0:
        print(f"No {test_season} plays in the dataset yet - nothing to evaluate.")
        return None, None
    X_test = X[is_test].reindex(columns=artifact["feature_columns"], fill_value=0)
    actual_concept = meta.loc[is_test, "play_concept"].to_numpy()
    actual_epa = meta.loc[is_test, "epa"].to_numpy()
    actual_success = y[is_test].to_numpy()

    recommended = model_recommendations(X_test, artifact)
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
    print(f"RECOMMENDER EVALUATION - {test_season} season")
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
    X_test, _ = get_test_set(artifact)

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
    # Each function returns None when the test season has no plays yet
    # (e.g. right after the season rollover, before week 1 is played).
    fig_cm = evaluate()
    if fig_cm is None:
        raise SystemExit(0)
    fig_cm.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)

    evaluate_epa_model()

    fig_cal = calibration()
    fig_cal.savefig(os.path.join(MODEL_DIR, "calibration.png"), dpi=150)

    _, fig_agree = recommendation_agreement()
    fig_agree.savefig(os.path.join(MODEL_DIR, "recommendation_agreement.png"), dpi=150)

    fig_shap = shap_summary()
    fig_shap.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), dpi=150)
    print(f"\nCharts saved to {MODEL_DIR}")
