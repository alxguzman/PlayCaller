"""
train.py
--------
Trains the play-success model that powers the recommender.

THE KEY IDEA (what makes this a recommender, not a classifier):
We do NOT predict "what play will the coach call" - that would just teach
the model to copy historical coaching tendencies. Instead we predict
P(success) for a play given (game situation + team form + the play call).
The play call - play_concept and formation - is an INPUT. At
recommendation time we score the same situation once per candidate call
and rank them: "in this spot, against this defense, a play-action deep
shot succeeds 48% of the time, an inside run 41%..."

Train/test split is BY SEASON (the newest season is held out). Random
splitting time-series data leaks information: plays from the same game
would land in both train and test.

Run with:  python -m src.model.train
"""

import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pbp_features.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "play_success_model.joblib")

TARGET = "success"

# Legacy artifact defaults. New training derives the split from the data
# and persists it in the artifact, so evaluation survives season rollover.
TEST_SEASON = 2025
VAL_SEASON = 2024
VAL_WEEK_FROM = 15    # first week of VAL_SEASON that goes to validation

# Situation features: everything here is knowable BEFORE the snap.
# Outcome columns (epa, yards_gained, air_yards...) must never appear -
# that would be the model "predicting" with the answer in hand.
NUMERIC_FEATURES = [
    "down", "ydstogo", "yardline_100",
    "qtr", "game_seconds_remaining", "score_differential",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "goal_to_go", "spread_line", "wp", "temp", "wind",
    "defenders_in_box",  # pre-snap look (NaN when not charted - XGBoost handles it)
    "box_x_run", "box_x_pass",  # built by add_box_interactions()
]

RUN_CONCEPTS = {"inside_run", "off_tackle_run", "outside_run", "qb_sneak"}
OPTIONAL_CONTEXT_COLUMNS = {"defenders_in_box", "box_x_run", "box_x_pass"}
# Selected by tune_epa on 2024/2025 temporal validation, not 2026 outcomes.
EPA_MODEL_PARAMS = {"max_depth": 3, "min_child_weight": 100, "reg_lambda": 20}


def mask_optional_context(X):
    """Match in-season/live requests where participation is unavailable."""
    masked = X.copy()
    for column in OPTIONAL_CONTEXT_COLUMNS.intersection(masked.columns):
        masked[column] = np.nan
    masked.loc[:, masked.columns.str.startswith("personnel_")] = False
    return masked


def augment_optional_context(X, y):
    """Give each historical play a charted and an uncharted representation.

    Half weights preserve each original play's total training weight.
    This teaches missing-value paths without discarding known box/personnel
    inputs, and never uses held-out outcomes to construct training data.
    """
    return (
        pd.concat([X, mask_optional_context(X)], ignore_index=True),
        pd.concat([y, y], ignore_index=True),
        np.full(len(X) * 2, 0.5),
    )


def add_box_interactions(df):
    """
    Split the box count into two columns: one that's live on run plays,
    one that's live on pass plays (the other is 0).

    Why: a loaded box hurts runs but HELPS passes (fewer defenders deep) -
    opposite directions. As a single column the tree has to discover that
    interaction with play_concept on its own, and in practice it learned
    ~1/4 of the real effect. These columns hand it the interaction
    directly. NaN (uncharted play) stays NaN in both.
    """
    df = df.copy()
    box = df["defenders_in_box"] if "defenders_in_box" in df.columns else np.nan
    is_run = df["play_concept"].isin(RUN_CONCEPTS)
    df["box_x_run"] = box * is_run
    df["box_x_pass"] = box * ~is_run
    return df

# Rolling team-form features built in features.py (off_* and def_*) get
# added automatically in prepare_data().

# The play call itself + context categories. play_concept and formation
# are the "knobs" the recommender turns.
CATEGORICAL_FEATURES = [
    "play_concept", "formation", "personnel",
    "posteam", "defteam", "roof",
]


def load_processed():
    """Load the processed dataset built by the data pipeline."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found - run `python -m src.data.features` first."
        )
    # dtype=str on personnel: it's a code like "11", not the number eleven.
    return pd.read_csv(DATA_PATH, low_memory=False, dtype={"personnel": str})


def prepare_data(df):
    """
    Filter to usable rows and split into features (X) and target (y).

    Drops plays with no concept label (scrambles, depth-unknown sacks)
    or no success value. One-hot encodes the categorical columns.

    Returns
    -------
    X : pd.DataFrame  (one-hot encoded, all numeric)
    y : pd.Series     (1 = successful play, 0 = not)
    meta : pd.DataFrame  (season/week for splitting, plus the actual
                          play_concept and epa for recommender evaluation)
    """
    usable = df[df["play_concept"].notna() & df[TARGET].notna()].copy()
    usable = add_box_interactions(usable)

    # qbf_ = rolling QB form (qb_ would also match outcome columns like
    # qb_epa - that prefix must never be auto-included).
    team_form = [c for c in usable.columns if c.startswith(("off_", "def_", "qbf_"))]
    numeric = [c for c in NUMERIC_FEATURES if c in usable.columns] + team_form
    categorical = [c for c in CATEGORICAL_FEATURES if c in usable.columns]

    X = usable[numeric + categorical]
    # One-hot encode: each category value becomes its own 0/1 column
    # (e.g. play_concept_screen_pass). XGBoost needs numeric inputs.
    X = pd.get_dummies(X, columns=categorical)

    y = usable[TARGET].astype(int)
    meta = usable[["season", "week", "play_concept", "epa"]]
    return X, y, meta


def split_by_season(X, y, meta, test_season=None, val_season=None):
    """
    Hold out the latest season and validate on the previous season's late
    weeks. Explicit seasons reproduce a saved artifact's evaluation split.
    """
    test_season = int(meta["season"].max()) if test_season is None else test_season
    val_season = test_season - 1 if val_season is None else val_season
    is_test = meta["season"] == test_season
    is_val = (meta["season"] == val_season) & (meta["week"] >= VAL_WEEK_FROM)
    is_train = (meta["season"] < test_season) & ~is_val

    return (
        X[is_train], y[is_train],
        X[is_val], y[is_val],
        X[is_test], y[is_test],
    )


# Situation-only feature filter for the coach-call model: the play call
# itself (concept/formation/personnel one-hots) and anything derived from
# it (the box-x-call interactions) must be excluded - this model predicts
# WHAT COACHES CALL from the situation alone. defenders_in_box is also
# out: the box count is the DEFENSE's pre-snap read of the offense (they
# stack it when they expect run), so it predicts the call circularly -
# and at recommendation time it's usually unknown (NaN), which sent the
# model down a wildly miscalibrated missing-value branch (96% pass
# flipped to 73% run on a 3rd-and-7).
CALL_EXCLUDED_PREFIXES = ("play_concept_", "formation_", "personnel_")
CALL_EXCLUDED_COLS = {"box_x_run", "box_x_pass", "defenders_in_box"}


def call_feature_columns(all_columns):
    """The subset of training columns the coach-call model may see."""
    return [
        c for c in all_columns
        if not c.startswith(CALL_EXCLUDED_PREFIXES) and c not in CALL_EXCLUDED_COLS
    ]


def train_call_model(X_train, X_val, concepts_train, concepts_val, classes):
    """
    Model 3 (multiclass): P(coach calls each concept | situation).

    This is deliberately the thing the recommender is NOT: a model of
    coaching behavior. It powers the call-credibility prior in
    recommend.py - a ranking-only credit that keeps the recommender from
    living on plays coaches almost never call (whose EPA estimates are
    the most selection-biased). See docs/experiments.md.
    """
    features = call_feature_columns(X_train.columns)
    y_train = pd.Categorical(concepts_train, categories=classes).codes
    y_val = pd.Categorical(concepts_val, categories=classes).codes

    call_model = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=300,
        early_stopping_rounds=20,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    print("\nTraining coach-call model (multiclass, takes a few minutes) ...")
    call_model.fit(
        X_train[features], y_train,
        eval_set=[(X_val[features], y_val)], verbose=50,
    )
    print(f"Best iteration: {call_model.best_iteration} trees")
    return call_model, features


def build_team_form_lookup(df):
    """
    Snapshot each team's most recent rolling form, so the recommender can
    fill in 'how good is KC's offense right now?' without the user typing
    17 numbers. Returns (offense_lookup, defense_lookup) dataframes
    indexed by team.
    """
    latest = df.sort_values(["season", "week"]).drop_duplicates(
        subset=["posteam"], keep="last"
    )
    # QB form rides along with the offense lookup: both are keyed by
    # posteam and answer "what does this team bring on offense right now?"
    off_cols = [c for c in df.columns if c.startswith(("off_", "qbf_"))]
    off_lookup = latest.set_index("posteam")[off_cols]

    latest_d = df.sort_values(["season", "week"]).drop_duplicates(
        subset=["defteam"], keep="last"
    )
    def_cols = [c for c in df.columns if c.startswith("def_")]
    def_lookup = latest_d.set_index("defteam")[def_cols]
    return off_lookup, def_lookup


def train():
    """Train BOTH models and save everything recommend.py and evaluate.py
    need into one artifact file.

    Model 1 (classifier): P(success) - "how often does this call work?"
    Model 2 (regressor):  expected EPA - "how many points is this call
    worth on average?" The regressor sees the payoff SIZE the classifier
    can't: a deep shot that hits is worth far more than a checkdown that
    'succeeds'. The call sheet ranks by expected EPA and shows both.
    """
    print("Loading processed data ...")
    df = load_processed()
    X, y, meta = prepare_data(df)
    test_season = int(meta["season"].max())
    val_season = test_season - 1
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(X, y, meta)
    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("Training requires earlier seasons, a validation season, and a holdout season.")

    print(f"Train: {len(X_train):,} plays (before {val_season} wk{VAL_WEEK_FROM})")
    print(f"Valid: {len(X_val):,} plays ({val_season} wk{VAL_WEEK_FROM}+, for early stopping)")
    print(f"Test : {len(X_test):,} plays ({test_season}, held out)")
    print(f"Features: {X.shape[1]} columns after one-hot encoding")

    success_X, success_y, success_weights = augment_optional_context(X_train, y_train)
    success_val_X, success_val_y, _ = augment_optional_context(X_val, y_val)

    model = xgb.XGBClassifier(
        n_estimators=600,          # up to 600 trees...
        early_stopping_rounds=30,  # ...but stop when validation stalls
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,             # each tree sees 80% of rows/columns -
        colsample_bytree=0.8,      # a standard guard against overfitting
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    print("\nTraining success classifier (takes ~a minute) ...")
    model.fit(success_X, success_y, sample_weight=success_weights,
              eval_set=[(success_val_X, success_val_y)], verbose=50)
    print(f"Best iteration: {model.best_iteration} trees")

    # --- Model 2: EPA regressor (same features, same split) ---------------
    epa = meta["epa"]
    epa_train = epa.loc[X_train.index]
    epa_val = epa.loc[X_val.index]
    epa_X, epa_y, epa_weights = augment_optional_context(X_train, epa_train)
    epa_val_X, epa_val_y, _ = augment_optional_context(X_val, epa_val)

    epa_model = xgb.XGBRegressor(
        n_estimators=600,
        early_stopping_rounds=30,
        learning_rate=0.05,
        **EPA_MODEL_PARAMS,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=42,
        n_jobs=4,
    )
    print("\nTraining EPA regressor (takes ~a minute) ...")
    epa_model.fit(epa_X, epa_y, sample_weight=epa_weights,
                  eval_set=[(epa_val_X, epa_val_y)], verbose=50)
    print(f"Best iteration: {epa_model.best_iteration} trees")

    # Compare with a deployable baseline fitted on training outcomes only.
    # Validate both known and missing context before replacing the artifact.
    baseline_epa = float(epa_train.mean())
    validation = {
        "epa_charted_rmse": float(mean_squared_error(epa_val, epa_model.predict(X_val)) ** 0.5),
        "epa_missing_context_rmse": float(mean_squared_error(epa_val, epa_model.predict(mask_optional_context(X_val))) ** 0.5),
        "epa_baseline_rmse": float(mean_squared_error(epa_val, np.full(len(epa_val), baseline_epa)) ** 0.5),
    }
    print(f"Validation: {validation}")
    if max(validation["epa_charted_rmse"], validation["epa_missing_context_rmse"]) >= validation["epa_baseline_rmse"]:
        raise ValueError("EPA model failed validation against the training-mean baseline; existing artifact retained.")

    # --- Model 3: coach-call model (same split, situation-only features) --
    concept_classes = sorted(df["play_concept"].dropna().unique())
    concept_meta = meta["play_concept"]
    call_model, call_features = train_call_model(
        X_train, X_val,
        concept_meta.loc[X_train.index], concept_meta.loc[X_val.index],
        concept_classes,
    )

    off_lookup, def_lookup = build_team_form_lookup(df)

    artifact = {
        "test_season": test_season,
        "val_season": val_season,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_through": str(df["game_date"].max()),
        "data_week": int(df.loc[df["season"] == test_season, "week"].max()),
        "epa_baseline": baseline_epa,
        "validation": validation,
        "training_strategy": "half-weight charted/missing context; no future-season imputation",
        "model": model,
        "epa_model": epa_model,
        "call_model": call_model,               # P(coach's call | situation)
        "call_features": call_features,         # its (situation-only) columns
        "feature_columns": list(X.columns),     # exact training column order
        "numeric_defaults": X_train.median(numeric_only=True),  # fill for unspecified inputs
        "concept_classes": concept_classes,
        "formation_classes": sorted(df["formation"].dropna().unique()),
        "off_form_lookup": off_lookup,
        "def_form_lookup": def_lookup,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH + ".part")
    os.replace(MODEL_PATH + ".part", MODEL_PATH)
    print(f"\nSaved model artifact -> {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train()
