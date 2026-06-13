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

Train/test split is BY SEASON (train 2023-24, test 2025). Random
splitting time-series data leaks information: plays from the same game
would land in both train and test.

Run with:  python -m src.model.train
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "pbp_features.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "play_success_model.joblib")

TARGET = "success"

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


def split_by_season(X, y, meta):
    """
    Temporal split:
      train      = 2023 + 2024 weeks 1-14
      validation = 2024 weeks 15+   (for early stopping)
      test       = 2025             (never touched until evaluation)
    """
    is_test = meta["season"] == 2025
    is_val = (meta["season"] == 2024) & (meta["week"] >= 15)
    is_train = ~is_test & ~is_val

    return (
        X[is_train], y[is_train],
        X[is_val], y[is_val],
        X[is_test], y[is_test],
    )


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
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_season(X, y, meta)

    print(f"Train: {len(X_train):,} plays (2023 + 2024 wk1-14)")
    print(f"Valid: {len(X_val):,} plays (2024 wk15+, for early stopping)")
    print(f"Test : {len(X_test):,} plays (2025, held out)")
    print(f"Features: {X.shape[1]} columns after one-hot encoding")

    model = xgb.XGBClassifier(
        n_estimators=600,          # up to 600 trees...
        early_stopping_rounds=30,  # ...but stop when validation stalls
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,             # each tree sees 80% of rows/columns -
        colsample_bytree=0.8,      # a standard guard against overfitting
        eval_metric="logloss",
        random_state=42,
    )
    print("\nTraining success classifier (takes ~a minute) ...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Best iteration: {model.best_iteration} trees")

    # --- Model 2: EPA regressor (same features, same split) ---------------
    epa = meta["epa"]
    epa_train = epa.loc[X_train.index]
    epa_val = epa.loc[X_val.index]

    epa_model = xgb.XGBRegressor(
        n_estimators=600,
        early_stopping_rounds=30,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        random_state=42,
    )
    print("\nTraining EPA regressor (takes ~a minute) ...")
    epa_model.fit(X_train, epa_train, eval_set=[(X_val, epa_val)], verbose=50)
    print(f"Best iteration: {epa_model.best_iteration} trees")

    off_lookup, def_lookup = build_team_form_lookup(df)

    artifact = {
        "model": model,
        "epa_model": epa_model,
        "feature_columns": list(X.columns),     # exact training column order
        "numeric_defaults": X_train.median(numeric_only=True),  # fill for unspecified inputs
        "concept_classes": sorted(df["play_concept"].dropna().unique()),
        "formation_classes": sorted(df["formation"].dropna().unique()),
        "off_form_lookup": off_lookup,
        "def_form_lookup": def_lookup,
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model artifact -> {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train()
