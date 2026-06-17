"""
recommend.py
------------
The play caller: turns the trained success model into recommendations.

Given a game situation, we build one candidate row per play concept
(optionally per formation too), run them all through the model, and rank
by predicted success probability. The user only supplies the situation;
each team's recent form is filled in automatically from the lookup
tables saved at training time.

Example
-------
from src.model.recommend import recommend_play
result = recommend_play({
    "posteam": "KC", "defteam": "BUF",
    "down": 3, "ydstogo": 7, "yardline_100": 35,
    "qtr": 4, "game_seconds_remaining": 240,
    "score_differential": -4,
})
print(result["best_call"])      # e.g. "pa_short_pass"
print(result["ranking"])        # dataframe: every concept with P(success)
"""

import numpy as np
import pandas as pd

from src.model.evaluate import load_artifact
from src.model.train import NUMERIC_FEATURES, RUN_CONCEPTS, add_box_interactions

# Common-sense guardrails on top of the model:
# a sneak only makes sense in very short yardage.
SNEAK_MAX_YDSTOGO = 2

# The run/pass balance prior (game theory the static model can't see).
# A single-play EPA maximizer always leans pass, because pass plays carry
# structurally higher EPA than runs league-wide. But play-calling is a
# mixed strategy: if you never run, defenses stop respecting it and your
# pass EPA collapses. The model scores each play assuming the defense's
# AVERAGE posture, so it never pays this predictability tax. We add an EPA
# credit to run concepts for RANKING ONLY (the displayed expected_epa stays
# honest). 0.0 = pure EPA (pass-heavy on every down).
#
# The credit is scaled by how CREDIBLE a run is in the situation, so it
# never recommends a run when you obviously must pass (3rd-and-long): a run
# stays a real threat through ~1st/2nd-and-medium, but on 3rd/4th down only
# in short yardage. With the default the recommender's 2025-holdout run
# share is ~34% (vs coaches' 42% - deliberately a touch more pass-leaning,
# as the analytics consensus prescribes) and it runs on 3rd/4th-and-7+ just
# 0.7% of the time (coaches 8.7%). See docs/experiments.md.
RUN_PASS_BALANCE = 0.70

# Yards-to-go over which a run stops being a credible call, by down. The
# credit decays linearly to 0 across this many yards (downs 1-2 keep runs
# live much longer than 3rd/4th down).
_RUN_HORIZON_EARLY = 16
_RUN_HORIZON_LATE = 4


def run_credibility(down, ydstogo):
    """Weight in [0, 1]: how credible a run is at this down & distance.
    1.0 in short yardage, decaying to 0.0 on long yardage (faster on
    3rd/4th down). Scales the run/pass balance credit so it never fires on
    an obvious passing down."""
    horizon = _RUN_HORIZON_EARLY if (down or 1) <= 2 else _RUN_HORIZON_LATE
    return max(0.0, min(1.0, 1.0 - max(0, ydstogo - 1) / horizon))

# The floor-and-ceiling blend: the EPA regressor picks the play (ceiling),
# but only among plays whose success probability is within this many
# points of the best available option (floor). This stops the inflated
# EPA of rarely-called aggressive plays (selection bias) from recommending
# a 35%-success deep shot when a 65%-success option exists.
# 0.05 (was 0.10): selection bias inflates the classifier's view of rare
# aggressive plays too, so at 0.10 a play-action deep shot could clear the
# floor over a QB sneak on 3rd-and-1 by a few thousandths. A 2025 backtest
# showed 0.05 keeps nearly all the success edge while doubling the EPA
# edge of the picks.
SUCCESS_FLOOR_GAP = 0.05

# These stay NaN when the caller doesn't pin a box count: NaN means
# "not charted", which is exactly how ~2/3 of training plays look.
# Filling a median here would fabricate information.
BOX_FEATURES = {"defenders_in_box", "box_x_run", "box_x_pass"}


def _build_candidates(situation, artifact):
    """
    Build one model-input row per candidate play concept.

    Fill order (most specific wins):
      1. values the user passed in `situation`
      2. the teams' latest rolling form from the training-time lookups
      3. training-set medians for anything still missing
    """
    concepts = artifact["concept_classes"]
    base = dict(situation)

    # Auto-fill team form from the lookup tables ("how good is this
    # offense/defense lately?") unless the caller supplied their own.
    posteam = base.get("posteam")
    defteam = base.get("defteam")
    off = artifact["off_form_lookup"]
    de = artifact["def_form_lookup"]
    if posteam in off.index:
        for col, val in off.loc[posteam].items():
            base.setdefault(col, val)
    if defteam in de.index:
        for col, val in de.loc[defteam].items():
            base.setdefault(col, val)

    rows = []
    for concept in concepts:
        row = dict(base)
        row["play_concept"] = concept
        # If the user didn't pin a formation, give each concept its
        # natural home (sneaks go under center, everything else shotgun -
        # the single most common formation). Pass formation="..." to test
        # a specific look.
        row.setdefault(
            "formation",
            "under_center_1back" if concept == "qb_sneak" else "shotgun_1back",
        )
        rows.append(row)
    return pd.DataFrame(rows), concepts


def _encode_like_training(candidates, artifact):
    """
    One-hot encode the candidate rows and align them to the exact column
    set the model was trained on.

    Fill rules for columns the caller didn't supply:
      * numeric features  -> training median ("a typical play"). The old
        code reindexed with fill_value=0, which silently fed the model
        wp=0, temp=0, 0 timeouts etc. for every live recommendation.
      * box features      -> stay NaN ("not charted", same as training)
      * one-hot dummies   -> 0 (the category simply isn't present;
        NaN would send XGBoost down its learned "missing" branch)
    """
    candidates = add_box_interactions(candidates)
    cat_cols = [
        c for c in ["play_concept", "formation", "personnel", "posteam", "defteam", "roof"]
        if c in candidates.columns
    ]
    X = pd.get_dummies(candidates, columns=cat_cols)
    X = X.reindex(columns=artifact["feature_columns"])  # absent columns -> NaN

    defaults = artifact["numeric_defaults"]
    for col in X.columns[X.isna().any()]:
        if col in BOX_FEATURES:
            continue
        is_numeric = col in NUMERIC_FEATURES or col.startswith(("off_", "def_", "qbf_"))
        if is_numeric and col in defaults.index:
            X[col] = X[col].fillna(defaults[col])
        else:
            X[col] = X[col].fillna(0)
    return X


def recommend_play(
    situation,
    artifact=None,
    success_floor_gap=SUCCESS_FLOOR_GAP,
    run_pass_balance=RUN_PASS_BALANCE,
):
    """
    Rank every play concept for one game situation.

    Parameters
    ----------
    situation : dict
        Anything the model knows about: down, ydstogo, yardline_100, qtr,
        game_seconds_remaining, score_differential, posteam, defteam,
        formation, personnel... Missing values are auto-filled (team form
        from lookups, the rest from training medians).
    artifact : dict, optional
        Pass a preloaded artifact to avoid re-reading from disk in loops.
    success_floor_gap : float
        How far (in probability points) a play's success chance may trail
        the safest option and still be recommended. Smaller = more
        conservative recommendations.
    run_pass_balance : float
        EPA credit added to run concepts for ranking only, modelling the
        value of staying unpredictable. 0.0 = pure EPA (pass-heavy); higher
        pulls the run share up. Orthogonal to success_floor_gap (that's the
        conversion-risk axis; this is the play-type axis).

    Returns
    -------
    dict with:
        best_call    : name of the top-ranked play concept
        best_prob    : its predicted success probability
        best_epa     : its expected EPA (None if no regressor in artifact)
        ranking      : DataFrame of all concepts sorted by expected EPA,
                       with a meets_floor column showing eligibility
        situation    : the situation as the model saw it (after fills)

    Ranking logic - the floor-and-ceiling blend:
      * CEILING (model 2, EPA regressor): plays are sorted by expected
        EPA, because payoff size matters - a deep shot that hits 35% of
        the time can still be the highest-value call.
      * FLOOR (model 1, success classifier): but the recommendation must
        be a play whose success probability is within `success_floor_gap`
        of the best available option. This keeps the selection-bias-
        inflated EPA of rare aggressive calls from winning everywhere.
    Falls back to pure success_prob ranking if the artifact predates the
    EPA regressor.
    """
    artifact = artifact or load_artifact()
    model = artifact["model"]
    epa_model = artifact.get("epa_model")  # None for old artifacts

    candidates, concepts = _build_candidates(situation, artifact)
    X = _encode_like_training(candidates, artifact)
    probs = model.predict_proba(X)[:, 1]

    ranking = pd.DataFrame({"play_concept": concepts, "success_prob": probs})
    if epa_model is not None:
        ranking["expected_epa"] = epa_model.predict(X)
        # Rank by EPA plus the run/pass balance credit (runs only, scaled by
        # how credible a run is here). The credit drives ordering but is NOT
        # shown - expected_epa stays the model's honest prediction.
        is_run = ranking["play_concept"].isin(RUN_CONCEPTS)
        credit = run_pass_balance * run_credibility(
            situation.get("down", 1), situation.get("ydstogo", 10)
        )
        ranking["_rank_score"] = ranking["expected_epa"] + credit * is_run
        sort_col = "_rank_score"
    else:
        sort_col = "success_prob"
    ranking = ranking.sort_values(sort_col, ascending=False, ignore_index=True)

    # Guardrails: never recommend a sneak with real yardage to gain, and
    # don't let trick plays take the top slot (the data is biased - teams
    # only call them when they expect them to work).
    ydstogo = situation.get("ydstogo", 10)
    if ydstogo > SNEAK_MAX_YDSTOGO:
        ranking = ranking[ranking["play_concept"] != "qb_sneak"].reset_index(drop=True)
    recommendable = ranking[ranking["play_concept"] != "trick_play"]

    # The success floor: model 1 sets the bar, measured against the
    # safest recommendable play.
    floor = recommendable["success_prob"].max() - success_floor_gap
    ranking["meets_floor"] = ranking["success_prob"] >= floor
    eligible = recommendable[recommendable["success_prob"] >= floor]

    # Best = top of the (balance-adjusted) ranking among eligible plays -
    # ranking is already sorted, so the first eligible row wins.
    best = eligible.iloc[0]
    ranking = ranking.drop(columns="_rank_score", errors="ignore")
    return {
        "best_call": best["play_concept"],
        "best_prob": float(best["success_prob"]),
        "best_epa": float(best["expected_epa"]) if epa_model is not None else None,
        "ranking": ranking,
        "situation": situation,
    }


if __name__ == "__main__":
    # Demo: 3rd & 7, KC down 4 at the BUF 35, four minutes left.
    result = recommend_play({
        "posteam": "KC", "defteam": "BUF",
        "down": 3, "ydstogo": 7, "yardline_100": 35,
        "qtr": 4, "game_seconds_remaining": 240,
        "score_differential": -4,
    })
    print(
        f"Best call: {result['best_call']} "
        f"(expected EPA {result['best_epa']:+.3f}, "
        f"success {result['best_prob']:.1%})\n"
    )
    print(result["ranking"].round(3).to_string(index=False))
