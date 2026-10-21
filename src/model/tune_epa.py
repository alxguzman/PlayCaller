"""Compare EPA fixes on past-season validation only; never score the holdout.

Run ``python -m src.model.tune_epa``. The small, fixed candidate set tests
missing-context training and stronger regularization on two temporal folds.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.model.train import (
    PROJECT_ROOT, VAL_WEEK_FROM, augment_optional_context, load_processed,
    mask_optional_context, prepare_data,
)

CANDIDATES = {
    "original": (False, {"max_depth": 6}),
    "missing_context": (True, {"max_depth": 6}),
    "regularized_depth4": (True, {"max_depth": 4, "min_child_weight": 30, "reg_lambda": 10}),
    "regularized_depth3": (True, {"max_depth": 3, "min_child_weight": 100, "reg_lambda": 20}),
}


def main():
    df = load_processed()
    holdout = int(df["season"].max())
    # Exclude the final season before preparing any candidate features.
    X, _, meta = prepare_data(df[df["season"] < holdout])
    results = []
    for season in (holdout - 2, holdout - 1):
        train_mask = (meta["season"] < season) | ((meta["season"] == season) & (meta["week"] < VAL_WEEK_FROM))
        valid_mask = (meta["season"] == season) & (meta["week"] >= VAL_WEEK_FROM)
        train_X, valid_X = X[train_mask], X[valid_mask]
        train_y, valid_y = meta.loc[train_mask, "epa"], meta.loc[valid_mask, "epa"]
        if train_X.empty or valid_X.empty:
            raise ValueError(f"Insufficient history for validation season {season}")
        missing_X = mask_optional_context(valid_X)
        for name, (augment, params) in CANDIDATES.items():
            fit_X, fit_y, weights = train_X, train_y, np.ones(len(train_y))
            eval_X, eval_y = valid_X, valid_y
            if augment:
                fit_X, fit_y, weights = augment_optional_context(train_X, train_y)
                eval_X, eval_y, _ = augment_optional_context(valid_X, valid_y)
            model = XGBRegressor(
                n_estimators=600, early_stopping_rounds=30, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="rmse",
                random_state=42, n_jobs=4, **params,
            )
            model.fit(fit_X, fit_y, sample_weight=weights, eval_set=[(eval_X, eval_y)], verbose=False)
            observed = float(mean_squared_error(valid_y, model.predict(valid_X)))
            missing = float(mean_squared_error(valid_y, model.predict(missing_X)))
            baseline = float(mean_squared_error(valid_y, np.full(len(valid_y), train_y.mean())))
            result = {
                "candidate": name, "validation_season": season,
                "charted_rmse": observed ** 0.5, "missing_rmse": missing ** 0.5,
                "combined_mse": (observed + missing) / 2,
                "baseline_rmse": baseline ** 0.5, "trees": model.best_iteration + 1,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    scores = {name: float(np.mean([r["combined_mse"] for r in results if r["candidate"] == name])) for name in CANDIDATES}
    selected = min(scores, key=scores.get)
    report = {"excluded_holdout_season": holdout, "selection_metric": "mean validation MSE, equal charted/missing weight",
              "results": results, "selected": selected, "candidate_scores": scores}
    path = Path(PROJECT_ROOT) / "data/processed/epa_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected on validation only: {selected}; report: {path}", flush=True)


if __name__ == "__main__":
    main()
