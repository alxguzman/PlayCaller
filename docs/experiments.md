# Model experiments log

Every model change gets evaluated on the same held-out 2025 season
before it can replace the current model. "Did it get worse?" should be
a number comparison, not a memory test.

## Baseline — Phase 2 model (main, June 2026)

Features: situation + rolling 4-week team form (off_*/def_*) + play call.

### Success classifier (31,924 test plays)

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 0.597 |
| ROC-AUC   | 0.621 |
| Log-loss  | 0.665 |

Base rate 0.452 (always-predict-fail accuracy: 0.548).

### EPA regressor

| Metric        | Value  |
| ------------- | ------ |
| RMSE          | 1.3297 |
| Baseline RMSE | 1.3386 |
| R-squared     | 0.0133 |

### Recommender agreement test

| Group                        | Plays  | Success rate | EPA/play |
| ---------------------------- | ------ | ------------ | -------- |
| Coach agreed with model      | 1,485  | 0.513        | +0.395   |
| Coach called something else  | 30,439 | 0.449        | +0.052   |

Agreement rate 4.7%.

---

## Experiment: QB rolling form (`qb-features` branch)

Hypothesis: team form smears over the actual QB. Adding the starting
QB's rolling 4-game form (EPA/dropback, CPOE, sack rate) should help,
especially around QB changes. Features are prefixed `qbf_` to avoid
colliding with outcome columns like `qb_epa`.

Verdict: pending — results go here after retraining.
