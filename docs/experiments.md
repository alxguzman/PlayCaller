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

Implementation: per-QB weekly EPA/dropback, CPOE, and sack rate rolled
over the QB's previous 4 games played; each team-week's plays get the
primary passer's (most dropbacks) rolling form. First starts / week 1
fall back to the season mean, same as team form.

### Results (same 31,924 test plays, baseline -> QB features)

| Metric                        | Baseline | QB form | Change |
| ----------------------------- | -------- | ------- | ------ |
| Classifier accuracy           | 0.597    | 0.599   | +0.002 |
| Classifier ROC-AUC            | 0.621    | 0.621   | flat   |
| Classifier log-loss           | 0.665    | 0.665   | flat   |
| EPA regressor RMSE            | 1.3297   | 1.3292  | -0.0005 |
| EPA regressor R-squared       | 0.0133   | 0.0140  | +0.0007 |
| Agreed-plays success rate     | 0.513    | 0.526   | +0.013 |
| Agreed-plays EPA/play         | +0.395   | +0.444  | +0.049 |
| Agreement rate                | 4.7%     | 4.5%    | -0.2pt |

Feature importance: qbf_epa_per_dropback ranks #39, qbf_cpoe #45,
qbf_sack_rate #87 of 128 features - used, mid-pack.

### Verdict: MERGED

Every metric is flat or slightly better; nothing got worse. The gains
are modest (the play-level ceiling here is low) and the agreed-plays
EPA jump (+0.049 on 1,442 plays) is within ~1.5 standard errors, so
treat it as suggestive, not proof. Merged because: no metric regressed,
the football rationale is sound (team form lags QB changes by ~4
weeks), and it unlocks the injury what-if scenario for the API later.
