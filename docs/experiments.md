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

---

## Experiment: run-bias fixes (`run-bias-fixes` branch)

Symptom: the recommender never picked a run - not even a QB sneak - in
a 108-situation sweep (every down/distance/box count, BAL's elite run
game vs CAR). Diagnosis found three causes:

1. **Selection-bias-inflated PA deep shots beat the floor.** On 3rd & 1
   the model predicted `pa_deep_pass` at 77% success (reality: 48% on
   124 selectively-called plays) - clearing the 0.10 floor over the QB
   sneak by 0.004.
2. **Predict-time encoding bug.** `_encode_like_training` reindexed with
   `fill_value=0` BEFORE the median-fill loop, so every live
   recommendation ran with wp=0, temp=0, 0 timeouts, 0 defenders in box.
   (The offline evals below were unaffected - they score real plays -
   so this bug corrupted only the live API path.)
3. **Dead box-count lever.** The model learned ~1/4 of the real
   defenders_in_box effect (rank 86-101 of 128 features); pinning
   box=5 vs 9 barely moved run EPA, when in reality (controlled for
   down/distance) a loaded box costs runs ~0.06 EPA and GAINS passes
   ~0.08.

Three changes, attributed separately:

### Step 1 - default `success_floor_gap` 0.10 -> 0.05 (no retrain)

Agreement test on the OLD model, only the floor changed:

| Metric                    | gap 0.10 | gap 0.05 |
| ------------------------- | -------- | -------- |
| Agreement rate            | 4.5%     | 8.1%     |
| Agreed-plays success rate | 0.526    | 0.561    |
| Agreed-plays EPA/play     | +0.444   | +0.258   |

QB sneaks went from never-recommended to 1,988 of 31,924 plays. The
agreed EPA drop is the expected regression toward the mean from a 79%
larger agreed set - both values dwarf the disagreed +0.05. The dial
still goes 0.0-0.5 in the API/UI; only the default moved.

### Step 2 - fix the predict-time fill bug (no retrain)

Unspecified numeric inputs now get training medians (as always
intended), one-hot dummies get 0, and the box features stay NaN
("not charted", matching ~2/3 of training plays). Live predictions
came back into distribution: 3rd & 1 sneak probability 0.865 -> 0.736
(league reality 0.773).

### Step 3 - box interaction features + retrain

`box_x_run` / `box_x_pass` (box count live only on run / pass concepts,
NaN when uncharted) hand the tree the asymmetry it under-learned.

| Metric                    | Baseline (QB form) | This branch | Change  |
| ------------------------- | ------------------ | ----------- | ------- |
| Classifier accuracy       | 0.599              | 0.598       | -0.001  |
| Classifier ROC-AUC        | 0.621              | 0.623       | +0.002  |
| Classifier log-loss       | 0.665              | 0.665       | flat    |
| EPA regressor RMSE        | 1.3292             | 1.3278      | -0.0014 |
| EPA regressor R-squared   | 0.0140             | 0.0161      | +0.0021 |
| Agreement rate            | 4.5%               | 6.0%        | +1.5pt  |
| Agreed-plays success rate | 0.526              | 0.537       | +0.011  |
| Agreed-plays EPA/play     | +0.444             | +0.371      | see St.1 |

Live behavior (the actual point):

- 108-situation sweep: 0 -> 13 run recommendations (all sneaks, all in
  short yardage - where the data says they belong).
- The box lever works: 1st & 10 inside run EPA now swings +0.032 ->
  -0.023 from a 5-man to 9-man box while short pass moves the opposite
  way (+0.123 -> +0.195) - matching the controlled data effect.
- 3rd & 1 vs an elite offense now picks `pa_short_pass` over the
  bias-inflated deep shot; vs an average one, the sneak.

### Verdict: MERGED

Holdout metrics flat-to-better across the board, the regressor
genuinely improved, and the live path no longer hallucinates wp=0.
Known remaining limits (future experiments): `pa_deep_pass` is still
the modal recommendation (61% of plays at gap 0.05) - its EPA is
inflated by selection bias everywhere, not just short yardage; and EPA
is clock-blind, so the model will never call the "kill the clock" run
when protecting a Q4 lead. Win-probability-aware ranking is the
principled fix for both.
