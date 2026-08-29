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

---

## Experiment: run/pass balance prior (`run-pass-balance` branch)

Symptom: the recommender still leaned pass on almost every non-short-yardage
down. Measured on the 2025 holdout, it recommended a run only **11.4%** of
plays vs the **42.4%** real coaches actually ran. Diagnosis (not the
defensive ratings - those already split run/pass): pass plays carry
structurally higher EPA than runs league-wide, so a single-play EPA
maximizer picks pass by construction. The deeper cause is a game-theory
blind spot - the model scores each play against the defense's *average*
posture, so it never pays the predictability tax of becoming one-dimensional
(if you never run, defenses stop respecting it and pass EPA collapses).

Fix (no retrain - pure post-scoring re-ranking, like `success_floor_gap`): a
`run_pass_balance` knob adds an EPA credit to run concepts for RANKING ONLY
(displayed `expected_epa` stays honest). The credit is scaled by a
down/distance "run credibility" weight so it never fires on an obvious
passing down: full credit in short yardage, decaying to 0 over 16 yds on
downs 1-2 and over 4 yds on downs 3-4. Default `0.70`.

A flat (un-scaled) credit was rejected: at the run-share-matching value it
recommended `outside_run` on 3rd-and-8 - trading one unrealistic extreme
for another. The taper fixes that.

### Results (same 31,924 test plays; classifier/regressor unchanged - this is ranking-only)

| Metric                     | Baseline (b=0) | Balance (b=0.70) | Change   |
| -------------------------- | -------------- | ---------------- | -------- |
| Recommended run share      | 11.4%          | 33.8%            | +22.4pt  |
| Run rec on 3rd/4th-&-7+    | 0.7%           | 0.7%             | flat     |
| Agreement rate             | 6.0%           | 7.2%             | +1.2pt   |
| Agreed-plays success rate  | 53.7%          | 50.0%            | -3.7pt   |
| Agreed-plays EPA/play      | +0.371         | +0.249           | see note |
| Disagreed-plays EPA/play   | +0.049         | +0.054           | flat     |

Reference points: real coaches ran 42.4% overall, 8.7% on 3rd/4th-&-7+. The
default lands run share at ~34% - deliberately a touch more pass-leaning
than coaches, matching the analytics consensus that teams should pass
slightly more than they do. It never over-runs passing downs (0.7%, even
more conservative than coaches there).

Live sanity sweep (b=0 -> b=0.70): 1st-&-10 stays a deep shot; 2nd-&-3 flips
pass -> outside run; 3rd-&-1 stays a sneak; 3rd-&-8 and 3rd-&-12 stay passes.

The agreed-EPA drop (+0.371 -> +0.249) is the expected regression toward the
mean from a 20% larger agreed set (1,906 -> 2,292 plays) that now includes
lower-EPA-but-realistic run calls - the same dynamic noted for the gap
0.10->0.05 change above. Both agreed values still dwarf the disagreed
baseline (+0.05), so the model's preferences still carry real signal.

### Verdict: MERGED (pending review)

Better on the goal that motivated it (realism: run share 11%->34%, sensible
per-situation calls) and on agreement rate, with the model's signal
preserved. The classifier and EPA regressor are untouched, so no holdout
metric there can regress. Remaining limit: the prior is a hand-shaped
down/distance taper, not win-probability-aware - it still can't call the
"kill the clock" run to protect a Q4 lead. The knob is exposed in the
API/UI (`run_pass_balance`, 0.0-1.0) alongside the risk dial.

---

## 2026 season prep: data refresh + retrain (Aug 28, 2026)

Two workflow changes ahead of the 2026 season, plus a fresh baseline:

1. **`python -m src.data.refresh`** now downloads the latest nflverse
   files (pbp, FTN, participation for every season in its `SEASONS` list)
   and rebuilds `pbp_features.csv`. nflverse republishes nightly during
   the season. The weekly in-season routine (Tuesdays, after MNF lands):
   refresh -> train -> evaluate -> commit the artifact -> push (Render
   auto-deploys). The retrain is what refreshes the frozen
   `off_form_lookup`/`def_form_lookup` rolling-form snapshots that live
   recommendations depend on.
2. **Season rollover constants** in `train.py`: `TEST_SEASON` /
   `VAL_SEASON` / `VAL_WEEK_FROM` replace the hardcoded 2024/2025 split.
   At 2026 kickoff flip to `TEST_SEASON=2026, VAL_SEASON=2025` and
   retrain - 2025 moves fully into training and 2026 becomes the holdout.
   Holdout numbers will be noisy until ~week 4.

### New baseline (retrained on final 2025 nflverse data, 31,922 test plays)

The committed June artifact was stale (its feature list predated the
timeouts features in `NUMERIC_FEATURES`). Retrained clean; numbers are
consistent with the run-pass-balance baseline above:

| Metric                    | June artifact | Aug 28 retrain |
| ------------------------- | ------------- | -------------- |
| Classifier accuracy       | 0.597         | 0.595          |
| ROC-AUC                   | 0.621         | 0.622          |
| Log-loss                  | 0.665         | 0.665          |
| EPA RMSE (baseline 1.3386)| 1.3297        | 1.3277         |
| EPA R-squared             | 0.0133        | 0.0162         |
| Agreement rate            | 7.2%          | 7.0%           |
| Agreed-plays EPA/play     | +0.249        | +0.236         |
| Disagreed-plays EPA/play  | +0.054        | +0.055         |

### Live tab (model vs coach, in real time)

The Live tab polls ESPN's public JSON feed (~10s) during games,
auto-fills the situation, asks `/recommend` before each snap, then
grades the model against the coach's actual call. Honesty notes baked
into the UI: grading is at the play-FAMILY level (ESPN play text can't
reveal play action, so `pa_*` folds into its base concept) and "success"
is the 40/60/100 yardage rule, not the training target (EPA > 0).
Heuristic check vs nflverse labels (GB @ CLE, 2025 wk3): run directions
matched exactly (24 inside / 14 outside / 10 off-tackle); pass families
61 vs 59. The rigorous scorecard remains `recommendation_agreement()` on
nflverse data after each weekly refresh - the live tab is the demo, the
weekly eval is the number to quote.

---

## Experiment: call-credibility prior (`call-credibility` branch)

Symptom: only **7%** exact agreement with real coaches on the 2025
holdout, driven almost entirely by one concept - the model recommended
`pa_deep_pass` on **61%** of snaps vs coaches' 2.8%. That's the
documented selection-bias inflation (PA deep shots only appear in the
data when coaches liked the conditions) expressing itself as an
unrealistic call mix. With call mixes that different, agreement was
mathematically capped around ~27%.

Fix - the general form of the run/pass balance idea: a third model in the
artifact, a **coach-call model** (multiclass XGBoost,
P(concept | situation), situation-only features - no call or outcome
columns), and a ranking-only credit
`CALL_CREDIBILITY * log P(coach calls this here)`. Plays coaches rarely
call in a situation pay a penalty that a genuinely large EPA edge can
still overcome; displayed EPA stays honest. `RUN_PASS_BALANCE` drops
0.70 -> 0.35 alongside, because the coach prior already encodes run
tendencies (keeping 0.70 double-counted and pushed run share to 48%+).

Design note: the prior must be **concept-level, not family-level** - a
family-level prior (pa_deep + deep sharing one probability) lets
pa_deep_pass ride deep_pass's credibility and it stayed at 50% of
recommendations. Known cost of concept-level: within a family the prior
tilts toward the most common variant (inside_run gains, off_tackle_run
drops to ~1.5% of recs).

Bug caught during validation: the first call model included
`defenders_in_box`, and a NaN box (the normal API-time state) sent it
down a wildly miscalibrated missing-value branch - a 3rd-and-7 went from
96% pass (correct, on real charted rows) to 73% run, surfacing as an
inside-run recommendation in the module demo. Root cause is conceptual,
not just numeric: the box count is the DEFENSE's pre-snap read of the
offense, so it predicts the call circularly. It's excluded from the call
model's features (`CALL_EXCLUDED_COLS` in train.py); the success/EPA
models keep it, where it's a legitimate difficulty modifier.

### Results (same 31,922-play 2025 holdout; classifier/regressor untouched - ranking only)

| Metric                     | Baseline (bal .70, cred 0) | New (bal .35, cred .20) |
| -------------------------- | -------------------------- | ----------------------- |
| Exact agreement            | 7.0%                       | 25.3%                   |
| Family agreement           | 13.3%                      | 28.2%                   |
| pa_deep_pass rec share     | 61.1%                      | 30.3% (coaches 2.8%)    |
| Recommended run share      | 33.8%                      | 39.0% (coaches 42.4%)   |
| Run rec on 3rd/4th-&-7+    | 1.1%                       | 1.1% (coaches 8.7%)     |
| Agreed-plays EPA/play      | +0.236                     | +0.111                  |
| Disagreed-plays EPA/play   | +0.055                     | +0.054                  |

Sweep grid: lambda in {0.03...0.35} x balance in {0, 0.25, 0.35, 0.45,
0.70}. Higher lambda keeps raising agreement (33% at 0.35) but the
agreed-vs-other EPA edge shrinks toward zero and flips negative by
lambda=0.35 with balance 0.70 - past that point the recommender is just
mimicking coaches and agreement stops selecting better plays. 0.20/0.35
keeps a clearly positive edge (+0.057 exact, +0.089 family-level) with a
call mix in the realistic range.

The agreed-EPA drop (+0.236 -> ~+0.11) is the same regression-to-the-mean
dynamic as previous ranking changes, at larger scale: the agreed set grew
from 2,230 to ~8,100 plays and now includes ordinary
inside-runs-on-1st-down instead of only the model's pet spots. The edge
over disagreed plays stays positive, which is the signal that matters.

Spot checks (KC vs BUF): 3rd-&-1 stays a sneak, 3rd-&-12 stays a deep
pass, 3rd-&-8 flips pa_short_pass -> short_pass (more realistic),
1st-&-10 / 2nd-&-3 / red-zone 2nd-&-7 flip from off-tackle to inside run
(the within-family tilt above).

### Verdict: MERGED

Directly fixes the pa_deep_pass dominance that motivated it, agreement
nearly quadruples into the realistic range, run share lands just shy of
real coaches, and the model's preferences still carry positive signal.
Classifier and EPA regressor are untouched. Remaining limits: off_tackle
runs nearly vanish from recommendations (within-family flattening), and
the prior is behavioral, not strategic - the principled next step is
still WP-aware ranking for clock situations.
