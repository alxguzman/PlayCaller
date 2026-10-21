# Model repair: missing in-season participation

The October 21, 2026 audit found that `defenders_in_box` was present for all
2023-2025 processed plays but missing for every 2026 play. Personnel was
about 97% available historically and entirely missing in 2026. The original
EPA model predicted an average +0.345 EPA on 2026 plays whose actual mean
was +0.058. Hiding participation in older validation data reproduced the
failure; this was an input availability mismatch, not just a need for more
trees.

The repair makes these changes:

- Each training play has a charted and a missing-participation representation,
  with half weight each. The success and EPA models learn both cases without
  changing each play's total training weight. User-supplied box/personnel
  information is still usable.
- Early-season rolling features remain missing instead of being filled with
  full-season averages that include future outcomes.
- EPA uses depth 3, `min_child_weight=100`, and `reg_lambda=20`. This was
  selected from four fixed candidates on 2024/2025 temporal validation.
- Historical hypothetical calls now recompute run/pass box interactions.
- The saved artifact includes a training-mean EPA baseline and validation
  metrics. Training refuses to replace it if either charted or missing-context
  validation RMSE fails to beat that baseline.

## Validation-only model selection

Run `python -m src.model.tune_epa` after building features. It excludes the
latest season before preparing the tuning data, validates on weeks 15 onward
of each of the two preceding seasons, and selects the lowest average MSE
across both context conditions and both folds. No final-season labels are
used to choose candidate hyperparameters. The report is saved to
`data/processed/epa_validation.json`.

| Candidate | 2024 charted RMSE | 2024 missing RMSE | 2025 charted RMSE | 2025 missing RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original depth 6 | 1.3278 | 1.3340 | 1.3198 | 1.3640 |
| Missing-context training, depth 6 | 1.3291 | 1.3292 | 1.3227 | 1.3232 |
| Missing-context training, regularized depth 4 | 1.3267 | 1.3266 | 1.3217 | 1.3218 |
| Missing-context training, regularized depth 3 (selected) | 1.3280 | 1.3282 | 1.3201 | 1.3201 |

All candidates above use the corrected rolling features. The training-mean
baseline RMSE was 1.3383 in the 2024 fold and 1.3378 in the 2025 fold.

Rebuild and train with `python -m src.data.refresh --update-model`. The
pipeline version forces existing API caches to rebuild after these changes.
Run `python -m unittest discover -s tests -v` for regression checks.

## Final 2026 check

After selecting the parameters, the rebuilt artifact was evaluated on 10,630
2026 plays. These compare the previously shipped pipeline with the repaired
pipeline; the before model also used the old rolling-feature imputation.

| Metric | Before | After |
| --- | ---: | ---: |
| EPA RMSE (lower is better) | 1.3527 | 1.3036 |
| Training-mean baseline RMSE | 1.3182 | 1.3182 |
| EPA R-squared | -0.0529 | 0.0221 |
| Mean predicted EPA | +0.3452 | +0.0605 |
| Mean actual EPA | +0.0582 | +0.0582 |
| Success ROC-AUC (higher is better) | 0.603 | 0.626 |
| Success log loss (lower is better) | 0.671 | 0.662 |

The repaired model beats the constant baseline, but the predictive advantage
remains modest. Fourteen regression tests passed, including missing-context
weighting, future-outcome invariance, hypothetical box interactions, and API
handling of teams without history. API checks passed for the season scorecard,
team rankings, and recommendations with unknown, six-man, and eight-man boxes.

## Limits

The 2026 failure was already inspected to diagnose missing inputs, so 2026 is
an out-of-training check rather than a pristine untouched research benchmark.
Future weeks are needed to confirm durability. EPA error measures prediction
on the coach's actual play; it does not prove the proposed alternative call
would have done better. Concept labels also exclude many sacks and scrambles
where intended depth is unknown, which can bias pass-versus-run comparisons.

Reference: [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html)
describes depth/child-weight/regularization controls; [scikit-learn's leakage
guidance](https://scikit-learn.org/stable/common_pitfalls.html) explains why
preprocessing must not use future/test observations.
