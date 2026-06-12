# NFL Play Predictor & AI Play Caller

A recommender system that suggests what play an NFL offense **should** call
(run vs pass) based on the game situation and how both teams have been
performing recently — not just what coaches historically called.

Built with: pandas · XGBoost · FastAPI · React · Claude API

## How it works

1. **Data pipeline** — 3 seasons of nflverse play-by-play data, cleaned to
   ~150k real run/pass decisions, enriched with rolling 4-week team-form
   features (offensive/defensive EPA, success rates) with no data leakage.
2. **Model** — two XGBoost models with the play call as an input: a
   classifier for P(success) and a regressor for expected EPA. Scoring
   the same situation once per play concept (10 candidate calls) turns
   prediction into recommendation (see "How the recommender picks a
   play" below). Trained on 2023–24, evaluated on a fully held-out 2025
   season (calibration, SHAP, and a coach-agreement outcome test).
3. **API** — FastAPI backend (`src/api/`): POST `/recommend` returns the
   full ranked call sheet, with a `success_floor_gap` risk dial and
   optional `formation`/`personnel` pins. GET `/teams` and GET `/options`
   list the legal values for dropdowns.
4. **AI play caller** *(Phase 4)* — Claude turns the numbers into a
   coordinator-style explanation: *"3rd & 7, their secondary has been
   leaking EPA for a month — air it out."*
5. **Dashboard** *(Phase 5)* — React + Recharts frontend.

## How the recommender picks a play: floor and ceiling

Two models score every candidate play call for the same situation, and
each one covers the other's blind spot:

- **The ceiling — EPA regressor.** Candidates are ranked by *expected
  points added*: how much a call is worth on average. This is the right
  objective because payoff size matters — a deep shot that hits only
  35% of the time can still be the highest-value call, and ranking by
  success probability alone would systematically bury high-variance
  plays.

- **The floor — success classifier.** But raw EPA ranking has a known
  failure mode: **selection bias**. Aggressive plays (especially
  play-action deep shots) only appear in the data when coaches liked
  the conditions, so their historical EPA is inflated, and a pure-EPA
  ranker recommends them everywhere — even 3rd-and-1, over a 74% QB
  sneak. So the recommendation must come from plays whose success
  probability is within `SUCCESS_FLOOR_GAP` (default 0.10) of the
  safest available option.

**The pick = highest expected EPA among plays that clear the floor.**

In practice: on 1st-and-10 the play-action deep shot wins legitimately
(its conversion odds are competitive). On 3rd-and-1 it gets blocked —
it trails the sneak by 11 points of conversion probability — and a
high-percentage play-action short pass wins instead. The full ranking
table always shows every play with both numbers and a `meets_floor`
flag, so nothing is hidden.

Two extra guardrails: QB sneaks are never recommended beyond 2 yards to
go, and trick plays never take the top slot (they're only called when
coaches expect them to work, so their stats are flattered — the same
selection bias in its purest form).

```
PlayCaller/
├── data/
│   ├── raw/              # nflverse CSVs go here (pbp + participation)
│   └── processed/        # pipeline output: pbp_features.csv
├── model/                # trained model artifact + evaluation charts
├── notebooks/
│   ├── 01_load_and_explore.ipynb         # start here
│   ├── 02_clean_and_feature_engineer.ipynb
│   ├── 03_build_team_stats.ipynb         # situational ratings + matchup cards
│   └── 04_model_training.ipynb           # train, evaluate, recommend
├── src/
│   ├── data/             # loader.py, cleaner.py, concepts.py, features.py
│   ├── model/            # train.py, evaluate.py, recommend.py
│   ├── api/              # FastAPI app (Phase 3)
│   └── ai/               # Claude play caller (Phase 4)
├── frontend/             # React dashboard (Phase 5)
└── requirements.txt
```

## Getting started

### 1. Install requirements

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Data files

Play-by-play files (already in place):
- `data/raw/play_by_play_2023.csv.gz` ✓
- `data/raw/play_by_play_2024.csv.gz` ✓
- `data/raw/play_by_play_2025.csv.gz` ✓

FTN charting (play-action/screen/RPO/formation flags; already in place):
- `data/raw/ftn_charting_2023.csv` ✓ (from https://github.com/nflverse/nflverse-data/releases/tag/ftn_charting)
- `data/raw/ftn_charting_2024.csv` ✓
- `data/raw/ftn_charting_2025.csv` ✓

Participation data (formation/personnel/coverage; already in place):
- `data/raw/participation_2023.csv` ✓
- `data/raw/participation_2024.csv` ✓
- `data/raw/participation_2025.csv` ✓
- (from https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation — note the `pbp_participation` tag; files are named `pbp_participation_YYYY.csv`)

The pipeline treats FTN and participation files as optional — missing
seasons just leave those columns empty.

### 3. Run the notebooks in order

Open `notebooks/01_load_and_explore.ipynb`, select the `venv` kernel, and
run all cells. Then 02 (builds the dataset), 03 (team-stats deep dive),
and 04 (trains the model and demos the play caller).

### 4. Terminal shortcuts

```powershell
python -m src.data.features    # rebuild data/processed/pbp_features.csv
python -m src.model.train      # train + save the model artifact
python -m src.model.evaluate   # metrics + confusion matrix + SHAP charts
python -m src.model.recommend  # demo recommendation (3rd & 7, KC vs BUF)
uvicorn src.api.main:app --reload   # start the API, docs at /docs
```

### 5. What success looks like

- Notebook 02 ends with `data/processed/pbp_features.csv` (~105k rows)
  and team-form charts where elite teams clearly separate from bad ones.
- Notebook 04 ends with a ranked "call sheet" for any situation you type
  in — e.g. on 3rd & 1 it recommends a QB sneak (74%), on 3rd & 12 a
  play-action pass (33%), matching real-world analytics.

## Roadmap

- [x] Phase 1 — data pipeline + exploration notebooks
- [x] Phase 2 — XGBoost model + recommender (`src/model/`)
- [x] Phase 3 — FastAPI backend (`src/api/`)
- [ ] Phase 4 — Claude AI play caller (`src/ai/`)
- [ ] Phase 5 — React dashboard (`frontend/`)
