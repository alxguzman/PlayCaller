# NFL Play Predictor & AI Play Caller

A recommender system that suggests what play an NFL offense **should** call
(run vs pass) based on the game situation and how both teams have been
performing recently — not just what coaches historically called.

Built with: pandas · XGBoost · FastAPI · React · Claude API

## How it works

1. **Data pipeline** — 3 seasons of nflverse play-by-play data, cleaned to
   ~150k real run/pass decisions, enriched with rolling 4-week team-form
   features (offensive/defensive EPA, success rates) with no data leakage.
2. **Model** — three XGBoost models. Two take the play call as an input: a
   classifier for P(success) and a regressor for expected EPA. Scoring
   the same situation once per play concept (10 candidate calls) turns
   prediction into recommendation (see "How the recommender picks a
   play" below). Trained on 2023–24, evaluated on a fully held-out 2025
   season (calibration, SHAP, and a coach-agreement outcome test).
3. **API** — FastAPI backend (`src/api/`): POST `/recommend` returns the
   full ranked call sheet, with a `success_floor_gap` risk dial and
   optional `formation`/`personnel`/`defenders_in_box` pins. GET `/teams`
   and GET `/options` list the legal values for dropdowns.
4. **AI play caller** *(Phase 4)* — Claude turns the numbers into a
   coordinator-style explanation: *"3rd & 7, their secondary has been
   leaking EPA for a month — air it out."* POST `/explain` re-runs the
   recommendation, hands Claude the call sheet + both teams' form ranks,
   and returns 2–4 headset-style sentences (`src/ai/play_caller.py`).
   Identical situations are served from an in-process cache so demo
   clicks don't burn API credits. Needs `ANTHROPIC_API_KEY` in `.env`.
5. **Dashboard** *(Phase 5)* — React frontend (`frontend/`): set the
   situation, watch the formation diagram and ranked call sheet react
   live. `npm run dev` in `frontend/` with the API running. The app sits
   behind a simple sign-in (POST `/login`, credentials from
   `APP_USERNAME`/`APP_PASSWORD` in `.env`) so strangers can't spend the
   Claude tokens — `/explain` rejects requests without a valid session.
6. **Live games** *(Phase 6)* — during a real NFL game, the Live tab
   polls ESPN's public JSON feed (`src/live/espn.py`, GET `/live/games`
   and `/live/games/{id}`), auto-fills the current situation, shows the
   model's call *before* the snap, then grades it against what the coach
   actually ran — a running model-vs-coach scorecard (family-level
   agreement, success rate and yards when they agreed vs not). Play
   descriptions are classified into the model's concepts by text
   heuristics (validated against nflverse labels: run directions match
   exactly; play action isn't detectable live, so play-action variants
   grade as their base family). `python -m src.live.replay <event_id>`
   replays any finished game through the same pipeline for testing. The
   rigorous week-after comparison still comes from
   `recommendation_agreement()` on nflverse data (true EPA and labels).

## How the recommender picks a play: floor, ceiling, and priors

Two models score every candidate play call for the same situation, and
two priors correct blind spots they share. Four pieces:

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
  sneak. So a play is only eligible if its success probability is within
  `success_floor_gap` (the **risk dial**, default 0.05) of the safest
  available option. The dial is capped at **0.06**: beyond that the floor
  drops far enough that the inflated-EPA deep pass clears it almost
  everywhere, so ~70%+ of picks collapse to deep passes and the dial
  stops doing anything useful.

- **The balance — run/pass prior.** Even within the floor, pass plays
  carry structurally higher EPA than runs league-wide, so a single-play
  EPA maximizer leans pass on nearly every down. That ignores game
  theory: play-calling is a mixed strategy, and if you never run,
  defenses stop respecting it and your pass EPA collapses. The model
  can't see this — it scores each play against the defense's *average*
  posture — so we add an EPA credit to run concepts for **ranking only**
  (`run_pass_balance`, the **balance dial**, default 0.35; the displayed
  EPA stays the model's honest number). The credit is scaled by how
  *credible* a run is at that down and distance: full in short yardage,
  decaying to zero by ~1st/2nd-and-long and much faster on 3rd/4th down,
  so it never recommends a run when you obviously must pass.

- **The credibility prior — coach-call model.** The floor and the balance
  credit still left the recommender calling play-action deep shots on
  61% of snaps (real coaches: 3%), because selection bias inflates the
  EPA of exactly those rarely-called plays. So a third model learns
  P(concept | situation) from what coaches actually call — situation
  features only, no play call, no outcome — and the ranking gets a credit
  of `CALL_CREDIBILITY × log P(coach calls this here)` (default 0.20).
  Plays coaches almost never call in a spot pay a penalty a genuinely
  large EPA edge can still overcome. This is *not* the recommender
  copying coaches: the credit is a soft tiebreak on top of the EPA
  ranking, and coach agreement stays selective (agreed plays average
  +0.111 EPA vs +0.054 for the rest). With both priors, the holdout run
  share is 39% (coaches 42%), PA-deep drops to 30%, and exact agreement
  rises from 7% to 25%.

**The pick = the highest prior-adjusted EPA among plays that clear the
floor.**

In practice (KC vs BUF): 1st-and-10 and 2nd-and-3 come back inside run —
the balance and credibility credits together lift a run over a
marginally-higher-EPA pass. On 3rd-and-1 the deep shot is blocked by the
floor and the QB sneak wins. On 3rd-and-8 the run credit is zero and the
credibility prior favors the call coaches actually make there, so it's a
short pass; on 3rd-and-12 it stays a pass too. The full ranking table
always shows every play with both model numbers and a `meets_floor` flag,
so nothing is hidden.

Two extra guardrails: QB sneaks are never recommended beyond 2 yards to
go, and trick plays never take the top slot (they're only called when
coaches expect them to work, so their stats are flattered — the same
selection bias in its purest form).

Two of the knobs are per-request API dials on independent axes:
`success_floor_gap` is the **conversion-risk** axis ("how much risk for
upside?", also the dashboard's risk slider) and `run_pass_balance` is the
**play-type** axis ("how predictable am I willing to be?").
`CALL_CREDIBILITY` is a tuned constant in `recommend.py` rather than a
dial — it's a bias correction, not a taste setting. See
`docs/experiments.md` for the holdout numbers behind all three defaults.

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

One command downloads everything from nflverse (play-by-play, FTN
charting, participation) and rebuilds the processed dataset:

```powershell
python -m src.data.refresh
```

Seasons are listed in `SEASONS` at the top of `src/data/refresh.py` —
edit that list once a year. Files that nflverse hasn't published yet
(e.g. the new season before week 1, or FTN/participation for recent
seasons) are noted and skipped; the pipeline treats them as optional.

**Weekly in-season routine** (Tuesdays, after Monday Night Football has
landed in nflverse's nightly build):

```powershell
python -m src.data.refresh     # latest play-by-play
python -m src.model.train      # retrain (also refreshes team-form lookups)
python -m src.model.evaluate   # sanity-check the holdout metrics
python -m src.model.season_log # rebuild the per-play model-vs-coach log
# then commit model/play_success_model.joblib and push - Render redeploys
```

`season_log` writes `data/processed/season_log_<year>.csv` — every play of
the test season with the model's call (made from pre-snap info only) next
to the coach's actual call and the real outcome, plus a week-by-week
agreement table. Since the model is deterministic and never sees the
coach's call, regenerating this after the fact gives the exact same
recommendations it would have made live. Interpretation caveat: when the
coach called something else, the model's play never ran — those rows are
a comparison, not proof of who was right.

It also writes `model/season_scorecard.json` (committed with the weekly
retrain), which GET `/season-scorecard` serves and the dashboard's
**Season tab** renders — the week-by-week table in the app. The Live
Games tab glows red whenever a game is in progress.

The retrain matters for live use: each team's rolling 4-week form is
frozen into the artifact at train time, so a weekly retrain is what
keeps recommendations current.

**Season rollover** (once, at kickoff of a new season): in
`src/model/train.py` bump `TEST_SEASON` and `VAL_SEASON` (e.g. 2026/2025),
add the year to `SEASONS` in `refresh.py`, and retrain. The finished
season moves into training and the new one becomes the holdout.

### 3. Run the notebooks in order

Open `notebooks/01_load_and_explore.ipynb`, select the `venv` kernel, and
run all cells. Then 02 (builds the dataset), 03 (team-stats deep dive),
and 04 (trains the model and demos the play caller).

### 4. Terminal shortcuts

```powershell
python -m src.data.refresh     # download latest nflverse data + rebuild features
python -m src.data.features    # rebuild data/processed/pbp_features.csv only
python -m src.model.train      # train + save the model artifact
python -m src.model.evaluate   # metrics + confusion matrix + SHAP charts
python -m src.model.recommend  # demo recommendation (3rd & 7, KC vs BUF)
python -m src.live.replay      # list today's games (add an id to replay one)

.\venv\Scripts\Activate.ps1 
uvicorn src.api.main:app --reload   # start the API, docs at /docs

# in seperate terminal run
cd frontend    #activate frontend 
npm run dev

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
- [x] Phase 4 — Claude AI play caller (`src/ai/`) + sign-in gate
- [x] Phase 5 — React dashboard (`frontend/`)
