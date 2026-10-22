# Frontend (Phase 5 — React dashboard)

A live play-calling dashboard: set the game situation on the left, and
the model's recommendation, an interactive formation diagram, and the
full ranked call sheet update as you type.

Built with Vite + React (no other runtime dependencies — the field is
hand-rolled SVG, the bars/ring are CSS).

## Run it

The dashboard opens directly without a login. Play Caller, Live Games,
Season, and AI explanations are available to all visitors.

```powershell
# 1. Start the API from the repo root (serves the model and refreshes NFL data):
python -m uvicorn src.api.main:app --reload

# 2. Start the dev server from this frontend/ folder:
npm install
npm run dev      # opens on http://localhost:5173
```

For the hosted site, GitHub Actions trains and publishes the model on Tuesdays
at 9:00 AM Pacific; Render auto-deploys the resulting commit. Your computer
does not need to stay on. You can trigger **Actions > Refresh NFL model > Run
workflow** in GitHub for an immediate update. Render's Auto-Deploy setting
must be **On Commit** for `main`.

Locally, the API checks nflverse on startup and Tuesdays at 9:00 AM Pacific,
then rebuilds the model and Season scorecard in the background. No data API
key is required. For an immediate complete update, run this from the repo root:

```powershell
python -m src.data.refresh --update-model
```

Check `http://127.0.0.1:8000/data/status` for the loaded model's season, week,
data coverage, and any refresh errors. The Live tab uses ESPN for in-game
updates: the selected game refreshes every 10 seconds and the scoreboard
every 30 seconds. Those polls are independent of the weekly nflverse refresh. Set
`NFL_DATA_REFRESH_ENABLED=0` in `.env` to disable background training.

## What's where

```
src/
├── api.js                      # fetch wrappers for GET /options, POST /recommend
├── App.jsx                     # state + debounced re-recommend on every change
├── lib/
│   ├── labels.js               # display names for concepts/formations/personnel
│   └── formations.js           # inputs -> 11v11 player coordinates + route art
└── components/
    ├── SituationPanel.jsx      # every /recommend input (teams, down, ..., risk dial)
    ├── RecommendationCard.jsx  # best call, expected EPA, success-probability ring
    ├── Field.jsx               # SVG field: LOS, 1st-down line, players, routes
    ├── PlayComparison.jsx      # best pass vs best run, head to head
    ├── CallSheet.jsx           # all concepts ranked by EPA, floor made visible
    └── HistoryStrip.jsx        # saved recommendations (localStorage)
```

The field reacts to the inputs: formation moves the QB/backfield,
personnel swaps TEs/WRs, defenders-in-box shifts the defense, and the
ball spot/first-down marker follow the situation. Route arrows show the
recommended concept.
