# Frontend (Phase 5 — React dashboard)

A live play-calling dashboard: set the game situation on the left, and
the model's recommendation, an interactive formation diagram, and the
full ranked call sheet update as you type.

Built with Vite + React (no other runtime dependencies — the field is
hand-rolled SVG, the bars/ring are CSS).

## Run it

```powershell
# 1. Start the API from the repo root (it serves the model):
uvicorn src.api.main:app --reload

# 2. Start the dev server from this frontend/ folder:
npm install
npm run dev      # opens on http://localhost:5173
```

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
