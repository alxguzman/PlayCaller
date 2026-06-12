# Frontend (Phase 5 — not built yet)

React dashboard that lets you set a game situation with sliders/dropdowns
and shows the recommended play, success probabilities (Recharts bar chart),
and Claude's natural-language explanation.

## When you're ready to build it

```powershell
# From this frontend/ folder:
npm create vite@latest . -- --template react
npm install
npm install recharts
npm run dev
```

Then build:
- `SituationForm` — down, distance, field position, quarter, clock, score, teams
- `RecommendationCard` — RUN vs PASS with success probabilities
- `ProbabilityChart` — Recharts `<BarChart>` comparing the two options
- `CoachExplanation` — text from the `/recommend` endpoint's Claude output

The API will run at http://localhost:8000 (see `src/api/main.py`).
