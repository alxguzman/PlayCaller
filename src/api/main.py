"""
main.py (Phase 3)
-----------------
FastAPI backend exposing the recommender to the React frontend.

Run with:  uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.

Endpoints
---------
GET  /healthz   health check
GET  /teams     valid team codes (for frontend dropdowns)
POST /recommend full ranked call sheet for one game situation
POST /login     username/password -> bearer token (see src/api/auth.py)
POST /explain   Claude's coordinator-style explanation (login required)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.ai.play_caller import CLAUDE_MODEL, PlayCallerError, explain_recommendation
from src.api.auth import check_credentials, create_token, require_auth
from src.model.evaluate import load_artifact
from src.model.recommend import RUN_PASS_BALANCE, SUCCESS_FLOOR_GAP, recommend_play

# The artifact (model + lookups) is ~1 MB and slow-ish to deserialize,
# so load it once at startup and share it across requests instead of
# re-reading it from disk on every call.
state = {"artifact": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["artifact"] = load_artifact()
    except FileNotFoundError:
        # Don't crash the server - /recommend returns a clear 503 instead.
        state["artifact"] = None
    yield
    state["artifact"] = None


app = FastAPI(
    title="NFL Play Caller API",
    description="Recommends what play an NFL offense *should* call, "
    "using the floor-and-ceiling blend of two XGBoost models.",
    lifespan=lifespan,
)

# Let the React dev server (Phase 5) call this API from the browser. In
# production the frontend is served from this same origin (see the static
# mount below), so CORS is a no-op there. ALLOWED_ORIGINS can override the
# list for a split frontend/backend deploy.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", _default_origins).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class GameSituation(BaseModel):
    """One game situation, exactly as you'd read it off the broadcast."""

    posteam: str = Field(description="Offense team code, e.g. KC", examples=["KC"])
    defteam: str = Field(description="Defense team code, e.g. BUF", examples=["BUF"])
    down: int = Field(ge=1, le=4)
    ydstogo: int = Field(ge=1, le=99, description="Yards to the first down")
    yardline_100: int = Field(
        ge=1, le=99, description="Yards to the opponent's end zone"
    )
    qtr: int = Field(ge=1, le=5, description="Quarter (5 = overtime)")
    game_seconds_remaining: int = Field(ge=0, le=3600)
    score_differential: int = Field(
        ge=-60, le=60, description="Offense score minus defense score"
    )
    formation: Optional[str] = Field(
        default=None,
        description="Pin a formation (e.g. shotgun_1back); otherwise each "
        "concept gets its natural home",
    )
    personnel: Optional[str] = Field(
        default=None,
        description="Personnel grouping as RB+TE counts, e.g. '11' = 1 RB "
        "1 TE, '12' = 1 RB 2 TE. Omit for the model's typical-play default.",
        examples=["11"],
    )
    defenders_in_box: Optional[int] = Field(
        default=None,
        ge=3,
        le=11,
        description="Pre-snap box count, e.g. 8 = run-stuffing loaded box. "
        "Omit for a typical box.",
    )
    success_floor_gap: float = Field(
        default=SUCCESS_FLOOR_GAP,
        ge=0.0,
        le=0.06,
        description="The risk dial: how far a play's success probability may "
        "trail the safest option and still be recommended. "
        "Smaller = safer calls, larger = chase upside. Capped at 0.06: "
        "beyond that the pick collapses to deep passes everywhere.",
    )
    run_pass_balance: float = Field(
        default=RUN_PASS_BALANCE,
        ge=0.0,
        le=1.0,
        description="The balance dial: EPA credit added to run plays for "
        "ranking, modelling the value of staying unpredictable. "
        "0 = pure EPA (pass-heavy); the default matches real NFL run rates.",
    )

    @field_validator("posteam", "defteam")
    @classmethod
    def uppercase_team(cls, v: str) -> str:
        return v.strip().upper()


class RankedPlay(BaseModel):
    """One row of the call sheet."""

    play_concept: str
    success_prob: float
    expected_epa: Optional[float] = None
    meets_floor: bool


class Recommendation(BaseModel):
    best_call: str
    best_prob: float
    best_epa: Optional[float]
    ranking: list[RankedPlay]


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str


class Explanation(BaseModel):
    explanation: str
    best_call: str
    model: str


def get_artifact():
    if state["artifact"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found - run `python -m src.model.train` "
            "on the server first.",
        )
    return state["artifact"]


def trained_values(artifact, column):
    """
    The values the model was trained on for a one-hot encoded column,
    read off the feature names (e.g. "personnel_11" -> "11"). Anything
    else would encode to all-zeros and silently mean "no information".
    """
    prefix = column + "_"
    return sorted(
        c[len(prefix):] for c in artifact["feature_columns"] if c.startswith(prefix)
    )


@app.get("/healthz")
def health():
    """Simple health check so you can verify the server is running.

    Lives at /healthz (not /) because in production / serves the React app.
    """
    return {
        "status": "ok",
        "message": "NFL Play Caller API is running",
        "model_loaded": state["artifact"] is not None,
    }


@app.get("/teams")
def teams():
    """Team codes the model has form data for (frontend dropdowns)."""
    artifact = get_artifact()
    return {"teams": sorted(artifact["off_form_lookup"].index)}


@app.get("/options")
def options():
    """Every legal value for the request's categorical fields."""
    artifact = get_artifact()
    return {
        "teams": sorted(artifact["off_form_lookup"].index),
        "formations": trained_values(artifact, "formation"),
        "personnel": trained_values(artifact, "personnel"),
    }


def _rank_table(df, specs):
    """
    Build a {n_teams, ranks:{TEAM:{key_rank, key_epa, ...}}} payload from a
    team-form lookup. `specs` is a list of (key, column, ascending) - rank 1
    always means "best", so pass ascending=True where lower is better
    (defense: EPA allowed) and ascending=False where higher is better
    (offense: EPA gained).
    """
    rank_series = {
        key: df[column].rank(ascending=asc, method="min") for key, column, asc in specs
    }
    ranks = {
        team: {
            **{f"{key}_rank": int(rank_series[key][team]) for key, _, _ in specs},
            **{f"{key}_epa": round(float(df.loc[team, col]), 3) for key, col, _ in specs},
        }
        for team in df.index
    }
    return {"n_teams": int(len(df)), "ranks": ranks}


@app.get("/defense-ranks")
def defense_ranks():
    """
    Each team's defense rank from its latest rolling 4-week form (the same
    snapshot the recommender fills in). Rank 1 = stingiest, i.e. lowest EPA
    allowed per play. Static snapshot, so the frontend fetches it once and
    looks up whichever defense is selected.
    """
    de = get_artifact()["def_form_lookup"]
    # Lower EPA allowed = better defense, so rank ascending (1 = best).
    return _rank_table(de, [
        ("overall", "def_epa_allowed_per_play", True),
        ("pass", "def_epa_allowed_pass", True),
        ("run", "def_epa_allowed_run", True),
    ])


@app.get("/offense-ranks")
def offense_ranks():
    """
    Each team's offense rank from its latest rolling 4-week form. Rank 1 =
    best, i.e. highest EPA gained per play. Mirror of /defense-ranks.
    """
    off = get_artifact()["off_form_lookup"]
    # Higher EPA = better offense, so rank descending (1 = best).
    return _rank_table(off, [
        ("overall", "off_epa_per_play", False),
        ("pass", "off_epa_pass", False),
        ("run", "off_epa_run", False),
    ])


def _run_recommendation(situation: GameSituation):
    """Validate a situation and run the recommender. Shared by /recommend
    and /explain so both endpoints agree on the same call."""
    artifact = get_artifact()

    # Unknown team codes would silently fall back to league-median form,
    # which looks like a valid answer but isn't - reject them instead.
    known = set(artifact["off_form_lookup"].index)
    for team in (situation.posteam, situation.defteam):
        if team not in known:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown team code '{team}'. See GET /teams for valid codes.",
            )

    # Same trap for formation/personnel: a value the model never saw
    # one-hot encodes to all-zeros, i.e. "no information", not an error.
    for field in ("formation", "personnel"):
        value = getattr(situation, field)
        valid = trained_values(artifact, field)
        if value is not None and value not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown {field} '{value}'. Valid values: {valid}",
            )

    # The dials are API knobs, not model features - keep them out of the
    # situation dict we hand to the model.
    sit = situation.model_dump(
        exclude_none=True, exclude={"success_floor_gap", "run_pass_balance"}
    )
    result = recommend_play(
        sit,
        artifact=artifact,
        success_floor_gap=situation.success_floor_gap,
        run_pass_balance=situation.run_pass_balance,
    )
    result["ranking"] = result["ranking"].to_dict(orient="records")
    return result


@app.post("/recommend", response_model=Recommendation)
def recommend(situation: GameSituation):
    """
    Rank every play concept for one game situation.

    The pick = highest expected EPA among plays whose success probability
    clears the floor (within `success_floor_gap` of the safest option).
    """
    result = _run_recommendation(situation)
    return Recommendation(
        best_call=result["best_call"],
        best_prob=result["best_prob"],
        best_epa=result["best_epa"],
        ranking=result["ranking"],
    )


@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
    """Trade the .env username/password for a 12-hour bearer token."""
    if not check_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Wrong username or password.")
    return LoginResponse(token=create_token(body.username), username=body.username)


@app.post("/explain", response_model=Explanation)
def explain(situation: GameSituation, user: str = Depends(require_auth)):
    """
    Claude's coordinator-style explanation of the recommended call.

    Login-required: this is the one endpoint that spends Claude API tokens,
    so it sits behind the sign-in (see src/api/auth.py).
    """
    result = _run_recommendation(situation)

    # Hand Claude the same team-form ranks the dashboard shows.
    off = _rank_table(get_artifact()["off_form_lookup"], [
        ("overall", "off_epa_per_play", False),
        ("pass", "off_epa_pass", False),
        ("run", "off_epa_run", False),
    ])["ranks"].get(situation.posteam)
    de = _rank_table(get_artifact()["def_form_lookup"], [
        ("overall", "def_epa_allowed_per_play", True),
        ("pass", "def_epa_allowed_pass", True),
        ("run", "def_epa_allowed_run", True),
    ])["ranks"].get(situation.defteam)

    try:
        text = explain_recommendation(result["situation"], result, off, de)
    except PlayCallerError as e:
        raise HTTPException(status_code=e.status, detail=str(e))

    return Explanation(explanation=text, best_call=result["best_call"], model=CLAUDE_MODEL)


# Serve the built React app (frontend/dist) from this same server, so the
# whole project is one URL in production. Registered LAST so the API routes
# above win; the mount only catches "/" and static asset paths. Guarded by
# an existence check so local dev (no build) still starts fine.
_DIST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "frontend",
    "dist",
)
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="frontend")
