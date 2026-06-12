"""
main.py (Phase 3)
-----------------
FastAPI backend exposing the recommender to the React frontend.

Run with:  uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.

Endpoints
---------
GET  /          health check
GET  /teams     valid team codes (for frontend dropdowns)
POST /recommend full ranked call sheet for one game situation
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.model.evaluate import load_artifact
from src.model.recommend import SUCCESS_FLOOR_GAP, recommend_play

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

# Let the React dev server (Phase 5) call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
        le=0.5,
        description="The risk dial: how far a play's success probability may "
        "trail the safest option and still be recommended. "
        "Smaller = safer calls, larger = chase upside.",
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


@app.get("/")
def health():
    """Simple health check so you can verify the server is running."""
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


@app.post("/recommend", response_model=Recommendation)
def recommend(situation: GameSituation):
    """
    Rank every play concept for one game situation.

    The pick = highest expected EPA among plays whose success probability
    clears the floor (within `success_floor_gap` of the safest option).
    """
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

    # The risk dial is an API knob, not a model feature - keep it out of
    # the situation dict we hand to the model.
    sit = situation.model_dump(exclude_none=True, exclude={"success_floor_gap"})
    result = recommend_play(
        sit, artifact=artifact, success_floor_gap=situation.success_floor_gap
    )

    return Recommendation(
        best_call=result["best_call"],
        best_prob=result["best_prob"],
        best_epa=result["best_epa"],
        ranking=result["ranking"].to_dict(orient="records"),
    )
