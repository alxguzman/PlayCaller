import { useCallback, useEffect, useRef, useState } from "react";
import { fetchLiveGames, fetchLiveGameState, fetchRecommendation } from "../api.js";
import { conceptLabel, downDistanceLabel, fieldPositionLabel } from "../lib/labels.js";
import RecommendationCard from "./RecommendationCard.jsx";
import CallSheet from "./CallSheet.jsx";
import LiveScorecard from "./LiveScorecard.jsx";

const GAMES_POLL_MS = 30_000;
const STATE_POLL_MS = 10_000;

// The model can't-see-play-action honesty rule, mirrored from the
// backend: grading happens at the family level (pa_* folds in).
const FAMILY = { pa_deep_pass: "deep_pass", pa_short_pass: "short_pass" };
const family = (c) => (c ? FAMILY[c] ?? c : null);

function loadRows(gameId) {
  try {
    return JSON.parse(localStorage.getItem(`pc-live-${gameId}`) ?? "[]");
  } catch {
    return [];
  }
}

function saveRows(gameId, rows) {
  localStorage.setItem(`pc-live-${gameId}`, JSON.stringify(rows));
}

/** Does a completed play's start state match the situation we recommended for? */
function playMatchesSituation(play, situation) {
  return (
    play.down === situation.down &&
    play.distance === situation.ydstogo &&
    play.yardline_100 === situation.yardline_100 &&
    play.posteam === situation.posteam
  );
}

function GamePicker({ games, error, onPick }) {
  const groups = [
    ["Live now", games.filter((g) => g.state === "in")],
    ["Upcoming", games.filter((g) => g.state === "pre")],
    ["Final", games.filter((g) => g.state === "post")],
  ];
  return (
    <div className="panel live-picker">
      <h2 className="panel-title">Pick a game</h2>
      {error && (
        <>
          <p className="muted">{error}</p>
          <p className="muted live-fineprint">
            Live data comes from ESPN's public feed, which rate-limits by IP
            and blocks shared cloud hosts. This tab always works when you run
            the app locally during a game.
          </p>
        </>
      )}
      {!error && games.length === 0 && (
        <p className="muted">No NFL games on today's scoreboard. Check back on game day!</p>
      )}
      {groups.map(
        ([label, list]) =>
          list.length > 0 && (
            <div key={label}>
              <div className="live-group-label">{label}</div>
              {list.map((g) => (
                <button key={g.id} className="live-game-row" onClick={() => onPick(g.id)}>
                  <span className="live-game-name">{g.short_name}</span>
                  <span className="live-game-score">
                    {g.state !== "pre" && `${g.away?.score ?? 0}–${g.home?.score ?? 0}`}
                  </span>
                  <span className="muted">{g.detail}</span>
                </button>
              ))}
            </div>
          )
      )}
    </div>
  );
}

function SituationStrip({ game, situation, reason }) {
  return (
    <div className="panel live-strip">
      <div className="live-score">
        <span>
          {game.away.abbr} <b>{game.away.score}</b>
        </span>
        <span className="muted"> @ </span>
        <span>
          {game.home.abbr} <b>{game.home.score}</b>
        </span>
        <span className="muted"> · {game.detail}</span>
      </div>
      {situation ? (
        <div className="live-situation">
          <b>{situation.posteam}</b> ball ·{" "}
          {downDistanceLabel(situation.down, situation.ydstogo, situation.yardline_100)}{" "}
          · ball on{" "}
          {fieldPositionLabel(
            situation.yardline_100,
            situation.posteam,
            situation.defteam
          )}
        </div>
      ) : (
        <div className="live-situation muted">{reason || "Waiting for the next snap…"}</div>
      )}
    </div>
  );
}

export default function LiveView() {
  const [games, setGames] = useState([]);
  const [gamesError, setGamesError] = useState(null);
  const [gameId, setGameId] = useState(null);
  const [state, setState] = useState(null); // latest /live/games/{id} response
  const [rec, setRec] = useState(null);
  const [recLoading, setRecLoading] = useState(false);
  const [rows, setRows] = useState([]);
  // pending = the recommendation we made for the CURRENT situation, waiting
  // to be graded against whatever the coach actually calls next.
  const pendingRef = useRef(null);

  const pollGames = useCallback(() => {
    fetchLiveGames()
      .then((d) => {
        setGames(d.games);
        setGamesError(null);
      })
      .catch((e) => setGamesError(`Can't load the scoreboard (${e.message}).`));
  }, []);

  // Scoreboard poll while no game is selected.
  useEffect(() => {
    if (gameId) return;
    pollGames();
    const t = setInterval(pollGames, GAMES_POLL_MS);
    return () => clearInterval(t);
  }, [gameId, pollGames]);

  const pickGame = useCallback((id) => {
    setGameId(id);
    setState(null);
    setRec(null);
    pendingRef.current = null;
    setRows(loadRows(id));
  }, []);

  // Grade any new play against the pending recommendation, then (if the
  // situation moved on) ask the model about the new situation.
  const handleState = useCallback((d) => {
    setState(d);

    const pending = pendingRef.current;
    if (pending) {
      setRows((prev) => {
        const gradedIds = new Set(prev.map((r) => r.playId));
        const fresh = d.recent_plays.find(
          (p) => !gradedIds.has(p.id) && playMatchesSituation(p, pending.situation)
        );
        if (!fresh) return prev;
        pendingRef.current = null;
        // Store the full situation with the row: the scorecard is
        // reloaded from localStorage, so it can't depend on live state.
        const defteam =
          [d.game.home.abbr, d.game.away.abbr].find((t) => t !== fresh.posteam) ?? "";
        const row = {
          playId: fresh.id,
          ts: Date.now(),
          posteam: fresh.posteam,
          defteam,
          down: fresh.down,
          distance: fresh.distance,
          yardline_100: fresh.yardline_100,
          model_call: pending.rec.best_call,
          model_family: family(pending.rec.best_call),
          coach_concept: fresh.concept, // null = ungradable (sack/scramble)
          coach_family: fresh.family,
          agreed: fresh.family !== null && family(pending.rec.best_call) === fresh.family,
          yards: fresh.yards,
          success: fresh.success,
          text: fresh.text,
        };
        return [row, ...prev];
      });
    }

    // A play that wiped the situation (penalty, turnover mid-count...)
    // invalidates the pending rec if the situation on the field changed.
    const sit = d.situation;
    const pendingSit = pendingRef.current?.situation;
    const sameSituation =
      sit &&
      pendingSit &&
      sit.down === pendingSit.down &&
      sit.ydstogo === pendingSit.ydstogo &&
      sit.yardline_100 === pendingSit.yardline_100 &&
      sit.posteam === pendingSit.posteam;

    if (!sit) {
      pendingRef.current = null;
      setRec(null);
    } else if (!sameSituation) {
      pendingRef.current = null;
      setRecLoading(true);
      fetchRecommendation(sit)
        .then((r) => {
          setRec(r);
          pendingRef.current = { situation: sit, rec: r };
        })
        .catch(() => setRec(null))
        .finally(() => setRecLoading(false));
    }
  }, []);

  // Game-state poll while a game is selected (stops once it's final).
  useEffect(() => {
    if (!gameId) return;
    let stopped = false;
    const poll = () =>
      fetchLiveGameState(gameId)
        .then((d) => {
          if (!stopped) handleState(d);
          if (d.game.state === "post") clearInterval(t);
        })
        .catch(() => {});
    poll();
    const t = setInterval(poll, STATE_POLL_MS);
    return () => {
      stopped = true;
      clearInterval(t);
    };
  }, [gameId, handleState]);

  // Persist the scorecard so a page refresh mid-game keeps it.
  useEffect(() => {
    if (gameId) saveRows(gameId, rows);
  }, [gameId, rows]);

  if (!gameId) {
    return (
      <main className="layout live-layout">
        <GamePicker games={games} error={gamesError} onPick={pickGame} />
      </main>
    );
  }

  if (!state) {
    return (
      <main className="layout live-layout">
        <div className="panel">
          <p className="muted">Loading game…</p>
        </div>
      </main>
    );
  }

  return (
    <main className="layout live-layout">
      <section className="center-col">
        <button className="reset-btn live-back" onClick={() => setGameId(null)}>
          ← All games
        </button>
        <SituationStrip game={state.game} situation={state.situation} reason={state.reason} />
        {state.situation && (
          <RecommendationCard rec={rec} loading={recLoading} onSave={() => {}} />
        )}
        {state.game.state === "post" && (
          <div className="panel">
            <p className="muted">
              Final. The scorecard below is this game's model-vs-coach record
              {rows.length === 0 && " (no snaps were graded)"}.
            </p>
          </div>
        )}
        <LiveScorecard rows={rows} />
      </section>
      <section className="right-col">
        {state.situation && <CallSheet rec={rec} floorGap={0.05} />}
        <div className="panel live-feed">
          <h2 className="panel-title">Recent plays</h2>
          {state.recent_plays.length === 0 && <p className="muted">No plays yet.</p>}
          <ul className="live-play-list">
            {state.recent_plays.map((p) => (
              <li key={p.id} className="live-play">
                <span className="live-play-meta">
                  {p.posteam} · {p.concept ? conceptLabel(p.concept) : "ungraded"} ·{" "}
                  {p.yards >= 0 ? "+" : ""}
                  {p.yards} yds
                </span>
                <span className="live-play-text">{p.text}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
