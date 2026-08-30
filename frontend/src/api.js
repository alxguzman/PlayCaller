// Thin wrapper around the FastAPI backend (src/api/main.py).
// Start it with:  uvicorn src.api.main:app --reload

// In production the frontend is served by the same FastAPI server, so an
// empty base means "call this same origin". In `npm run dev` the Vite server
// runs on :5173 while the API runs on :8000, so fall back to the local API.
// Override either with VITE_API_BASE (e.g. a split frontend/backend deploy).
const API_BASE =
  import.meta.env.VITE_API_BASE ??
  (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

async function getJSON(path, init) {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) {
        detail =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* keep the status text */
    }
    const err = new Error(detail);
    err.status = res.status; // lets callers spot 401 = session expired
    throw err;
  }
  return res.json();
}

/** Legal dropdown values: { teams, formations, personnel } */
export function fetchOptions() {
  return getJSON("/options");
}

/** Per-team defense ranks from rolling 4-week form: { n_teams, ranks } */
export function fetchDefenseRanks() {
  return getJSON("/defense-ranks");
}

/** Per-team offense ranks from rolling 4-week form: { n_teams, ranks } */
export function fetchOffenseRanks() {
  return getJSON("/offense-ranks");
}

/** Full ranked call sheet for one game situation. */
export function fetchRecommendation(situation) {
  return getJSON("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(situation),
  });
}

/* ---- live games (ESPN-powered, see src/api/live.py) ---------------

   ESPN rate-limits by IP and blocks shared cloud ranges, so a deployed
   server (Render's free tier) often gets a 403 while a laptop doesn't.
   So: try the server first, and if it can't reach ESPN, fetch the feed
   from THIS browser instead - the viewer's own IP and quota, the same
   request espn.com itself makes - and POST the payload to the server to
   be parsed. The parsing (play-concept mapping) stays in Python either
   way; only the fetching moves. */

const ESPN_SCOREBOARD =
  "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
const ESPN_SUMMARY =
  "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event=";

// Flips to true the first time the server proves it can't reach ESPN, so
// we don't pay a failing round trip on every poll after that.
let relayViaBrowser = false;

function postJSON(path, payload) {
  return getJSON(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function espnJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ESPN returned ${res.status}`);
  return res.json();
}

/** Today's NFL games: { games: [{id, short_name, state, detail, home, away}] } */
export async function fetchLiveGames() {
  if (!relayViaBrowser) {
    try {
      return await getJSON("/live/games");
    } catch {
      relayViaBrowser = true;
    }
  }
  return postJSON("/live/parse/scoreboard", await espnJSON(ESPN_SCOREBOARD));
}

/** One game's live state: { game, situation, reason, last_play_id, recent_plays } */
export async function fetchLiveGameState(id) {
  if (!relayViaBrowser) {
    try {
      return await getJSON(`/live/games/${id}`);
    } catch {
      relayViaBrowser = true;
    }
  }
  const raw = await espnJSON(ESPN_SUMMARY + encodeURIComponent(id));
  // The summary carries boxscore/news/video/odds we never read; sending
  // only what the parser uses keeps the upload small on a 10s poll.
  return postJSON("/live/parse/game", { header: raw.header, drives: raw.drives });
}

/** Week-by-week model-vs-coach summary: { available, season, weeks, totals } */
export function fetchSeasonScorecard() {
  return getJSON("/season-scorecard");
}

/* ---- sign-in ------------------------------------------------------
   The token gates /explain (the endpoint that spends Claude credits).
   It lives in localStorage so a refresh doesn't log you out; the server
   expires it after 12 hours. */

const TOKEN_KEY = "pc-token";
const USER_KEY = "pc-user";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function getUsername() {
  return localStorage.getItem(USER_KEY);
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

/** POST /login -> stores the bearer token. Throws on bad credentials. */
export async function login(username, password) {
  const res = await getJSON("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  localStorage.setItem(TOKEN_KEY, res.token);
  localStorage.setItem(USER_KEY, res.username);
  return res;
}

/**
 * Claude's coordinator-style explanation of the current recommendation.
 * Requires a login token; a 401 means the session expired - the caller
 * should clear it and show the sign-in page again.
 */
export function fetchExplanation(situation) {
  return getJSON("/explain", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken() ?? ""}`,
    },
    body: JSON.stringify(situation),
  });
}
