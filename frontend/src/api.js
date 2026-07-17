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
