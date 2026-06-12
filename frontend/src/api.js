// Thin wrapper around the FastAPI backend (src/api/main.py).
// Start it with:  uvicorn src.api.main:app --reload

const API_BASE = "http://127.0.0.1:8000";

async function getJSON(path, init) {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return res.json();
}

/** Legal dropdown values: { teams, formations, personnel } */
export function fetchOptions() {
  return getJSON("/options");
}

/** Full ranked call sheet for one game situation. */
export function fetchRecommendation(situation) {
  return getJSON("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(situation),
  });
}
