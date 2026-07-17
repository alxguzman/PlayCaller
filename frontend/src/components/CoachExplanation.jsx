import { useEffect, useState } from "react";
import { fetchExplanation } from "../api.js";

/**
 * Claude's coordinator-style read on the current recommendation.
 *
 * Deliberately button-triggered (not automatic like /recommend): every
 * call spends real Claude API tokens, so the user decides when to ask.
 * The answer is tied to the recommendation it explains - when the
 * situation changes and a new rec arrives, the stale text is cleared.
 */
export default function CoachExplanation({ rec, situation, onAuthError }) {
  const [text, setText] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // New recommendation -> the old explanation no longer applies.
  useEffect(() => {
    setText(null);
    setError(null);
  }, [rec]);

  function ask() {
    setLoading(true);
    setError(null);
    fetchExplanation(situation)
      .then((r) => setText(r.explanation))
      .catch((e) => {
        if (e.status === 401) onAuthError();
        else setError(e.message);
      })
      .finally(() => setLoading(false));
  }

  return (
    <div className="panel coach-panel">
      <h2 className="panel-title">
        Coach&apos;s Call <span className="coach-badge">Claude AI</span>
      </h2>
      {text ? (
        <p className="coach-text">“{text}”</p>
      ) : (
        <p className="muted coach-empty">
          {loading
            ? "The coordinator is thinking…"
            : "Ask Claude to break down why this is the right call."}
        </p>
      )}
      {error && <div className="coach-error">{error}</div>}
      <button
        className="coach-btn"
        onClick={ask}
        disabled={loading || !rec}
        title="Uses Claude API credits"
      >
        {loading ? "Thinking…" : text ? "🎧 Ask again" : "🎧 Explain this call"}
      </button>
    </div>
  );
}
