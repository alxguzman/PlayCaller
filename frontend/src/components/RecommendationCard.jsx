import { conceptLabel, conceptType, epa, pct } from "../lib/labels.js";

/** SVG progress ring showing the recommended play's success probability. */
function Ring({ value }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 84 84" className="ring">
      <circle cx="42" cy="42" r={r} className="ring-track" />
      <circle
        cx="42"
        cy="42"
        r={r}
        className="ring-fill"
        strokeDasharray={`${c * value} ${c}`}
        transform="rotate(-90 42 42)"
      />
      <text x="42" y="47" textAnchor="middle" className="ring-text">
        {pct(value)}
      </text>
    </svg>
  );
}

export default function RecommendationCard({ rec, loading, onSave }) {
  if (!rec) {
    return (
      <div className="panel rec-card empty">
        <h2 className="panel-title">AI Recommendation</h2>
        <p className="muted">Waiting for the model…</p>
      </div>
    );
  }

  // The counterfactual: how the best call of the *other* family scored.
  const type = conceptType(rec.best_call);
  const otherType = type === "run" ? "pass" : "run";
  const bestOther = rec.ranking.find((p) => conceptType(p.play_concept) === otherType);

  return (
    <div className={`panel rec-card${loading ? " loading" : ""}`}>
      <h2 className="panel-title">AI Recommendation</h2>
      <div className="rec-body">
        <div className="rec-main">
          <div className="rec-call">{conceptLabel(rec.best_call)}</div>
          <div className="rec-sub">Recommended play · {type.toUpperCase()}</div>
        </div>
        <div className="rec-epa">
          <div className="stat-label">Expected EPA</div>
          <div className="stat-big green">{epa(rec.best_epa)}</div>
          {bestOther && (
            <div className="rec-vs">
              vs best {otherType}{" "}
              <span className={bestOther.expected_epa >= 0 ? "green" : "red"}>
                {epa(bestOther.expected_epa)}
              </span>
            </div>
          )}
        </div>
        <div className="rec-ring">
          <div className="stat-label">Success Probability</div>
          <Ring value={rec.best_prob} />
        </div>
      </div>
      <button className="save-btn" onClick={onSave}>
        ☆ Save Recommendation
      </button>
    </div>
  );
}
