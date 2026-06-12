import { conceptLabel, conceptType, epa, pct } from "../lib/labels.js";

/**
 * The right panel: every play concept the model scored, sorted by
 * expected EPA, with the floor-and-ceiling logic made visible.
 */
export default function CallSheet({ rec, floorGap }) {
  return (
    <aside className="panel callsheet">
      <h2 className="panel-title">Call Sheet</h2>
      <p className="callsheet-hint">
        Ranked by expected EPA. Greyed plays trail the safest option's
        success rate by more than {floorGap.toFixed(2)}, so they can't be
        the call.
      </p>
      {!rec && <p className="muted">Waiting for the model…</p>}
      <ol className="callsheet-list">
        {rec?.ranking.map((p) => {
          const type = conceptType(p.play_concept);
          const isPick = p.play_concept === rec.best_call;
          return (
            <li
              key={p.play_concept}
              className={`sheet-row${p.meets_floor ? "" : " below-floor"}${isPick ? " pick" : ""}`}
            >
              <span className={`type-chip ${type}`}>{type === "trick" ? "✦" : type}</span>
              <span className="sheet-name" title={isPick ? "The model's pick" : undefined}>
                {conceptLabel(p.play_concept)}
                {isPick && <span className="pick-badge">★</span>}
              </span>
              <span className="sheet-bars">
                <span className="bar-track">
                  <span
                    className={`bar-fill ${p.expected_epa >= 0 ? "green" : "red"}`}
                    style={{ width: `${Math.min(100, Math.abs(p.expected_epa) * 150)}%` }}
                  />
                </span>
              </span>
              <span className={`sheet-epa ${p.expected_epa >= 0 ? "green" : "red"}`}>
                {epa(p.expected_epa)}
              </span>
              <span className="sheet-prob">{pct(p.success_prob)}</span>
            </li>
          );
        })}
      </ol>
      {rec && (
        <div className="sheet-footer">
          <span>EPA</span>
          <span>P(success)</span>
        </div>
      )}
    </aside>
  );
}
