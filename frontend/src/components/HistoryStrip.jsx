import { conceptLabel, conceptType, epa, fieldPositionLabel } from "../lib/labels.js";

const ORDINAL = { 1: "1st", 2: "2nd", 3: "3rd", 4: "4th" };

/** Saved recommendations, newest first (kept in localStorage). */
export default function HistoryStrip({ history, onClear }) {
  return (
    <footer className="history">
      <div className="history-head">
        <h2 className="panel-title">Saved Recommendations</h2>
        {history.length > 0 && (
          <button className="clear-btn" onClick={onClear}>
            Clear
          </button>
        )}
      </div>
      {history.length === 0 ? (
        <p className="muted">
          Nothing saved yet — hit ☆ Save Recommendation on a call you like.
        </p>
      ) : (
        <div className="history-cards">
          {history.map((h) => (
            <div key={h.ts} className="history-card">
              <div className="history-sit">
                {ORDINAL[h.down]} & {h.ydstogo} ·{" "}
                {fieldPositionLabel(h.yardline_100, h.posteam, h.defteam)}
              </div>
              <div className={`history-call ${conceptType(h.best_call)}`}>
                {conceptLabel(h.best_call)}
              </div>
              <div className="history-meta">
                {h.posteam} vs {h.defteam} · EPA {epa(h.best_epa)}
              </div>
            </div>
          ))}
        </div>
      )}
    </footer>
  );
}
