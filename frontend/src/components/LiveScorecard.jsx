import { conceptLabel, conceptType } from "../lib/labels.js";

const runOrPass = (fam) => (conceptType(fam) === "run" ? "run" : "pass");

/**
 * Running model-vs-coach tally for one live game.
 *
 * "Agreement" is at the play-FAMILY level (play-action variants fold into
 * their base concept) because live play descriptions can't reveal play
 * action. Success uses the standard 40/60/100 yardage rule - the closest
 * live stand-in for the model's real training target (EPA > 0).
 */
export default function LiveScorecard({ rows }) {
  const graded = rows.filter((r) => r.coach_family !== null);
  const agreed = graded.filter((r) => r.agreed);
  const disagreed = graded.filter((r) => !r.agreed);
  const rpAgreed = graded.filter(
    (r) => runOrPass(r.model_family) === runOrPass(r.coach_family)
  );

  const pct = (part, whole) =>
    whole.length ? `${Math.round((part.length / whole.length) * 100)}%` : "–";
  const successRate = (subset) =>
    subset.length
      ? `${Math.round((subset.filter((r) => r.success).length / subset.length) * 100)}%`
      : "–";
  const avgYards = (subset) =>
    subset.length
      ? (subset.reduce((s, r) => s + r.yards, 0) / subset.length).toFixed(1)
      : "–";

  return (
    <div className="panel live-scorecard">
      <h2 className="panel-title">Model vs Coach</h2>
      {graded.length === 0 ? (
        <p className="muted">
          No snaps graded yet. Once a play runs, it lands here: what the model
          recommended before the snap vs what the coach actually called.
        </p>
      ) : (
        <>
          <div className="live-stat-row">
            <div className="live-stat">
              <div className="stat-label">Snaps graded</div>
              <div className="stat-big">{graded.length}</div>
            </div>
            <div className="live-stat">
              <div className="stat-label">Same play family</div>
              <div className="stat-big">{pct(agreed, graded)}</div>
            </div>
            <div className="live-stat">
              <div className="stat-label">Same run/pass</div>
              <div className="stat-big">{pct(rpAgreed, graded)}</div>
            </div>
          </div>
          <table className="live-table">
            <thead>
              <tr>
                <th></th>
                <th>plays</th>
                <th>success*</th>
                <th>avg yds</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Coach agreed with model</td>
                <td>{agreed.length}</td>
                <td>{successRate(agreed)}</td>
                <td>{avgYards(agreed)}</td>
              </tr>
              <tr>
                <td>Coach called differently</td>
                <td>{disagreed.length}</td>
                <td>{successRate(disagreed)}</td>
                <td>{avgYards(disagreed)}</td>
              </tr>
            </tbody>
          </table>
          <p className="live-fineprint">
            *Success = the 40/60/100 yardage rule. Play-action variants grade
            as their base family (live play text can't reveal play action).
          </p>
          <ul className="live-log">
            {rows.slice(0, 12).map((r) => (
              <li key={r.playId} className="live-log-row">
                <span className={`live-log-badge ${r.coach_family === null ? "na" : r.agreed ? "hit" : "miss"}`}>
                  {r.coach_family === null ? "–" : r.agreed ? "✓" : "✗"}
                </span>
                <span>
                  model: <b>{conceptLabel(r.model_call)}</b> · coach:{" "}
                  <b>{r.coach_concept ? conceptLabel(r.coach_concept) : "ungradable"}</b> ·{" "}
                  {r.yards >= 0 ? "+" : ""}
                  {r.yards} yds
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
