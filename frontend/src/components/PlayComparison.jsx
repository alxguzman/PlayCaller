import { conceptLabel, conceptType, epa, pct } from "../lib/labels.js";

function Bar({ value, max, tone }) {
  const w = Math.max(0, Math.min(1, max > 0 ? value / max : 0)) * 100;
  return (
    <div className="bar-track">
      <div className={`bar-fill ${tone}`} style={{ width: `${w}%` }} />
    </div>
  );
}

/**
 * Pass vs run, head to head: the best concept from each family,
 * straight out of the ranking.
 */
export default function PlayComparison({ rec }) {
  if (!rec) return null;

  const bestOf = (type) =>
    rec.ranking.find((p) => conceptType(p.play_concept) === type);
  const rows = [bestOf("pass"), bestOf("run")].filter(Boolean);
  if (rows.length < 2) return null;

  const better = rows[0].expected_epa >= rows[1].expected_epa ? 0 : 1;

  return (
    <div className="panel">
      <h2 className="panel-title">Pass vs Run</h2>
      <table className="compare-table">
        <thead>
          <tr>
            <th>Best option</th>
            <th>Expected EPA</th>
            <th>Success rate</th>
            <th>Clears floor</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p, i) => {
            const type = conceptType(p.play_concept);
            const tone = i === better ? "green" : "red";
            return (
              <tr key={p.play_concept}>
                <td>
                  <span className={`type-chip ${type}`}>{type}</span>{" "}
                  {conceptLabel(p.play_concept)}
                </td>
                <td className={tone}>{epa(p.expected_epa)}</td>
                <td>
                  <span className="cell-num">{pct(p.success_prob)}</span>
                  <Bar value={p.success_prob} max={1} tone={tone} />
                </td>
                <td>{p.meets_floor ? "✓" : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
