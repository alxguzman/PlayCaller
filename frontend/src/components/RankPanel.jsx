import { epa } from "../lib/labels.js";

/** 1 -> "1st", 2 -> "2nd", 23 -> "23rd". */
function ordinal(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

/**
 * Continuous league-rank color: rank 1 (best) is green, fading through
 * orange in the middle to red for the worst. Rank already encodes quality
 * for both panels (1 = best), so the same mapping works for offense and
 * defense.
 */
function rankColor(rank, n) {
  const f = n > 1 ? (rank - 1) / (n - 1) : 0; // 0 (best) .. 1 (worst)
  // Piecewise hue so the middle reads orange, not yellow:
  // green(140) -> orange(32) -> red(0).
  const hue =
    f <= 0.5 ? 140 + (32 - 140) * (f / 0.5) : 32 + (0 - 32) * ((f - 0.5) / 0.5);
  return `hsl(${Math.round(hue)}, 85%, 58%)`;
}

function RankRow({ label, rank, epaValue, n, epaSuffix }) {
  return (
    <div className="rank-row">
      <span className="rank-label">{label}</span>
      <span className="rank-value" style={{ color: rankColor(rank, n) }}>
        {ordinal(rank)}
        <span className="rank-of"> / {n}</span>
      </span>
      <span className="rank-epa">
        {epa(epaValue)} {epaSuffix}
      </span>
    </div>
  );
}

/**
 * A team-form panel: Overall / vs Pass / vs Run league ranks from the
 * rolling 4-week snapshot. Used for both the offense (posteam) and defense
 * (defteam) below the call sheet.
 */
export default function RankPanel({ title, rows, nTeams, epaSuffix, hint, emptyText }) {
  return (
    <aside className="panel rank-panel">
      <h2 className="panel-title">{title}</h2>
      {!rows ? (
        <p className="muted">{emptyText}</p>
      ) : (
        <>
          {rows.map((r) => (
            <RankRow
              key={r.label}
              label={r.label}
              rank={r.rank}
              epaValue={r.epa}
              n={nTeams}
              epaSuffix={epaSuffix}
            />
          ))}
          <p className="rank-hint">{hint}</p>
        </>
      )}
    </aside>
  );
}
