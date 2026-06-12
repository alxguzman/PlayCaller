import {
  FIELD_WIDTH,
  defensePlayers,
  offensePlayers,
  playArt,
} from "../lib/formations.js";

// Pixels per yard. The viewBox is 120yd x 53.3yd scaled by this.
const S = 9;
const W = 120 * S;
const H = FIELD_WIDTH * S;
const MID = FIELD_WIDTH / 2;

const clampY = (yd) => Math.max(2, Math.min(FIELD_WIDTH - 2, yd));

/**
 * The interactive field: line of scrimmage from the situation, players
 * from formation/personnel/box, route art from the recommended call.
 * Offense always drives left-to-right.
 */
export default function Field({ situation, formation, personnel, box, bestCall }) {
  // yardline_100 = yards to the opponent's end zone (which sits at x=110).
  const losYd = 110 - situation.yardline_100;
  const firstDownYd = Math.min(110, losYd + situation.ydstogo);

  const offense = offensePlayers(formation, personnel);
  const defense = defensePlayers(box, offense);
  const arrows = playArt(bestCall, offense);

  const px = (p) => (losYd + p.dx) * S;
  const py = (p) => clampY(MID + p.dy) * S;

  return (
    <div className="panel field-panel">
      <svg viewBox={`0 0 ${W} ${H}`} className="field">
        {/* turf + end zones */}
        <rect x="0" y="0" width={W} height={H} className="turf" />
        <rect x="0" y="0" width={10 * S} height={H} className="endzone" />
        <rect x={110 * S} y="0" width={10 * S} height={H} className="endzone" />
        <text x={5 * S} y={H / 2} className="ez-text" transform={`rotate(-90 ${5 * S} ${H / 2})`}>
          TOUCHDOWN
        </text>
        <text x={115 * S} y={H / 2} className="ez-text" transform={`rotate(90 ${115 * S} ${H / 2})`}>
          TOUCHDOWN
        </text>

        {/* yard lines + numbers every 10 */}
        {Array.from({ length: 21 }, (_, i) => 10 + i * 5).map((yd) => (
          <line key={yd} x1={yd * S} y1="0" x2={yd * S} y2={H} className="yardline" />
        ))}
        {[20, 30, 40, 50, 60, 70, 80, 90, 100].map((yd) => {
          const n = yd <= 60 ? yd - 10 : 110 - yd; // 10..50..10
          return (
            <g key={yd} className="yard-num">
              <text x={yd * S} y={H - 14}>{n}</text>
              <text x={yd * S} y={26}>{n}</text>
            </g>
          );
        })}

        {/* line of scrimmage + first-down marker */}
        <line x1={losYd * S} y1="0" x2={losYd * S} y2={H} className="los" />
        <line x1={firstDownYd * S} y1="0" x2={firstDownYd * S} y2={H} className="first-down" />

        {/* route / run art for the recommended play */}
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="arrow-head" />
          </marker>
        </defs>
        {arrows.map((a, i) => (
          <polyline
            key={`${bestCall}-${i}`}
            className={`route${a.dashed ? " dashed" : ""}`}
            markerEnd="url(#arrow)"
            points={a.points
              .map(([dx, dy]) => `${px(a.from) + dx * S},${clampY(MID + a.from.dy + dy) * S}`)
              .join(" ")}
          />
        ))}

        {/* players - CSS transitions on transform make them slide when inputs change */}
        {defense.map((p) => (
          <g key={p.id} className="player def" style={{ transform: `translate(${px(p)}px, ${py(p)}px)` }}>
            <text className="x-mark" textAnchor="middle" dy="5">✕</text>
            <title>{p.label}</title>
          </g>
        ))}
        {offense.map((p) => (
          <g key={p.id} className={`player off ${p.role}`} style={{ transform: `translate(${px(p)}px, ${py(p)}px)` }}>
            <circle r="7" />
            <title>{p.label}</title>
          </g>
        ))}
      </svg>
      <div className="field-legend">
        <span><i className="dot off-dot" /> {situation.posteam} offense</span>
        <span><i className="dot qb-dot" /> QB</span>
        <span className="x-legend">✕ {situation.defteam} defense ({box} in box)</span>
        <span className="los-legend">— LOS</span>
        <span className="fd-legend">— 1st down</span>
      </div>
    </div>
  );
}
