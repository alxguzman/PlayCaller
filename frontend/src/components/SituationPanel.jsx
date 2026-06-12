import { fieldPositionLabel, formationLabel, personnelLabel } from "../lib/labels.js";

const DOWN_LABELS = { 1: "1st", 2: "2nd", 3: "3rd", 4: "4th" };

function Row({ label, children }) {
  return (
    <label className="row">
      <span className="row-label">{label}</span>
      <span className="row-control">{children}</span>
    </label>
  );
}

/**
 * The left panel: every input the /recommend endpoint accepts, plus the
 * scoreboard fields it's derived from.
 */
export default function SituationPanel({ situation: s, options, onChange, onReset }) {
  const teams = options?.teams ?? [s.posteam, s.defteam];
  const num = (key, lo, hi) => (e) => {
    const v = Number(e.target.value);
    if (Number.isFinite(v)) onChange({ [key]: Math.max(lo, Math.min(hi, v)) });
  };

  return (
    <aside className="panel sidebar">
      <h2 className="panel-title">Game Context</h2>

      <Row label="Offense">
        <select
          value={s.posteam}
          onChange={(e) => {
            const team = e.target.value;
            // Same team on both sides makes no sense - swap instead.
            onChange(team === s.defteam ? { posteam: team, defteam: s.posteam } : { posteam: team });
          }}
        >
          {teams.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
      </Row>
      <Row label="Defense">
        <select
          value={s.defteam}
          onChange={(e) => {
            const team = e.target.value;
            onChange(team === s.posteam ? { defteam: team, posteam: s.defteam } : { defteam: team });
          }}
        >
          {teams.map((t) => (
            <option key={t}>{t}</option>
          ))}
        </select>
      </Row>

      <Row label="Down">
        <div className="segmented">
          {[1, 2, 3, 4].map((d) => (
            <button
              key={d}
              className={s.down === d ? "on" : ""}
              onClick={() => onChange({ down: d })}
            >
              {DOWN_LABELS[d]}
            </button>
          ))}
        </div>
      </Row>
      <Row label="Distance">
        <input type="number" min="1" max="99" value={s.ydstogo} onChange={num("ydstogo", 1, 99)} />
      </Row>
      <Row label={`Ball on · ${fieldPositionLabel(s.yardline_100, s.posteam, s.defteam)}`}>
        <input
          type="range"
          min="1"
          max="99"
          value={100 - s.yardline_100}
          onChange={(e) => onChange({ yardline_100: 100 - Number(e.target.value) })}
        />
      </Row>

      <Row label="Score">
        <div className="score-inputs">
          <input type="number" min="0" max="99" value={s.offScore} onChange={num("offScore", 0, 99)} />
          <span className="score-sep">{s.posteam} – {s.defteam}</span>
          <input type="number" min="0" max="99" value={s.defScore} onChange={num("defScore", 0, 99)} />
        </div>
      </Row>

      <Row label="Quarter">
        <div className="segmented">
          {[1, 2, 3, 4, 5].map((q) => (
            <button
              key={q}
              className={s.qtr === q ? "on" : ""}
              onClick={() => onChange({ qtr: q })}
            >
              {q === 5 ? "OT" : q}
            </button>
          ))}
        </div>
      </Row>
      <Row label="Time left in qtr">
        <input
          type="text"
          className="clock"
          value={s.clock}
          placeholder="mm:ss"
          onChange={(e) => onChange({ clock: e.target.value })}
          onBlur={(e) => {
            if (!/^\d{1,2}:[0-5]\d$/.test(e.target.value.trim()))
              onChange({ clock: "15:00" });
          }}
        />
      </Row>

      <h2 className="panel-title">Offensive Look</h2>

      <Row label="Formation">
        <select value={s.formation} onChange={(e) => onChange({ formation: e.target.value })}>
          <option value="">Auto (model picks)</option>
          {(options?.formations ?? []).map((f) => (
            <option key={f} value={f}>
              {formationLabel(f)}
            </option>
          ))}
        </select>
      </Row>
      <Row label="Personnel">
        <select value={s.personnel} onChange={(e) => onChange({ personnel: e.target.value })}>
          <option value="">Auto (typical play)</option>
          {(options?.personnel ?? []).map((p) => (
            <option key={p} value={p}>
              {personnelLabel(p)}
            </option>
          ))}
        </select>
      </Row>
      <Row label="Defenders in box">
        <select
          value={s.defenders_in_box}
          onChange={(e) => onChange({ defenders_in_box: e.target.value })}
        >
          <option value="">Auto (typical box)</option>
          {[3, 4, 5, 6, 7, 8, 9, 10, 11].map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </Row>
      <Row label={`Risk dial · ${s.success_floor_gap.toFixed(2)}`}>
        <input
          type="range"
          min="0"
          max="0.5"
          step="0.01"
          value={s.success_floor_gap}
          onChange={(e) => onChange({ success_floor_gap: Number(e.target.value) })}
        />
      </Row>
      <div className="risk-hint">
        <span>safer</span>
        <span>chase upside</span>
      </div>

      <button className="reset-btn" onClick={onReset}>
        ⟳ Reset Situation
      </button>
    </aside>
  );
}
