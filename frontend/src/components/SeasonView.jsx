import { useEffect, useState } from "react";
import { fetchSeasonScorecard } from "../api.js";

// nflverse numbers the playoffs as weeks 19-22.
const PLAYOFF_WEEKS = { 19: "WC", 20: "DIV", 21: "CONF", 22: "SB" };
const weekLabel = (w) => PLAYOFF_WEEKS[w] ?? w;

const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const epa = (x) =>
  x == null ? "–" : `${x >= 0 ? "+" : ""}${x.toFixed(3)}`;
const epaClass = (x) => (x == null ? "" : x >= 0 ? "green" : "red");

/**
 * The season-long model-vs-coach record, built from nflverse data by
 * `python -m src.model.season_log` after each weekly refresh. Unlike the
 * Live tab's scorecard (family-level, yardage-rule success, only plays
 * you watched), this covers EVERY play of EVERY game with true EPA and
 * exact play concepts.
 */
export default function SeasonView() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSeasonScorecard().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error || (data && !data.available)) {
    return (
      <main className="layout live-layout">
        <div className="panel live-picker">
          <h2 className="panel-title">Season scorecard</h2>
          <p className="muted">
            {error
              ? `Can't load the scorecard (${error}).`
              : "No season scorecard yet. It appears after the first weekly " +
                "data refresh of the season (python -m src.model.season_log)."}
          </p>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="layout live-layout">
        <div className="panel live-picker">
          <p className="muted">Loading…</p>
        </div>
      </main>
    );
  }

  const t = data.totals;
  return (
    <main className="layout live-layout">
      <div className="panel season-panel">
        <h2 className="panel-title">
          Model vs Coach · {data.season} season
        </h2>
        <p className="muted season-sub">
          Every {data.season} play, graded after the weekly nflverse data
          refresh (last updated {data.generated_at}). The model sees only
          pre-snap information — same recommendations it would have made live.
        </p>

        <div className="live-stat-row">
          <div className="live-stat">
            <div className="stat-label">Plays graded</div>
            <div className="stat-big">{t.plays.toLocaleString()}</div>
          </div>
          <div className="live-stat">
            <div className="stat-label">Exact-call agreement</div>
            <div className="stat-big">{pct(t.agreement)}</div>
          </div>
          <div className="live-stat">
            <div className="stat-label">EPA/play when agreed</div>
            <div className={`stat-big ${epaClass(t.agreed_epa)}`}>{epa(t.agreed_epa)}</div>
          </div>
          <div className="live-stat">
            <div className="stat-label">EPA/play otherwise</div>
            <div className={`stat-big ${epaClass(t.other_epa)}`}>{epa(t.other_epa)}</div>
          </div>
        </div>

        <div className="season-table-wrap">
          <table className="live-table">
            <thead>
              <tr>
                <th>week</th>
                <th>plays</th>
                <th>agreement</th>
                <th>agreed EPA</th>
                <th>other EPA</th>
                <th>agreed success</th>
                <th>other success</th>
              </tr>
            </thead>
            <tbody>
              {data.weeks.map((w) => (
                <tr key={w.week}>
                  <td>{weekLabel(w.week)}</td>
                  <td>{w.plays.toLocaleString()}</td>
                  <td>{pct(w.agreement)}</td>
                  <td className={epaClass(w.agreed_epa)}>{epa(w.agreed_epa)}</td>
                  <td className={epaClass(w.other_epa)}>{epa(w.other_epa)}</td>
                  <td>{pct(w.agreed_success)}</td>
                  <td>{pct(w.other_success)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="live-fineprint">
          "Agreed" = the coach called exactly what the model would have
          recommended in that situation. When the coach called something
          else, the model's play never ran — so "other" rows are a
          comparison, not proof the model was wrong (or right). Agreement
          also clusters in easier situations, so read the EPA gap as
          evidence, not a controlled experiment.
        </p>
      </div>
    </main>
  );
}
