import { useCallback, useEffect, useRef, useState } from "react";
import { fetchOptions, fetchRecommendation } from "./api.js";
import Header from "./components/Header.jsx";
import SituationPanel from "./components/SituationPanel.jsx";
import RecommendationCard from "./components/RecommendationCard.jsx";
import Field from "./components/Field.jsx";
import PlayComparison from "./components/PlayComparison.jsx";
import CallSheet from "./components/CallSheet.jsx";
import HistoryStrip from "./components/HistoryStrip.jsx";

const DEFAULT_SITUATION = {
  posteam: "KC",
  defteam: "BUF",
  down: 1,
  ydstogo: 10,
  yardline_100: 75,
  qtr: 1,
  clock: "15:00", // time left in the quarter, mm:ss
  offScore: 0,
  defScore: 0,
  formation: "", // "" = let the model pick each concept's natural home
  personnel: "",
  defenders_in_box: "", // "" = typical box
  success_floor_gap: 0.1,
};

/** mm:ss left in the quarter -> seconds left in the game. */
function gameSecondsRemaining(qtr, clock) {
  const m = /^(\d{1,2}):([0-5]\d)$/.exec(clock?.trim() ?? "");
  const inQuarter = m ? Math.min(900, Number(m[1]) * 60 + Number(m[2])) : 900;
  return Math.max(0, (4 - qtr) * 900 + inQuarter);
}

/** UI state -> the exact JSON body POST /recommend expects. */
function toRequest(s) {
  const body = {
    posteam: s.posteam,
    defteam: s.defteam,
    down: s.down,
    ydstogo: s.ydstogo,
    yardline_100: s.yardline_100,
    qtr: s.qtr,
    game_seconds_remaining: gameSecondsRemaining(s.qtr, s.clock),
    score_differential: s.offScore - s.defScore,
    success_floor_gap: s.success_floor_gap,
  };
  if (s.formation) body.formation = s.formation;
  if (s.personnel) body.personnel = s.personnel;
  if (s.defenders_in_box !== "") body.defenders_in_box = Number(s.defenders_in_box);
  return body;
}

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem("pc-history") ?? "[]");
  } catch {
    return [];
  }
}

export default function App() {
  const [situation, setSituation] = useState(DEFAULT_SITUATION);
  const [options, setOptions] = useState(null); // { teams, formations, personnel }
  const [rec, setRec] = useState(null); // last /recommend response
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [history, setHistory] = useState(loadHistory);
  const debounceRef = useRef(null);

  const update = useCallback(
    (patch) => setSituation((s) => ({ ...s, ...patch })),
    []
  );

  const loadOptions = useCallback(() => {
    fetchOptions()
      .then((o) => {
        setOptions(o);
        setApiError(null);
      })
      .catch((e) => setApiError(`Can't reach the API (${e.message}).`));
  }, []);

  useEffect(loadOptions, [loadOptions]);

  // Re-ask the model whenever the situation changes (debounced so
  // dragging a slider doesn't fire a request per pixel).
  useEffect(() => {
    if (!options) return;
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setLoading(true);
      fetchRecommendation(toRequest(situation))
        .then((r) => {
          setRec(r);
          setApiError(null);
        })
        .catch((e) => setApiError(`Recommendation failed: ${e.message}`))
        .finally(() => setLoading(false));
    }, 350);
    return () => clearTimeout(debounceRef.current);
  }, [situation, options]);

  const saveToHistory = useCallback(() => {
    if (!rec) return;
    setHistory((h) => {
      const entry = {
        ts: Date.now(),
        posteam: situation.posteam,
        defteam: situation.defteam,
        down: situation.down,
        ydstogo: situation.ydstogo,
        yardline_100: situation.yardline_100,
        best_call: rec.best_call,
        best_epa: rec.best_epa,
        best_prob: rec.best_prob,
      };
      const next = [entry, ...h].slice(0, 10);
      localStorage.setItem("pc-history", JSON.stringify(next));
      return next;
    });
  }, [rec, situation]);

  const clearHistory = useCallback(() => {
    localStorage.removeItem("pc-history");
    setHistory([]);
  }, []);

  // What the field actually shows: pinned values win, otherwise the
  // recommender's defaults (its natural-home formation, a typical box).
  const fieldFormation =
    situation.formation ||
    (rec?.best_call === "qb_sneak" ? "under_center_1back" : "shotgun_1back");
  const fieldPersonnel = situation.personnel || "11";
  const fieldBox =
    situation.defenders_in_box === "" ? 7 : Number(situation.defenders_in_box);

  return (
    <div className="app">
      <Header />
      {apiError && (
        <div className="api-banner">
          <span>
            {apiError} Start it with{" "}
            <code>uvicorn src.api.main:app --reload</code>
          </span>
          <button onClick={loadOptions}>Retry</button>
        </div>
      )}
      <main className="layout">
        <SituationPanel
          situation={situation}
          options={options}
          onChange={update}
          onReset={() => setSituation(DEFAULT_SITUATION)}
        />
        <section className="center-col">
          <RecommendationCard
            rec={rec}
            loading={loading}
            onSave={saveToHistory}
          />
          <Field
            situation={situation}
            formation={fieldFormation}
            personnel={fieldPersonnel}
            box={fieldBox}
            bestCall={rec?.best_call}
          />
          <PlayComparison rec={rec} />
        </section>
        <CallSheet rec={rec} floorGap={situation.success_floor_gap} />
      </main>
      <HistoryStrip history={history} onClear={clearHistory} />
    </div>
  );
}
