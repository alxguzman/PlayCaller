import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearSession,
  fetchDefenseRanks,
  fetchOffenseRanks,
  fetchOptions,
  fetchRecommendation,
  getToken,
  getUsername,
} from "./api.js";
import Header from "./components/Header.jsx";
import Login from "./components/Login.jsx";
import SituationPanel from "./components/SituationPanel.jsx";
import RecommendationCard from "./components/RecommendationCard.jsx";
import CoachExplanation from "./components/CoachExplanation.jsx";
import Field from "./components/Field.jsx";
import PlayComparison from "./components/PlayComparison.jsx";
import CallSheet from "./components/CallSheet.jsx";
import RankPanel from "./components/RankPanel.jsx";
import HistoryStrip from "./components/HistoryStrip.jsx";

/** Build the three Overall/Pass/Run rows a RankPanel renders. */
function rankRows(ranks) {
  if (!ranks) return null;
  return [
    { label: "Overall", rank: ranks.overall_rank, epa: ranks.overall_epa },
    { label: "vs Pass", rank: ranks.pass_rank, epa: ranks.pass_epa },
    { label: "vs Run", rank: ranks.run_rank, epa: ranks.run_epa },
  ];
}

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
  success_floor_gap: 0.05,
};

const randInt = (lo, hi) => Math.floor(Math.random() * (hi - lo + 1)) + lo;
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];

/**
 * Roll a fresh, valid game situation for the demo "Randomize" button.
 * Every Game Context and Offensive Look input gets a new value; the risk
 * dial is left where the user set it (it's a tuning knob, not a game fact).
 */
function randomSituation(options, prev) {
  const teams = options?.teams ?? [prev.posteam, prev.defteam];
  const posteam = pick(teams);
  // Same team on both sides is invalid - pick the defense from the rest.
  const others = teams.filter((t) => t !== posteam);
  const defteam = others.length ? pick(others) : prev.defteam;

  const yardline_100 = randInt(1, 99);
  return {
    ...prev,
    posteam,
    defteam,
    down: randInt(1, 4),
    // Never ask for more yards than there are to the end zone (goal-to-go).
    ydstogo: Math.min(randInt(1, 15), yardline_100),
    yardline_100,
    qtr: randInt(1, 4),
    clock: `${randInt(0, 14)}:${String(randInt(0, 59)).padStart(2, "0")}`,
    offScore: randInt(0, 35),
    defScore: randInt(0, 35),
    // "" = Auto; include it so demos sometimes show the model's own pick.
    formation: pick(["", ...(options?.formations ?? [])]),
    personnel: pick(["", ...(options?.personnel ?? [])]),
    defenders_in_box: pick(["", "3", "4", "5", "6", "7", "8", "9", "10", "11"]),
  };
}

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
  // null = not signed in -> show the login page instead of the dashboard.
  const [user, setUser] = useState(() => (getToken() ? getUsername() : null));
  const [situation, setSituation] = useState(DEFAULT_SITUATION);
  const [options, setOptions] = useState(null); // { teams, formations, personnel }
  const [defense, setDefense] = useState(null); // { n_teams, ranks }
  const [offense, setOffense] = useState(null); // { n_teams, ranks }
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
    // Team-form ranks are a static snapshot - fetch once alongside options.
    fetchOffenseRanks()
      .then(setOffense)
      .catch(() => setOffense(null));
    fetchDefenseRanks()
      .then(setDefense)
      .catch(() => setDefense(null));
  }, []);

  // Don't hit the API until someone is signed in - the login page
  // shouldn't fire requests in the background.
  useEffect(() => {
    if (user) loadOptions();
  }, [user, loadOptions]);

  // Re-ask the model whenever the situation changes (debounced so
  // dragging a slider doesn't fire a request per pixel).
  useEffect(() => {
    if (!options || !user) return;
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
  }, [situation, options, user]);

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

  const randomize = useCallback(
    () => setSituation((s) => randomSituation(options, s)),
    [options]
  );

  const clearHistory = useCallback(() => {
    localStorage.removeItem("pc-history");
    setHistory([]);
  }, []);

  const signOut = useCallback(() => {
    clearSession();
    setUser(null);
  }, []);

  if (!user) {
    return <Login onSignIn={setUser} />;
  }

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
      <Header username={user} onSignOut={signOut} />
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
          onRandomize={randomize}
        />
        <section className="center-col">
          <RecommendationCard
            rec={rec}
            loading={loading}
            onSave={saveToHistory}
          />
          <CoachExplanation
            rec={rec}
            situation={toRequest(situation)}
            onAuthError={signOut}
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
        <section className="right-col">
          <CallSheet rec={rec} floorGap={situation.success_floor_gap} />
          <RankPanel
            title={`${situation.posteam} Offense · last 4 weeks`}
            rows={rankRows(offense?.ranks?.[situation.posteam])}
            nTeams={offense?.n_teams ?? 32}
            epaSuffix="EPA/play"
            hint="League rank by EPA per play (1 = best). Rolling average of the offense's previous 4 games — higher is better."
            emptyText="No form data for this offense."
          />
          <RankPanel
            title={`${situation.defteam} Defense · last 4 weeks`}
            rows={rankRows(defense?.ranks?.[situation.defteam])}
            nTeams={defense?.n_teams ?? 32}
            epaSuffix="EPA/play allowed"
            hint="League rank by EPA allowed per play (1 = stingiest). Rolling average of the defense's previous 4 games — lower is better."
            emptyText="No form data for this defense."
          />
        </section>
      </main>
      <HistoryStrip history={history} onClear={clearHistory} />
    </div>
  );
}
