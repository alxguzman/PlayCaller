import { useEffect, useRef, useState } from "react";
import SituationPanel from "./SituationPanel.jsx";
import RecommendationCard from "./RecommendationCard.jsx";
import CoachExplanation from "./CoachExplanation.jsx";
import Field from "./Field.jsx";
import PlayComparison from "./PlayComparison.jsx";
import CallSheet from "./CallSheet.jsx";
import RankPanel from "./RankPanel.jsx";
import HistoryStrip from "./HistoryStrip.jsx";
import { downDistanceLabel, fieldPositionLabel } from "../lib/labels.js";

function SituationEditor({ situation, options, onApply, onDismiss, defaults, randomize }) {
  const [draft, setDraft] = useState(situation);
  const dialog = useRef(null);
  useEffect(() => {
    const el = dialog.current;
    const previousOverflow = document.body.style.overflow;
    el.showModal();
    document.body.style.overflow = "hidden";
    return () => {
      el.close();
      document.body.style.overflow = previousOverflow;
    };
  }, []);
  return (
    <dialog ref={dialog} className="situation-sheet" aria-labelledby="situation-title"
      onCancel={(event) => { event.preventDefault(); onDismiss(); }}>
      <div className="sheet-heading">
        <h2 id="situation-title">Game situation</h2>
        <button autoFocus className="mobile-button" aria-label="Close situation editor" onClick={onDismiss}>✕</button>
      </div>
      <div className="sheet-scroll">
        <SituationPanel mobile situation={draft} options={options}
          onChange={(patch) => setDraft((s) => ({ ...s, ...patch }))}
          onReset={() => setDraft(defaults)}
          onRandomize={() => setDraft((s) => randomize(options, s))} />
      </div>
      <div className="sheet-apply">
        <button className="mobile-button primary" onClick={() => onApply(draft)}>Apply situation</button>
      </div>
    </dialog>
  );
}

export default function MobileDashboard({ situation: s, options, rec, loading, onSave,
  onApply, defaults, randomize, request, formation, personnel, box, offense, defense,
  rankRows, history, onClearHistory }) {
  const [editing, setEditing] = useState(false);
  const [teamSide, setTeamSide] = useState("offense");
  const editButton = useRef(null);
  const isOffense = teamSide === "offense";
  const team = isOffense ? s.posteam : s.defteam;
  const ranks = isOffense ? offense : defense;
  function dismiss() {
    setEditing(false);
    requestAnimationFrame(() => editButton.current?.focus());
  }
  return (
    <main className="mobile-dashboard">
      <section className="panel mobile-situation" aria-label="Current game situation">
        <div className="mobile-matchup"><span>{s.posteam} offense</span><span>{s.defteam} defense</span></div>
        <h2>{downDistanceLabel(s.down, s.ydstogo, s.yardline_100)} · {fieldPositionLabel(s.yardline_100, s.posteam, s.defteam)}</h2>
        <p>{s.qtr === 5 ? "OT" : `Q${s.qtr}`} {s.clock} · {s.posteam} {s.offScore} – {s.defteam} {s.defScore}</p>
        <button ref={editButton} className="mobile-button" onClick={() => setEditing(true)} aria-haspopup="dialog">Edit situation</button>
      </section>
      <RecommendationCard rec={rec} loading={loading} onSave={onSave} />
      <Field situation={s} formation={formation} personnel={personnel} box={box} bestCall={rec?.best_call} />
      <details className="panel mobile-disclosure">
        <summary>Why this call?</summary>
        <CoachExplanation rec={rec} situation={request} />
      </details>
      <details className="panel mobile-disclosure">
        <summary>Pass vs Run</summary>
        <PlayComparison rec={rec} />
      </details>
      <CallSheet rec={rec} floorGap={s.success_floor_gap} compact />
      <section className="mobile-team-form" aria-label="Team form">
        <h2>Team form · Last 4 weeks</h2>
        <div className="segmented mobile-team-toggle" role="group" aria-label="Choose team rankings">
          <button className={isOffense ? "on" : ""} aria-pressed={isOffense} onClick={() => setTeamSide("offense")}>{s.posteam} Offense</button>
          <button className={!isOffense ? "on" : ""} aria-pressed={!isOffense} onClick={() => setTeamSide("defense")}>{s.defteam} Defense</button>
        </div>
        <RankPanel title={`${team} ${isOffense ? "Offense" : "Defense"}`}
          rows={rankRows(ranks?.ranks?.[team], isOffense)} nTeams={ranks?.n_teams ?? 32}
          epaSuffix={isOffense ? "EPA/play" : "EPA/play allowed"}
          hint={`Previous 4 games. ${isOffense ? "Higher EPA is better." : "Lower EPA allowed is better."} Rank 1 = best.`}
          emptyText="No form data for this team." />
      </section>
      <details className="panel mobile-disclosure">
        <summary>Saved recommendations ({history.length})</summary>
        <HistoryStrip history={history} onClear={onClearHistory} />
      </details>
      {editing && <SituationEditor situation={s} options={options} defaults={defaults} randomize={randomize}
        onDismiss={dismiss} onApply={(draft) => { onApply(draft); dismiss(); }} />}
    </main>
  );
}
