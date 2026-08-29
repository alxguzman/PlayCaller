export default function Header({ username, onSignOut, view, onViewChange, liveNow }) {
  return (
    <header className="header">
      <div className="brand">
        <img className="brand-logo" src="/football.png" alt="AI Play Pilot logo" />
        <div>
          <div className="brand-name">AI Play Pilot</div>
        </div>
      </div>
      <nav className="nav">
        <button
          className={`nav-tab${view === "sim" ? " active" : ""}`}
          onClick={() => onViewChange("sim")}
        >
          ▶ Play Caller
        </button>
        <button
          className={`nav-tab${view === "live" ? " active" : ""}${liveNow ? " live-on" : ""}`}
          onClick={() => onViewChange("live")}
          title={liveNow ? "Games are live right now!" : undefined}
        >
          ● Live Games
        </button>
        <button
          className={`nav-tab${view === "season" ? " active" : ""}`}
          onClick={() => onViewChange("season")}
        >
          ▦ Season
        </button>
      </nav>
      <div className="header-user">
        <span className="muted">{username}</span>
        <button className="signout-btn" onClick={onSignOut}>
          Sign out
        </button>
      </div>
    </header>
  );
}
