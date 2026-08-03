export default function Header({ username, onSignOut }) {
  return (
    <header className="header">
      <div className="brand">
        <img className="brand-logo" src="/football.png" alt="AI Play Pilot logo" />
        <div>
          <div className="brand-name">AI Play Pilot</div>
        </div>
      </div>
      <nav className="nav">
        <button className="nav-tab active">▶ Live Caller</button>
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
