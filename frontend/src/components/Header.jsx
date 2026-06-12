export default function Header() {
  return (
    <header className="header">
      <div className="brand">
        <div className="brand-logo">🏈</div>
        <div>
          <div className="brand-name">AI Play Caller</div>
          <div className="brand-tag">Smarter calls. Better results.</div>
        </div>
      </div>
      <nav className="nav">
        <button className="nav-tab active">▶ Live Caller</button>
        <button className="nav-tab" disabled title="Coming soon">
          Dashboard
        </button>
        <button className="nav-tab" disabled title="Coming soon">
          History
        </button>
      </nav>
    </header>
  );
}
