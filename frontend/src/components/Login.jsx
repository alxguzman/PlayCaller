import { useState } from "react";
import { login } from "../api.js";

/**
 * Full-screen sign-in gate. Credentials live in the server's .env
 * (APP_USERNAME / APP_PASSWORD); the point is to keep strangers from
 * spending the Claude API tokens behind /explain.
 */
export default function Login({ onSignIn }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await login(username.trim(), password);
      onSignIn(res.username);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      {/* Blurred dashboard screenshot (frontend/public/login-bg.png) - a
          teaser of what signing in unlocks. If the file is missing the
          div just renders the plain dark background. */}
      <div className="login-bg" aria-hidden="true" />
      <form className="panel login-card" onSubmit={submit}>
        <div className="login-brand">
          <div className="brand-logo">🏈</div>
          <div>
            <div className="brand-name">AI Play Caller</div>
            <div className="brand-tag">Smarter calls. Better results.</div>
          </div>
        </div>
        <h2 className="panel-title">Sign in</h2>
        <p className="muted login-hint">
          The AI coordinator uses Claude API credits, so the dashboard is
          login-only.
        </p>
        <label className="login-label">
          Username
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            autoFocus
            required
          />
        </label>
        <label className="login-label">
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>
        {error && <div className="login-error">{error}</div>}
        <button className="login-btn" type="submit" disabled={busy}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
