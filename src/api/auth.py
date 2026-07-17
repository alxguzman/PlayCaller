"""
auth.py
-------
Minimal sign-in for the dashboard: one username + password from .env, and
a signed expiring token so the browser doesn't have to re-send the password
on every request. The point is to keep strangers from burning the Claude
API key behind /explain - it's a padlock, not a bank vault.

Env vars (see .env.example):
  APP_USERNAME    login name (default "coach")
  APP_PASSWORD    login password (REQUIRED - login is disabled without it)
  APP_SECRET_KEY  signs the tokens; if unset a random one is generated at
                  startup, which just means everyone re-logs-in after a
                  server restart.

Token format:  base64url("username|expiry_unix") + "." + HMAC-SHA256 hex
Stdlib only - no JWT library needed for a single-user app.
"""

import base64
import hashlib
import hmac
import os
import secrets
import time

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

TOKEN_TTL_SECONDS = 12 * 60 * 60  # 12 hours, then sign in again

_SECRET = os.environ.get("APP_SECRET_KEY") or secrets.token_hex(32)


def _sign(payload: bytes) -> str:
    return hmac.new(_SECRET.encode(), payload, hashlib.sha256).hexdigest()


def check_credentials(username: str, password: str) -> bool:
    """Constant-time comparison against the .env credentials."""
    expected_user = os.environ.get("APP_USERNAME", "coach")
    expected_pass = os.environ.get("APP_PASSWORD")
    if not expected_pass:
        raise HTTPException(
            status_code=503,
            detail="Login is not configured - set APP_PASSWORD in .env on the server.",
        )
    ok_user = secrets.compare_digest(username.encode(), expected_user.encode())
    ok_pass = secrets.compare_digest(password.encode(), expected_pass.encode())
    return ok_user and ok_pass


def create_token(username: str) -> str:
    expiry = int(time.time()) + TOKEN_TTL_SECONDS
    payload = f"{username}|{expiry}".encode()
    return base64.urlsafe_b64encode(payload).decode() + "." + _sign(payload)


def verify_token(token: str) -> str:
    """Return the username if the token is valid, else raise 401."""
    try:
        payload_b64, signature = token.split(".", 1)
        payload = base64.urlsafe_b64decode(payload_b64.encode())
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Malformed token.")
    if not hmac.compare_digest(_sign(payload), signature):
        raise HTTPException(status_code=401, detail="Invalid token - sign in again.")
    username, _, expiry = payload.decode().rpartition("|")
    if int(expiry) < time.time():
        raise HTTPException(status_code=401, detail="Session expired - sign in again.")
    return username


# FastAPI dependency: put `user: str = Depends(require_auth)` on any route
# that should be login-only. auto_error=False lets us return our own 401
# message instead of FastAPI's generic 403.
_bearer = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Sign in required.")
    return verify_token(credentials.credentials)
