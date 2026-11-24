import hashlib
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from ..schemas.auth import LoginResponse, User


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class AuthStore:
    def __init__(self, path: Path, session_ttl: int, refresh_ttl: int):
        self.path = path
        self.session_ttl = session_ttl
        self.refresh_ttl = refresh_ttl
        self.hasher = PasswordHasher()
        # Unknown accounts go through the same password verification as existing accounts.
        self.dummy_hash = self.hasher.hash(secrets.token_urlsafe(32))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            yield db

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_digest BLOB PRIMARY KEY NOT NULL,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS refresh_tokens (
                    token_digest BLOB PRIMARY KEY NOT NULL,
                    access_digest BLOB NOT NULL UNIQUE
                        REFERENCES sessions(token_digest) ON DELETE CASCADE,
                    expires_at INTEGER NOT NULL
                );
            """)

    def signup(self, email: str, password: str) -> User | None:
        password_hash = self.hasher.hash(password)
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO users(email, password_hash) VALUES (?, ?) "
                "ON CONFLICT(email) DO NOTHING",
                (email, password_hash),
            )
            if not cursor.rowcount:
                return None
            return User(id=cursor.lastrowid, email=email)

    def login(self, email: str, password: str) -> LoginResponse | None:
        with self._connect() as db:
            user = db.execute(
                "SELECT id, password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
            password_hash = user["password_hash"] if user else self.dummy_hash
            try:
                self.hasher.verify(password_hash, password)
            except VerifyMismatchError:
                return None
            if user is None:
                return None
            if self.hasher.check_needs_rehash(password_hash):
                db.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ? AND password_hash = ?",
                    (self.hasher.hash(password), user["id"], password_hash),
                )
            now = int(time.time())
            db.execute(
                """DELETE FROM sessions WHERE expires_at <= ? AND NOT EXISTS (
                    SELECT 1 FROM refresh_tokens
                    WHERE access_digest = sessions.token_digest AND expires_at > ?
                )""",
                (now, now),
            )
            return self._issue_tokens(db, user["id"], now, now + self.refresh_ttl)

    def _issue_tokens(
        self, db: sqlite3.Connection, user_id: int, now: int, refresh_expires_at: int
    ) -> LoginResponse:
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        access_digest = _token_digest(access_token)
        remaining = refresh_expires_at - now
        access_ttl = min(self.session_ttl, remaining)
        db.execute(
            "INSERT INTO sessions(token_digest, user_id, expires_at) VALUES (?, ?, ?)",
            (access_digest, user_id, now + access_ttl),
        )
        db.execute(
            "INSERT INTO refresh_tokens(token_digest, access_digest, expires_at) VALUES (?, ?, ?)",
            (_token_digest(refresh_token), access_digest, refresh_expires_at),
        )
        return LoginResponse(
            access_token=access_token, refresh_token=refresh_token,
            expires_in=access_ttl, refresh_expires_in=remaining,
        )

    def refresh(self, token: str) -> LoginResponse | None:
        with self._connect() as db:
            # Serialize redemption so two requests cannot consume the same token.
            db.execute("BEGIN IMMEDIATE")
            now = int(time.time())
            session = db.execute(
                """SELECT s.user_id, r.access_digest, r.expires_at
                   FROM refresh_tokens AS r JOIN sessions AS s ON s.token_digest = r.access_digest
                   WHERE r.token_digest = ? AND r.expires_at > ?""",
                (_token_digest(token), now),
            ).fetchone()
            if session is None:
                return None
            db.execute("DELETE FROM sessions WHERE token_digest = ?", (session["access_digest"],))
            return self._issue_tokens(db, session["user_id"], now, session["expires_at"])

    def current_user(self, token: str) -> User | None:
        with self._connect() as db:
            user = db.execute(
                "SELECT users.id, users.email FROM sessions JOIN users ON users.id = user_id "
                "WHERE token_digest = ? AND expires_at > ?",
                (_token_digest(token), int(time.time())),
            ).fetchone()
        return User(**user) if user else None

    def logout(self, token: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE token_digest = ?", (_token_digest(token),))
