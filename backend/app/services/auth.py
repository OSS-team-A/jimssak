import hashlib
import secrets
import time

import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from ..db import Database
from ..schemas.auth import LoginResponse, User


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


class AuthStore:
    def __init__(self, db: Database, session_ttl: int, refresh_ttl: int):
        self.db = db
        self.session_ttl = session_ttl
        self.refresh_ttl = refresh_ttl
        self.hasher = PasswordHasher()
        # Unknown accounts go through the same password verification as existing accounts.
        self.dummy_hash = self.hasher.hash(secrets.token_urlsafe(32))

    def signup(self, email: str, password: str) -> User | None:
        password_hash = self.hasher.hash(password)
        with self.db.connect() as db:
            row = db.execute(
                "INSERT INTO users(email, password_hash) VALUES (%s, %s) "
                "ON CONFLICT(email) DO NOTHING RETURNING id",
                (email, password_hash),
            ).fetchone()
        return User(id=row["id"], email=email) if row else None

    def login(self, email: str, password: str) -> LoginResponse | None:
        with self.db.connect() as db:
            user = db.execute(
                "SELECT id, password_hash FROM users WHERE email = %s", (email,)
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
                    "UPDATE users SET password_hash = %s WHERE id = %s AND password_hash = %s",
                    (self.hasher.hash(password), user["id"], password_hash),
                )
            now = int(time.time())
            db.execute(
                """DELETE FROM sessions WHERE expires_at <= %s AND NOT EXISTS (
                    SELECT 1 FROM refresh_tokens
                    WHERE access_digest = sessions.token_digest AND expires_at > %s
                )""",
                (now, now),
            )
            return self._issue_tokens(db, user["id"], now, now + self.refresh_ttl)

    def _issue_tokens(
        self, db: psycopg.Connection, user_id: int, now: int, refresh_expires_at: int
    ) -> LoginResponse:
        access_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        access_digest = _token_digest(access_token)
        remaining = refresh_expires_at - now
        access_ttl = min(self.session_ttl, remaining)
        db.execute(
            "INSERT INTO sessions(token_digest, user_id, expires_at) VALUES (%s, %s, %s)",
            (access_digest, user_id, now + access_ttl),
        )
        db.execute(
            "INSERT INTO refresh_tokens(token_digest, access_digest, expires_at) "
            "VALUES (%s, %s, %s)",
            (_token_digest(refresh_token), access_digest, refresh_expires_at),
        )
        return LoginResponse(
            access_token=access_token, refresh_token=refresh_token,
            expires_in=access_ttl, refresh_expires_in=remaining,
        )

    def refresh(self, token: str) -> LoginResponse | None:
        with self.db.connect() as db:
            now = int(time.time())
            # Deleting is what redeems the token: a second request for the same token waits
            # on the row, then finds it gone, so a token can only be used once.
            redeemed = db.execute(
                "DELETE FROM refresh_tokens WHERE token_digest = %s AND expires_at > %s "
                "RETURNING access_digest, expires_at",
                (_token_digest(token), now),
            ).fetchone()
            if redeemed is None:
                return None
            session = db.execute(
                "DELETE FROM sessions WHERE token_digest = %s RETURNING user_id",
                (redeemed["access_digest"],),
            ).fetchone()
            if session is None:
                return None
            return self._issue_tokens(db, session["user_id"], now, redeemed["expires_at"])

    def current_user(self, token: str) -> User | None:
        with self.db.connect() as db:
            user = db.execute(
                "SELECT users.id, users.email FROM sessions JOIN users ON users.id = user_id "
                "WHERE token_digest = %s AND expires_at > %s",
                (_token_digest(token), int(time.time())),
            ).fetchone()
        return User(**user) if user else None

    def logout(self, token: str) -> None:
        with self.db.connect() as db:
            db.execute("DELETE FROM sessions WHERE token_digest = %s", (_token_digest(token),))
