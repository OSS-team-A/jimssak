import hashlib
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


@dataclass(frozen=True)
class Move:
    id: str
    user_id: int | None
    created_at: int
    # Guest moves only: access ends this many seconds after the last use.
    expires_at: int | None

    @property
    def is_guest(self) -> bool:
        return self.user_id is None


class MoveStore:
    """Moves live next to users so a member move can reference its account."""

    def __init__(self, path: Path, guest_ttl: int):
        self.path = path
        self.guest_ttl = guest_ttl

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
                CREATE TABLE IF NOT EXISTS moves (
                    id TEXT PRIMARY KEY NOT NULL,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    guest_digest BLOB UNIQUE,
                    created_at INTEGER NOT NULL,
                    last_used_at INTEGER NOT NULL,
                    CHECK ((user_id IS NULL) <> (guest_digest IS NULL))
                );
                CREATE INDEX IF NOT EXISTS moves_user ON moves(user_id);
            """)

    def _move(self, row: sqlite3.Row) -> Move:
        expires_at = None if row["user_id"] else row["last_used_at"] + self.guest_ttl
        return Move(row["id"], row["user_id"], row["created_at"], expires_at)

    def create_for_member(self, user_id: int) -> Move:
        now = int(time.time())
        move_id = secrets.token_urlsafe(12)
        with self._connect() as db:
            db.execute(
                "INSERT INTO moves(id, user_id, created_at, last_used_at) VALUES (?, ?, ?, ?)",
                (move_id, user_id, now, now),
            )
        return Move(move_id, user_id, now, None)

    def create_for_guest(self) -> tuple[Move, str]:
        now = int(time.time())
        move_id = secrets.token_urlsafe(12)
        guest_token = secrets.token_urlsafe(32)
        with self._connect() as db:
            db.execute(
                "INSERT INTO moves(id, guest_digest, created_at, last_used_at) VALUES (?, ?, ?, ?)",
                (move_id, _digest(guest_token), now, now),
            )
        return Move(move_id, None, now, now + self.guest_ttl), guest_token

    def get_for_member(self, move_id: str, user_id: int) -> Move | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM moves WHERE id = ? AND user_id = ?", (move_id, user_id)
            ).fetchone()
        return self._move(row) if row else None

    def get_for_guest(self, move_id: str, guest_token: str) -> Move | None:
        """Return an unexpired guest move and extend its inactivity deadline."""
        now = int(time.time())
        with self._connect() as db:
            row = db.execute(
                "UPDATE moves SET last_used_at = ? "
                "WHERE id = ? AND guest_digest = ? AND last_used_at > ? RETURNING *",
                (now, move_id, _digest(guest_token), now - self.guest_ttl),
            ).fetchone()
        return self._move(row) if row else None
