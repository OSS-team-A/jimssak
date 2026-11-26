"""PostgreSQL connection pool and the schema every store shares.

Each `connect()` block is one transaction: it commits when the block ends and rolls back
if it raises. Rows come back as dicts.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# Tables are created in order, since later ones reference earlier ones. Times are Unix
# seconds. New features add tables instead of changing old ones, so no migrations yet.
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_digest BYTEA PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token_digest BYTEA PRIMARY KEY,
    access_digest BYTEA NOT NULL UNIQUE REFERENCES sessions(token_digest) ON DELETE CASCADE,
    expires_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS moves (
    id TEXT PRIMARY KEY,
    -- Creation order, to sort moves made in the same second.
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    guest_digest BYTEA UNIQUE,
    created_at BIGINT NOT NULL,
    last_used_at BIGINT NOT NULL,
    CHECK ((user_id IS NULL) <> (guest_digest IS NULL))
);
CREATE INDEX IF NOT EXISTS moves_user ON moves(user_id);
-- Small one-room moves: one room video per move.
CREATE TABLE IF NOT EXISTS move_videos (
    id TEXT PRIMARY KEY,
    move_id TEXT NOT NULL UNIQUE REFERENCES moves(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    storage_path TEXT NOT NULL,
    created_at BIGINT NOT NULL
);
-- Keep the post-move recording separate from the room video used for the initial inventory.
CREATE TABLE IF NOT EXISTS move_after_videos (
    id TEXT PRIMARY KEY,
    move_id TEXT NOT NULL UNIQUE REFERENCES moves(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    storage_path TEXT NOT NULL,
    created_at BIGINT NOT NULL
);
-- SC-06: origin and destination are kept apart, since one end can have stairs and the
-- other an elevator.
CREATE TABLE IF NOT EXISTS move_places (
    move_id TEXT NOT NULL REFERENCES moves(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('origin', 'destination')),
    road_address TEXT NOT NULL,
    detail_address TEXT,
    floor INTEGER,
    elevator TEXT NOT NULL,
    elevator_source TEXT,
    updated_at BIGINT NOT NULL,
    PRIMARY KEY (move_id, role)
);
CREATE TABLE IF NOT EXISTS move_building_lookups (
    move_id TEXT NOT NULL REFERENCES moves(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('origin', 'destination')),
    status TEXT NOT NULL,
    adm_cd TEXT,
    mountain BOOLEAN NOT NULL DEFAULT FALSE,
    lot_main INTEGER,
    lot_sub INTEGER,
    rows_json TEXT,
    building_json TEXT,
    checked_at BIGINT,
    PRIMARY KEY (move_id, role)
);
CREATE TABLE IF NOT EXISTS move_routes (
    move_id TEXT PRIMARY KEY REFERENCES moves(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    origin_address TEXT NOT NULL,
    destination_address TEXT NOT NULL,
    distance_m INTEGER,
    duration_s INTEGER,
    origin_x DOUBLE PRECISION,
    origin_y DOUBLE PRECISION,
    destination_x DOUBLE PRECISION,
    destination_y DOUBLE PRECISION,
    checked_at BIGINT
);
CREATE TABLE IF NOT EXISTS move_date_info (
    move_id TEXT PRIMARY KEY REFERENCES moves(id) ON DELETE CASCADE,
    move_date TEXT NOT NULL,
    holiday_status TEXT NOT NULL,
    lunar_status TEXT NOT NULL,
    info_json TEXT NOT NULL,
    checked_at BIGINT NOT NULL
);
-- Shared by every move: 'holidays:2026', 'lunar:2026-11'.
CREATE TABLE IF NOT EXISTS calendar_cache (
    key TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    data_json TEXT NOT NULL,
    fetched_at BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS move_schedules (
    move_id TEXT PRIMARY KEY REFERENCES moves(id) ON DELETE CASCADE,
    move_date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    updated_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS move_analyses (
    analysis_id TEXT PRIMARY KEY,
    -- Creation order, to pick the newest of jobs started in the same second.
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    move_id TEXT NOT NULL REFERENCES moves(id) ON DELETE CASCADE,
    media_id TEXT NOT NULL,
    status TEXT NOT NULL,
    progress DOUBLE PRECISION NOT NULL DEFAULT 0,
    demo BOOLEAN NOT NULL DEFAULT FALSE,
    demo_fail BOOLEAN NOT NULL DEFAULT FALSE,
    result_json TEXT,
    error_json TEXT,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS move_analyses_move ON move_analyses(move_id);
CREATE INDEX IF NOT EXISTS move_analyses_queue ON move_analyses(status, created_at);
"""


class Database:
    def __init__(self, url: str, max_size: int = 10):
        self.pool = ConnectionPool(
            url, min_size=1, max_size=max_size, open=False,
            kwargs={"row_factory": dict_row},
        )

    def open(self) -> None:
        self.pool.open(wait=True)

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def connect(self) -> Iterator[psycopg.Connection]:
        with self.pool.connection() as db:
            yield db

    def initialize(self) -> None:
        with self.connect() as db:
            # Several API processes starting at once must not create tables twice.
            db.execute("SELECT pg_advisory_xact_lock(4247)")
            db.execute(SCHEMA)
