import hashlib
import json
import secrets
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

from psycopg import Connection

from ..db import Database
from ..schemas.moves import ReviewDetails
from .building_register import count, default_ladder


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


@dataclass(frozen=True)
class Media:
    id: str
    move_id: str
    filename: str
    content_type: str
    size_bytes: int
    # Relative to the media directory, so the folder can be moved.
    storage_path: str
    created_at: int


@dataclass(frozen=True)
class PlaceRecord:
    road_address: str
    detail_address: str | None
    floor: int | None
    elevator: str
    elevator_source: str | None
    # The user's choice only (use / not_use); None means the default applies.
    ladder: str | None = None


@dataclass(frozen=True)
class ScheduleRecord:
    # ISO strings: YYYY-MM-DD and HH:MM:SS.
    move_date: str
    start_time: str


@dataclass(frozen=True)
class LookupRecord:
    """Building register result for one end of a move, kept so it is fetched only once."""

    # found / ambiguous / not_found / failed / not_requested
    status: str
    # The lot the lookup was for; null when the address had no search codes.
    adm_cd: str | None = None
    mountain: bool = False
    lot_main: int | None = None
    lot_sub: int | None = None
    # Every main-building row of the lot as JSON, or null when the fetch failed.
    rows_json: str | None = None
    # The chosen row as JSON when status is found.
    building_json: str | None = None
    checked_at: int | None = None


def place_ladder(record: PlaceRecord, lookup: LookupRecord | None) -> tuple[str, str]:
    """The ladder value to use and where it came from: the user, or the default rule."""
    if record.ladder is not None:
        return record.ladder, "user"
    building_floors = None
    if lookup is not None and lookup.building_json:
        building_floors = count(json.loads(lookup.building_json).get("grndFlrCnt"))
    return default_ladder(record.elevator, building_floors, record.floor), "default"


@dataclass(frozen=True)
class RouteRecord:
    """Road route between the two addresses, kept so it is computed only once."""

    # found / not_found (an address could not be located) / failed
    status: str
    # The two road addresses this result is for; a change to either means recompute.
    origin_address: str
    destination_address: str
    distance_m: int | None = None
    duration_s: int | None = None
    origin_x: float | None = None
    origin_y: float | None = None
    destination_x: float | None = None
    destination_y: float | None = None
    checked_at: int | None = None


@dataclass(frozen=True)
class MoveSummaryRecord:
    id: str
    created_at: int
    move_date: str | None
    start_time: str | None
    origin_address: str | None
    destination_address: str | None
    stage: str = "preparing"


@dataclass(frozen=True)
class DateInfoRecord:
    """What the move date falls on, judged when the date was saved."""

    move_date: str
    # confirmed / unpublished (the year's holidays are not out yet) / failed
    holiday_status: str
    # found / failed
    lunar_status: str
    # calendar.DayInfo as JSON
    info_json: str
    checked_at: int


@dataclass(frozen=True)
class CalendarEntry:
    """Shared KASI data: a year of holidays or a month of lunar dates."""

    status: str
    data_json: str
    fetched_at: int


MEDIA_COLUMNS = "id, move_id, filename, content_type, size_bytes, storage_path, created_at"


class MoveStore:
    """Moves live next to users so a member move can reference its account."""

    def __init__(self, db: Database, guest_ttl: int):
        self.db = db
        self.guest_ttl = guest_ttl

    def _move(self, row: dict) -> Move:
        expires_at = None if row["user_id"] else row["last_used_at"] + self.guest_ttl
        return Move(row["id"], row["user_id"], row["created_at"], expires_at)

    def create_for_member(self, user_id: int) -> Move:
        now = int(time.time())
        move_id = str(uuid.uuid4())
        with self.db.connect() as db:
            db.execute(
                "INSERT INTO moves(id, user_id, created_at, last_used_at) VALUES (%s, %s, %s, %s)",
                (move_id, user_id, now, now),
            )
        return Move(move_id, user_id, now, None)

    def create_for_guest(self) -> tuple[Move, str]:
        now = int(time.time())
        move_id = str(uuid.uuid4())
        guest_token = secrets.token_urlsafe(32)
        with self.db.connect() as db:
            db.execute(
                "INSERT INTO moves(id, guest_digest, created_at, last_used_at) "
                "VALUES (%s, %s, %s, %s)",
                (move_id, _digest(guest_token), now, now),
            )
        return Move(move_id, None, now, now + self.guest_ttl), guest_token

    def get_for_member(self, move_id: str, user_id: int) -> Move | None:
        with self.db.connect() as db:
            row = db.execute(
                "SELECT * FROM moves WHERE id = %s AND user_id = %s", (move_id, user_id)
            ).fetchone()
        return self._move(row) if row else None

    def replace_video(self, media: Media) -> Media | None:
        """Make media the move's only video and return the one it replaced.

        Delete and insert happen in one transaction, so a failure keeps the old
        video and a move never has zero or two videos in between.
        """
        with self.db.connect() as db:
            # Locking the move makes two uploads to it take turns.
            db.execute("SELECT 1 FROM moves WHERE id = %s FOR UPDATE", (media.move_id,))
            old = db.execute(
                f"DELETE FROM move_videos WHERE move_id = %s RETURNING {MEDIA_COLUMNS}",
                (media.move_id,),
            ).fetchone()
            db.execute(
                f"INSERT INTO move_videos({MEDIA_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (media.id, media.move_id, media.filename, media.content_type,
                 media.size_bytes, media.storage_path, media.created_at),
            )
        return Media(**dict(old)) if old else None

    def replace_after_video(self, media: Media) -> Media | None:
        """Store an after-move video without touching the initial room video."""
        with self.db.connect() as db:
            db.execute("SELECT 1 FROM moves WHERE id = %s FOR UPDATE", (media.move_id,))
            old = db.execute(
                f"DELETE FROM move_after_videos WHERE move_id = %s RETURNING {MEDIA_COLUMNS}",
                (media.move_id,),
            ).fetchone()
            db.execute(
                f"INSERT INTO move_after_videos({MEDIA_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (media.id, media.move_id, media.filename, media.content_type,
                 media.size_bytes, media.storage_path, media.created_at),
            )
        return Media(**dict(old)) if old else None

    def get_details(
        self, move_id: str, *, conn: Connection | None = None,
    ) -> tuple[dict[str, PlaceRecord], ScheduleRecord | None]:
        with self.db.connect() if conn is None else nullcontext(conn) as db:
            places = {
                row["role"]: PlaceRecord(
                    row["road_address"], row["detail_address"], row["floor"],
                    row["elevator"], row["elevator_source"], row["ladder"],
                )
                for row in db.execute(
                    "SELECT * FROM move_places WHERE move_id = %s", (move_id,)
                ).fetchall()
            }
            row = db.execute(
                "SELECT move_date, start_time FROM move_schedules WHERE move_id = %s", (move_id,)
            ).fetchone()
        return places, ScheduleRecord(row["move_date"], row["start_time"]) if row else None

    def get_review(self, move_id: str, *, conn: Connection | None = None) -> ReviewDetails | None:
        with self.db.connect() if conn is None else nullcontext(conn) as db:
            row = db.execute(
                "SELECT delivery_date, notes FROM move_review_details WHERE move_id = %s",
                (move_id,),
            ).fetchone()
        return ReviewDetails.model_validate(row) if row else None

    def get_lookups(
        self, move_id: str, *, conn: Connection | None = None,
    ) -> dict[str, LookupRecord]:
        with self.db.connect() if conn is None else nullcontext(conn) as db:
            rows = db.execute(
                "SELECT * FROM move_building_lookups WHERE move_id = %s", (move_id,)
            ).fetchall()
        return {
            row["role"]: LookupRecord(
                status=row["status"], adm_cd=row["adm_cd"], mountain=bool(row["mountain"]),
                lot_main=row["lot_main"], lot_sub=row["lot_sub"], rows_json=row["rows_json"],
                building_json=row["building_json"], checked_at=row["checked_at"],
            )
            for row in rows
        }

    def get_calendar(self, key: str) -> CalendarEntry | None:
        with self.db.connect() as db:
            row = db.execute(
                "SELECT status, data_json, fetched_at FROM calendar_cache WHERE key = %s", (key,)
            ).fetchone()
        return CalendarEntry(**dict(row)) if row else None

    def put_calendar(self, key: str, entry: CalendarEntry) -> None:
        with self.db.connect() as db:
            db.execute(
                "INSERT INTO calendar_cache(key, status, data_json, fetched_at) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT(key) DO UPDATE SET status = excluded.status, "
                "data_json = excluded.data_json, fetched_at = excluded.fetched_at",
                (key, entry.status, entry.data_json, entry.fetched_at),
            )

    def get_date_info(
        self, move_id: str, *, conn: Connection | None = None,
    ) -> DateInfoRecord | None:
        with self.db.connect() if conn is None else nullcontext(conn) as db:
            row = db.execute(
                "SELECT move_date, holiday_status, lunar_status, info_json, checked_at "
                "FROM move_date_info WHERE move_id = %s", (move_id,)
            ).fetchone()
        return DateInfoRecord(**dict(row)) if row else None

    def get_route(self, move_id: str, *, conn: Connection | None = None) -> RouteRecord | None:
        with self.db.connect() if conn is None else nullcontext(conn) as db:
            row = db.execute("SELECT * FROM move_routes WHERE move_id = %s", (move_id,)).fetchone()
        if row is None:
            return None
        fields = dict(row)
        fields.pop("move_id")
        return RouteRecord(**fields)

    def update_details(
        self,
        move_id: str,
        places: dict[str, PlaceRecord | None],
        schedule: ScheduleRecord | None,
        schedule_sent: bool,
        lookups: dict[str, LookupRecord | None] | None = None,
        route: RouteRecord | None = None,
        route_sent: bool = False,
        date_info: DateInfoRecord | None = None,
        date_info_sent: bool = False,
        review: ReviewDetails | None = None,
        review_sent: bool = False,
        *,
        conn: Connection | None = None,
    ) -> None:
        """Replace only the sections that were sent, all in one transaction.

        A place, schedule, lookup, route or date info given as None is cleared.
        """
        now = int(time.time())
        with self.db.connect() if conn is None else nullcontext(conn) as db:
            db.execute("SELECT 1 FROM moves WHERE id = %s FOR NO KEY UPDATE", (move_id,))
            if date_info_sent:
                db.execute("DELETE FROM move_date_info WHERE move_id = %s", (move_id,))
                if date_info is not None:
                    db.execute(
                        "INSERT INTO move_date_info(move_id, move_date, holiday_status, "
                        "lunar_status, info_json, checked_at) VALUES (%s, %s, %s, %s, %s, %s)",
                        (move_id, date_info.move_date, date_info.holiday_status,
                         date_info.lunar_status, date_info.info_json, date_info.checked_at),
                    )
            if route_sent:
                db.execute("DELETE FROM move_routes WHERE move_id = %s", (move_id,))
                if route is not None:
                    db.execute(
                        "INSERT INTO move_routes(move_id, status, origin_address, "
                        "destination_address, distance_m, duration_s, origin_x, origin_y, "
                        "destination_x, destination_y, checked_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (move_id, route.status, route.origin_address, route.destination_address,
                         route.distance_m, route.duration_s, route.origin_x, route.origin_y,
                         route.destination_x, route.destination_y, route.checked_at),
                    )
            for role, lookup in (lookups or {}).items():
                db.execute(
                    "DELETE FROM move_building_lookups WHERE move_id = %s AND role = %s",
                    (move_id, role),
                )
                if lookup is not None:
                    db.execute(
                        "INSERT INTO move_building_lookups(move_id, role, status, adm_cd, "
                        "mountain, lot_main, lot_sub, rows_json, building_json, checked_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (move_id, role, lookup.status, lookup.adm_cd, lookup.mountain,
                         lookup.lot_main, lookup.lot_sub, lookup.rows_json,
                         lookup.building_json, lookup.checked_at),
                    )
            for role, place in places.items():
                if place is None:
                    db.execute(
                        "DELETE FROM move_places WHERE move_id = %s AND role = %s", (move_id, role)
                    )
                    continue
                db.execute(
                    "INSERT INTO move_places(move_id, role, road_address, detail_address, floor, "
                    "elevator, elevator_source, ladder, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT(move_id, role) DO UPDATE SET "
                    "road_address = excluded.road_address, "
                    "detail_address = excluded.detail_address, floor = excluded.floor, "
                    "elevator = excluded.elevator, elevator_source = excluded.elevator_source, "
                    "ladder = excluded.ladder, updated_at = excluded.updated_at",
                    (move_id, role, place.road_address, place.detail_address, place.floor,
                     place.elevator, place.elevator_source, place.ladder, now),
                )
            if schedule_sent:
                if schedule is None:
                    db.execute("DELETE FROM move_schedules WHERE move_id = %s", (move_id,))
                else:
                    db.execute(
                        "INSERT INTO move_schedules(move_id, move_date, start_time, updated_at) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT(move_id) DO UPDATE SET "
                        "move_date = excluded.move_date, start_time = excluded.start_time, "
                        "updated_at = excluded.updated_at",
                        (move_id, schedule.move_date, schedule.start_time, now),
                    )
            if review_sent:
                if review is None:
                    db.execute("DELETE FROM move_review_details WHERE move_id = %s", (move_id,))
                else:
                    delivery = review.delivery_date.isoformat() if review.delivery_date else None
                    db.execute(
                        "INSERT INTO move_review_details(move_id, delivery_date, notes) "
                        "VALUES (%s, %s, %s) ON CONFLICT(move_id) DO UPDATE SET "
                        "delivery_date = excluded.delivery_date, notes = excluded.notes",
                        (move_id, delivery, review.notes),
                    )

    def get_video(self, move_id: str) -> Media | None:
        with self.db.connect() as db:
            row = db.execute(
                f"SELECT {MEDIA_COLUMNS} FROM move_videos WHERE move_id = %s", (move_id,)
            ).fetchone()
        return Media(**dict(row)) if row else None

    def claim(self, move_id: str, guest_token: str, user_id: int) -> Move | None:
        """Turn an unexpired guest move into the member's, keeping everything attached.

        Both proofs are needed: the guest token shows it is this person's move, the
        login shows whose account it goes to. Asking again once it is already that
        member's succeeds, so a double tap or refresh is harmless.
        """
        now = int(time.time())
        with self.db.connect() as db:
            row = db.execute(
                "UPDATE moves SET user_id = %s, guest_digest = NULL, last_used_at = %s "
                "WHERE id = %s AND guest_digest = %s AND last_used_at > %s RETURNING *",
                (user_id, now, move_id, _digest(guest_token), now - self.guest_ttl),
            ).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT * FROM moves WHERE id = %s AND user_id = %s", (move_id, user_id)
                ).fetchone()
        return self._move(row) if row else None

    def list_for_member(
        self, user_id: int, limit: int, offset: int
    ) -> tuple[list[MoveSummaryRecord], int]:
        """The member's moves, newest first, with what a history card shows."""
        with self.db.connect() as db:
            total = db.execute(
                "SELECT COUNT(*) AS total FROM moves WHERE user_id = %s", (user_id,)
            ).fetchone()["total"]
            # Stage: done once the after-move review is finished, comparing while an
            # after-move comparison of the current list exists, after_move once the list is
            # confirmed, preparing before that.
            rows = db.execute(
                "SELECT m.id, m.created_at, s.move_date, s.start_time, "
                "o.road_address AS origin_address, d.road_address AS destination_address, "
                "CASE WHEN c.reviewed_at IS NOT NULL THEN 'done' "
                "WHEN c.comparison_id IS NOT NULL THEN 'comparing' "
                "WHEN i.confirmed THEN 'after_move' ELSE 'preparing' END AS stage "
                "FROM moves AS m "
                "LEFT JOIN move_schedules AS s ON s.move_id = m.id "
                "LEFT JOIN move_places AS o ON o.move_id = m.id AND o.role = 'origin' "
                "LEFT JOIN move_places AS d ON d.move_id = m.id AND d.role = 'destination' "
                "LEFT JOIN LATERAL (SELECT analysis_id FROM move_analyses "
                "  WHERE move_id = m.id AND status != 'cancelled' "
                "  ORDER BY created_at DESC, seq DESC LIMIT 1) AS a ON TRUE "
                "LEFT JOIN move_inventories AS i ON i.analysis_id = a.analysis_id "
                "LEFT JOIN LATERAL (SELECT comparison_id, reviewed_at FROM move_comparisons "
                "  WHERE move_id = m.id AND analysis_id = a.analysis_id "
                "  AND status != 'cancelled' "
                "  ORDER BY created_at DESC, seq DESC LIMIT 1) AS c ON TRUE "
                "WHERE m.user_id = %s ORDER BY m.created_at DESC, m.seq DESC "
                "LIMIT %s OFFSET %s",
                (user_id, limit, offset),
            ).fetchall()
        return [MoveSummaryRecord(**dict(row)) for row in rows], total

    def get_for_guest(self, move_id: str, guest_token: str) -> Move | None:
        """Return an unexpired guest move and extend its inactivity deadline."""
        now = int(time.time())
        with self.db.connect() as db:
            row = db.execute(
                "UPDATE moves SET last_used_at = %s "
                "WHERE id = %s AND guest_digest = %s AND last_used_at > %s RETURNING *",
                (now, move_id, _digest(guest_token), now - self.guest_ttl),
            ).fetchone()
        return self._move(row) if row else None
