"""After-move comparison jobs: started by the after-move video, finished by an analyzer.

Like the first analysis, a demo analyzer (ANALYSIS_DEMO=true) finishes jobs with a sample
result worked out from the time since upload, and stuck jobs time out.
"""

import json
import secrets
import time
from dataclasses import dataclass, replace

from psycopg.types.json import Jsonb

from ..db import Database
from .analysis import ACTIVE, DEMO_PHOTOS, StaleAnalysis, demo_photo_url

# Results that need the user's answer before the review can be finished.
NEEDS_CHECK = ("changed", "missing")


@dataclass(frozen=True)
class ComparisonTask:
    comparison_id: str
    move_id: str
    analysis_id: str
    before_media_id: str
    after_media_id: str
    inventory: list[dict]
    status: str  # queued / running / completed / failed / cancelled
    progress: float
    demo: bool
    demo_fail: bool
    result_json: str | None
    error_json: str | None
    checks: dict[str, str]
    reviewed_at: int | None
    created_at: int
    updated_at: int

    def results(self) -> dict[str, dict]:
        """The analyzer's result per item_id, once completed."""
        if not self.result_json:
            return {}
        return {item["item_id"]: item for item in json.loads(self.result_json)["items"]}


class ComparisonReviewed(Exception):
    """The user already finished this review."""


class ComparisonStore:
    def __init__(self, db: Database, timeout_seconds: int, demo_seconds: float):
        self.db = db
        self.timeout_seconds = timeout_seconds
        self.demo_seconds = demo_seconds
        # Replaceable so tests can move time forward.
        self.clock = time.time

    @staticmethod
    def _task(row: dict) -> ComparisonTask:
        fields = dict(row)
        fields.pop("seq", None)
        return ComparisonTask(**fields)

    def start(
        self, move_id: str, analysis_id: str, before_media_id: str, after_media_id: str,
        inventory: list[dict], demo: bool, demo_fail: bool,
    ) -> ComparisonTask:
        """A new after-move video replaces the move's earlier comparison and its answers."""
        now = int(self.clock())
        comparison_id = f"cmp_{secrets.token_hex(8)}"
        with self.db.connect() as db:
            db.execute(
                "UPDATE move_comparisons SET status = 'cancelled', updated_at = %s "
                "WHERE move_id = %s AND status != 'cancelled'", (now, move_id),
            )
            row = db.execute(
                "INSERT INTO move_comparisons(comparison_id, move_id, analysis_id, "
                "before_media_id, after_media_id, inventory, status, demo, demo_fail, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s) RETURNING *",
                (comparison_id, move_id, analysis_id, before_media_id, after_media_id,
                 Jsonb(inventory), demo, demo_fail, now, now),
            ).fetchone()
        return self._task(row)

    def current(self, move_id: str, analysis_id: str | None) -> ComparisonTask | None:
        """The comparison against the move's current list; one for an older list is ignored."""
        if analysis_id is None:
            return None
        with self.db.connect() as db:
            row = db.execute(
                "SELECT * FROM move_comparisons WHERE move_id = %s AND analysis_id = %s "
                "AND status != 'cancelled' ORDER BY created_at DESC, seq DESC LIMIT 1",
                (move_id, analysis_id),
            ).fetchone()
        return self._settle(self._task(row)) if row else None

    def get(self, comparison_id: str) -> ComparisonTask | None:
        with self.db.connect() as db:
            row = db.execute(
                "SELECT * FROM move_comparisons WHERE comparison_id = %s", (comparison_id,)
            ).fetchone()
        return self._settle(self._task(row)) if row else None

    # --- the analyzer's side ---

    def claim_next(self) -> ComparisonTask | None:
        """Hand the oldest waiting (non-demo) comparison to an analyzer and mark it running."""
        now = int(self.clock())
        with self.db.connect() as db:
            row = db.execute(
                "UPDATE move_comparisons SET status = 'running', updated_at = %s "
                "WHERE comparison_id = (SELECT comparison_id FROM move_comparisons "
                "WHERE status = 'queued' AND NOT demo AND created_at > %s "
                "ORDER BY created_at, seq LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING *",
                (now, now - self.timeout_seconds),
            ).fetchone()
        return self._task(row) if row else None

    def active(self, comparison_id: str) -> ComparisonTask:
        """The comparison, if an analyzer may still report on it; otherwise StaleAnalysis."""
        task = self.get(comparison_id)
        if task is None or task.status not in ACTIVE or task.demo:
            raise StaleAnalysis(comparison_id)
        return task

    def report_progress(self, comparison_id: str, progress: float) -> ComparisonTask:
        task = self.active(comparison_id)
        now = int(self.clock())
        with self.db.connect() as db:
            db.execute(
                "UPDATE move_comparisons SET progress = %s, updated_at = %s "
                "WHERE comparison_id = %s", (progress, now, comparison_id),
            )
        return replace(task, progress=progress, updated_at=now)

    def submit(self, comparison_id: str, status: str, result_json: str | None,
               error_json: str | None) -> ComparisonTask:
        task = self.active(comparison_id)
        return self._finish(task, status, result_json, error_json, int(self.clock()))

    # --- the user's review ---

    def set_check(self, comparison_id: str, item_id: str, answer: str) -> None:
        """Save the user's answer for one item; answers can change until the review is done."""
        now = int(self.clock())
        with self.db.connect() as db:
            row = db.execute(
                "SELECT reviewed_at FROM move_comparisons WHERE comparison_id = %s FOR UPDATE",
                (comparison_id,),
            ).fetchone()
            if row["reviewed_at"] is not None:
                raise ComparisonReviewed(comparison_id)
            db.execute(
                "UPDATE move_comparisons SET checks = checks || %s, updated_at = %s "
                "WHERE comparison_id = %s", (Jsonb({item_id: answer}), now, comparison_id),
            )

    def finish_review(self, comparison_id: str, needs_check: list[str]) -> bool:
        """Mark the review done if every item needing a check has an answer.

        Returns False when an answer is missing. Finishing twice is harmless.
        """
        now = int(self.clock())
        with self.db.connect() as db:
            row = db.execute(
                "SELECT checks, reviewed_at FROM move_comparisons "
                "WHERE comparison_id = %s FOR UPDATE", (comparison_id,),
            ).fetchone()
            if row["reviewed_at"] is not None:
                return True
            if any(item_id not in row["checks"] for item_id in needs_check):
                return False
            db.execute(
                "UPDATE move_comparisons SET reviewed_at = %s, updated_at = %s "
                "WHERE comparison_id = %s", (now, now, comparison_id),
            )
        return True

    def _settle(self, task: ComparisonTask) -> ComparisonTask:
        if task.status not in ACTIVE:
            return task
        now = int(self.clock())
        if task.demo:
            return self._advance_demo(task, now)
        if now - task.created_at >= self.timeout_seconds:
            error = {"code": "timeout", "retryable": True,
                     "message": "비교가 오래 걸려 중단됐어요. 이사 후 영상을 다시 올려 주세요."}
            return self._finish(task, "failed", None, json.dumps(error, ensure_ascii=False), now)
        return task

    def _advance_demo(self, task: ComparisonTask, now: int) -> ComparisonTask:
        elapsed = now - task.created_at
        if elapsed < self.demo_seconds:
            if elapsed < 1:
                return task
            return replace(task, status="running",
                           progress=round(min(0.95, elapsed / self.demo_seconds), 2))
        if task.demo_fail:
            error = {"code": "video_too_dark", "retryable": True,
                     "message": "영상이 너무 어두워 물품을 비교하지 못했어요. "
                                "밝은 곳에서 다시 찍어 주세요."}
            return self._finish(task, "failed", None, json.dumps(error, ensure_ascii=False), now)
        result = demo_comparison(task)
        return self._finish(task, "completed", json.dumps(result, ensure_ascii=False), None, now)

    def _finish(self, task: ComparisonTask, status: str, result_json: str | None,
                error_json: str | None, now: int) -> ComparisonTask:
        with self.db.connect() as db:
            db.execute(
                "UPDATE move_comparisons SET status = %s, progress = 1, result_json = %s, "
                "error_json = %s, updated_at = %s WHERE comparison_id = %s "
                "AND status IN ('queued', 'running')",
                (status, result_json, error_json, now, task.comparison_id),
            )
        return replace(task, status=status, progress=1.0, result_json=result_json,
                       error_json=error_json, updated_at=now)


# --- the demo analyzer's sample: one item looks changed, one is missing, the rest are fine ---

def demo_comparison(task: ComparisonTask) -> dict:
    photos = list(DEMO_PHOTOS)
    items = []
    for index, item in enumerate(task.inventory):
        before = [{"time_ms": shot["time_ms"], "image_url": shot.get("image_url")}
                  for shot in item.get("snapshots", [])]
        photo = demo_photo_url(task.move_id, photos[index % len(photos)])
        if index == 0:
            result, reasons = "changed", ["모서리에 긁힌 자국이 생긴 것 같아요"]
            after = [{"time_ms": 5200, "image_url": photo}]
        elif index == 1:
            result, reasons, after = "missing", ["이사 후 영상에서 찾지 못했어요"], []
        else:
            result, reasons = "ok", []
            after = [{"time_ms": 3000 + 1500 * index, "image_url": photo}]
        items.append({"item_id": item["item_id"], "result": result, "reasons": reasons,
                      "before_snapshots": before, "after_snapshots": after})
    return {
        "schema_version": "1.0", "comparison_id": task.comparison_id, "status": "completed",
        "progress": 1.0, "video": {"video_id": task.after_media_id}, "items": items,
        "warnings": [], "error": None,
    }

