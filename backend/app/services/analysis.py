"""Video analysis jobs: created on upload, taken by an analyzer, finished with a v2.0 result.

Until the AI analyzer exists, a demo analyzer (ANALYSIS_DEMO=true) finishes jobs with a
fixed sample so the front end can build its screens. Demo progress is worked out from the
time since upload whenever the job is read, so no background thread is needed.
"""

import json
import secrets
import time
from dataclasses import dataclass, replace

from ..db import Database

# Statuses kept in the table; "cancelled" marks a job whose video was replaced.
ACTIVE = ("queued", "running")


@dataclass(frozen=True)
class AnalysisJob:
    analysis_id: str
    move_id: str
    media_id: str
    status: str  # queued / running / completed / failed / cancelled
    progress: float
    demo: bool
    demo_fail: bool
    result_json: str | None
    error_json: str | None
    created_at: int
    updated_at: int


class StaleAnalysis(Exception):
    """The job is not the move's current one, or is already finished."""


class AnalysisStore:
    def __init__(self, db: Database, timeout_seconds: int, demo_seconds: float):
        self.db = db
        self.timeout_seconds = timeout_seconds
        self.demo_seconds = demo_seconds
        # Replaceable so tests can move time forward.
        self.clock = time.time

    @staticmethod
    def _job(row: dict) -> AnalysisJob:
        fields = dict(row)
        fields.pop("seq", None)
        return AnalysisJob(**fields)

    def start(self, move_id: str, media_id: str, demo: bool, demo_fail: bool) -> AnalysisJob:
        """A new video replaces the move's earlier job; its late results are refused."""
        now = int(self.clock())
        job = AnalysisJob(
            analysis_id=f"an_{secrets.token_hex(8)}", move_id=move_id, media_id=media_id,
            status="queued", progress=0.0, demo=demo, demo_fail=demo_fail,
            result_json=None, error_json=None, created_at=now, updated_at=now,
        )
        with self.db.connect() as db:
            db.execute(
                "UPDATE move_analyses SET status = 'cancelled', updated_at = %s "
                "WHERE move_id = %s AND status IN ('queued', 'running')", (now, move_id),
            )
            db.execute(
                "INSERT INTO move_analyses(analysis_id, move_id, media_id, status, progress, "
                "demo, demo_fail, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, 0, %s, %s, %s, %s)",
                (job.analysis_id, move_id, media_id, job.status, demo, demo_fail,
                 now, now),
            )
        return job

    def current(self, move_id: str) -> AnalysisJob | None:
        with self.db.connect() as db:
            row = db.execute(
                "SELECT * FROM move_analyses WHERE move_id = %s AND status != 'cancelled' "
                "ORDER BY created_at DESC, seq DESC LIMIT 1", (move_id,),
            ).fetchone()
        if row is None:
            return None
        return self._settle(self._job(row))

    def get(self, analysis_id: str) -> AnalysisJob | None:
        with self.db.connect() as db:
            row = db.execute(
                "SELECT * FROM move_analyses WHERE analysis_id = %s", (analysis_id,)
            ).fetchone()
        return self._settle(self._job(row)) if row else None

    def _settle(self, job: AnalysisJob) -> AnalysisJob:
        """Bring a job up to date: advance a demo job, or time out a stuck one."""
        if job.status not in ACTIVE:
            return job
        now = int(self.clock())
        if job.demo:
            return self._advance_demo(job, now)
        if now - job.created_at >= self.timeout_seconds:
            error = {"code": "timeout", "retryable": True,
                     "message": "분석이 오래 걸려 중단됐어요. 영상을 다시 올려 주세요."}
            return self._finish(job, "failed", None, json.dumps(error, ensure_ascii=False), now)
        return job

    def _advance_demo(self, job: AnalysisJob, now: int) -> AnalysisJob:
        elapsed = now - job.created_at
        if elapsed < self.demo_seconds:
            # A moment in the queue, then running with steadily rising progress.
            if elapsed < 1:
                return job
            return replace(job, status="running",
                           progress=round(min(0.95, elapsed / self.demo_seconds), 2))
        if job.demo_fail:
            error = {"code": "video_too_dark", "retryable": True,
                     "message": "영상이 너무 어두워 물품을 찾지 못했어요. "
                                "밝은 곳에서 다시 찍어 주세요."}
            return self._finish(job, "failed", None, json.dumps(error, ensure_ascii=False), now)
        result = demo_result(job)
        return self._finish(job, "completed", json.dumps(result, ensure_ascii=False), None, now)

    def _finish(self, job: AnalysisJob, status: str, result_json: str | None,
                error_json: str | None, now: int) -> AnalysisJob:
        with self.db.connect() as db:
            updated = db.execute(
                "UPDATE move_analyses SET status = %s, progress = 1, result_json = %s, "
                "error_json = %s, updated_at = %s WHERE analysis_id = %s "
                "AND status IN ('queued', 'running')",
                (status, result_json, error_json, now, job.analysis_id),
            )
            if updated.rowcount and status == "completed":
                db.execute(
                    "INSERT INTO move_inventories (analysis_id, items) "
                    "VALUES (%s, %s::jsonb -> 'items')", (job.analysis_id, result_json),
                )
        return replace(job, status=status, progress=1.0, result_json=result_json,
                       error_json=error_json, updated_at=now)

    # --- the analyzer's side ---

    def claim_next(self) -> AnalysisJob | None:
        """Hand the oldest waiting (non-demo) job to an analyzer and mark it running."""
        now = int(self.clock())
        with self.db.connect() as db:
            # SKIP LOCKED: two analyzers asking at once get different jobs.
            row = db.execute(
                "UPDATE move_analyses SET status = 'running', updated_at = %s "
                "WHERE analysis_id = (SELECT analysis_id FROM move_analyses "
                "WHERE status = 'queued' AND NOT demo AND created_at > %s "
                "ORDER BY created_at, seq LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING *",
                (now, now - self.timeout_seconds),
            ).fetchone()
        return self._job(row) if row else None

    def report_progress(self, analysis_id: str, progress: float) -> AnalysisJob:
        job = self._active(analysis_id)
        now = int(self.clock())
        with self.db.connect() as db:
            db.execute(
                "UPDATE move_analyses SET progress = %s, updated_at = %s WHERE analysis_id = %s",
                (progress, now, analysis_id),
            )
        return replace(job, progress=progress, updated_at=now)

    def submit(self, analysis_id: str, status: str, result_json: str | None,
               error_json: str | None) -> AnalysisJob:
        job = self._active(analysis_id)
        return self._finish(job, status, result_json, error_json, int(self.clock()))

    def _active(self, analysis_id: str) -> AnalysisJob:
        job = self.get(analysis_id)
        if job is None or job.status not in ACTIVE or job.demo:
            raise StaleAnalysis(analysis_id)
        return job


# --- the demo analyzer's fixed result: the Figma SC-07 example room ---

DEMO_PHOTOS = {"chair": "chair.jpg", "desk": "desk.jpg"}


def demo_photo_url(move_id: str, photo: str) -> str:
    return f"/api/v1/moves/{move_id}/analysis/photos/{DEMO_PHOTOS[photo]}"


def demo_result(job: AnalysisJob) -> dict:
    """The v2.0 example agreed with the AI side, with this job's IDs and sample photos."""
    def photos(*times: int, photo: str) -> list[dict]:
        return [{"time_ms": ms, "image_url": demo_photo_url(job.move_id, photo)} for ms in times]

    return {
        "schema_version": "2.0",
        "analysis_id": job.analysis_id,
        "status": "completed",
        "progress": 1.0,
        "video": {"video_id": job.media_id, "duration_s": 42.5, "capture_source": "camera"},
        "items": [
            {
                "item_id": "item-1", "source": "demo", "label": "desk", "label_ko": "책상",
                "quantity": 1, "confidence": 0.91,
                "dimensions_cm": {"width": 120.3, "depth": 60.1, "height": 73.5},
                "dimensions_status": "estimated", "disassembly_required": None,
                "needs_review": False, "review_reasons": [],
                "snapshots": photos(4200, photo="desk"),
            },
            {
                "item_id": "item-2", "source": "demo", "label": "chair", "label_ko": "의자",
                "quantity": 1, "confidence": 0.84,
                "dimensions_cm": {"width": 141.0, "depth": None, "height": 185.4},
                "dimensions_status": "partial", "disassembly_required": None,
                "needs_review": True,
                "review_reasons": ["옆면이 촬영되지 않아 깊이를 구하지 못했어요"],
                "snapshots": photos(9800, photo="chair"),
            },
        ],
        "warnings": [],
        "error": None,
    }

