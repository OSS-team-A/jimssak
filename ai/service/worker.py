"""AI analyzer worker: take jobs from the backend, analyze the video, send the result.

Uses the backend's internal API (``backend/app/api/analysis_worker.py``):
claim → download video → progress → result (v2.0). Run it next to the backend::

    ANALYSIS_WORKER_KEY=... python -m ai.service.worker --server http://localhost:8000 \
        --recognizer my_package.recognition:recognize

Without a recognition stage, ``--prepared-job job.json`` reuses the observations of a
measurement job (for testing with a known video). ``--once`` handles at most one job.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ai.service import result_v2
from ai.service.pipeline import (
    SERVICE_DEPTH_MODEL,
    NoFurniture,
    Recognizer,
    analyze,
    load_recognizer,
    prepared_job,
)

API = "/api/v1/internal/analyses"
log = logging.getLogger("ai.worker")

FAILURES = {
    "no_furniture_found": ("가구를 찾지 못했어요. 가구 전체가 보이도록 천천히 다시 촬영해 주세요",
                           True),
    "recognition_failed": ("영상에서 가구를 인식하지 못했어요. 다시 촬영해 주세요", True),
    "measurement_failed": ("가구 크기를 재지 못했어요. 가구 주위를 천천히 돌며 다시 촬영해 주세요",
                           True),
}


class Stale(Exception):
    """The backend no longer expects this job (video replaced, timed out or finished)."""


class Backend:
    """Thin client over the internal API. ``client`` is any ``httpx.Client``."""

    def __init__(self, client: httpx.Client, key: str):
        self.client = client
        self.headers = {"X-Worker-Key": key}

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code == 409:
            raise Stale(response.text)
        response.raise_for_status()
        return response

    def claim(self) -> dict | None:
        response = self._check(self.client.post(f"{API}/claim", headers=self.headers))
        return None if response.status_code == 204 else response.json()

    def download(self, job: dict, folder: Path) -> Path:
        suffix = ".mov" if job.get("content_type") == "video/quicktime" else ".mp4"
        path = folder / f"{job['video_id']}{suffix}"
        with self.client.stream("GET", job["video_url"], headers=self.headers) as response:
            self._check(response)
            with path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return path

    def progress(self, analysis_id: str, value: float) -> None:
        self._check(self.client.post(f"{API}/{analysis_id}/progress", headers=self.headers,
                                     json={"progress": round(value, 3)}))

    def submit(self, analysis_id: str, body: dict) -> dict:
        return self._check(self.client.post(f"{API}/{analysis_id}/result",
                                            headers=self.headers, json=body)).json()


def default_measure(device: str = "auto") -> Callable:
    """``estimate_video`` with one Depth Anything V2 model, loaded on first use."""
    state = {}

    def measure(clip):
        from ai.measurement.metric_depth import MetricDepth
        from ai.measurement.moving_video import estimate_video
        if "model" not in state:
            state["model"] = MetricDepth(SERVICE_DEPTH_MODEL, device=device)
        return estimate_video(clip, state["model"])

    return measure


class Worker:
    def __init__(self, backend: Backend, recognizer: Recognizer, measure: Callable, *,
                 region: str = "kr", work_dir: Path | None = None):
        self.backend = backend
        self.recognizer = recognizer
        self.measure = measure
        self.region = region
        self.work_dir = work_dir

    def _fail(self, job: dict, code: str) -> dict:
        message, retryable = FAILURES[code]
        return result_v2.failed(analysis_id=job["analysis_id"], video_id=job["video_id"],
                                code=code, message=message, retryable=retryable)

    def handle(self, job: dict) -> dict:
        """Analyze one claimed job and return the result that was sent."""
        analysis_id = job["analysis_id"]
        with tempfile.TemporaryDirectory(dir=self.work_dir) as folder:
            path = self.backend.download(job, Path(folder))
            self.backend.progress(analysis_id, 0.05)
            try:
                recognition = self.recognizer(path)
            except NoFurniture:
                body = self._fail(job, "no_furniture_found")
            except Exception:
                log.exception("recognition failed for %s", analysis_id)
                body = self._fail(job, "recognition_failed")
            else:
                self.backend.progress(analysis_id, 0.2)
                try:
                    body = analyze(path, recognition, self.measure, analysis_id=analysis_id,
                                   video_id=job["video_id"], region=self.region,
                                   progress=lambda v: self.backend.progress(analysis_id, v))
                except NoFurniture:
                    body = self._fail(job, "no_furniture_found")
                except Stale:
                    raise
                except Exception:
                    log.exception("measurement failed for %s", analysis_id)
                    body = self._fail(job, "measurement_failed")
        self.backend.submit(analysis_id, body)
        log.info("%s: %s, %d items", analysis_id, body["status"], len(body["items"]))
        return body

    def run_once(self) -> dict | None:
        job = self.backend.claim()
        if job is None:
            return None
        try:
            return self.handle(job)
        except Stale:
            log.warning("%s is no longer current; result dropped", job["analysis_id"])
            return None

    def run_forever(self, poll_s: float = 5.0) -> None:
        while True:
            try:
                if self.run_once() is None:
                    time.sleep(poll_s)
            except httpx.HTTPError as exc:
                log.warning("backend unreachable: %s", exc)
                time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--key", default=os.environ.get("ANALYSIS_WORKER_KEY"),
                        help="defaults to $ANALYSIS_WORKER_KEY")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recognizer", help="package.module:function")
    source.add_argument("--prepared-job", type=Path, help="measurement job file to replay")
    parser.add_argument("--region", choices=("kr", "us"), default="kr")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between checks")
    parser.add_argument("--once", action="store_true", help="handle at most one job")
    args = parser.parse_args(argv)
    if not args.key:
        parser.error("--key or ANALYSIS_WORKER_KEY is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    recognizer = (load_recognizer(args.recognizer) if args.recognizer
                  else prepared_job(args.prepared_job))
    with httpx.Client(base_url=args.server, timeout=60) as client:
        worker = Worker(Backend(client, args.key), recognizer, default_measure(args.device),
                        region=args.region)
        if args.once:
            body = worker.run_once()
            log.info("no job waiting" if body is None else f"sent {body['status']}")
        else:
            worker.run_forever(args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
