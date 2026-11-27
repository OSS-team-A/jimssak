"""Turn a measured video into the v2.0 analysis result the backend stores.

The format is defined in ``backend/app/schemas/analysis.py`` (``AnalysisResultV2``). One
measured track becomes one item. Sizes are expected to be calibrated and beds snapped
already (see ``ai.service.pipeline``); this module only reshapes and adds review reasons.

    python -m ai.service.result_v2 outputs/result-beds.json --analysis-id an_1 \
        --video-id room-01 --video outputs/room-01.mp4 --output outputs/room-01-v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.measurement.moving_contract import MeasurementResponse, VideoResult
from ai.measurement.refine import LABEL_KO, canonical_label, review_track
from ai.measurement.standard_sizes import (
    APPLIANCE_QUALIFIER,
    FURNITURE_QUALIFIER,
    NONSTANDARD,
    QUALIFIER,
    STANDARDS,
)

SCHEMA_VERSION = "2.0"
AXES = ("width", "depth", "height")
MAX_SNAPSHOTS = 3
# Used when the recognizer gives no confidence; marked with confidence_source.
DEFAULT_CONFIDENCE = 1.0
NONSTANDARD_REASON = "침대 크기가 일반 규격과 많이 달라 확인이 필요해요"
WARNINGS = {
    "approximate_phone_fov": "camera_intrinsics_estimated",
    "size_not_calibrated": "size_not_calibrated",
}


def snapshot_frames(frames: list[int], count: int = MAX_SNAPSHOTS) -> list[int]:
    """Up to ``count`` frames spread over the frames the track was measured in."""
    frames = sorted(set(frames))
    if len(frames) <= count:
        return frames
    step = (len(frames) - 1) / (count - 1)
    return sorted({frames[round(i * step)] for i in range(count)})


def _standard(track) -> dict | None:
    codes = [q.split(":", 1)[1] for q in track.qualifiers if q.startswith(f"{QUALIFIER}:")]
    if not codes:
        return None
    for region, standards in STANDARDS.items():
        for key, name, width, length in standards:
            if key == codes[0]:
                return {"code": key, "label_ko": name, "mattress_cm": [width, length],
                        "region": region}
    return {"code": codes[0]}


def item(track, index: int, *, fps: float | None, confidence: float | None) -> dict:
    label = canonical_label(track.label)
    review = review_track(track)
    reasons = list(review["review_reasons"])
    if NONSTANDARD in track.qualifiers and NONSTANDARD_REASON not in reasons:
        reasons.append(NONSTANDARD_REASON)
    dims = {axis: (round(d.cm, 1) if (d := track.dimensions.get(axis)) else None)
            for axis in AXES}
    snapshots = []
    if fps:
        snapshots = [{"time_ms": round(frame / fps * 1000), "image_url": None,
                      "frame_index": frame} for frame in snapshot_frames(track.frames)]
    body = {
        "item_id": f"item-{index}",
        "source": "ai",
        "label": label,
        "label_ko": LABEL_KO.get(label, track.label),
        "quantity": 1,
        "confidence": round(DEFAULT_CONFIDENCE if confidence is None else confidence, 3),
        "dimensions_cm": dims,
        "dimensions_status": track.status,
        "disassembly_required": None,
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "snapshots": snapshots,
        # Extra fields (the backend keeps them): for edits and debugging.
        "track_id": track.track_id,
        "review_codes": review["review_codes"]
        + (["bed_size_nonstandard"] if NONSTANDARD in track.qualifiers else []),
    }
    if confidence is None:
        body["confidence_source"] = "not_provided"
    standard = _standard(track)
    if standard:
        body["standard_size"] = standard
    if {APPLIANCE_QUALIFIER, FURNITURE_QUALIFIER} & set(track.qualifiers):
        body["size_adjustment"] = "catalog_prior"
    return body


def completed(result: VideoResult, *, analysis_id: str, video_id: str,
              fps: float | None = None, duration_s: float | None = None,
              confidences: dict[str, float] | None = None, capture_source: str = "camera",
              warnings: list[str] | None = None) -> dict:
    confidences = confidences or {}
    found = list(warnings or [])
    if result.intrinsics_source == "approximate_phone_fov":
        found.append(WARNINGS["approximate_phone_fov"])
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": "completed",
        "progress": 1.0,
        "video": {"video_id": video_id,
                  "duration_s": round(duration_s, 2) if duration_s else None,
                  "capture_source": capture_source},
        "items": [item(track, i, fps=fps, confidence=confidences.get(track.track_id))
                  for i, track in enumerate(result.objects, start=1)],
        "warnings": sorted(set(found)),
        "error": None,
    }


def failed(*, analysis_id: str, video_id: str, code: str, message: str, retryable: bool,
           duration_s: float | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "status": "failed",
        "progress": 1.0,
        "video": {"video_id": video_id, "duration_s": duration_s, "capture_source": "camera"},
        "items": [],
        "warnings": [],
        "error": {"code": code, "message": message, "retryable": retryable},
    }


def video_info(path: Path) -> tuple[float | None, float | None]:
    """Frames per second and duration in seconds, or None when the file is unreadable."""
    try:
        import cv2
    except ImportError:
        return None, None
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or None
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or None
    finally:
        capture.release()
    if not fps or fps <= 0:
        return None, None
    return fps, (frames / fps if frames else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result", type=Path, help="measurement result (one video)")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, help="the video, for snapshot times")
    parser.add_argument("--index", type=int, default=0, help="which video in the result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    response = MeasurementResponse.model_validate_json(args.result.read_text())
    fps, duration = video_info(args.video) if args.video else (None, None)
    body = completed(response.videos[args.index], analysis_id=args.analysis_id,
                     video_id=args.video_id, fps=fps, duration_s=duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
