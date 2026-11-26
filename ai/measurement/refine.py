"""Combine depth models per dimension and flag sizes that need a user check.

Runs after ``ai.measurement.moving_video`` and needs no model weights::

    python -m ai.measurement.refine fuse \
        --result depth-pro=outputs/dp.json --result depth-anything-v2=outputs/dav2.json \
        --output outputs/fused.json
    python -m ai.measurement.refine review outputs/fused.json --output outputs/review.json
    python -m ai.measurement.refine ranges outputs/arkitscenes-dev-objects.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from ai.measurement.moving_contract import (
    DimensionEstimate,
    MeasuredTrack,
    MeasurementResponse,
)

AXES = ("width", "depth", "height")
AXIS_KO = {"width": "가로", "depth": "깊이", "height": "높이"}

# Chosen on the six ARKitScenes tuning scenes with capture poses. Depth Pro had the
# lowest width error in every leave-one-scene-out fold; depth and height stay with
# Depth Anything V2. Videos without capture poses keep the automatic single model
# until a development-set comparison supports a policy for them.
FUSION_POLICY: dict[str, dict[str, str]] = {
    "capture_poses": {
        "width": "depth-pro",
        "depth": "depth-anything-v2",
        "height": "depth-anything-v2",
    },
}
RGB_PRIMARY = "depth-pro"
DISAGREEMENT_RATIO = 0.25

# Gravity-aligned sizes in cm: width is the longer floor side, depth the shorter.
# Values outside the range are flagged, never changed. ARKitScenes labels come from
# ``python -m ai.measurement.refine ranges`` on development-side 3DOD boxes (361 scans,
# 3,420 objects; 1st-99th percentile widened by 20%), which flags 1.0% of true sizes.
# desk, wardrobe and drawer are not ARKitScenes classes, so their ranges are manual.
SIZE_RANGES_CM: dict[str, dict[str, tuple[float, float]]] = {
    "bed": {"width": (135, 292), "depth": (69, 236), "height": (28, 184)},
    "cabinet": {"width": (26, 334), "depth": (16, 92), "height": (24, 299)},
    "chair": {"width": (28, 107), "depth": (24, 100), "height": (42, 147)},
    "refrigerator": {"width": (44, 123), "depth": (36, 104), "height": (90, 243)},
    "shelf": {"width": (22, 239), "depth": (13, 69), "height": (28, 306)},
    "sofa": {"width": (51, 357), "depth": (35, 150), "height": (35, 128)},
    "stool": {"width": (15, 198), "depth": (14, 85), "height": (13, 87)},
    "table": {"width": (23, 301), "depth": (16, 156), "height": (28, 113)},
    "washer": {"width": (43, 96), "depth": (30, 83), "height": (60, 113)},
    "tv_monitor": {"width": (30, 230), "depth": (3, 60), "height": (20, 150)},
    "desk": {"width": (50, 240), "depth": (35, 110), "height": (55, 110)},
    "wardrobe": {"width": (40, 320), "depth": (35, 90), "height": (120, 270)},
    "drawer": {"width": (25, 200), "depth": (25, 80), "height": (25, 170)},
}
RANGE_MIN_SAMPLES = 20
LABEL_ALIASES = {
    "couch": "sofa", "dining table": "table", "tv": "tv_monitor", "bookshelf": "shelf",
    "bookcase": "shelf", "dresser": "drawer", "closet": "wardrobe",
    "washing machine": "washer", "fridge": "refrigerator",
}
LABEL_KO = {
    "bed": "침대", "sofa": "소파", "chair": "의자", "stool": "스툴", "table": "테이블",
    "desk": "책상", "cabinet": "수납장", "shelf": "선반", "wardrobe": "옷장",
    "drawer": "서랍장", "refrigerator": "냉장고", "washer": "세탁기", "tv_monitor": "TV",
}
MISSING_REASON = {
    "width": "양 끝이 함께 보이지 않아 가로를 구하지 못했어요",
    "depth": "옆면이 충분히 촬영되지 않아 깊이를 구하지 못했어요",
    "height": "바닥과 윗면이 함께 보이지 않아 높이를 구하지 못했어요",
}


# Checked in order, so "bookshelf" maps to shelf and "nightstand" to cabinet.
LABEL_KEYWORDS = (
    ("wardrobe", "wardrobe"), ("closet", "wardrobe"), ("refrigerator", "refrigerator"),
    ("fridge", "refrigerator"), ("washer", "washer"), ("bookshelf", "shelf"),
    ("bookcase", "shelf"), ("shelf", "shelf"), ("nightstand", "cabinet"),
    ("dresser", "drawer"), ("drawer", "drawer"), ("cabinet", "cabinet"), ("desk", "desk"),
    ("table", "table"), ("sofa", "sofa"), ("couch", "sofa"), ("stool", "stool"),
    ("chair", "chair"), ("bed", "bed"),
)


def canonical_label(label: str) -> str:
    name = label.strip().lower().replace("-", " ").replace("_", " ")
    if name in LABEL_ALIASES:
        return LABEL_ALIASES[name]
    if name.replace(" ", "_") in SIZE_RANGES_CM:
        return name.replace(" ", "_")
    words = name.split()
    for keyword, canonical in LABEL_KEYWORDS:
        if keyword in words or (len(keyword) > 4 and keyword in name):
            return canonical
    return name


def _family_order(pose_source: str, axis: str, families: list[str]) -> list[str]:
    preferred = FUSION_POLICY.get(pose_source, {}).get(axis)
    if preferred is None:
        preferred = RGB_PRIMARY if RGB_PRIMARY in families else families[0]
    return [preferred] + [f for f in families if f != preferred]


def _match(responses: dict[str, MeasurementResponse]) -> list[str]:
    families = list(responses)
    if len(families) < 2:
        raise ValueError("Fusion needs results from at least two depth models")
    expected = [(v.video_id, v.source_sha256, v.pose_source)
                for v in responses[families[0]].videos]
    for family in families[1:]:
        found = [(v.video_id, v.source_sha256, v.pose_source)
                 for v in responses[family].videos]
        if found != expected:
            raise ValueError("Every model must measure the same videos in the same order")
    return families


def fuse(responses: dict[str, MeasurementResponse]) -> MeasurementResponse:
    """Pick each primary dimension from the policy's model, falling back when it is missing."""
    families = _match(responses)
    fused_videos = []
    for index, base_clip in enumerate(responses[families[0]].videos):
        clips = {family: responses[family].videos[index] for family in families}
        pose_source = base_clip.pose_source
        anchor = _family_order(pose_source, "height", families)[0]
        tracks = {f: {t.track_id: t for t in clips[f].objects} for f in families}
        objects = []
        for track in clips[anchor].objects:
            dimensions = dict(track.dimensions)
            qualifiers = set(track.qualifiers)
            for axis in AXES:
                values = {}
                for family in families:
                    other = tracks[family].get(track.track_id)
                    estimate = other.dimensions.get(axis) if other else None
                    if estimate is not None:
                        values[family] = estimate
                chosen = next(
                    (f for f in _family_order(pose_source, axis, families) if f in values), None
                )
                dimensions[axis] = (
                    None if chosen is None
                    else values[chosen].model_copy(update={"depth_model_family": chosen})
                )
                if len(values) >= 2:
                    numbers = [v.cm for v in values.values()]
                    if (max(numbers) - min(numbers)) / median(numbers) > DISAGREEMENT_RATIO:
                        qualifiers.add(f"depth_models_disagree_{axis}")
            status = track.status
            if all(dimensions.get(a) is not None for a in AXES):
                status = "estimated"
            elif any(v is not None for v in dimensions.values()):
                status = "partial"
            objects.append(track.model_copy(update={
                "dimensions": dimensions, "status": status, "qualifiers": sorted(qualifiers),
            }))
        fused_videos.append(clips[anchor].model_copy(update={
            "objects": objects,
            "depth_model_family": "fused(" + ",".join(families) + ")",
        }))
    return MeasurementResponse(videos=fused_videos)


def review_track(track: MeasuredTrack) -> dict:
    """Decide whether the user should check a track, with reasons shown in the app."""
    label = canonical_label(track.label)
    name = LABEL_KO.get(label, track.label)
    codes: list[str] = []
    reasons: list[str] = []

    def flag(code: str, reason: str) -> None:
        if code not in codes:
            codes.append(code)
            reasons.append(reason)

    primary: dict[str, DimensionEstimate | None] = {a: track.dimensions.get(a) for a in AXES}
    if track.status == "unavailable" or all(v is None for v in primary.values()):
        flag("unavailable", "크기를 추정하지 못했어요. 직접 입력해 주세요")
    else:
        for axis, estimate in primary.items():
            if estimate is None:
                flag(f"missing_{axis}", MISSING_REASON[axis])
    measured = [v for v in primary.values() if v is not None]
    if any(v.method == "metric_depth_view_median" or v.views < 3 for v in measured):
        flag("few_views", "여러 각도에서 촬영되지 않아 크기가 부정확할 수 있어요")
    if "bounding_box_is_approximate" in track.qualifiers:
        flag("approximate_outline", "윤곽 대신 사각형으로 측정해 크기가 부정확할 수 있어요")
    if any(q.startswith("depth_models_disagree_") for q in track.qualifiers):
        flag("models_disagree", "추정 방식마다 값 차이가 커서 확인이 필요해요")
    for axis, (low, high) in SIZE_RANGES_CM.get(label, {}).items():
        estimate = primary.get(axis)
        if estimate is not None and not low <= estimate.cm <= high:
            flag(
                f"out_of_range_{axis}",
                f"{name} {AXIS_KO[axis]} {estimate.cm:.0f}cm는 일반적인 크기"
                f"({low:.0f}~{high:.0f}cm)를 벗어나요",
            )
    return {
        "track_id": track.track_id,
        "label": track.label,
        "needs_review": bool(codes),
        "review_codes": codes,
        "review_reasons": reasons,
    }


def review(response: MeasurementResponse) -> dict:
    return {"videos": [
        {"video_id": clip.video_id, "tracks": [review_track(t) for t in clip.objects]}
        for clip in response.videos
    ]}


def derive_size_ranges(rows: list[dict], widen: float = 0.2) -> dict:
    """Percentile ranges per label from ``arkitscenes_split objects`` CSV rows."""
    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        label = canonical_label(row["label"])
        for axis in AXES:
            values.setdefault(label, {a: [] for a in AXES})[axis].append(
                float(row[f"{axis}_cm"])
            )

    def percentile(data: list[float], q: float) -> float:
        data = sorted(data)
        position = (len(data) - 1) * q
        low = int(position)
        high = min(low + 1, len(data) - 1)
        return data[low] + (data[high] - data[low]) * (position - low)

    ranges = {}
    for label, axes in sorted(values.items()):
        if len(axes["width"]) < RANGE_MIN_SAMPLES:
            continue
        ranges[label] = {
            axis: (
                round(max(5.0, percentile(data, 0.01) * (1 - widen))),
                round(percentile(data, 0.99) * (1 + widen)),
            )
            for axis, data in axes.items()
        }
    return ranges


def _parse_result(text: str) -> tuple[str, Path]:
    family, sep, path = text.partition("=")
    if not sep or not family or not path:
        raise argparse.ArgumentTypeError("Use FAMILY=PATH, e.g. depth-pro=outputs/dp.json")
    return family, Path(path)


def _load(path: Path) -> MeasurementResponse:
    return MeasurementResponse.model_validate_json(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fuse_cmd = sub.add_parser("fuse", help="combine per-model results per dimension")
    fuse_cmd.add_argument("--result", type=_parse_result, action="append", required=True)
    fuse_cmd.add_argument("--output", type=Path, required=True)
    review_cmd = sub.add_parser("review", help="list tracks that need a user check")
    review_cmd.add_argument("result", type=Path)
    review_cmd.add_argument("--output", type=Path, required=True)
    ranges_cmd = sub.add_parser("ranges", help="derive size ranges from an objects CSV")
    ranges_cmd.add_argument("objects_csv", type=Path)
    args = parser.parse_args(argv)
    if args.command == "ranges":
        with args.objects_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(json.dumps(derive_size_ranges(rows), indent=2))
        return 0
    if args.command == "fuse":
        responses = {}
        for family, path in args.result:
            if family in responses:
                raise SystemExit(f"duplicate model family: {family}")
            responses[family] = _load(path)
        output = fuse(responses).model_dump_json(indent=2) + "\n"
    else:
        output = json.dumps(review(_load(args.result)), indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
