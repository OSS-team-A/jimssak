"""Run local COCO furniture detection and class-specific BoT-SORT on a video."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

CLASSES = {"chair": 56, "dining table": 60}
MIN_CONF = 0.15
MIN_TRACK_FRAMES = 3


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(path):
    video = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not video.isOpened():
        raise ValueError(f"Cannot decode {path} with OpenCV FFmpeg")
    video.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return video


def frames(video, stride):
    index = 0
    while video.grab():
        if index % stride == 0:
            ok, frame = video.retrieve()
            if not ok:
                raise RuntimeError(f"Could not retrieve frame {index}")
            yield index, video.get(cv2.CAP_PROP_POS_MSEC) / 1000, frame
        index += 1


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def track_video(source, weights, output, stride, image_size):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model = YOLO(str(weights))
    config = YAML.load(check_yaml("botsort.yaml"))
    config.update(
        track_high_thresh=0.25,
        track_low_thresh=MIN_CONF,
        new_track_thresh=0.25,
        track_buffer=10,
        with_reid=False,
    )
    trackers = {label: BOTSORT(args=IterableSimpleNamespace(**config)) for label in CLASSES}
    video = capture(source)
    fps = video.get(cv2.CAP_PROP_FPS)
    summary = {
        "video": {
            "name": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "reported_frame_count": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
            "average_fps": fps,
            "orientation_metadata_deg": int(video.get(cv2.CAP_PROP_ORIENTATION_META)),
            "codec": int(video.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii"),
        },
        "model": {
            "file": weights.name,
            "sha256": sha256(weights),
            "classes": CLASSES,
            "minimum_confidence": MIN_CONF,
            "nms_iou": 0.5,
            "image_size": image_size,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "torch": torch.__version__,
        },
        "sample_stride": stride,
        "tracker": config,
        "min_track_observations": MIN_TRACK_FRAMES,
    }
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "overlays").mkdir()
    observations = defaultdict(list)
    raw_counts = Counter()
    tracked_counts = Counter()
    samples = 0
    try:
        with (output / "detections.csv").open("w", newline="") as stream:
            columns = ["frame", "time_s", "class", "track_id", "confidence", "x1", "y1", "x2", "y2"]
            writer = csv.writer(stream)
            writer.writerow(columns)
            for index, time_s, frame in frames(video, stride):
                samples += 1
                height, width = frame.shape[:2]
                summary["video"]["display_width_px"] = width
                summary["video"]["display_height_px"] = height
                name = f"{index:06d}.jpg"
                write_image(output / "frames" / name, frame)
                overlay = frame.copy()
                result = model.predict(
                    frame,
                    classes=list(CLASSES.values()),
                    conf=MIN_CONF,
                    iou=0.5,
                    imgsz=image_size,
                    verbose=False,
                    device="cpu",
                )[0]
                boxes = result.boxes.cpu().numpy()
                for label, category in CLASSES.items():
                    selected = boxes[boxes.cls == category]
                    tracks = trackers[label].update(selected, frame)
                    ids = {int(row[-1]): f"{category}-{int(row[4])}" for row in tracks}
                    raw_counts[label] += len(selected)
                    for local, (box, score) in enumerate(zip(selected.xyxy, selected.conf)):
                        track = ids.get(local)
                        writer.writerow(
                            [
                                index,
                                f"{time_s:.6f}",
                                label,
                                track or "",
                                f"{score:.5f}",
                                *[f"{v:.2f}" for v in box],
                            ]
                        )
                        if track:
                            tracked_counts[label] += 1
                            observations[(label, track)].append((index, time_s, float(score)))
                        color = (255, 180, 20) if label == "chair" else (30, 230, 245)
                        x1, y1, x2, y2 = [int(v) for v in box]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        caption = f"{label} {track or '?'} {score:.2f}"
                        cv2.putText(
                            overlay,
                            caption,
                            (x1, max(20, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            (0, 0, 0),
                            3,
                        )
                        cv2.putText(
                            overlay,
                            caption,
                            (x1, max(20, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.48,
                            color,
                            1,
                        )
                write_image(output / "overlays" / name, overlay)
    finally:
        video.release()

    summary["video"]["sampled_frames"] = samples
    with (output / "tracks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "class",
                "track_id",
                "first_frame",
                "last_frame",
                "first_time_s",
                "last_time_s",
                "observations",
                "max_confidence",
            ]
        )
        for (label, track), rows in observations.items():
            writer.writerow(
                [
                    label,
                    track,
                    rows[0][0],
                    rows[-1][0],
                    round(rows[0][1], 6),
                    round(rows[-1][1], 6),
                    len(rows),
                    max(row[2] for row in rows),
                ]
            )
    summary["by_class"] = {}
    for label in CLASSES:
        counts = [len(rows) for (kind, _), rows in observations.items() if label == kind]
        summary["by_class"][label] = {
            "box_observations": raw_counts[label],
            "observations_with_track": tracked_counts[label],
            "track_ids": len(counts),
            "track_ids_min_observations": sum(count >= MIN_TRACK_FRAMES for count in counts),
            "verified_unique_objects_in_video": None,
        }
    (output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=960)
    args = parser.parse_args()
    if args.stride < 1 or args.image_size < 32:
        parser.error("stride must be positive and image size must be at least 32")
    if not args.weights.is_file():
        parser.error("provide a local YOLO weight file")
    track_video(args.video, args.weights, args.output, args.stride, args.image_size)


if __name__ == "__main__":
    main()
