"""Replay explicit visual identity corrections and inventory a reviewed frame."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from ai.training.video_inventory import write_image


def review(inventory, annotations, output):
    metadata = json.loads((inventory / "inventory.json").read_text())
    if metadata["video"]["sha256"] != annotations["source_sha256"]:
        raise ValueError("Visual review was made for a different video")
    if metadata["model"]["sha256"] != annotations["model_sha256"]:
        raise ValueError("Visual review was made for different model weights")
    if metadata["sample_stride"] != annotations["sample_stride"]:
        raise ValueError("Visual review was made for a different frame stride")
    aliases = {}
    for unit in annotations["verified_units"]:
        for track in unit["tracks"]:
            if track in aliases:
                raise ValueError(f"Track {track} is assigned more than once")
            aliases[track] = unit

    audited = defaultdict(list)
    with (inventory / "detections.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            index = int(row["frame"])
            unit = aliases.get(row["track_id"])
            if unit and unit["first_frame"] <= index <= unit["last_frame"]:
                audited[index, unit["unit_id"]].append(row)
    counts = Counter(unit_id for _, unit_id in audited)
    expected = {unit["unit_id"] for unit in annotations["verified_units"]}
    if expected != counts.keys():
        raise ValueError("One or more reviewed units have no matching boxes")

    output.mkdir(parents=True)
    with (output / "verified_units.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "time_s",
                "unit_id",
                "classes",
                "track_ids",
                "source_boxes",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        for (index, unit_id), rows in sorted(audited.items()):
            bounds = [
                select(float(row[key]) for row in rows)
                for key, select in zip(("x1", "y1", "x2", "y2"), (min, min, max, max))
            ]
            writer.writerow(
                [
                    index,
                    rows[0]["time_s"],
                    unit_id,
                    "|".join(sorted({row["class"] for row in rows})),
                    "|".join(sorted({row["track_id"] for row in rows})),
                    len(rows),
                    *bounds,
                ]
            )

    census = annotations["single_frame_chair_census"]
    photo = cv2.imread(str(inventory / "frames" / f"{census['frame']:06d}.jpg"))
    if photo is None:
        raise ValueError("The census frame is missing")
    height, width = photo.shape[:2]
    base_width, base_height = census["coordinate_size_px"]
    for index, (x, y) in enumerate(census["chair_back_points"], 1):
        point = (round(x * width / base_width), round(y * height / base_height))
        cv2.circle(photo, point, 15, (0, 0, 0), -1)
        cv2.circle(photo, point, 14, (0, 250, 250), 2)
        cv2.putText(
            photo,
            str(index),
            (point[0] - 9, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (0, 250, 250),
            1,
            cv2.LINE_AA,
        )
    write_image(output / "chair_census.jpg", photo)

    frame_counts = Counter(frame for frame, _ in audited)
    report = {
        "verified_units_in_reviewed_intervals": len(counts),
        "reviewed_unit_observations": len(audited),
        "reviewed_source_box_observations": sum(map(len, audited.values())),
        "within_review_duplicate_box_observations_removed": sum(
            len(rows) - 1 for rows in audited.values()
        ),
        "frames_with_both_reviewed_units": sum(value == 2 for value in frame_counts.values()),
        "unit_frames": counts,
        "single_frame_chair_census": {
            "frame": census["frame"],
            "confirmed_at_least": len(census["chair_back_points"]),
        },
        "verified_whole_video_totals": None,
    }
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.inventory, json.loads(args.annotations.read_text()), args.output)


if __name__ == "__main__":
    main()
