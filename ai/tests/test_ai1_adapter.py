import csv
import json

import numpy as np
import pytest

from ai.service import ai1_adapter
from ai.service.pipeline import NoFurniture

FIELDS = ["pred_item_id", "status", "category", "mean_conf", "track_ids"]


def square(x, y, side=100):
    return [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]


def det(track, polygons, box, area=10_000):
    return {"track_id": track, "category": "chair", "conf": 0.8, "box_xyxy": box,
            "mask_area_px": area, "mask_polygons": polygons}


def write_run(tmp_path, frames, items, size=(1080, 1920), relink=None):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps(
        {"video": {"decoded_width": size[0], "decoded_height": size[1]}}))
    (run / "frames.jsonl").write_text("\n".join(
        json.dumps({"frame_idx": i, "detections": d}) for i, d in frames))
    for folder, rows in ((run, items), (run / "relink_x", relink)):
        if rows is None:
            continue
        folder.mkdir(exist_ok=True)
        with (folder / "items.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return run


def item(pid, tracks, status="confirmed", category="chair", conf="0.7"):
    return {"pred_item_id": pid, "status": status, "category": category,
            "mean_conf": conf, "track_ids": ";".join(map(str, tracks))}


def test_relinked_tracks_become_one_object_and_unlisted_tracks_are_dropped(tmp_path):
    box = [100, 100, 200, 200]
    frames = [(i, [det(1 if i < 5 else 2, [square(100, 100)], box),
                   det(9, [square(400, 400)], [400, 400, 500, 500])]) for i in range(10)]
    run = write_run(tmp_path, frames, items=[item("T00001", [1]), item("T00002", [2])],
                    relink=[item("G00001", [1, 2], conf="0.75"),
                            item("T00009", [9], status="dropped")])
    result = ai1_adapter.from_run(run, tmp_path / "v.mp4")
    clip = result.video
    assert {v.track_id for v in clip.views} == {"G00001"}
    assert len(clip.views) == ai1_adapter.VIEWS_PER_ITEM
    assert {v.frame_index for v in clip.views} >= {0, 9}  # spread over both tracks
    assert clip.pixel_rotation_ccw == 0 and clip.annotation_size_px == (1080, 1920)
    assert result.confidences == {"G00001": 0.75}


def test_split_outline_is_sent_whole_as_rle(tmp_path):
    pieces = [square(100, 100), square(300, 100)]
    run = write_run(tmp_path, [(0, [det(1, pieces, [100, 100, 400, 200])])],
                    items=[item("T00001", [1])], size=(640, 480))
    view = ai1_adapter.from_run(run, tmp_path / "v.mp4").video.views[0]
    assert view.mask_polygon_px is None
    runs = np.asarray(view.mask_rle)
    mask = np.repeat(np.arange(len(runs)) % 2, runs).reshape(480, 640)
    assert mask[150, 150] == 1 and mask[150, 350] == 1 and mask[150, 250] == 0
    assert runs.sum() == 640 * 480


def test_4k_coordinates_are_scaled_and_border_boxes_marked_incomplete(tmp_path):
    frames = [(0, [det(1, [square(200, 200, 400)], [200, 200, 600, 600]),
                   det(2, [square(0, 1000, 400)], [0, 1000, 400, 1400])])]
    run = write_run(tmp_path, frames, items=[item("T00001", [1]), item("T00002", [2])],
                    size=(2160, 3840))
    clip = ai1_adapter.from_run(run, tmp_path / "v.mp4").video
    assert clip.annotation_size_px == (1080, 1920)
    first, edge = sorted(clip.views, key=lambda v: v.track_id)
    assert first.mask_polygon_px[0] == pytest.approx((100, 100))
    assert first.complete and not edge.complete


def test_outline_a_few_pixels_short_of_the_border_is_incomplete(tmp_path):
    # Upscaled model masks stop a few pixels short of the border they were cut at.
    near = [[700, 800], [1075, 800], [1075, 1200], [700, 1200]]
    frames = [(0, [det(1, [near], [700, 800, 1070, 1200]),
                   det(2, [square(300, 300, 300)], [300, 300, 600, 600])])]
    run = write_run(tmp_path, frames, items=[item("T00001", [1]), item("T00002", [2])])
    cut, inside = sorted(ai1_adapter.from_run(run, tmp_path / "v.mp4").video.views,
                         key=lambda v: v.track_id)
    assert not cut.complete and inside.complete


def test_no_listed_furniture_raises(tmp_path):
    run = write_run(tmp_path, [(0, [det(1, [square(0, 0)], [0, 0, 100, 100])])],
                    items=[item("T00001", [1], status="dropped")])
    with pytest.raises(NoFurniture):
        ai1_adapter.from_run(run, tmp_path / "v.mp4")


def test_recognize_needs_weights(monkeypatch, tmp_path):
    monkeypatch.delenv("JIMSSAK_AI1_WEIGHTS", raising=False)
    with pytest.raises(RuntimeError, match="JIMSSAK_AI1_WEIGHTS"):
        ai1_adapter.recognize(tmp_path / "v.mp4")
