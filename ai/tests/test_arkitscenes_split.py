import json

import pytest

from ai.training.arkitscenes_split import (
    check_jobs,
    find_used_scans,
    furniture_objects,
    main,
    make_split,
    scan_ids_in_job,
)


def metadata(rooms=60, scans_per_room=3):
    rows = {}
    for room in range(rooms):
        for index in range(scans_per_room):
            scan = f"4{room:04d}{index:03d}"
            rows[scan] = {
                "video_id": scan,
                "visit_id": str(400000 + room),
                "fold": "Validation",
                "is_in_threedod": "True",
            }
    rows["40999999"] = {
        "video_id": "40999999", "visit_id": "1", "fold": "Training", "is_in_threedod": "True"
    }
    return rows


def test_split_keeps_rooms_together_and_excludes_used_rooms():
    rows = metadata()
    used = {"40000001": ["old-experiment"]}
    split = make_split(rows, used, holdout_fraction=0.5)
    holdout = set(split["holdout_scans"])
    assert holdout
    assert not {"40000000", "40000001", "40000002"} & holdout
    assert "40999999" not in holdout
    for visit in split["holdout_rooms"]:
        room_scans = {s for s, r in rows.items() if r["visit_id"] == visit}
        assert room_scans <= holdout
    assert make_split(rows, used, 0.5) == split


def test_find_used_scans_reads_names_and_text(tmp_path):
    (tmp_path / "exp-a" / "data" / "40000000").mkdir(parents=True)
    (tmp_path / "exp-b").mkdir()
    (tmp_path / "exp-b" / "truth.json").write_text('{"video_id": "40001000-rgb"}')
    (tmp_path / "exp-b" / "metadata.csv").write_text("40002000\n")
    used = find_used_scans(tmp_path, {"40000000", "40001000", "40002000"})
    assert used == {"40000000": ["exp-a"], "40001000": ["exp-b"]}


def test_check_rejects_holdout_in_development_and_others_in_final(tmp_path):
    split = {"version": 1, "holdout_scans": ["41125718"]}
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"videos": [{"video_id": "48458732-rgb"}]}))
    final = tmp_path / "final.json"
    final.write_text(json.dumps({"videos": [{"video_id": "x", "path": "41125718/a.mov"}]}))
    assert scan_ids_in_job(json.loads(final.read_text())) == {"41125718"}
    assert check_jobs(split, [dev]) == []
    assert check_jobs(split, [final], final=True) == []
    assert check_jobs(split, [final])
    assert check_jobs(split, [dev], final=True)


def box(label, lengths, vertical=1.0):
    return {
        "uid": label,
        "label": label,
        "segments": {"obbAligned": {
            "axesLengths": lengths,
            "normalizedAxes": [1, 0, 0, 0, 1, 0, 0, 0, vertical],
            "centroid": [0, 0, 0],
        }},
    }


def test_furniture_objects_orders_floor_axes_and_filters():
    annotation = {"data": [
        box("table", [0.4947, 0.7402, 0.4261]),
        box("fireplace", [1.0, 0.5, 1.0]),
        box("chair", [0.1, 0.4, 0.8]),
        box("sofa", [2.0, 0.9, 0.8], vertical=0.5),
    ]}
    assert furniture_objects(annotation, "48458732") == [{
        "scan": "48458732", "uid": "table", "label": "table",
        "width_cm": 74.02, "depth_cm": 49.47, "height_cm": 42.61,
    }]


def test_make_refuses_to_overwrite_frozen_split(tmp_path):
    output = tmp_path / "split.json"
    output.write_text("{}")
    source = tmp_path / "metadata.csv"
    source.write_text("video_id,visit_id,fold,is_in_threedod\n40000000,1,Validation,True\n")
    with pytest.raises(SystemExit):
        main(["make", str(source), "--output", str(output)])
