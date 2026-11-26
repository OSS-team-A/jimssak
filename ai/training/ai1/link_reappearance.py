#!/usr/bin/env python
"""재등장 연결(B2)을 기존 추론 결과 폴더에 적용한다. 추론을 다시 하지 않으므로 B0/B1과 검출·추적이
동일하다.

예) python -u link_reappearance.py --run runs/B0/<run-name>
결과: <run>/relink_<시각>/  (items.csv, merge_decisions.csv, evidence/, relink_manifest.json)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import (  # noqa: E402
    PROJECT_ROOT,
    code_fingerprint,
    fail,
    load_yaml,
    log,
    make_run_dir,
    now_iso,
    now_stamp,
    read_csv,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from jimssak_ai1.pipeline import ITEM_FIELDS  # noqa: E402
from jimssak_ai1.relink import evaluate_pairs, evidence_image, group_tracks, load_track_items  # noqa: E402

PAIR_FIELDS = [
    "track_a",
    "track_b",
    "category",
    "gap_sec",
    "box_iou",
    "appearance",
    "decision",
    "applied",
    "reason",
    "evidence_image",
]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="재등장 연결: 같은 품목·시간 겹침 없음·근거 조건을 만족하는 트랙만 묶고, "
        "애매한 쌍은 확인 요청으로 남긴다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", required=True, type=Path, help="infer.py 결과 폴더")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="relink 설정이 든 YAML. 생략 시 그 실행에 쓰인 pipeline 설정 파일의 relink 섹션",
    )
    p.add_argument("--out-name", default=None, help="출력 하위 폴더 이름. 기본 relink_<시각>")
    args = p.parse_args(argv)

    run_dir = args.run.resolve()
    if not (run_dir / "run_manifest.json").is_file():
        fail(f"infer.py 결과 폴더가 아닙니다(run_manifest.json 없음): {run_dir}")
    man = read_json(run_dir / "run_manifest.json")
    cfg_path = args.config or Path(man["pipeline_config_path"])
    if not cfg_path.is_file():
        cfg_path = PROJECT_ROOT / "configs" / "pipeline_default.yaml"
        log(f"실행 당시 설정 파일을 찾지 못해 {cfg_path} 를 사용합니다")
    cfg = load_yaml(cfg_path).get("relink")
    if not cfg:
        fail(f"설정 파일에 relink 섹션이 없습니다: {cfg_path}")

    items = read_csv(run_dir / "items.csv")
    tracks = read_csv(run_dir / "tracks.csv")
    tis = load_track_items(run_dir, items, tracks)
    pairs = evaluate_pairs(tis, cfg)
    groups, pairs = group_tracks(tis, pairs)

    out = make_run_dir(run_dir, args.out_name or f"relink_{now_stamp()}")
    (out / "evidence").mkdir(exist_ok=True)
    for r in pairs:
        if r["decision"] in ("auto_merge", "needs_review"):
            name = f"{r['decision']}_{r['track_a']}_{r['track_b']}.jpg"
            txt = (
                f"{r['decision']} {r['category']} A={r['track_a']}(last) B={r['track_b']}(first) "
                f"gap={r['gap_sec']}s iou={r['box_iou']} app={r['appearance']}"
            )
            evidence_image(run_dir, r["track_a"], r["track_b"], txt, out / "evidence" / name)
            r["evidence_image"] = f"evidence/{name}"

    # 확인 요청 표시: 어느 그룹이 어느 그룹과 같은 실물일 수 있는지
    pid_to_group = {}
    new_items = []
    gi = 0
    for g in groups:
        if len(g) == 1:
            gid = g[0].pid  # 단일 트랙은 B0와 같은 ID 유지 → 트랙 단위 검수표를 그대로 사용 가능
        else:
            gi += 1
            gid = f"G{gi:05d}"
        for t in g:
            pid_to_group[t.pid] = gid
    review_notes: dict[str, set] = {}
    for r in pairs:
        if r["decision"] == "needs_review":
            ga, gb = pid_to_group[r["track_a"]], pid_to_group[r["track_b"]]
            if ga != gb:
                review_notes.setdefault(ga, set()).add(gb)
                review_notes.setdefault(gb, set()).add(ga)

    for g in groups:
        gid = pid_to_group[g[0].pid]
        rows = [t.row for t in g]
        status = "confirmed" if any(t.status == "confirmed" for t in g) else "needs_review"
        best = max(rows, key=lambda r: float(r["max_conf"]))
        nf = [int(r["n_frames"]) for r in rows]
        mean_conf = sum(float(r["mean_conf"]) * n for r, n in zip(rows, nf)) / max(1, sum(nf))
        note = []
        if len(g) > 1:
            note.append("재등장 자동 연결: " + "+".join(t.pid for t in g))
        if gid in review_notes:
            note.append("같은 실물일 수 있음(확인 필요): " + ",".join(sorted(review_notes[gid])))
        new_items.append(
            {
                **best,
                "pred_item_id": gid,
                "status": status,
                "n_frames": sum(nf),
                "mean_conf": round(mean_conf, 4),
                "max_conf": max(float(r["max_conf"]) for r in rows),
                "first_frame": min(int(r["first_frame"]) for r in rows),
                "last_frame": max(int(r["last_frame"]) for r in rows),
                "first_time_sec": min(float(r["first_time_sec"]) for r in rows),
                "last_time_sec": max(float(r["last_time_sec"]) for r in rows),
                "rep_images": ";".join(r["rep_images"] for r in rows if r["rep_images"]),
                "track_ids": ";".join(r["track_ids"] for r in rows),
                "track_n_frames": ";".join(str(n) for n in nf),
                "note": " | ".join(note),
                "run_id": f"{run_dir.name}/{out.name}",
            }
        )
    write_csv(out / "items.csv", new_items, ITEM_FIELDS)
    write_csv(out / "merge_decisions.csv", pairs, PAIR_FIELDS)

    dec = Counter(r["decision"] for r in pairs)
    applied = Counter(r.get("applied") for r in pairs if r["decision"] == "auto_merge")
    st_before = Counter(r["status"] for r in items)
    st_after = Counter(r["status"] for r in new_items)
    manifest = {
        "created_at": now_iso(),
        "base_run": str(run_dir),
        "base_run_label": man.get("run_label"),
        "relink_config_path": str(cfg_path),
        "relink_config_sha256": sha256_file(cfg_path),
        "relink_config": cfg,
        "code_sha256": code_fingerprint(),
        "appearance_descriptor": "마스크 영역 HSV 히스토그램(16x4x4), 1-Bhattacharyya",
        "uses_dinov2": False,
        "pair_decisions": dict(dec),
        "auto_merge_applied": dict(applied),
        "items_before_by_status": dict(st_before),
        "items_after_by_status": dict(st_after),
        "review_requests": dec.get("needs_review", 0),
        "note": "자동 병합은 짧은 공백+위치+외형 근거를 모두 만족할 때만. 외형만 비슷한 쌍은 "
        "병합하지 않고 확인 요청으로 남김.",
    }
    write_json(out / "relink_manifest.json", manifest)
    log(f"재등장 연결 완료: {out}")
    log(f"  쌍 판정 {dict(dec)} | 자동 병합 적용 {dict(applied)}")
    log(
        f"  물품 수(상태별) 이전 {dict(st_before)} → 이후 {dict(st_after)} | 확인 "
        f"요청 {dec.get('needs_review', 0)}쌍"
    )


if __name__ == "__main__":
    main()
