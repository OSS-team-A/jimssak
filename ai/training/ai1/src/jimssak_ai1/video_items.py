"""입력 영상 결과 → 임베딩 기반 재등장 연결 + 백엔드/Depth 연동용 물품 JSON.

실제 치수·부피 측정 기능은 없으므로 dimensions_cm=null, box_fit='unknown'(판단 보류).
이사 후 영상이 없으면 missing_status='not_evaluated', 전후 비교가 없으면
damage_status='not_comparable'.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .common import imread, read_csv, write_csv, write_json
from .pipeline import ITEM_FIELDS
from .tracking.bench import group_tracks

SCHEMA_VERSION = "jimssak.item.v1"


def track_crops_from_run(run_dir: Path, pid: str) -> list[np.ndarray]:
    paths = sorted((run_dir / "items" / pid).glob("rep*_crop.jpg")) + [
        run_dir / "tracks" / f"{pid}_first.jpg",
        run_dir / "tracks" / f"{pid}_last.jpg",
    ]
    return [imread(p) for p in paths if p.is_file()]


def embed_tracks(run_dir: Path, embedder) -> dict[str, np.ndarray]:
    items = read_csv(run_dir / "items.csv")
    E = {}
    for r in items:
        cs = track_crops_from_run(run_dir, r["pred_item_id"])
        if cs:
            z = embedder.embed(cs).mean(0)
            E[r["pred_item_id"]] = z / max(np.linalg.norm(z), 1e-12)
    return E


def embedding_relink(run_dir: Path, E: dict, thr: float, out_dir: Path, method: str) -> list[dict]:
    items = read_csv(run_dir / "items.csv")
    feats = np.load(run_dir / "track_features.npz")
    tracks = {
        r["pred_item_id"]: {
            "class": r["category"],
            "frames": feats[f"{r['pred_item_id']}__frames"].tolist(),
            "row": r,
        }
        for r in items
    }

    def ok(a, b):
        if a not in E or b not in E:
            return None
        s = float(E[a] @ E[b])
        return s if s >= thr else None

    gmap = group_tracks(tracks, ok)
    groups = defaultdict(list)
    for t, g in gmap.items():
        groups[g].append(t)
    out, gi = [], 0
    for g, ts in sorted(
        groups.items(), key=lambda kv: min(float(tracks[t]["row"]["first_time_sec"]) for t in kv[1])
    ):
        rows = [
            tracks[t]["row"]
            for t in sorted(ts, key=lambda t: float(tracks[t]["row"]["first_time_sec"]))
        ]
        if len(rows) == 1:
            gid = rows[0]["pred_item_id"]
        else:
            gi += 1
            gid = f"G{gi:05d}"
        nf = [int(r["n_frames"]) for r in rows]
        best = max(rows, key=lambda r: float(r["max_conf"]))
        out.append(
            {
                **best,
                "pred_item_id": gid,
                "status": "confirmed"
                if any(r["status"] == "confirmed" for r in rows)
                else "needs_review",
                "n_frames": sum(nf),
                "mean_conf": round(
                    sum(float(r["mean_conf"]) * n for r, n in zip(rows, nf)) / max(1, sum(nf)), 4
                ),
                "first_time_sec": min(float(r["first_time_sec"]) for r in rows),
                "last_time_sec": max(float(r["last_time_sec"]) for r in rows),
                "rep_images": ";".join(r["rep_images"] for r in rows if r["rep_images"]),
                "track_ids": ";".join(r["track_ids"] for r in rows),
                "track_n_frames": ";".join(map(str, nf)),
                "note": (
                    f"{method} 임베딩 재등장 연결(코사인≥{thr:.2f}): "
                    + "+".join(r["pred_item_id"] for r in rows)
                )
                if len(rows) > 1
                else "",
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "items.csv", out, ITEM_FIELDS)
    return out


def _rep_detection(run_dir: Path, rows: list[dict]) -> dict:
    """대표 프레임의 박스·마스크 다각형을 frames.jsonl에서 찾는다."""
    want = {}
    for r in rows:
        tids = [int(x) for x in r["track_ids"].split(";") if x]
        want[(int(r["rep_frame_idx"]), tuple(tids))] = r["pred_item_id"]
    by_frame = defaultdict(list)
    for (f, tids), pid in want.items():
        by_frame[f].append((set(tids), pid))
    found = {}
    with open(run_dir / "frames.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["frame_idx"] not in by_frame:
                continue
            for tids, pid in by_frame[d["frame_idx"]]:
                cands = [x for x in d["detections"] if x["track_id"] in tids]
                if cands:
                    x = max(cands, key=lambda z: z["conf"])
                    found[pid] = {
                        "frame_idx": d["frame_idx"],
                        "box_xyxy": x["box_xyxy"],
                        "polygons": x["mask_polygons"],
                        "conf": x["conf"],
                    }
    return found


def build_item_json(
    *,
    run_dir: Path,
    items_csv: Path,
    video_id: str,
    model_label: str,
    emb_ref: str | None,
    emb_keys: dict | None,
    out_path: Path,
    damage: dict | None = None,
    rel_base: str = "",
) -> dict:
    """rel_base: 결과 폴더 기준 영상 실행 폴더 경로(예: 'video/B1/B1') — 이미지 경로를 결과 압축
    안에서 열 수 있게.
    """
    rows = [r for r in read_csv(items_csv) if r["status"] in ("confirmed", "needs_review")]
    reps = _rep_detection(run_dir, rows)
    items = []
    for k, r in enumerate(rows, 1):
        tids = [int(x) for x in r["track_ids"].split(";") if x]
        rep = reps.get(r["pred_item_id"], {})
        review, reasons = r["status"] == "needs_review", []
        if review:
            reasons.append("짧거나 신뢰도 낮은 트랙(확인 필요)")
        if "확인 필요" in (r.get("note") or ""):
            review = True
            reasons.append("재등장 동일 물품 여부 확인 필요")
        dmg = (damage or {}).get(r["pred_item_id"])
        items.append(
            {
                "item_id": f"{video_id}_I{k:04d}",
                "source_item_id": r["pred_item_id"],
                "tracking_id": tids[0] if tids else None,
                "linked_tracking_ids": tids,
                "class_name": r["category"],
                "class_name_ko": r.get("category_ko"),
                "model_class": r["model_class"],
                "confidence": float(r["mean_conf"]),
                "status": r["status"],
                "first_seen_timestamp": float(r["first_time_sec"]),
                "last_seen_timestamp": float(r["last_time_sec"]),
                "representative_image": (f"{rel_base}/" if rel_base else "")
                + (r["rep_images"].split(";") or [""])[0],
                "bounding_box": {
                    "frame_idx": rep.get("frame_idx"),
                    "xyxy": rep.get("box_xyxy"),
                    "coordinate_system": "회전 적용 후 원본 해상도 픽셀",
                },
                "segmentation_mask": {"format": "polygons_xy", "polygons": rep.get("polygons")},
                "reid_embedding_reference": {
                    "file": emb_ref,
                    "key": (emb_keys or {}).get(r["pred_item_id"]),
                }
                if emb_ref
                else None,
                "matched_item_id": None,
                "matching_confidence": None,
                "missing_status": "not_evaluated",
                "damage_status": "not_comparable",
                "damage_type": dmg.get("type") if dmg else None,
                "damage_region": dmg.get("region") if dmg else None,
                "damage_confidence": dmg.get("confidence") if dmg else None,
                "damage_note": dmg.get("note") if dmg else "이사 전후 쌍이 없어 상태 비교 불가",
                "dimensions_cm": None,
                "volume_liter": None,
                "box_fit_60x40x40": "unknown",
                "box_fit_reason": "치수 추정(Depth) 모듈 미구현 — 판단 보류",
                "review_required": review,
                "review_reason": "; ".join(reasons),
            }
        )
    doc = {
        "schema": SCHEMA_VERSION,
        "video_id": video_id,
        "model": model_label,
        "n_items": len(items),
        "n_auto_confirmed": sum(1 for i in items if i["status"] == "confirmed"),
        "class_counts_confirmed": dict(
            Counter(i["class_name"] for i in items if i["status"] == "confirmed")
        ),
        "items": items,
        "notes": [
            "정답 물품 목록이 없어 이 영상의 정확도는 미측정(검수표로 사람이 확인)",
            "missing_status는 이사 후 영상과 비교해야 결정됨 (compare_moves.py)",
        ],
    }
    write_json(out_path, doc)
    return doc
