"""재등장 연결(B2): 추적이 끊겨 같은 실물이 여러 트랙으로 나뉜 경우를 묶는다.

원칙
- 같은 품목(category)끼리만 비교한다.
- 한 프레임에라도 함께 보인(시간이 겹친) 트랙은 서로 다른 실물로 보고 절대 병합하지 않는다.
  병합 결과 그룹 안의 모든 트랙 쌍에도 같은 조건을 적용한다.
- 외형 유사도만으로 자동 확정하지 않는다. 자동 병합은
    짧은 공백(auto_max_gap_sec) + 끊긴 위치의 박스 IoU(auto_min_box_iou) + 외형
    유사도(auto_min_appearance)
  를 모두 만족할 때만 한다.
- 공백이 길거나 위치 근거가 없지만 외형이 비슷한 쌍은 병합하지 않고 '확인 요청'으로만 남긴다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .appearance import track_similarity
from .common import imread, imwrite


@dataclass
class TrackItem:
    pid: str
    row: dict
    category: str
    status: str
    frames: np.ndarray
    t0: float
    t1: float
    first_box: list
    last_box: list
    hists: np.ndarray


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def load_track_items(run_dir: Path, items: list[dict], tracks: list[dict]) -> list[TrackItem]:
    feats = np.load(run_dir / "track_features.npz")
    tmap = {f"T{int(t['track_id']):05d}": t for t in tracks}
    out = []
    for r in items:
        pid = r["pred_item_id"]
        t = tmap[pid]
        out.append(
            TrackItem(
                pid=pid,
                row=r,
                category=r["category"],
                status=r["status"],
                frames=feats[f"{pid}__frames"],
                t0=float(r["first_time_sec"]),
                t1=float(r["last_time_sec"]),
                first_box=json.loads(t["first_box"]),
                last_box=json.loads(t["last_box"]),
                hists=feats[f"{pid}__hists"],
            )
        )
    return out


def _overlap(a: TrackItem, b: TrackItem) -> bool:
    if a.t1 < b.t0 or b.t1 < a.t0:
        return False
    return bool(np.intersect1d(a.frames, b.frames, assume_unique=False).size)


def evaluate_pairs(tis: list[TrackItem], cfg: dict) -> list[dict]:
    rows = []
    for i, a in enumerate(tis):
        for b in tis[i + 1 :]:
            if a.category != b.category:
                continue
            first, second = (a, b) if a.t0 <= b.t0 else (b, a)
            rec = {"track_a": first.pid, "track_b": second.pid, "category": a.category}
            if _overlap(first, second):
                rec.update(decision="never_merge", reason="같은 프레임에 함께 보임(서로 다른 실물)")
                rows.append(rec)
                continue
            gap = second.t0 - first.t1
            iou = box_iou(first.last_box, second.first_box)
            sim = (
                track_similarity(first.hists, second.hists)
                if len(first.hists) and len(second.hists)
                else float("nan")
            )
            rec.update(
                gap_sec=round(gap, 3),
                box_iou=round(iou, 3),
                appearance=round(sim, 3) if sim == sim else "",
            )
            if (
                gap <= cfg["auto_max_gap_sec"]
                and iou >= cfg["auto_min_box_iou"]
                and sim == sim
                and sim >= cfg["auto_min_appearance"]
            ):
                rec.update(
                    decision="auto_merge", reason="짧은 공백 + 끊긴 위치 박스 IoU + 외형 유사"
                )
            elif (
                gap <= cfg["suggest_max_gap_sec"]
                and sim == sim
                and sim >= cfg["suggest_min_appearance"]
            ):
                rec.update(
                    decision="needs_review",
                    reason="외형은 비슷하지만 위치·시간 근거 부족 → 사람 확인",
                )
            else:
                rec.update(decision="separate", reason="조건 미달")
            rows.append(rec)
    return rows


def group_tracks(
    tis: list[TrackItem], pair_rows: list[dict]
) -> tuple[list[list[TrackItem]], list[dict]]:
    """auto_merge 쌍을 외형 유사도 높은 순으로 합치되, 그룹 안에 시간이 겹치는 쌍이 생기면
    거부한다.
    """
    by = {t.pid: t for t in tis}
    parent = {t.pid: t.pid for t in tis}
    members = {t.pid: [t.pid] for t in tis}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    autos = sorted(
        [r for r in pair_rows if r["decision"] == "auto_merge"],
        key=lambda r: -float(r["appearance"]),
    )
    for r in autos:
        ra, rb = find(r["track_a"]), find(r["track_b"])
        if ra == rb:
            r["applied"] = "already_grouped"
            continue
        conflict = any(_overlap(by[x], by[y]) for x in members[ra] for y in members[rb])
        if conflict:
            r["applied"] = "rejected_group_overlap"
            continue
        parent[rb] = ra
        members[ra] += members.pop(rb)
        r["applied"] = "merged"
    groups = [sorted((by[p] for p in m), key=lambda t: t.t0) for m in members.values()]
    groups.sort(key=lambda g: g[0].t0)
    return groups, pair_rows


def evidence_image(run_dir: Path, a_pid: str, b_pid: str, text: str, out: Path) -> None:
    """끊긴 트랙의 마지막 모습(A)과 다음 트랙의 첫 모습(B)을 나란히 저장."""
    a = imread(run_dir / "tracks" / f"{a_pid}_last.jpg")
    b = imread(run_dir / "tracks" / f"{b_pid}_first.jpg")
    h = 300
    a = cv2.resize(a, (max(1, int(a.shape[1] * h / a.shape[0])), h))
    b = cv2.resize(b, (max(1, int(b.shape[1] * h / b.shape[0])), h))
    canvas = np.full((h + 60, a.shape[1] + b.shape[1] + 20, 3), 255, np.uint8)
    canvas[60:, : a.shape[1]] = a
    canvas[60:, a.shape[1] + 20 :] = b
    cv2.putText(canvas, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


__all__ = ["load_track_items", "evaluate_pairs", "group_tracks", "evidence_image"]
