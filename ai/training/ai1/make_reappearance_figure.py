#!/usr/bin/env python
"""발표용 핵심 장면: 같은 물건이 다시 등장한 두 프레임과 재등장 연결 전(B0/B1)·후(B2) 결과를 한
장으로.

예) python make_reappearance_figure.py --run runs/B0/<run> --relink runs/B0/<run>/relink_<t> --a
T00003 --b T00136
    (--a/--b 생략 시 merge_decisions.csv의 auto_merge → needs_review 순으로 첫 쌍)

주의: 이 그림은 '추적 개선(재등장 연결)' 장면이다. 학습(B1)으로 개선된 장면과 섞지 않는다.
실제로 같은 실물인지는 사람이 검수표(match_review)로 확인한 뒤 발표에 사용한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import fail, imread, imwrite, log, read_csv  # noqa: E402


def first_rep(run: Path, items: dict, pid: str) -> tuple[np.ndarray, str]:
    r = items[pid]
    reps = [x for x in r["rep_images"].split(";") if x]
    if not reps:
        fail(f"{pid} 대표 이미지 없음")
    return imread(
        run / reps[0]
    ), f"{pid} {r['model_class']} {float(r['first_time_sec']):.1f}-{float(r['last_time_sec']):.1f}s"


def main(argv=None):
    ap = argparse.ArgumentParser(description="재등장 장면 전후 비교 그림")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--relink", required=True, type=Path)
    ap.add_argument("--a", default=None)
    ap.add_argument("--b", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    items = {r["pred_item_id"]: r for r in read_csv(args.run / "items.csv")}
    pairs = read_csv(args.relink / "merge_decisions.csv")
    if args.a and args.b:
        pr = next((p for p in pairs if {p["track_a"], p["track_b"]} == {args.a, args.b}), None)
    else:
        pr = next((p for p in pairs if p["decision"] == "auto_merge"), None) or next(
            (p for p in pairs if p["decision"] == "needs_review"), None
        )
    if pr is None:
        fail("해당 쌍이 merge_decisions.csv에 없습니다")
    a, b = pr["track_a"], pr["track_b"]
    ia, ta = first_rep(args.run, items, a)
    ib, tb = first_rep(args.run, items, b)
    h = 640
    ia = cv2.resize(ia, (int(ia.shape[1] * h / ia.shape[0]), h))
    ib = cv2.resize(ib, (int(ib.shape[1] * h / ib.shape[0]), h))
    W = ia.shape[1] + ib.shape[1] + 30
    canvas = np.full((h + 170, W, 3), 255, np.uint8)
    canvas[60 : 60 + h, : ia.shape[1]] = ia
    canvas[60 : 60 + h, ia.shape[1] + 30 :] = ib

    def put(t, xy, c=(0, 0, 0), s=0.7):
        cv2.putText(canvas, t, xy, cv2.FONT_HERSHEY_SIMPLEX, s, c, 2)

    put(ta, (5, 40))
    put(tb, (ia.shape[1] + 35, 40))
    n_conf = sum(items[x]["status"] == "confirmed" for x in (a, b))
    put(f"Before (no re-link): 2 separate items, {n_conf} auto-counted", (5, h + 95), (0, 0, 200))
    after = {
        "auto_merge": "After (re-link B2): merged into 1 item",
        "needs_review": "After (re-link B2): NOT merged, flagged for human check",
    }.get(pr["decision"], f"After: {pr['decision']}")
    put(after, (5, h + 130), (0, 120, 0))
    put(
        f"gap {pr.get('gap_sec')}s | box IoU {pr.get('box_iou')} | "
        f"appearance {pr.get('appearance')}",
        (5, h + 160),
        (80, 80, 80),
        0.55,
    )
    out = args.out or (args.relink / "figures" / f"reappear_{a}_{b}.jpg")
    imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"저장: {out}  (판정: {pr['decision']} / {pr.get('applied', '')})")


if __name__ == "__main__":
    main()
