#!/usr/bin/env python
"""실제 이사 전·후 영상의 물품 목록 비교 (분실 의심·추가 발견·수량 차이). 두 영상 모두 infer.py로
분석한 뒤 사용.

    python compare_moves.py --before outputs/<run>/video/B1/B1 --after <이사 후 영상 분석 폴더> \
        --threshold 0.62 --out outputs/move_compare_01

임베딩은 각 실행 폴더의 대표 crop으로 다시 계산한다(--embedder trained|dinov2|hist).
결과의 missing_candidate는 '분실 의심'이며 확정이 아니다. 모든 항목에 사람 확인이 필요하다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import fail, log, read_csv, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="이사 전후 물품 매칭·분실 의심 판단")
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="B3 평가에서 검증 데이터로 정한 임계값 (outputs/<run>/eval/matching/b3_eval.json)",
    )
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--embedder", default="dinov2", choices=["hist", "dinov2", "trained"])
    ap.add_argument(
        "--reid-ckpt", type=Path, default=None, help="--embedder trained 일 때 models/reid/best.pt"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    from jimssak_ai1.common import resolve_device
    from jimssak_ai1.matching.before_after import match_items
    from jimssak_ai1.reidentification.embedders import (
        DinoBackbone,
        DinoEmbedder,
        HistEmbedder,
        TrainedEmbedder,
    )
    from jimssak_ai1.video_items import embed_tracks

    dev = resolve_device(a.device)
    if a.embedder == "hist":
        E = HistEmbedder()
    else:
        bb = DinoBackbone(device=dev)
        if a.embedder == "trained":
            if not a.reid_ckpt:
                fail("--embedder trained 에는 --reid-ckpt 필요")
            E = TrainedEmbedder(bb, a.reid_ckpt)
        else:
            E = DinoEmbedder(bb)
    sides = {}
    for name, d in (("before", a.before), ("after", a.after)):
        rows = {
            r["pred_item_id"]: r for r in read_csv(d / "items.csv") if r["status"] == "confirmed"
        }
        Z = embed_tracks(d, E)
        sides[name] = [
            {"item_id": k, "class": rows[k]["category"], "emb": Z[k]} for k in rows if k in Z
        ]
        log(f"{name}: 자동 확정 물품 {len(sides[name])}개")
    res = match_items(sides["before"], sides["after"], a.threshold, a.band)
    a.out.mkdir(parents=True, exist_ok=True)
    res["note"] = "missing_candidate = 분실 의심(미관측 가능). 사용자 확인 필요"
    write_json(a.out / "move_comparison.json", res)
    from collections import Counter

    log(
        f"이전 상태: {dict(Counter(v['status'] for v in res['before'].values()))} | 이후 "
        f"상태: {dict(Counter(v['status'] for v in res['after'].values()))}"
    )
    log(f"저장: {a.out / 'move_comparison.json'}")


if __name__ == "__main__":
    main()
