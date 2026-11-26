#!/usr/bin/env python
"""매칭 검수표(match_review.csv) 생성: 예측 물품마다 한 행, 근거 이미지·시각을 미리 채우고
사람이 gt_item_id(정답 물품 ID)를 적도록 비워 둔다. 모델 출력은 정답 칸에 복사하지 않는다.

예) python make_review_sheet.py --items runs/B0/<run>/items.csv --out
annotations/match_review_B0.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import PROJECT_ROOT, fail, log, read_csv, write_csv  # noqa: E402


def _portable(p: Path) -> str:
    """프로젝트 안의 경로는 프로젝트 루트 기준 상대 경로로 적어 PC·서버 어디서든 같은 뜻이 되게
    한다.
    """
    try:
        return p.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


REVIEW_FIELDS = [
    "video_id",
    "pred_source",
    "pred_item_id",
    "pred_status",
    "pred_category",
    "pred_model_class",
    "first_time_sec",
    "last_time_sec",
    "rep_time_sec",
    "rep_images",
    "track_ids",
    # ↓ 사람이 채우는 칸
    "gt_item_id",
    "extra_gt_item_ids",
    "error_type",
    "reviewer",
    "review_note",
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="예측 → 정답 물품 매칭 검수표 생성 (사람이 gt_item_id를 채운다)"
    )
    ap.add_argument(
        "--items",
        required=True,
        type=Path,
        nargs="+",
        help="infer.py 또는 link_reappearance.py의 items.csv",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--include-needs-review",
        action="store_true",
        default=True,
        help="확인 필요 항목도 검수 행에 포함 (자동 집계에는 들어가지 않지만 확인 요청 중 실제 "
        "대상 수를 세기 위함)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        fail(
            f"이미 있습니다: {args.out} (사람이 채운 내용을 지우지 않도록 덮어쓰지 않음. 필요하면 "
            f"--overwrite)"
        )
    rows = []
    for p in args.items:
        base = p.resolve().parent
        for r in read_csv(p):
            if r.get("status") == "needs_review" and not args.include_needs_review:
                continue
            rows.append(
                {
                    "video_id": r["video_id"],
                    "pred_source": r.get("run_id", ""),
                    "pred_item_id": r["pred_item_id"],
                    "pred_status": r.get("status", ""),
                    "pred_category": r.get("category", ""),
                    "pred_model_class": r.get("model_class", ""),
                    "first_time_sec": r.get("first_time_sec", ""),
                    "last_time_sec": r.get("last_time_sec", ""),
                    "rep_time_sec": r.get("rep_time_sec", ""),
                    "rep_images": ";".join(
                        _portable(base / x) for x in r.get("rep_images", "").split(";") if x
                    ),
                    "track_ids": r.get("track_ids", ""),
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows, REVIEW_FIELDS)
    log(f"검수표 {len(rows)}행 생성: {args.out}")
    log(
        "  각 행의 rep_images를 열어 보고 gt_item_id 칸에 ground_truth_items.csv의 item_id를 "
        "적으세요."
    )
    log(
        "  실재하지 않는 물체면 gt_item_id를 비우고 error_type=오검출. 놓친 정답은 pred_item_id를 "
        "비운 새 행으로 추가."
    )


if __name__ == "__main__":
    main()
