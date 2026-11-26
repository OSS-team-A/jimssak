#!/usr/bin/env python
"""최종 물품 목록 채점: 사람이 작성한 정답·매칭 검수표로 TP/FP/FN, Precision/Recall/F1, 수량
절대오차를 계산한다.

규칙: docs/evaluation_rules.md (학습 전에 고정)

예) B0 목록만 비교
    python evaluate_list.py --label B0 --gt annotations/ground_truth_items.csv \
        --pred runs/B0/<run>/items.csv --review annotations/match_review_B0.csv \
        --support-csv runs/B0/<run>/category_support.csv
예) B2 (재등장 연결; 단일 트랙 물품은 B0 검수표를 그대로, 묶인 물품은 트랙 검수에서 유도)
    python evaluate_list.py --label B2 --gt ... --pred runs/B0/<run>/relink_<t>/items.csv --review
    annotations/match_review_B0.csv
예) ChatGPT
    python evaluate_list.py --label ChatGPT --pred-format chatgpt --gt ... --pred
    annotations/chatgpt_items.csv \
        --review annotations/match_review_chatgpt.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import (  # noqa: E402
    PROJECT_ROOT,
    fail,
    log,
    make_run_dir,
    now_iso,
    now_stamp,
    read_csv,
    safe_id,
    sha256_file,
    write_csv,
    write_json,
)
from jimssak_ai1.listeval import attach_review, evaluate_video, prf, validate_gt  # noqa: E402
from jimssak_ai1.taxonomy import load_taxonomy  # noqa: E402


def fmt(x):
    return "" if x is None else round(x, 4)


def load_chatgpt(rows: list[dict]) -> tuple[list[dict], set[str], dict]:
    """ChatGPT 답변(사람이 옮겨 적은 CSV) → 예측 행. 개별 근거가 없으면 해당 영상은 F1 판정 불가."""
    preds, count_only_videos = [], set()
    counts: dict = defaultdict(Counter)
    for i, r in enumerate(rows, start=1):
        v = r["video_id"]
        q = int(r.get("quantity") or 1)
        counts[v][r["category"]] += q
        has_item_evidence = (
            bool(r.get("pred_item_id"))
            and q == 1
            and bool(r.get("evidence_time_sec") or r.get("evidence_frame"))
        )
        if not has_item_evidence:
            count_only_videos.add(v)
        preds.append(
            {
                "video_id": v,
                "pred_item_id": r.get("pred_item_id") or f"C{i:04d}",
                "status": "confirmed",
                "category": r["category"],
                "box_fit": r.get("box_fit", ""),
                "first_time_sec": r.get("evidence_time_sec") or 0,
                "rep_images": r.get("evidence_frame", ""),
                "quantity": q,
            }
        )
    return preds, count_only_videos, counts


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="최종 물품 목록 채점 (TP/FP/FN, P/R/F1, 수량 절대오차)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--label", required=True, help="비교표에 쓸 버전 이름 (B0, B1, B2, ChatGPT 등)")
    ap.add_argument("--gt", required=True, type=Path, help="ground_truth_items.csv")
    ap.add_argument(
        "--pred", required=True, type=Path, nargs="+", help="예측 items.csv (여러 영상이면 여러 개)"
    )
    ap.add_argument(
        "--review", type=Path, nargs="*", default=[], help="match_review.csv (사람이 작성한 매칭)"
    )
    ap.add_argument("--pred-format", choices=["ai1", "chatgpt"], default="ai1")
    ap.add_argument(
        "--mode",
        choices=["list", "box", "both"],
        default="both",
        help="list=물품 목록만(사람이 정한 대상 집합), box=박스 필터 포함",
    )
    ap.add_argument(
        "--box-fit-csv",
        type=Path,
        default=None,
        help="AI2·실측으로 정한 예측별 박스 판정 (video_id,pred_item_id,box_fit,basis). 없으면 "
        "AI1 box 모드는 미실행",
    )
    ap.add_argument(
        "--categories", default=None, help="이 품목만 채점 (쉼표 구분). 예: 공통 지원 품목만"
    )
    ap.add_argument(
        "--support-csv",
        type=Path,
        default=None,
        help="infer 결과의 category_support.csv (FN 원인 힌트용)",
    )
    ap.add_argument(
        "--videos", default=None, help="이 video_id만 채점 (쉼표 구분). 최종 test 영상 목록 고정용"
    )
    ap.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "taxonomy.yaml")
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "evaluations")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)

    tax = load_taxonomy(args.taxonomy)
    gt = read_csv(args.gt)
    errs = validate_gt(gt, set(tax.codes))
    if errs:
        fail("정답 CSV 오류:\n  " + "\n  ".join(errs[:30]))
    if not gt:
        fail(
            f"정답 CSV에 행이 없습니다: {args.gt}  (사람이 영상을 보고 실제 물품을 먼저 기록해야 "
            f"합니다)"
        )

    raw_preds = [r for p in args.pred for r in read_csv(p)]
    count_only, chat_counts = set(), {}
    if args.pred_format == "chatgpt":
        raw_preds, count_only, chat_counts = load_chatgpt(raw_preds)
    if args.box_fit_csv:
        bf = {(r["video_id"], r["pred_item_id"]): r["box_fit"] for r in read_csv(args.box_fit_csv)}
        for r in raw_preds:
            if (r["video_id"], r["pred_item_id"]) in bf:
                r["box_fit"] = bf[(r["video_id"], r["pred_item_id"])]
    reviews = [r for p in args.review for r in read_csv(p)]
    preds, warns = attach_review(raw_preds, reviews)

    videos = sorted({g["video_id"] for g in gt})
    if args.videos:
        sel = {v.strip() for v in args.videos.split(",") if v.strip()}
        missing = sel - set(videos)
        if missing:
            fail(f"정답 CSV에 없는 video_id: {sorted(missing)}")
        videos = sorted(sel)
    cats = {c.strip() for c in args.categories.split(",")} if args.categories else None
    unsupported = None
    if args.support_csv:
        unsupported = {r["category"] for r in read_csv(args.support_csv) if r["supported"] != "yes"}

    out = make_run_dir(args.output_root, args.run_name or f"{now_stamp()}_{safe_id(args.label)}")
    modes = ["list", "box"] if args.mode == "both" else [args.mode]
    summary = {
        "label": args.label,
        "created_at": now_iso(),
        "inputs": {
            "gt": str(args.gt),
            "gt_sha256": sha256_file(args.gt),
            "pred": [str(p) for p in args.pred],
            "review": [str(p) for p in args.review],
            "pred_format": args.pred_format,
            "categories_filter": sorted(cats) if cats else None,
            "videos": videos,
        },
        "rules": "docs/evaluation_rules.md",
        "warnings": warns[:50],
        "modes": {},
    }

    for mode in modes:
        has_box_claims = any((p.get("box_fit") or "") in ("fits", "no_fit") for p in preds)
        if mode == "box" and not has_box_claims:
            summary["modes"][mode] = {
                "status": "미실행",
                "reason": "예측에 박스 적합 판정이 없음 (AI1 단독은 박스 판정을 하지 않음)",
            }
            continue
        per_video, per_cat_rows, cases, count_rows = [], [], [], []
        tot = Counter()
        cat_tot: dict = defaultdict(Counter)
        f1_judgeable_all = True
        for v in videos:
            res = evaluate_video(
                v,
                gt,
                preds,
                reviews,
                mode=mode,
                categories=cats,
                unsupported_categories=unsupported,
            )
            judge = res.f1_judgeable and v not in count_only
            f1_judgeable_all &= judge
            p_, r_, f_ = prf(res.tp, res.fp, res.fn)
            per_video.append(
                {
                    "video_id": v,
                    "n_gt_target": res.n_gt_target,
                    "tp": res.tp if judge else "판정 불가",
                    "fp": res.fp if judge else "판정 불가",
                    "fn": res.fn if judge else "판정 불가",
                    "fp_duplicate": res.fp_duplicate if judge else "판정 불가",
                    "fp_nonexistent": res.fp_nonexistent if judge else "판정 불가",
                    "fp_misclass": res.fp_misclass if judge else "판정 불가",
                    "fp_box_fit": res.fp_box_fit,
                    "fp_unreviewed": res.fp_unreviewed,
                    "wrong_merge": res.wrong_merge if judge else "판정 불가",
                    "precision": fmt(p_) if judge else "판정 불가",
                    "recall": fmt(r_) if judge else "판정 불가",
                    "f1": fmt(f_) if judge else "판정 불가",
                    "needs_review_total": res.needs_review_total,
                    "needs_review_target": res.needs_review_target,
                    "out_of_scope_preds": res.out_of_scope,
                    "n_gt_box_fit_unknown": res.n_gt_unknown,
                }
            )
            if judge:
                for k in (
                    "tp",
                    "fn",
                    "fp_duplicate",
                    "fp_nonexistent",
                    "fp_misclass",
                    "fp_box_fit",
                    "fp_unreviewed",
                    "wrong_merge",
                    "needs_review_total",
                    "needs_review_target",
                    "out_of_scope",
                ):
                    tot[k] += getattr(res, k)
                tot["fp"] += res.fp
                for c, cnt in res.per_cat.items():
                    cat_tot[c].update(cnt)
            tot["n_gt_target"] += res.n_gt_target
            cases += res.cases
            # 수량 절대오차
            gt_c = Counter(
                g["category"]
                for g in gt
                if g["video_id"] == v
                and g["box_fit"] == "no_fit"
                and (cats is None or g["category"] in cats)
            )
            if args.pred_format == "chatgpt" and v in count_only:
                pc = Counter(
                    {c: n for c, n in chat_counts.get(v, {}).items() if cats is None or c in cats}
                )
                if mode == "box":
                    pc = Counter()
                    for p in preds:
                        if (
                            p["video_id"] == v
                            and p.get("box_fit") == "no_fit"
                            and (cats is None or p["category"] in cats)
                        ):
                            pc[p["category"]] += int(p.get("quantity") or 1)
            else:
                pc = Counter({c: n["pred"] for c, n in res.per_cat.items() if n.get("pred")})
            for c in sorted(set(gt_c) | set(pc)):
                if not c:
                    continue
                count_rows.append(
                    {
                        "video_id": v,
                        "category": c,
                        "pred_count": pc.get(c, 0),
                        "gt_count": gt_c.get(c, 0),
                        "abs_error": abs(pc.get(c, 0) - gt_c.get(c, 0)),
                    }
                )
        for c, cnt in sorted(cat_tot.items()):
            if not c:
                continue
            per_cat_rows.append(
                {
                    "category": c,
                    "category_ko": tax.ko(c),
                    "n_gt_target": cnt["gt"],
                    "tp": cnt["tp"],
                    "fn": cnt["fn"],
                    "pred_counted": cnt["pred"],
                    "recall": fmt(cnt["tp"] / cnt["gt"]) if cnt["gt"] else "",
                    "supported_by_model": (
                        "no(미지원)"
                        if unsupported and c in unsupported
                        else ("yes" if unsupported else "")
                    ),
                }
            )
        p_, r_, f_ = prf(tot["tp"], tot["fp"], tot["fn"])
        mae = (sum(r["abs_error"] for r in count_rows) / len(count_rows)) if count_rows else None
        msum = {
            "status": "완료",
            "n_videos": len(videos),
            "n_gt_target": tot["n_gt_target"],
            "f1_judgeable_all_videos": f1_judgeable_all,
            "tp": tot["tp"],
            "fp": tot["fp"],
            "fn": tot["fn"],
            "fp_duplicate": tot["fp_duplicate"],
            "fp_nonexistent": tot["fp_nonexistent"],
            "fp_misclass": tot["fp_misclass"],
            "fp_box_fit": tot["fp_box_fit"],
            "fp_unreviewed": tot["fp_unreviewed"],
            "wrong_merge": tot["wrong_merge"],
            "precision": fmt(p_),
            "recall": fmt(r_),
            "f1": fmt(f_),
            "needs_review_total": tot["needs_review_total"],
            "needs_review_target": tot["needs_review_target"],
            "out_of_scope_preds": tot["out_of_scope"],
            "count_mae": fmt(mae),
            "count_mae_scope": "각 영상의 (정답 대상 품목 ∪ 예측 품목) 모든 영상×품목 쌍",
            "n_count_pairs": len(count_rows),
        }
        if not f1_judgeable_all:
            msum["note"] = (
                "일부 영상은 개별 근거가 없어 F1을 판정하지 못함 → TP/FP/FN·F1은 판정 가능한 "
                "영상만 합산"
            )
        summary["modes"][mode] = msum
        write_csv(
            out / f"{mode}_per_video.csv",
            per_video,
            list(per_video[0].keys()) if per_video else ["video_id"],
        )
        write_csv(
            out / f"{mode}_per_category.csv",
            per_cat_rows,
            [
                "category",
                "category_ko",
                "n_gt_target",
                "tp",
                "fn",
                "pred_counted",
                "recall",
                "supported_by_model",
            ],
        )
        write_csv(
            out / f"{mode}_count_errors.csv",
            count_rows,
            ["video_id", "category", "pred_count", "gt_count", "abs_error"],
        )
        write_csv(
            out / f"{mode}_failure_cases.csv",
            cases,
            [
                "video_id",
                "outcome",
                "pred_item_id",
                "pred_category",
                "gt_item_id",
                "gt_category",
                "first_time_sec",
                "reviewer_error_type",
                "auto_hint",
                "rep_images",
            ],
        )

    unk = [g for g in gt if g["box_fit"] == "unknown" and g["video_id"] in videos]
    write_csv(out / "gt_box_fit_unknown.csv", unk, list(gt[0].keys()))
    write_json(out / "summary.json", summary)
    log(f"채점 완료: {out}")
    for m, s in summary["modes"].items():
        if s.get("status") != "완료":
            log(f"  [{m}] {s['status']}: {s['reason']}")
            continue
        log(
            f"  [{m}] 정답 대상 {s['n_gt_target']} | TP {s['tp']} FP {s['fp']} "
            f"(중복 {s['fp_duplicate']}, 오검출 {s['fp_nonexistent']}, "
            f"혼동 {s['fp_misclass']}, 미검수 {s['fp_unreviewed']}) FN {s['fn']} | 잘못된 "
            f"병합 {s['wrong_merge']} | "
            f"P {s['precision']} R {s['recall']} F1 {s['f1']} | 수량 MAE {s['count_mae']} | 확인 "
            f"요청 {s['needs_review_total']}"
        )
    if warns:
        log(f"  경고 {len(warns)}건 (summary.json warnings 참고). 예: {warns[0]}")


if __name__ == "__main__":
    main()
