"""최종 물품 목록 채점 (발표 핵심 지표).

규칙은 docs/evaluation_rules.md 에 고정되어 있다. 요약:
- 예측 물품 ↔ 실제 물품을 사람이 match_review.csv 로 1:1 연결한 근거만 사용한다.
- 맞는 실물·품목을 처음 한 번 연결 = TP. 같은 실물을 또 연결 = FP(중복).
  실재하지 않는 항목 = FP(오검출). 품목이 다르면 FP(클래스 혼동)이고 그 실물은 다른 TP가 없으면 FN.
- 한 예측이 여러 실물을 덮으면(extra_gt_item_ids) 잘못된 병합 건수로 센다. 추가 실물은 TP가 되지
  않는다.
- status=needs_review(확인 필요) 예측은 자동 목록에 넣지 않는다 → TP도 FP도 아님. 건수는 따로 보고.
- 정답 box_fit=unknown 물품은 대상 집합에서 빼고 별도 표로 남긴다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

GT_REQUIRED = ["video_id", "item_id", "category", "box_fit"]
BOX_FIT_VALUES = {"fits", "no_fit", "unknown"}

ERROR_TYPES = [
    "미지원 품목",
    "탐지 누락",
    "클래스 혼동",
    "마스크 오류",
    "추적 끊김",
    "재등장 중복",
    "잘못된 병합",
    "오검출",
    "기타",
]


def split_ids(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.replace(",", ";").split(";") if x.strip()]


@dataclass
class EvalResult:
    video_id: str
    tp: int = 0
    fp_nonexistent: int = 0
    fp_duplicate: int = 0
    fp_misclass: int = 0
    fp_box_fit: int = 0
    fp_unreviewed: int = 0
    fn: int = 0
    wrong_merge: int = 0
    out_of_scope: int = 0
    needs_review_total: int = 0
    needs_review_target: int = 0
    n_gt_target: int = 0
    n_gt_unknown: int = 0
    n_pred_counted: int = 0
    f1_judgeable: bool = True
    cases: list[dict] = field(default_factory=list)
    per_cat: dict = field(default_factory=lambda: defaultdict(Counter))

    @property
    def fp(self) -> int:
        return (
            self.fp_nonexistent
            + self.fp_duplicate
            + self.fp_misclass
            + self.fp_box_fit
            + self.fp_unreviewed
        )


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = tp / (tp + fp) if (tp + fp) > 0 else None
    r = tp / (tp + fn) if (tp + fn) > 0 else None
    f = (
        2 * p * r / (p + r)
        if (p is not None and r is not None and (p + r) > 0)
        else (0.0 if p is not None and r is not None else None)
    )
    return p, r, f


def validate_gt(gt_rows: list[dict], known_categories: set[str]) -> list[str]:
    errs = []
    seen = set()
    for i, r in enumerate(gt_rows, start=2):
        for k in GT_REQUIRED:
            if not r.get(k):
                errs.append(f"ground_truth {i}행: '{k}' 값이 비어 있음")
        key = (r.get("video_id"), r.get("item_id"))
        if key in seen:
            errs.append(f"ground_truth {i}행: 같은 video_id·item_id 중복 {key}")
        seen.add(key)
        if r.get("box_fit") and r["box_fit"] not in BOX_FIT_VALUES:
            errs.append(
                f"ground_truth {i}행: box_fit은 fits/no_fit/unknown 중 하나여야 함 (현재 "
                f"'{r['box_fit']}')"
            )
        if r.get("category") and r["category"] not in known_categories:
            errs.append(
                f"ground_truth {i}행: category '{r['category']}' 가 configs/taxonomy.yaml에 없음"
            )
    return errs


def attach_review(pred_rows: list[dict], review_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """예측 행에 사람이 작성한 매칭(gt_item_id, extra_gt_item_ids)을 붙인다.

    재등장 연결(B2) 결과처럼 예측 물품이 여러 트랙(T####)을 묶은 경우, 묶음 단위 검수가 없으면
    트랙 단위 검수(B0 검수표)에서 자동으로 유도한다:
      - 구성 트랙들의 gt_item_id가 모두 같으면 그 실물
      - 서로 다르면 가장 프레임이 많은 트랙의 실물을 대표로, 나머지 실물은 extra(잘못된 병합)
    """
    by_pred = {}
    for r in review_rows:
        pid = r.get("pred_item_id", "")
        if pid:
            by_pred[(r.get("video_id", ""), pid)] = r
    warnings = []
    out = []
    for p in pred_rows:
        p = dict(p)
        key = (p["video_id"], p["pred_item_id"])
        rv = by_pred.get(key)
        if rv is not None:
            p["_reviewed"] = True
            p["_gt"] = rv.get("gt_item_id", "")
            p["_extra"] = split_ids(rv.get("extra_gt_item_ids"))
            p["_error_type"] = rv.get("error_type", "")
        else:
            tracks = split_ids(p.get("track_ids"))
            members = [
                by_pred.get((p["video_id"], f"T{int(t):05d}")) for t in tracks if t.isdigit()
            ]
            if tracks and members and all(m is not None for m in members):
                frames = [int(x) for x in split_ids(p.get("track_n_frames"))] or [1] * len(members)
                gts = [m.get("gt_item_id", "") for m in members]
                extras = [e for m in members for e in split_ids(m.get("extra_gt_item_ids"))]
                order = sorted(
                    range(len(members)), key=lambda i: -frames[i] if i < len(frames) else 0
                )
                primary = gts[order[0]]
                others = sorted({g for g in gts if g and g != primary} | set(extras))
                p["_reviewed"] = True
                p["_gt"] = primary
                p["_extra"] = others
                p["_error_type"] = "잘못된 병합" if others else ""
                p["_derived_from_tracks"] = True
            else:
                p["_reviewed"] = False
                p["_gt"] = ""
                p["_extra"] = []
                p["_error_type"] = ""
                warnings.append(f"검수되지 않은 예측: {p['video_id']} {p['pred_item_id']}")
        out.append(p)
    return out, warnings


def evaluate_video(
    video_id: str,
    gt_rows: list[dict],
    pred_rows: list[dict],
    review_rows: list[dict],
    mode: str = "list",
    categories: set[str] | None = None,
    unsupported_categories: set[str] | None = None,
) -> EvalResult:
    """한 영상 채점. mode='list'(물품 목록만) 또는 'box'(박스 필터 포함).

    pred_rows는 attach_review를 거친 행이어야 한다.
    """
    res = EvalResult(video_id=video_id)
    gts = [g for g in gt_rows if g["video_id"] == video_id]
    if categories is not None:
        gts = [g for g in gts if g["category"] in categories]
    gt_by_id = {g["item_id"]: g for g in gts}
    all_gt_ids = {g["item_id"] for g in gt_rows if g["video_id"] == video_id}
    targets = {g["item_id"] for g in gts if g["box_fit"] == "no_fit"}
    res.n_gt_target = len(targets)
    res.n_gt_unknown = sum(1 for g in gts if g["box_fit"] == "unknown")

    preds = [p for p in pred_rows if p["video_id"] == video_id]
    if categories is not None:
        preds = [p for p in preds if p.get("category") in categories]
    preds.sort(key=lambda p: (float(p.get("first_time_sec") or 0), p["pred_item_id"]))

    claimed: dict[str, str] = {}
    misclass_gt: set[str] = set()
    merged_away: set[str] = set()
    needs_review_gt: set[str] = set()

    for p in preds:
        status = p.get("status", "confirmed") or "confirmed"
        g = p.get("_gt", "")
        if g and g not in all_gt_ids:
            raise ValueError(
                f"match_review의 gt_item_id '{g}' 가 {video_id}의 정답 목록에 없습니다 "
                f"(pred {p['pred_item_id']})"
            )
        if status == "needs_review":
            res.needs_review_total += 1
            if g in targets:
                res.needs_review_target += 1
                needs_review_gt.add(g)
            continue
        if mode == "box":
            if (p.get("box_fit") or "") != "no_fit":
                continue  # 박스 필터를 통과하지 못한(또는 판정이 없는) 예측은 최종 목록에 없음
        case = {
            "video_id": video_id,
            "pred_item_id": p["pred_item_id"],
            "pred_category": p.get("category", ""),
            "gt_item_id": g,
            "gt_category": gt_by_id.get(g, {}).get("category", ""),
            "first_time_sec": p.get("first_time_sec", ""),
            "rep_images": p.get("rep_images", ""),
            "reviewer_error_type": p.get("_error_type", ""),
        }
        if not p.get("_reviewed"):
            res.fp_unreviewed += 1
            res.f1_judgeable = False
            case["outcome"] = "FP(미검수)"
            res.cases.append(case)
            res.n_pred_counted += 1
            res.per_cat[p.get("category", "")]["pred"] += 1
            continue
        if g and g not in gt_by_id:
            # categories 필터로 범위 밖이 된 실물
            res.out_of_scope += 1
            continue
        if g and gt_by_id[g]["box_fit"] != "no_fit":
            if mode == "box" and gt_by_id[g]["box_fit"] == "fits":
                res.fp_box_fit += 1
                case["outcome"] = "FP(박스 적합 오판)"
                res.cases.append(case)
                res.n_pred_counted += 1
                res.per_cat[p.get("category", "")]["pred"] += 1
            else:
                res.out_of_scope += 1
            continue
        res.n_pred_counted += 1
        res.per_cat[p.get("category", "")]["pred"] += 1
        extras = [e for e in p.get("_extra", []) if e in targets]
        if extras:
            res.wrong_merge += len(extras)
            merged_away.update(extras)
        if not g:
            res.fp_nonexistent += 1
            case["outcome"] = "FP(오검출·실재하지 않음)"
        elif p.get("category") != gt_by_id[g]["category"]:
            res.fp_misclass += 1
            misclass_gt.add(g)
            res.per_cat[p.get("category", "")]["fp"] += 1
            case["outcome"] = "FP(클래스 혼동)"
        elif g in claimed:
            res.fp_duplicate += 1
            res.per_cat[p.get("category", "")]["fp"] += 1
            case["outcome"] = f"FP(중복: {claimed[g]}와 같은 실물)"
        else:
            claimed[g] = p["pred_item_id"]
            res.tp += 1
            res.per_cat[p.get("category", "")]["tp"] += 1
            case["outcome"] = "TP"
        if extras:
            case["outcome"] += f" + 잘못된 병합({';'.join(extras)})"
        if not case["outcome"].startswith("TP") or extras:
            res.cases.append(case)

    miss_causes = {}
    for r in review_rows:
        if r.get("video_id") == video_id and not r.get("pred_item_id") and r.get("gt_item_id"):
            miss_causes[r["gt_item_id"]] = r.get("error_type", "")
    for g in sorted(targets):
        cat = gt_by_id[g]["category"]
        res.per_cat[cat]["gt"] += 1
        if g in claimed:
            continue
        res.fn += 1
        res.per_cat[cat]["fn"] += 1
        hints = []
        if unsupported_categories and cat in unsupported_categories:
            hints.append("미지원 품목")
        if g in misclass_gt:
            hints.append("클래스 혼동")
        if g in merged_away:
            hints.append("잘못된 병합")
        if g in needs_review_gt:
            hints.append("확인 필요로만 존재")
        res.cases.append(
            {
                "video_id": video_id,
                "pred_item_id": "",
                "pred_category": "",
                "gt_item_id": g,
                "gt_category": cat,
                "first_time_sec": gt_by_id[g].get("rep_time_sec", ""),
                "rep_images": "",
                "reviewer_error_type": miss_causes.get(g, ""),
                "outcome": "FN(누락)",
                "auto_hint": ";".join(hints),
            }
        )
    for c, cnt in res.per_cat.items():
        cnt.setdefault("gt", 0)
    return res


def count_errors(per_video_counts: dict[str, dict[str, tuple[int, int]]]) -> list[dict]:
    """영상×품목 수량 절대오차. 입력: {video: {category: (pred, gt)}}.

    평균 범위(고정): 각 영상에서 '정답 대상 품목 ∪ 예측 품목'의 모든 (영상, 품목) 쌍.
    """
    rows = []
    for v, cats in sorted(per_video_counts.items()):
        for c, (pc, gc) in sorted(cats.items()):
            rows.append(
                {
                    "video_id": v,
                    "category": c,
                    "pred_count": pc,
                    "gt_count": gc,
                    "abs_error": abs(pc - gc),
                }
            )
    return rows
