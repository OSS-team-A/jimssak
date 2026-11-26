"""이사 전후 물품 매칭과 분실 의심 판단 (B3).

분실은 '탐지 클래스'가 아니라 전후 물품 인스턴스 비교 문제로 정의한다.
1) 전/후 물품 목록과 물품별 임베딩  2) 같은 클래스끼리 코사인 유사도  3) Hungarian 일대일 매칭
4) 유사도 ≥ thr → matched, [thr - band, thr) → uncertain(사람 확인), 그 미만 → 미매칭
5) 미매칭 이전 물품 → missing_candidate (분실 '의심' — 확정 아님, 사용자 확인 필요)
6) 미매칭 이후 물품 → unexpected_item   7) 클래스별 수량 차이 → quantity_mismatch 표시
영상에 안 보였다는 이유만으로 분실을 확정하지 않는다: 모든 missing_candidate는 review_required=True.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_items(
    before: list[dict], after: list[dict], thr: float, band: float = 0.05, use_class: bool = True
) -> dict:
    """before/after: [{'item_id', 'class', 'emb'(L2 정규화)}]"""
    nb, na = len(before), len(after)
    S = np.full((nb, na), -1.0)
    for i, b in enumerate(before):
        for j, a in enumerate(after):
            if use_class and b["class"] != a["class"]:
                continue
            S[i, j] = float(np.dot(b["emb"], a["emb"]))
    res_b = {
        b["item_id"]: {
            "status": "missing_candidate",
            "matched_item_id": None,
            "matching_confidence": None,
            "review_required": True,
            "review_reason": "이사 후 영상에서 같은 물품을 찾지 못함(미관측일 수 있음)",
        }
        for b in before
    }
    res_a = {
        a["item_id"]: {
            "status": "unexpected_item",
            "matched_item_id": None,
            "matching_confidence": None,
            "review_required": True,
            "review_reason": "이사 전 목록에 없는 물품",
        }
        for a in after
    }
    if nb and na:
        r, c = linear_sum_assignment(-S)
        for i, j in zip(r, c):
            s = S[i, j]
            if s < 0:
                continue
            bid, aid = before[i]["item_id"], after[j]["item_id"]
            if s >= thr:
                res_b[bid] = {
                    "status": "matched",
                    "matched_item_id": aid,
                    "matching_confidence": round(s, 4),
                    "review_required": False,
                    "review_reason": "",
                }
                res_a[aid] = {
                    "status": "matched",
                    "matched_item_id": bid,
                    "matching_confidence": round(s, 4),
                    "review_required": False,
                    "review_reason": "",
                }
            elif s >= thr - band:
                why = f"유사도 {s:.3f}가 판단 경계({thr - band:.3f}~{thr:.3f}) 안"
                res_b[bid] = {
                    "status": "uncertain",
                    "matched_item_id": aid,
                    "matching_confidence": round(s, 4),
                    "review_required": True,
                    "review_reason": why,
                }
                res_a[aid] = {
                    "status": "uncertain",
                    "matched_item_id": bid,
                    "matching_confidence": round(s, 4),
                    "review_required": True,
                    "review_reason": why,
                }
    cb = Counter(b["class"] for b in before)
    ca = Counter(a["class"] for a in after)
    qty = {
        c: {"before": cb.get(c, 0), "after": ca.get(c, 0), "diff": ca.get(c, 0) - cb.get(c, 0)}
        for c in sorted(set(cb) | set(ca))
    }
    mism = {c for c, v in qty.items() if v["diff"] != 0}
    for b in before:
        res_b[b["item_id"]]["quantity_mismatch"] = b["class"] in mism
    for a in after:
        res_a[a["item_id"]]["quantity_mismatch"] = a["class"] in mism
    return {
        "before": res_b,
        "after": res_a,
        "quantity": qty,
        "threshold": thr,
        "uncertain_band": band,
    }


def count_only_baseline(before: list[dict], after: list[dict]) -> dict:
    """기준선: 클래스별 수량만 비교. 부족한 수만큼 이전 물품을 (근거 없이) 앞에서부터 분실 의심으로
    표시.
    """
    ca = Counter(a["class"] for a in after)
    used = Counter()
    res_b = {}
    for b in before:
        c = b["class"]
        if used[c] < ca.get(c, 0):
            used[c] += 1
            res_b[b["item_id"]] = {"status": "matched", "matched_item_id": None}
        else:
            res_b[b["item_id"]] = {"status": "missing_candidate", "matched_item_id": None}
    cb = Counter(b["class"] for b in before)
    seen = Counter()
    res_a = {}
    for a in after:
        c = a["class"]
        seen[c] += 1
        res_a[a["item_id"]] = {
            "status": "unexpected_item" if seen[c] > cb.get(c, 0) else "matched",
            "matched_item_id": None,
        }
    return {"before": res_b, "after": res_a}


def score_scenario(
    result: dict, truth_missing: set, truth_unexpected: set, truth_pairs: dict
) -> dict:
    """truth_pairs: before_id -> after_id (실제 같은 물품)."""
    pb = result["before"]
    pa = result["after"]
    flagged = {k for k, v in pb.items() if v["status"] == "missing_candidate"}
    flagged_u = {k for k, v in pa.items() if v["status"] == "unexpected_item"}
    present = set(pb) - truth_missing
    tp = len(flagged & truth_missing)
    fp = len(flagged - truth_missing)
    fn = len(truth_missing - flagged)
    utp = len(flagged_u & truth_unexpected)
    ufp = len(flagged_u - truth_unexpected)
    ufn = len(truth_unexpected - flagged_u)
    correct_pairs = sum(
        1
        for b, a in truth_pairs.items()
        if pb.get(b, {}).get("status") == "matched" and pb[b].get("matched_item_id") == a
    )
    unc = {k for k, v in pb.items() if v["status"] == "uncertain"}
    return {
        "missing_tp": tp,
        "missing_fp": fp,
        "missing_fn": fn,
        "unexpected_tp": utp,
        "unexpected_fp": ufp,
        "unexpected_fn": ufn,
        "present_items": len(present),
        "false_missing": len(flagged & present),
        "true_pairs": len(truth_pairs),
        "correct_pairs": correct_pairs,
        "uncertain": len(unc),
        "uncertain_truly_missing": len(unc & truth_missing),
    }


def summarize(scores: list[dict]) -> dict:
    s = Counter()
    for x in scores:
        s.update(x)
    p = s["missing_tp"] / max(1, s["missing_tp"] + s["missing_fp"])
    r = s["missing_tp"] / max(1, s["missing_tp"] + s["missing_fn"])
    up = s["unexpected_tp"] / max(1, s["unexpected_tp"] + s["unexpected_fp"])
    ur = s["unexpected_tp"] / max(1, s["unexpected_tp"] + s["unexpected_fn"])
    has_missing = s["missing_tp"] + s["missing_fn"] > 0
    return {
        "scenarios": len(scores),
        "missing_precision": round(p, 4),
        "missing_recall": round(r, 4) if has_missing else None,
        "missing_f1": round(2 * p * r / max(1e-9, p + r), 4) if has_missing else None,
        "false_missing_rate": round(s["false_missing"] / max(1, s["present_items"]), 4),
        "missed_missing_rate": round(1 - r, 4) if has_missing else None,
        "unexpected_precision": round(up, 4),
        "unexpected_recall": round(ur, 4),
        "match_accuracy": round(s["correct_pairs"] / max(1, s["true_pairs"]), 4),
        "uncertain_items": s["uncertain"],
        "uncertain_truly_missing": s["uncertain_truly_missing"],
        **{k: int(v) for k, v in s.items()},
    }
