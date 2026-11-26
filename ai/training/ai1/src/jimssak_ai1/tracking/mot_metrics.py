"""다중 객체 추적 지표: HOTA(DetA·AssA), IDF1, CLEAR(MOTA·ID Switches).

TrackEval(https://github.com/JonathonLuiten/TrackEval)의 HOTA/Identity/CLEAR 정의를 따라 직접
구현했다.
입력: 프레임별 gt [(gt_id, box)], tr [(tr_id, box)] — 상자 IoU 기반, 클래스 무관.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..frameeval import box_iou_matrix


def _ids(frames, key):
    s = sorted({i for f in frames for i, _ in f[key]})
    return {v: k for k, v in enumerate(s)}


def compute_mot(frames: list[dict]) -> dict:
    """frames: [{'gt': [(id, box)], 'tr': [(id, box)]}, ...]"""
    gmap, tmap = _ids(frames, "gt"), _ids(frames, "tr")
    G, T = len(gmap), len(tmap)
    n_gt_det = sum(len(f["gt"]) for f in frames)
    n_tr_det = sum(len(f["tr"]) for f in frames)
    if n_gt_det == 0:
        return {"note": "정답 객체 없음"}
    sims = []
    for f in frames:
        gb = np.array([b for _, b in f["gt"]]) if f["gt"] else np.zeros((0, 4))
        tb = np.array([b for _, b in f["tr"]]) if f["tr"] else np.zeros((0, 4))
        sims.append(box_iou_matrix(gb, tb) if len(gb) and len(tb) else np.zeros((len(gb), len(tb))))
    # ---------- HOTA ----------
    alphas = np.arange(0.05, 0.99, 0.05)
    pot = np.zeros((G, T))
    gcnt = np.zeros(G)
    tcnt = np.zeros(T)
    for f, S in zip(frames, sims):
        gi = [gmap[i] for i, _ in f["gt"]]
        ti = [tmap[i] for i, _ in f["tr"]]
        if len(gi) and len(ti):
            denom = S.sum(0)[None, :] + S.sum(1)[:, None] - S
            pot[np.ix_(gi, ti)] += np.where(denom > 0, S / np.maximum(denom, 1e-12), 0)
        gcnt[gi] += 1
        tcnt[ti] += 1
    gas = pot / np.maximum(gcnt[:, None] + tcnt[None, :] - pot, 1e-12)
    tp = np.zeros(len(alphas))
    fn = np.zeros(len(alphas))
    fp = np.zeros(len(alphas))
    mc = np.zeros((len(alphas), G, T))
    for f, S in zip(frames, sims):
        gi = [gmap[i] for i, _ in f["gt"]]
        ti = [tmap[i] for i, _ in f["tr"]]
        if not gi or not ti:
            fn += len(gi)
            fp += len(ti)
            continue
        score = gas[np.ix_(gi, ti)] * S
        r, c = linear_sum_assignment(-score)
        for a_i, a in enumerate(alphas):
            ok = S[r, c] >= a - np.finfo(float).eps
            m = int(ok.sum())
            tp[a_i] += m
            fn[a_i] += len(gi) - m
            fp[a_i] += len(ti) - m
            for rr, cc in zip(r[ok], c[ok]):
                mc[a_i, gi[rr], ti[cc]] += 1
    det_a = tp / np.maximum(tp + fn + fp, 1)
    ass_a = np.zeros(len(alphas))
    for a_i in range(len(alphas)):
        M = mc[a_i]
        ass = M / np.maximum(gcnt[:, None] + tcnt[None, :] - M, 1)
        ass_a[a_i] = (M * ass).sum() / max(1, tp[a_i])
    hota = np.sqrt(det_a * ass_a)
    # ---------- Identity (IDF1) ----------
    match = np.zeros((G, T))
    for f, S in zip(frames, sims):
        gi = [gmap[i] for i, _ in f["gt"]]
        ti = [tmap[i] for i, _ in f["tr"]]
        if gi and ti:
            match[np.ix_(gi, ti)] += S >= 0.5
    if G and T:
        r, c = linear_sum_assignment(-match)
        idtp = match[r, c].sum()
    else:
        idtp = 0.0
    idf1 = 2 * idtp / max(1, n_gt_det + n_tr_det)
    # ---------- CLEAR ----------
    prev = {}
    c_tp = c_fp = c_fn = idsw = 0
    for f, S in zip(frames, sims):
        g_ids = [i for i, _ in f["gt"]]
        t_ids = [i for i, _ in f["tr"]]
        if not g_ids or not t_ids:
            c_fn += len(g_ids)
            c_fp += len(t_ids)
            continue
        score = S.copy()
        for a, g in enumerate(g_ids):
            for b, t in enumerate(t_ids):
                if prev.get(g) == t and S[a, b] >= 0.5:
                    score[a, b] += 1000
        r, c = linear_sum_assignment(-score)
        ok = S[r, c] >= 0.5
        for rr, cc in zip(r[ok], c[ok]):
            g, t = g_ids[rr], t_ids[cc]
            if g in prev and prev[g] != t:
                idsw += 1
            prev[g] = t
        m = int(ok.sum())
        c_tp += m
        c_fn += len(g_ids) - m
        c_fp += len(t_ids) - m
    mota = 1 - (c_fn + c_fp + idsw) / n_gt_det
    return {
        "HOTA": round(float(hota.mean()), 4),
        "DetA": round(float(det_a.mean()), 4),
        "AssA": round(float(ass_a.mean()), 4),
        "IDF1": round(float(idf1), 4),
        "MOTA": round(float(mota), 4),
        "IDSW": int(idsw),
        "CLEAR_TP": int(c_tp),
        "CLEAR_FP": int(c_fp),
        "CLEAR_FN": int(c_fn),
        "gt_ids": G,
        "tracker_ids": T,
        "gt_dets": n_gt_det,
        "tracker_dets": n_tr_det,
    }


def item_level(
    frames: list[dict],
    item_of_track: dict,
    gt_class: dict,
    item_class: dict,
    listable_gt: set,
    counted_items: set,
) -> dict:
    """서비스 관점 물품 목록 지표 (합성 영상 정답 기준).

    - 예측 물품 = 확정 조건을 넘은 트랙 묶음(counted_items). 매칭 프레임(IoU≥0.5) 다수결로 정답
      객체에 연결.
    - 중복: 이미 다른 물품이 차지한 정답에 연결된 물품. 잘못된 병합: 한 물품의 매칭 프레임 중 두
      번째로 많은 정답 비율 ≥ 20%.
    - 재등장 연결 정확도: 화면에서 사라졌다 다시 나타난 정답 객체 중, 모든 매칭 프레임이 한 물품에
      속한 비율.
    """
    from collections import Counter, defaultdict

    hits = defaultdict(Counter)  # item -> gt id -> frames
    gt_items = defaultdict(Counter)
    visible = defaultdict(list)
    for t, f in enumerate(frames):
        for g, _ in f["gt"]:
            visible[g].append(t)
        if not f["gt"] or not f["tr"]:
            continue
        gb = np.array([b for _, b in f["gt"]])
        tb = np.array([b for _, b in f["tr"]])
        S = box_iou_matrix(gb, tb)
        r, c = linear_sum_assignment(-S)
        for rr, cc in zip(r, c):
            if S[rr, cc] >= 0.5:
                g = f["gt"][rr][0]
                it = item_of_track[f["tr"][cc][0]]
                hits[it][g] += 1
                gt_items[g][it] += 1
    claimed = {}
    tp = fp = dup = wrong_merge = misclass = 0
    for it in sorted(counted_items, key=str):
        h = hits.get(it)
        if not h:
            fp += 1
            continue
        (g, n1), *rest = h.most_common()
        if rest and rest[0][1] >= 0.2 * sum(h.values()):
            wrong_merge += 1
        if g not in listable_gt:
            fp += 1
            continue
        if item_class.get(it) != gt_class.get(g):
            misclass += 1
            fp += 1
            continue
        if g in claimed:
            dup += 1
            fp += 1
        else:
            claimed[g] = it
            tp += 1
    fn = len(listable_gt) - len(claimed)
    reapp, reapp_ok = 0, 0
    for g in listable_gt:
        v = visible[g]
        gaps = [b - a for a, b in zip(v, v[1:]) if b - a > 1]
        if gaps:
            reapp += 1
            its = [i for i in gt_items[g] if i in counted_items]
            if len(its) == 1:
                reapp_ok += 1
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    cnt_pred = Counter(item_class[i] for i in counted_items)
    cnt_gt = Counter(gt_class[g] for g in listable_gt)
    cls = set(cnt_pred) | set(cnt_gt)
    return {
        "items_pred": len(counted_items),
        "items_gt": len(listable_gt),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "duplicates": dup,
        "misclass": misclass,
        "wrong_merges": wrong_merge,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / max(1e-9, p + r), 4),
        "duplicate_rate": round(dup / max(1, len(listable_gt)), 4),
        "wrong_merge_rate": round(wrong_merge / max(1, len(counted_items)), 4),
        "reappearing_gt": reapp,
        "reappearance_link_correct": reapp_ok,
        "reappearance_link_accuracy": round(reapp_ok / reapp, 4) if reapp else None,
        "count_abs_errors": {c: abs(cnt_pred.get(c, 0) - cnt_gt.get(c, 0)) for c in cls},
    }
