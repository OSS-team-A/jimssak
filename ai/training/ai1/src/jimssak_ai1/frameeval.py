"""프레임 단위 인식·분할 평가: box/mask mAP50-95, 품목별 Recall, 정답 개수.

- 정답: Ultralytics YOLO-seg 라벨(클래스명 = taxonomy code).
- 예측 클래스는 taxonomy 매핑(B0: COCO 동일 개념만, B1: code 그대로)으로만 변환한다. 매핑되지 않은
  예측은 버린다.
- AP: COCO 방식(101점 보간), IoU 0.50:0.05:0.95 평균. 매칭은 이미지·클래스별로 confidence 내림차순
  탐욕 매칭.
- Recall: IoU≥0.5, conf≥recall_conf(기본 0.25 = BoT-SORT track_high_thresh)에서 정답 인스턴스 중
  맞힌 비율.
  YOLO confidence 자체를 정확도·Recall로 쓰지 않는다.
- 정답 라벨에 다각형이 없으면(박스만) mask mAP는 '미측정'.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)


@dataclass
class GTInst:
    cls: str
    box: np.ndarray  # xyxy px
    poly: np.ndarray | None  # (N,2) px


@dataclass
class PredInst:
    cls: str
    conf: float
    box: np.ndarray
    mask: np.ndarray | None  # bool HxW


@dataclass
class ClassStats:
    n_gt: int = 0
    # 각 예측: (conf, tp[10] box, tp[10] mask)
    confs: list = field(default_factory=list)
    tp_box: list = field(default_factory=list)
    tp_mask: list = field(default_factory=list)
    recall_hits: int = 0


def read_yolo_seg_label(path, w: int, h: int, names: list[str]) -> tuple[list[GTInst], bool]:
    """YOLO 라벨 파일 → 정답 인스턴스. 반환 (인스턴스, 모두 다각형인지)."""
    insts, all_poly = [], True
    if not path.is_file():
        return insts, all_poly
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        t = line.split()
        if not t:
            continue
        cid = int(t[0])
        vals = np.array([float(x) for x in t[1:]], dtype=np.float64)
        if len(vals) == 4:  # 박스 라벨 (cx cy w h)
            cx, cy, bw, bh = vals * [w, h, w, h]
            box = np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            insts.append(GTInst(names[cid], box, None))
            all_poly = False
        else:
            pts = vals.reshape(-1, 2) * [w, h]
            box = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
            insts.append(GTInst(names[cid], box, pts))
    return insts, all_poly


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def rasterize(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def mask_iou_matrix(
    pm: list[np.ndarray], gm: list[np.ndarray], pbox: np.ndarray, gbox: np.ndarray
) -> np.ndarray:
    """박스가 겹치는 쌍만 겹친 영역에서 교집합을 세는 정확한 마스크 IoU."""
    out = np.zeros((len(pm), len(gm)))
    pa = [int(m.sum()) for m in pm]
    ga = [int(m.sum()) for m in gm]
    h, w = pm[0].shape if pm else (gm[0].shape if gm else (0, 0))
    for i in range(len(pm)):
        for j in range(len(gm)):
            x1 = int(max(0, np.floor(max(pbox[i, 0], gbox[j, 0]))))
            y1 = int(max(0, np.floor(max(pbox[i, 1], gbox[j, 1]))))
            x2 = int(min(w, np.ceil(min(pbox[i, 2], gbox[j, 2])) + 1))
            y2 = int(min(h, np.ceil(min(pbox[i, 3], gbox[j, 3])) + 1))
            if x2 <= x1 or y2 <= y1:
                continue
            inter = int(np.logical_and(pm[i][y1:y2, x1:x2], gm[j][y1:y2, x1:x2]).sum())
            u = pa[i] + ga[j] - inter
            out[i, j] = inter / u if u > 0 else 0.0
    return out


def greedy_match(iou: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """예측별 IoU 임계값 10개에 대한 TP 여부 (n_pred, 10)."""
    tp = np.zeros((iou.shape[0], len(IOU_THRS)), bool)
    order = np.argsort(-conf, kind="stable")
    for k, thr in enumerate(IOU_THRS):
        used = np.zeros(iou.shape[1], bool)
        for i in order:
            if iou.shape[1] == 0:
                break
            cand = np.where(~used & (iou[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[i, k] = True
    return tp


def accumulate_image(
    stats: dict,
    gts: list[GTInst],
    preds: list[PredInst],
    w: int,
    h: int,
    with_mask: bool,
    recall_conf: float,
) -> None:
    classes = {g.cls for g in gts} | {p.cls for p in preds}
    for c in classes:
        g = [x for x in gts if x.cls == c]
        p = [x for x in preds if x.cls == c]
        st = stats.setdefault(c, ClassStats())
        st.n_gt += len(g)
        if not p:
            continue
        conf = np.array([x.conf for x in p])
        gb = np.array([x.box for x in g]) if g else np.zeros((0, 4))
        pb = np.array([x.box for x in p])
        biou = box_iou_matrix(pb, gb)
        tpb = greedy_match(biou, conf)
        if with_mask and g:
            gm = [rasterize(x.poly, w, h) for x in g]
            pm = [x.mask for x in p]
            miou = mask_iou_matrix(pm, gm, pb, gb)
            tpm = greedy_match(miou, conf)
        else:
            tpm = np.zeros_like(tpb)
        st.confs += conf.tolist()
        st.tp_box += tpb.tolist()
        st.tp_mask += tpm.tolist()
        # Recall@0.5 (운영 임계값 이상 예측만)
        keep = conf >= recall_conf
        if keep.any() and g:
            st.recall_hits += int(greedy_match(biou[keep], conf[keep])[:, 0].sum())


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    """COCO 101점 보간 AP, IoU 임계값별 (10,)."""
    if n_gt == 0:
        return np.full(len(IOU_THRS), np.nan)
    if len(conf) == 0:
        return np.zeros(len(IOU_THRS))
    order = np.argsort(-conf, kind="stable")
    tp = tp[order]
    ap = np.zeros(tp.shape[1])
    rs = np.linspace(0, 1, 101)
    for k in range(tp.shape[1]):
        ctp = np.cumsum(tp[:, k])
        cfp = np.cumsum(~tp[:, k])
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        prec = np.maximum.accumulate(prec[::-1])[::-1]  # 단조 감소 포락선
        idx = np.searchsorted(rec, rs, side="left")
        q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
        ap[k] = q.mean()
    return ap


def summarize(stats: dict, with_mask: bool, recall_conf: float) -> list[dict]:
    rows = []
    for c, st in sorted(stats.items()):
        conf = np.array(st.confs)
        apb = average_precision(np.array(st.tp_box, bool).reshape(-1, len(IOU_THRS)), conf, st.n_gt)
        apm = (
            average_precision(np.array(st.tp_mask, bool).reshape(-1, len(IOU_THRS)), conf, st.n_gt)
            if with_mask
            else None
        )
        rows.append(
            {
                "category": c,
                "n_gt_instances": st.n_gt,
                "n_pred": len(conf),
                "box_AP50": None if np.isnan(apb[0]) else round(float(apb[0]), 4),
                "box_AP50_95": None if np.isnan(apb[0]) else round(float(apb.mean()), 4),
                "mask_AP50_95": (
                    "미측정"
                    if apm is None
                    else (None if np.isnan(apm[0]) else round(float(apm.mean()), 4))
                ),
                f"recall@iou0.5_conf{recall_conf}": round(st.recall_hits / st.n_gt, 4)
                if st.n_gt
                else None,
            }
        )
    return rows
