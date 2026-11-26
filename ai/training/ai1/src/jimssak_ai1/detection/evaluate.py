"""탐지·세그멘테이션 정량 평가 (공개 데이터셋 테스트 라벨 기준).

지표: box/mask AP50, AP50-95(COCO 101점 보간), 클래스별 AP, 운영점(conf≥0.25, IoU≥0.5)
Precision/Recall/F1·FP·FN,
      Confusion Matrix(배경 포함), 매칭된 쌍의 마스크 IoU·Dice, 이미지당 추론 지연시간·FPS.
LVIS 연합 주석 규칙: 이미지에서 확인되지 않은 클래스(pos∪neg 밖)의 예측은 무시, not_exhaustive
클래스의 미매칭 예측은 FP로 세지 않음.
클래스 매핑은 taxonomy만 사용(B0 COCO 클래스는 같은 개념만). 매핑되지 않은 예측은 버린다.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..common import imread, load_yaml, log, write_csv, write_json
from ..frameeval import IOU_THRS, box_iou_matrix, rasterize
from ..taxonomy import Taxonomy

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _label_path(ip: Path) -> Path:
    parts = list(ip.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_gt(lp: Path, w: int, h: int, names: list[str]):
    out = []
    if not lp.is_file():
        return out
    for line in lp.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if not t:
            continue
        cls = names[int(t[0])]
        v = np.array([float(x) for x in t[1:]])
        if len(v) == 4:
            cx, cy, bw, bh = v * [w, h, w, h]
            out.append((cls, np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), None))
        else:
            p = v.reshape(-1, 2) * [w, h]
            out.append(
                (cls, np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]), p)
            )
    return out


def _mask_iou_pairs(pm, gm, pbox, gbox):
    out = np.zeros((len(pm), len(gm)))
    if not len(pm) or not len(gm):
        return out
    h, w = gm[0].shape
    pa = [int(m.sum()) for m in pm]
    ga = [int(m.sum()) for m in gm]
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


def _greedy(iou, conf, thrs):
    tp = np.zeros((iou.shape[0], len(thrs)), bool)
    pair = [dict() for _ in thrs]
    order = np.argsort(-conf, kind="stable")
    for k, thr in enumerate(thrs):
        used = np.zeros(iou.shape[1], bool)
        for i in order:
            if iou.shape[1] == 0:
                break
            cand = np.where(~used & (iou[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[i, k] = True
                pair[k][i] = j
    return tp, pair


def ap_with_ignore(tp, ign, conf, n_gt):
    if n_gt == 0:
        return np.full(tp.shape[1] if tp.ndim == 2 else len(IOU_THRS), np.nan)
    aps = np.zeros(len(IOU_THRS))
    rs = np.linspace(0, 1, 101)
    for k in range(len(IOU_THRS)):
        keep = ~ign[:, k] if len(ign) else np.zeros(0, bool)
        c = conf[keep]
        t = tp[keep, k]
        if len(c) == 0:
            aps[k] = 0.0
            continue
        o = np.argsort(-c, kind="stable")
        t = t[o]
        ctp = np.cumsum(t)
        cfp = np.cumsum(~t)
        rec = ctp / n_gt
        prec = np.maximum.accumulate((ctp / np.maximum(ctp + cfp, 1e-9))[::-1])[::-1]
        idx = np.searchsorted(rec, rs, side="left")
        aps[k] = np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])
    return aps


def evaluate_model(
    *,
    model_path: str,
    data_yaml: Path,
    split: str,
    tax: Taxonomy,
    device: str,
    out_dir: Path,
    label: str,
    federated_meta: dict | None = None,
    conf_op: float = 0.25,
    imgsz: int = 640,
    max_images: int | None = None,
    with_masks: bool = True,
) -> dict:
    from ultralytics import YOLO

    data = load_yaml(data_yaml)
    root = Path(data["path"])
    names = data["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    imgs = sorted(p for p in (root / data[split]).glob("*") if p.suffix.lower() in IMG_EXT)
    if max_images:
        imgs = imgs[:max_images]
    model = YOLO(model_path)
    mnames = {int(k): v for k, v in model.names.items()}
    cls_map, mode = tax.model_class_to_code(mnames)
    is_seg = model.task == "segment"
    supported = set(cls_map.values()) & set(names)
    eval_classes = list(names)
    st = defaultdict(
        lambda: {
            "conf": [],
            "tpb": [],
            "ignb": [],
            "tpm": [],
            "ignm": [],
            "n_gt": 0,
            "TP": 0,
            "FP": 0,
            "FN": 0,
            "mask_iou": [],
            "dice": [],
        }
    )
    labels_cm = eval_classes + ["background"]
    cm = np.zeros((len(labels_cm), len(labels_cm)), int)  # [gt, pred]
    per_image = []
    lat = []
    gt_has_masks = True
    for n_i, ip in enumerate(imgs):
        img = imread(ip)
        h, w = img.shape[:2]
        gts = read_gt(_label_path(ip), w, h, names)
        if any(g[2] is None for g in gts):
            gt_has_masks = False
        meta = (federated_meta or {}).get(ip.name)
        evaluable = set(meta["pos"]) | set(meta["neg"]) if meta else set(eval_classes)
        not_ex = set(meta["not_exhaustive"]) if meta else set()
        t0 = time.perf_counter()
        r = model.predict(
            img,
            conf=0.001,
            imgsz=imgsz,
            retina_masks=with_masks and is_seg,
            device=device,
            verbose=False,
            max_det=300,
        )[0]
        lat.append(time.perf_counter() - t0)
        preds = []
        if len(r.boxes):
            masks = (
                (r.masks.data.cpu().numpy() > 0.5)
                if (with_masks and is_seg and r.masks is not None)
                else None
            )
            for i in range(len(r.boxes)):
                c = cls_map.get(int(r.boxes.cls[i]))
                if c is None or c not in evaluable or c not in eval_classes:
                    continue
                preds.append(
                    (
                        c,
                        float(r.boxes.conf[i]),
                        r.boxes.xyxy[i].cpu().numpy(),
                        masks[i] if masks is not None else None,
                    )
                )
        img_fp = img_fn = 0
        for c in set(g[0] for g in gts) | set(p[0] for p in preds):
            G = [g for g in gts if g[0] == c]
            P = [p for p in preds if p[0] == c]
            s = st[c]
            s["n_gt"] += len(G)
            gb = np.array([g[1] for g in G]) if G else np.zeros((0, 4))
            if P:
                conf = np.array([p[1] for p in P])
                pb = np.array([p[2] for p in P])
                biou = box_iou_matrix(pb, gb)
                tpb, _ = _greedy(biou, conf, IOU_THRS)
                ignb = (~tpb) & (c in not_ex)
                if with_masks and is_seg and G and all(g[2] is not None for g in G):
                    gm = [rasterize(g[2], w, h) for g in G]
                    miou = _mask_iou_pairs([p[3] for p in P], gm, pb, gb)
                    tpm, _ = _greedy(miou, conf, IOU_THRS)
                else:
                    miou = None
                    tpm = np.zeros_like(tpb)
                ignm = (~tpm) & (c in not_ex)
                s["conf"] += conf.tolist()
                s["tpb"] += tpb.tolist()
                s["ignb"] += ignb.tolist()
                s["tpm"] += tpm.tolist()
                s["ignm"] += ignm.tolist()
                keep = conf >= conf_op
                if keep.any():
                    t50, pair = _greedy(biou[keep], conf[keep], [0.5])
                    tp = int(t50[:, 0].sum())
                    fp = int((~t50[:, 0]).sum()) if c not in not_ex else 0
                    s["TP"] += tp
                    s["FP"] += fp
                    s["FN"] += len(G) - tp
                    img_fp += fp
                    img_fn += len(G) - tp
                    if miou is not None:
                        kidx = np.where(keep)[0]
                        for pi, gj in pair[0].items():
                            m_iou = miou[kidx[pi], gj]
                            s["mask_iou"].append(m_iou)
                            s["dice"].append(2 * m_iou / (1 + m_iou) if m_iou > 0 else 0.0)
                else:
                    s["FN"] += len(G)
                    img_fn += len(G)
            else:
                s["FN"] += len(G)
                img_fn += len(G)
        # confusion (클래스 무관 매칭, IoU≥0.5, conf≥conf_op)
        P = [p for p in preds if p[1] >= conf_op]
        if gts or P:
            gb = np.array([g[1] for g in gts]) if gts else np.zeros((0, 4))
            pb = np.array([p[2] for p in P]) if P else np.zeros((0, 4))
            iou = box_iou_matrix(pb, gb)
            used_g = set()
            for i in np.argsort([-p[1] for p in P]):
                cand = [jj for jj in range(iou.shape[1]) if jj not in used_g and iou[i, jj] >= 0.5]
                j = max(cand, key=lambda jj: iou[i, jj]) if cand else -1
                if j >= 0:
                    used_g.add(j)
                    cm[labels_cm.index(gts[j][0]), labels_cm.index(P[i][0])] += 1
                elif P[i][0] not in not_ex:
                    cm[-1, labels_cm.index(P[i][0])] += 1
            for j, g in enumerate(gts):
                if j not in used_g:
                    cm[labels_cm.index(g[0]), -1] += 1
        per_image.append({"image": ip.name, "fp": img_fp, "fn": img_fn, "n_gt": len(gts)})
        if (n_i + 1) % 200 == 0:
            log(f"    [{label}] {n_i + 1}/{len(imgs)}")

    rows = []
    for c in eval_classes:
        s = st[c]
        conf = np.array(s["conf"])
        k = len(IOU_THRS)
        tpb = np.array(s["tpb"], bool).reshape(-1, k)
        ignb = np.array(s["ignb"], bool).reshape(-1, k)
        tpm = np.array(s["tpm"], bool).reshape(-1, k)
        ignm = np.array(s["ignm"], bool).reshape(-1, k)
        apb = ap_with_ignore(tpb, ignb, conf, s["n_gt"])
        apm = (
            ap_with_ignore(tpm, ignm, conf, s["n_gt"])
            if (is_seg and with_masks and gt_has_masks)
            else None
        )
        P_ = s["TP"] / (s["TP"] + s["FP"]) if s["TP"] + s["FP"] else None
        R_ = s["TP"] / s["n_gt"] if s["n_gt"] else None
        F_ = (
            2 * P_ * R_ / (P_ + R_)
            if P_ and R_
            else (0.0 if P_ is not None and R_ is not None else None)
        )

        def f(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return round(float(x), 4)

        rows.append(
            {
                "class": c,
                "supported_by_model": "yes" if c in supported else "no(미지원)",
                "n_gt": s["n_gt"],
                "box_AP50": f(apb[0]),
                "box_AP50_95": f(np.mean(apb)) if s["n_gt"] else None,
                "mask_AP50": f(apm[0]) if apm is not None else "미측정",
                "mask_AP50_95": f(np.mean(apm))
                if apm is not None and s["n_gt"]
                else ("미측정" if apm is None else None),
                "precision": f(P_),
                "recall": f(R_),
                "f1": f(F_),
                "TP": s["TP"],
                "FP": s["FP"],
                "FN": s["FN"],
                "mask_IoU_mean": f(np.mean(s["mask_iou"])) if s["mask_iou"] else None,
                "dice_mean": f(np.mean(s["dice"])) if s["dice"] else None,
            }
        )

    def agg(classes, key):
        v = [
            r[key]
            for r in rows
            if r["class"] in classes and isinstance(r[key], (int, float)) and r["n_gt"] > 0
        ]
        return round(float(np.mean(v)), 4) if v else None

    gt_classes = [r["class"] for r in rows if r["n_gt"] > 0]
    tot_tp = sum(r["TP"] for r in rows)
    tot_fp = sum(r["FP"] for r in rows)
    tot_fn = sum(r["FN"] for r in rows)
    P = tot_tp / (tot_tp + tot_fp) if tot_tp + tot_fp else None
    R = tot_tp / (tot_tp + tot_fn) if tot_tp + tot_fn else None
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_class.csv", rows, list(rows[0].keys()))
    write_csv(
        out_dir / "confusion_matrix.csv",
        [
            {
                "gt\\pred": labels_cm[i],
                **{labels_cm[j]: int(cm[i, j]) for j in range(len(labels_cm))},
            }
            for i in range(len(labels_cm))
        ],
        ["gt\\pred"] + labels_cm,
    )
    write_csv(out_dir / "per_image.csv", per_image, ["image", "fp", "fn", "n_gt"])
    np.save(out_dir / "confusion_matrix.npy", cm)
    import ultralytics

    summary = {
        "label": label,
        "model": model_path,
        "task": model.task,
        "class_mapping_mode": mode,
        "split": split,
        "n_images": len(imgs),
        "imgsz": imgsz,
        "device": device,
        "conf_operating": conf_op,
        "gt_masks_available": gt_has_masks,
        "federated_rules": bool(federated_meta),
        "all_gt_classes": {
            "classes": gt_classes,
            "box_mAP50": agg(gt_classes, "box_AP50"),
            "box_mAP50_95": agg(gt_classes, "box_AP50_95"),
            "mask_mAP50": agg(gt_classes, "mask_AP50") if gt_has_masks and is_seg else "미측정",
            "mask_mAP50_95": agg(gt_classes, "mask_AP50_95")
            if gt_has_masks and is_seg
            else "미측정",
        },
        "supported_classes": sorted(supported & set(gt_classes)),
        "micro_at_operating_point": {
            "TP": tot_tp,
            "FP": tot_fp,
            "FN": tot_fn,
            "precision": round(P, 4) if P else P,
            "recall": round(R, 4) if R else R,
            "f1": round(2 * P * R / (P + R), 4) if P and R else None,
        },
        "mask_IoU_mean_all": round(
            float(np.mean([x for c in st.values() for x in c["mask_iou"]])), 4
        )
        if any(c["mask_iou"] for c in st.values())
        else None,
        "dice_mean_all": round(float(np.mean([x for c in st.values() for x in c["dice"]])), 4)
        if any(c["dice"] for c in st.values())
        else None,
        "latency_ms_mean": round(1000 * float(np.mean(lat[1:] or lat)), 2) if lat else None,
        "fps": round(1.0 / float(np.mean(lat[1:] or lat)), 2) if lat else None,
        "ultralytics_version": ultralytics.__version__,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def class_subset_scores(per_class_csv_rows: list[dict], classes: list[str]) -> dict:
    def v(k):
        return [
            float(r[k])
            for r in per_class_csv_rows
            if r["class"] in classes and r[k] not in ("", None, "미측정") and int(r["n_gt"]) > 0
        ]

    def m(k):
        return round(float(np.mean(v(k))), 4) if v(k) else None

    return {
        "classes": classes,
        "box_mAP50_95": m("box_AP50_95"),
        "mask_mAP50_95": m("mask_AP50_95"),
        "box_mAP50": m("box_AP50"),
        "recall_mean": m("recall"),
    }
