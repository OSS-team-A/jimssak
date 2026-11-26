"""B4 단계: (a) 균열 세그멘테이션 학습 전후, (b) PatchCore 이상 탐지(VisA 실제 / ABO 합성), (c) 이사
전후 상태 변화(합성).

실제 데이터와 합성 데이터 결과는 항상 별도 키로 기록한다. 가구 실제 손상 데이터는 확보하지
못했으므로 '미측정'.
"""

from __future__ import annotations

import csv
import random

import numpy as np

from .common import imread, log, read_csv, write_json
from .reidentification.train_reid import auroc


def _f1(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return round(p, 4), round(r, 4), round(2 * p * r / max(1e-9, p + r), 4)


def _pixel_metrics(maps, masks, thr, max_pixels=400000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.concatenate([m.reshape(-1) for m in maps])
    y = np.concatenate([g.reshape(-1) for g in masks]).astype(int)
    if len(s) > max_pixels:
        idx = rng.choice(len(s), max_pixels, replace=False)
        s, y = s[idx], y[idx]
    au = auroc(s, y)
    ious, dices = [], []
    for m, g in zip(maps, masks):
        if g.sum() == 0:
            continue
        p = m >= thr
        inter = (p & g).sum()
        uni = (p | g).sum()
        ious.append(inter / max(1, uni))
        dices.append(2 * inter / max(1, p.sum() + g.sum()))
    return {
        "pixel_auroc": round(au, 4) if au is not None else None,
        "region_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
        "dice_mean": round(float(np.mean(dices)), 4) if dices else None,
    }


def patchcore_eval(fx, train_imgs, calib_imgs, test_imgs, test_labels, test_masks, ratio, imgsz):
    import cv2

    from .damage.patchcore import PatchCore

    pc = PatchCore(fx, ratio).fit(train_imgs)
    cs, cmaps, cg = pc.score(calib_imgs)
    ts, tmaps, tg = pc.score(test_imgs)
    y = np.array(test_labels)
    thr_img = float(np.quantile(cs, 0.99)) if len(cs) else float(np.median(ts))
    thr_pix = (
        float(np.quantile(np.concatenate([m.reshape(-1) for m in cmaps]), 0.99))
        if len(cmaps)
        else float(np.median(tmaps))
    )
    pred = ts >= thr_img
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    P, R, F = _f1(tp, fp, fn)
    gm = [
        cv2.resize(m.astype(np.uint8), (imgsz, imgsz), interpolation=cv2.INTER_NEAREST).astype(bool)
        for m in test_masks
    ]
    pix = _pixel_metrics(list(tmaps), gm, thr_pix)
    return {
        "patchcore": {
            "image_auroc": round(auroc(ts, y), 4) if auroc(ts, y) is not None else None,
            "precision": P,
            "recall": R,
            "f1": F,
            "threshold_rule": "보정용 정상 이미지 점수 99% 분위",
            **pix,
        },
        "baseline_global_knn": {
            "image_auroc": round(auroc(tg, y), 4) if auroc(tg, y) is not None else None,
            "pixel_auroc": "미측정(위치 지도 없음)",
        },
        "n_train": len(train_imgs),
        "n_calib": len(calib_imgs),
        "n_test": len(test_imgs),
        "n_test_anomaly": int(y.sum()),
        "memory_bank_patches": int(len(pc.bank)),
    }


def run_damage(c) -> dict:
    from .damage.change import FeatureDiff, align, color_normalize, decide, pixel_diff_map
    from .damage.patchcore import FeatureExtractor
    from .damage.synth import add_damage, foreground, nuisance
    from .detection.evaluate import evaluate_model
    from .detection.train_yolo import train_yolo
    from .taxonomy import Taxonomy

    od = c.run / "eval" / "damage"
    od.mkdir(parents=True, exist_ok=True)
    res = {"real_data": {}, "synthetic_data": {}}
    # ---------- (a) 균열 세그멘테이션: 학습 전(COCO, crack 클래스 없음) vs 파인튜닝 ----------
    m = c.p["damage"]["crack_model"]
    c.monitor.stage = "train_crack"
    tr = train_yolo(
        weights=str(c.pre / f"{m['name']}.pt"),
        data_yaml=c.ds / "crackseg" / "data.yaml",
        project=c.run / "models" / "yolo",
        name=f"crack_{m['name']}",
        epochs=m["epochs"],
        batch=m["batch"],
        imgsz=m["imgsz"],
        device=c.ydev,
        common=c.p["train_common"],
    )
    c.set("crack", tr)
    crack_tax = Taxonomy(categories=[{"code": "crack", "coco": None}])
    ev = {}
    for label, w in (
        ("B0_COCO_pretrained", str(c.pre / f"{m['name']}.pt")),
        ("B4_crack_finetuned", tr["best_pt"]),
    ):
        ev[label] = evaluate_model(
            model_path=w,
            data_yaml=c.ds / "crackseg" / "data.yaml",
            split="test",
            tax=crack_tax,
            device=c.device,
            out_dir=od / "crackseg_test" / label,
            label=label,
            imgsz=m["imgsz"],
        )
    res["real_data"]["crack_segmentation"] = {
        k: {
            "mask_mAP50_95": v["all_gt_classes"]["mask_mAP50_95"],
            "box_mAP50": v["all_gt_classes"]["box_mAP50"],
            **v["micro_at_operating_point"],
            "mask_IoU_mean": v["mask_IoU_mean_all"],
            "dice_mean": v["dice_mean_all"],
        }
        for k, v in ev.items()
    }
    res["real_data"]["crack_segmentation"]["domain"] = "도로·벽 균열(crack-seg). 가구 손상 아님"
    res["real_data"]["crack_segmentation"]["note"] = (
        "COCO 사전학습 모델에는 crack 클래스가 없어 학습 전 재현율 0 (미지원)"
    )
    # ---------- (b) PatchCore ----------
    pcfg = c.p["damage"]["patchcore"]
    fx = FeatureExtractor(c.device, pcfg["imgsz"])
    rng = random.Random(0)
    cats = c.p["damage"].get("visa_categories") or []
    visa_rows = (
        list(
            csv.DictReader(
                open(c.data / "datasets" / "visa" / "split_csv" / "1cls.csv", encoding="utf-8")
            )
        )
        if cats
        else []
    )
    vis_res = {}
    for cat in cats:
        c.monitor.stage = f"patchcore_{cat}"
        rows = [r for r in visa_rows if r["object"] == cat]
        trn = [r for r in rows if r["split"] == "train"]
        rng.shuffle(trn)
        n_cal = max(5, len(trn) // 10)
        calib, trn = trn[:n_cal], trn[n_cal : n_cal + pcfg["max_train_images"]]
        tst = [r for r in rows if r["split"] == "test"]
        root = c.data / "datasets" / "visa"

        def load(rs):
            return [imread(root / r["image"]) for r in rs]

        masks = []
        for r in tst:
            if r["mask"]:
                mk = imread(root / r["mask"])[..., 0] > 0
            else:
                im = imread(root / r["image"])
                mk = np.zeros(im.shape[:2], bool)
            masks.append(mk)
        log(f"  PatchCore [{cat}] 학습 {len(trn)} / 보정 {len(calib)} / 테스트 {len(tst)}")
        vis_res[cat] = patchcore_eval(
            fx,
            load(trn),
            load(calib),
            load(tst),
            [1 if r["label"] == "anomaly" else 0 for r in tst],
            masks,
            pcfg["coreset_ratio"],
            pcfg["imgsz"],
        )
    if vis_res:
        res["real_data"]["visa_patchcore"] = vis_res
        res["real_data"]["visa_patchcore_note"] = (
            "산업 부품(VisA) 이상 탐지 — 가구 손상과 도메인이 다름"
        )
    else:
        res["real_data"]["visa_patchcore"] = "미실행(프로필에서 VisA 제외)"
    # ABO 합성 손상 이상 탐지 (같은 종류 정상 상품 사진으로 메모리 구축)
    rows = read_csv(c.ds / "abo" / "abo_manifest.csv")
    ptype = max(
        {r["product_type"] for r in rows},
        key=lambda t: sum(1 for r in rows if r["product_type"] == t and r["split"] == "train"),
    )
    tr_r = [r for r in rows if r["split"] == "train" and r["product_type"] == ptype][
        : pcfg["max_train_images"]
    ]
    te_r = [r for r in rows if r["split"] == "test" and r["product_type"] == ptype][:60]
    abo = c.ds / "abo"
    tr_imgs = [imread(abo / r["path"]) for r in tr_r]
    n_cal = max(5, len(tr_imgs) // 10)
    t_imgs, t_lab, t_masks = [], [], []
    for i, r in enumerate(te_r):
        im = imread(abo / r["path"])
        if i % 2:
            im, mk = add_damage(im, rng.choice(["scratch", "crack", "dent", "discoloration"]), rng)
            t_lab.append(1)
        else:
            mk = np.zeros(im.shape[:2], bool)
            t_lab.append(0)
        t_imgs.append(im)
        t_masks.append(mk)
    c.monitor.stage = "patchcore_abo_synth"
    res["synthetic_data"]["abo_patchcore"] = {
        "product_type": ptype,
        **patchcore_eval(
            fx,
            tr_imgs[n_cal:],
            tr_imgs[:n_cal],
            t_imgs,
            t_lab,
            t_masks,
            pcfg["coreset_ratio"],
            pcfg["imgsz"],
        ),
    }
    # ---------- (c) 이사 전후 상태 변화 (합성) ----------
    from ultralytics import YOLO

    crack_model = YOLO(tr["best_pt"])
    fd = FeatureDiff(fx)
    cb = c.p["damage"]["change_bench"]

    def make_pairs(split, n, seed):
        rr = random.Random(seed)
        pool = [r for r in rows if r["split"] == split]
        pairs = []
        for k in range(n):
            r = rr.choice(pool)
            base = imread(abo / r["path"])
            case = ["unchanged", "new", "pre_existing"][k % 3]
            kind = rr.choice(["scratch", "crack", "dent", "discoloration"])
            new_mask = np.zeros(base.shape[:2], bool)
            if case == "pre_existing":
                base, _ = add_damage(base, kind, rr)
            after_src = base
            if case == "new":
                after_src, new_mask = add_damage(base, kind, rr)
            after, M = nuisance(after_src, rr)
            mw = __import__("cv2").warpAffine(
                new_mask.astype(np.uint8), M, (base.shape[1], base.shape[0]), flags=0
            )
            pairs.append(
                {
                    "case": case,
                    "kind": kind,
                    "before": base,
                    "after": after,
                    "new_mask_before_coords": new_mask,
                    "mask_after": mw,
                }
            )
        return pairs

    def scores(pairs):
        out = {"pixel_diff": [], "feature_diff": []}
        maps = {"pixel_diff": [], "feature_diff": []}
        oks = []
        for p in pairs:
            al, cc = align(p["before"], p["after"])
            ok = al is not None and cc > 0.5
            oks.append(ok)
            if al is None:
                al = p["after"]
            al = color_normalize(al, p["before"])
            fg = foreground(p["before"])
            for name, fn in (("pixel_diff", pixel_diff_map), ("feature_diff", fd.map)):
                mp = fn(p["before"], al)
                v = mp[fg] if fg.any() else mp.reshape(-1)
                out[name].append(
                    float(np.mean(np.sort(v)[-max(1, len(v) // 200) :]))
                )  # 상위 0.5% 평균
                maps[name].append(mp)
        return out, maps, oks

    val_p, test_p = (
        make_pairs("val", cb["n_val_pairs"], 21),
        make_pairs("test", cb["n_test_pairs"], 22),
    )
    c.monitor.stage = "change_bench"
    vs, _, _ = scores(val_p)
    ts, tmaps, toks = scores(test_p)
    yv = np.array([p["case"] == "new" for p in val_p])
    yt = np.array([p["case"] == "new" for p in test_p])
    ch = {}
    for name in ("pixel_diff", "feature_diff"):
        sv = np.array(vs[name])
        st = np.array(ts[name])
        best = max(
            np.unique(np.quantile(sv, np.linspace(0.02, 0.98, 97))),
            key=lambda t: _f1(
                int(((sv >= t) & yv).sum()),
                int(((sv >= t) & ~yv).sum()),
                int(((sv < t) & yv).sum()),
            )[2],
        )
        pred = st >= best
        P, R, F = _f1(int((pred & yt).sum()), int((pred & ~yt).sum()), int((~pred & yt).sum()))
        fa_unchanged = float(
            np.mean([pr for pr, p in zip(pred, test_p) if p["case"] == "unchanged"])
        )
        fa_pre = float(np.mean([pr for pr, p in zip(pred, test_p) if p["case"] == "pre_existing"]))
        # 변화 지도 영역 IoU: 'new' 사례에서 상위 분위로 이진화한 지도 vs 새 손상 마스크(이전 좌표)
        ious = []
        for mp, p in zip(tmaps[name], test_p):
            if p["case"] != "new" or p["new_mask_before_coords"].sum() == 0:
                continue
            fgm = foreground(p["before"])
            thr_px = np.quantile(mp[fgm] if fgm.any() else mp, 0.98)
            pm = mp >= thr_px
            g = p["new_mask_before_coords"]
            ious.append((pm & g).sum() / max(1, (pm | g).sum()))
        # 상태 판정 정확도 (단일 이미지 손상 검출기 = 균열 모델)
        status_ok = {"unchanged": [0, 0], "new": [0, 0], "pre_existing": [0, 0]}
        expect = {
            "unchanged": "unchanged",
            "new": "newly_detected_damage",
            "pre_existing": "pre_existing_damage",
        }
        for s_, ok, p in zip(st, toks, test_p):
            bd = (
                len(
                    crack_model.predict(p["before"], conf=0.25, device=c.device, verbose=False)[
                        0
                    ].boxes
                )
                > 0
            )
            stt, _ = decide(float(s_), float(best), 0.0, None, None, ok, bd)
            status_ok[p["case"]][0] += stt == expect[p["case"]]
            status_ok[p["case"]][1] += 1
        ch[name] = {
            "threshold_from_val": round(float(best), 5),
            "new_damage_precision": P,
            "new_damage_recall": R,
            "new_damage_f1": F,
            "auroc": round(auroc(st, yt.astype(int)), 4)
            if auroc(st, yt.astype(int)) is not None
            else None,
            "false_alarm_rate_unchanged(조명·시점 변화만)": round(fa_unchanged, 4),
            "false_alarm_rate_pre_existing": round(fa_pre, 4),
            "change_region_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "status_accuracy": {k: round(v[0] / max(1, v[1]), 4) for k, v in status_ok.items()},
            "alignment_success_rate": round(float(np.mean(toks)), 4),
        }
    res["synthetic_data"]["before_after_change"] = {
        **ch,
        "pairs": {"val": len(val_p), "test": len(test_p)},
        "construction": "ABO 상품 사진 + 합성 손상(긁힘·균열·찍힘·변색) + 조명·색·작은 회전/이동",
        "pre_existing_detector": "B4 균열 모델(conf≥0.25) — 합성 손상 유형과 달라 낮게 나올 수 "
        "있음",
    }
    res["real_furniture_damage"] = "미측정 — 인증 없이 받을 수 있는 가구·가전 손상 라벨 데이터 없음"
    write_json(od / "b4_eval.json", res)
    return {
        "crack_test": res["real_data"]["crack_segmentation"],
        "change_f1": {k: v["new_damage_f1"] for k, v in ch.items()},
    }
