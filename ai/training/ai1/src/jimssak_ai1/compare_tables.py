"""학습 전후·알고리즘별 비교표(CSV/JSON) 생성. 측정된 값만 넣고 없는 값은 '미측정'."""

from __future__ import annotations

import json

import numpy as np

from .common import read_csv, read_json, write_csv, write_json
from .gpu import summarize_gpu_csv

NA = "미측정"


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _subset(rows, classes):
    rs = [r for r in rows if r["class"] in classes and int(r["n_gt"]) > 0]

    def mean(k):
        v = [_num(r[k]) for r in rs if _num(r[k]) is not None]
        return round(float(np.mean(v)), 4) if v else NA

    tp = sum(int(r["TP"]) for r in rs)
    fp = sum(int(r["FP"]) for r in rs)
    fn = sum(int(r["FN"]) for r in rs)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    return {
        "n_classes": len(rs),
        "box_mAP50": mean("box_AP50"),
        "box_mAP50_95": mean("box_AP50_95"),
        "mask_mAP50": mean("mask_AP50"),
        "mask_mAP50_95": mean("mask_AP50_95"),
        "precision": round(p, 4) if p is not None else NA,
        "recall": round(rc, 4) if rc is not None else NA,
        "f1": round(2 * p * rc / (p + rc), 4) if p and rc else NA,
        "FP": fp,
        "FN": fn,
        "mask_IoU_mean": mean("mask_IoU_mean"),
        "dice_mean": mean("dice_mean"),
    }


def build_all(c) -> dict:
    out = c.run / "compare"
    out.mkdir(parents=True, exist_ok=True)
    groups = c.res("class_groups") or {"common": [], "new": []}
    seg = c.res("seg_models") or {}
    det_rows, pc_rows = [], {}
    base = c.run / "eval" / "detection" / "lvis_test"
    for d in sorted(base.glob("*")) if base.is_dir() else []:
        s = read_json(d / "summary.json")
        rows = read_csv(d / "per_class.csv")
        tr = seg.get(d.name, {})
        for sub, cls in (
            ("all_gt_classes", s["all_gt_classes"]["classes"]),
            ("common_classes", groups["common"]),
            ("new_classes", groups["new"]),
        ):
            det_rows.append(
                {
                    "model": d.name,
                    "subset": sub,
                    **_subset(rows, cls),
                    "latency_ms": s["latency_ms_mean"],
                    "fps": s["fps"],
                    "params": tr.get("params", NA),
                    "weights_MB": round(tr["best_bytes"] / 2**20, 2)
                    if tr.get("best_bytes")
                    else NA,
                    "train_time_sec": tr.get(
                        "train_time_from_results_sec",
                        NA if d.name.startswith("B1") else "사전학습(학습 안 함)",
                    ),
                }
            )
        for r in rows:
            pc_rows.setdefault(
                r["class"],
                {
                    "class": r["class"],
                    "n_gt": r["n_gt"],
                    "group": "common" if r["class"] in groups["common"] else "new",
                },
            )
            pc_rows[r["class"]][f"{d.name}|mask_AP50_95"] = r["mask_AP50_95"]
            pc_rows[r["class"]][f"{d.name}|box_AP50_95"] = r["box_AP50_95"]
            pc_rows[r["class"]][f"{d.name}|recall"] = r["recall"]
    if det_rows:
        write_csv(out / "detection_before_after.csv", det_rows, list(det_rows[0].keys()))
        pcl = list(pc_rows.values())
        keys = sorted({k for r in pcl for k in r})
        write_csv(
            out / "per_class_before_after.csv",
            pcl,
            ["class", "group", "n_gt"] + [k for k in keys if "|" in k],
        )
    ho = c.run / "eval" / "detection" / "homeobjects_test"
    ho_rows = []
    for d in sorted(ho.glob("*")) if ho.is_dir() else []:
        s = read_json(d / "summary.json")
        rows = read_csv(d / "per_class.csv")
        ho_rows.append(
            {
                "model": d.name,
                **_subset(rows, s["all_gt_classes"]["classes"]),
                "mask": "미측정(정답 박스만)",
            }
        )
    if ho_rows:
        write_csv(out / "homeobjects_box_eval.csv", ho_rows, list(ho_rows[0].keys()))
    # ---- 추적·재식별 ablation (합성 영상) ----
    tc = c.run / "eval" / "tracking" / "tracking_comparison.csv"
    abl = []
    if tc.is_file():
        T = read_csv(tc)

        def pick(d, t, m):
            return next(
                (r for r in T if r["detector"] == d and r["tracker"] == t and r["relink"] == m),
                None,
            )

        # 그래프 축에 쓰이므로 영문 (서버에 한글 폰트가 없어도 깨지지 않게). 의미: A0 학습 전 → A1
        # 파인튜닝 → A1' 추적기 비교
        # → A2 기존 색 히스토그램 재등장 규칙 → A3 DINOv2 임베딩 재등장 연결 → A4 triplet 학습 헤드
        # 재등장 연결
        steps = [
            ("A0 B0 pretrained + BoT-SORT", "B0", "botsort", "none"),
            ("A1 + fine-tuned (B1)", "B1", "botsort", "none"),
            ("A1' B1 + ByteTrack", "B1", "bytetrack", "none"),
            ("A2 + hist relink (old rule)", "B1", "botsort", "hist_rule"),
            ("A3 + DINOv2 relink", "B1", "botsort", "dinov2"),
            ("A4 + trained re-ID relink", "B1", "botsort", "trained"),
        ]
        for name, d, t, m in steps:
            r = pick(d, t, m)
            if r is None:
                abl.append({"step": name, "status": "미실행"})
                continue
            abl.append(
                {
                    "step": name,
                    "detector": d,
                    "tracker": t,
                    "relink": m,
                    "threshold": r["threshold"],
                    "item_precision": r["item_precision"],
                    "item_recall": r["item_recall"],
                    "item_f1": r["item_f1"],
                    "duplicates": r["item_duplicates"],
                    "duplicate_rate": r["item_duplicate_rate"],
                    "wrong_merges": r["item_wrong_merges"],
                    "wrong_merge_rate": r["item_wrong_merge_rate"],
                    "reappearance_link_accuracy": r["item_reappearance_link_accuracy"],
                    "count_MAE": r["count_MAE"],
                    "IDF1": r["mot_IDF1"],
                    "HOTA": r["mot_HOTA"],
                    "IDSW": r["mot_IDSW"],
                    "MOTA": r["mot_MOTA"],
                    "sequences": r["sequences"],
                }
            )
        write_csv(
            out / "ablation_service_synthetic_video.csv",
            abl,
            sorted({k for r in abl for k in r}, key=lambda k: (k != "step", k)),
        )
    # ---- B3 / B4 / reid 요약 ----
    # ---- 비용 ----
    gsum = summarize_gpu_csv(c.run / "logs" / "gpu_usage.csv")
    cost = []
    for k, v in seg.items():
        g = gsum.get(f"train_{k.replace('B1_', '')}", {}) if isinstance(gsum, dict) else {}
        et = v.get("epoch_times_sec") or []
        cost.append(
            {
                "model": k,
                "task": "seg(LVIS)",
                "train_time_sec": v.get("train_time_from_results_sec"),
                "epochs": v.get("epochs_completed"),
                "epoch_time_mean_sec": round(float(np.mean(et)), 2) if et else NA,
                "batch": v.get("batch"),
                "device": v.get("device"),
                "gpu_mem_max_MiB": g.get("mem_max_MiB", NA),
                "gpu_util_mean_pct": g.get("util_mean_pct", NA),
                "params": v.get("params"),
                "weights_MB": round(v["best_bytes"] / 2**20, 2) if v.get("best_bytes") else NA,
            }
        )
    for key, task, st in (
        ("homeobj_det", "det(HomeObjects)", "train_homeobjects_det"),
        ("crack", "seg(crack)", "train_crack"),
    ):
        v = c.res(key)
        if v:
            g = gsum.get(st, {}) if isinstance(gsum, dict) else {}
            et = v.get("epoch_times_sec") or []
            cost.append(
                {
                    "model": key,
                    "task": task,
                    "train_time_sec": v.get("train_time_from_results_sec"),
                    "epochs": v.get("epochs_completed"),
                    "epoch_time_mean_sec": round(float(np.mean(et)), 2) if et else NA,
                    "batch": v.get("batch"),
                    "device": v.get("device"),
                    "gpu_mem_max_MiB": g.get("mem_max_MiB", NA),
                    "gpu_util_mean_pct": g.get("util_mean_pct", NA),
                    "params": v.get("params"),
                    "weights_MB": round(v["best_bytes"] / 2**20, 2) if v.get("best_bytes") else NA,
                }
            )
    rt = (c.res("reid") or {}).get("train", {})
    if rt:
        cost.append(
            {
                "model": "reid_head",
                "task": "reid(ABO triplet)",
                "train_time_sec": rt.get("train_sec"),
                "epochs": rt.get("epochs_run"),
                "epoch_time_mean_sec": NA,
                "batch": c.p["reid"]["batch_items"],
                "device": c.device,
                "gpu_mem_max_MiB": (
                    gsum.get("train_reid", {}) if isinstance(gsum, dict) else {}
                ).get("mem_max_MiB", NA),
                "gpu_util_mean_pct": NA,
                "params": NA,
                "weights_MB": NA,
            }
        )
    if cost:
        write_csv(out / "training_cost.csv", cost, list(cost[0].keys()))
    vid = []
    for lab in ("B0", "B1"):
        mp = c.run / "video" / lab / lab / "run_manifest.json"
        if mp.is_file():
            m = read_json(mp)
            t = m["timing"]
            v = m["video"]
            dur = v["frames_processed"] / v["fps_reported"] if v.get("fps_reported") else None
            vid.append(
                {
                    "model": lab,
                    "device": m["device"],
                    "frames": v["frames_processed"],
                    "video_sec_processed": round(dur, 2) if dur else NA,
                    "model_load_sec": t["model_load_and_warmup_s"],
                    "process_sec": t["frame_loop_total_s"],
                    "fps": t.get("processed_fps"),
                    "sec_per_video_minute": round(t["frame_loop_total_s"] / dur * 60, 2)
                    if dur
                    else NA,
                    "latency_ms_per_frame": round(
                        1000 * t["frame_loop_total_s"] / max(1, v["frames_processed"]), 2
                    ),
                }
            )
    if vid:
        write_csv(out / "video_processing_cost.csv", vid, list(vid[0].keys()))
    dl_bytes = 0
    dlp = c.run / "datasets" / "downloads.jsonl"
    if dlp.is_file():
        for ln in dlp.read_text(encoding="utf-8").splitlines():
            dl_bytes += json.loads(ln).get("bytes", 0)
    ds = (
        read_json(c.run / "datasets" / "dataset_stats.json")
        if (c.run / "datasets" / "dataset_stats.json").is_file()
        else {}
    )
    img_dl = 0
    for v in ds.values():
        d = v.get("image_downloads") if isinstance(v, dict) else None
        if not isinstance(d, dict):
            continue
        parts = [d] if "requested" in d else list(d.values())  # ABO: 단일 통계, LVIS: 분할별 통계
        img_dl += sum(p.get("downloaded_bytes", 0) for p in parts if isinstance(p, dict))
    total_gpu_sec = sum(
        v.get("seconds", 0)
        for k, v in c.state["stages"].items()
        if k
        in (
            "train_detection",
            "train_reid",
            "matching_b3",
            "damage_b4",
            "eval_detection",
            "eval_tracking",
            "video_reanalysis",
            "baseline_b0_video",
        )
    )
    summary = {
        "detection_rows": len(det_rows),
        "ablation_rows": len(abl),
        "dataset_archive_bytes": dl_bytes,
        "dataset_image_download_bytes_this_run": img_dl,
        "gpu_stage_wall_sec": round(total_gpu_sec, 1),
        "gpu_usage": gsum,
        "device": c.dev,
        "files": sorted(p.name for p in out.glob("*.csv")),
    }
    write_json(out / "compare_summary.json", summary)
    return {k: summary[k] for k in ("detection_rows", "ablation_rows", "gpu_stage_wall_sec")}
