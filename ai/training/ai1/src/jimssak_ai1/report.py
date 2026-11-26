"""그래프(PNG)·사례 이미지·최종 보고서(Markdown) 생성. 측정된 값만 사용하고, 없는 값은 '미측정'과
사유를 적는다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .common import imread, imwrite, load_yaml, log, read_csv, read_json

NA = "미측정"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for f in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        try:
            from matplotlib import font_manager

            if any(f in x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = f
                break
        except Exception:
            pass
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _results(run_dir: Path):
    p = Path(run_dir) / "results.csv"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def plot_training(plt, runs: dict, out: Path):
    files = []
    for name, rd in runs.items():
        rows = _results(rd)
        if not rows:
            continue
        ep = [int(float(r["epoch"])) for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for k in rows[0]:
            if k.startswith(("train/", "val/")) and "loss" in k:
                ax[0].plot(ep, [float(r[k]) for r in rows], label=k)
        ax[0].set_title(f"{name} loss")
        ax[0].set_xlabel("epoch")
        ax[0].legend(fontsize=6)
        for k in rows[0]:
            if k.startswith("metrics/precision") or k.startswith("metrics/recall"):
                ax[1].plot(ep, [float(r[k]) for r in rows], label=k)
        ax[1].set_title("Precision / Recall (val)")
        ax[1].legend(fontsize=7)
        ax[1].set_xlabel("epoch")
        for k in rows[0]:
            if k.startswith("metrics/mAP"):
                ax[2].plot(ep, [float(r[k]) for r in rows], label=k)
        ax[2].set_title("mAP (val)")
        ax[2].legend(fontsize=7)
        ax[2].set_xlabel("epoch")
        fig.tight_layout()
        p = out / f"train_curves_{name}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(p.name)
    return files


def plot_confusion(plt, eval_dir: Path, label: str, out: Path):
    p = eval_dir / "confusion_matrix.csv"
    if not p.is_file():
        return None
    rows = read_csv(p)
    labs = [r["gt\\pred"] for r in rows]
    M = np.array([[float(r[lb]) for lb in labs] for r in rows])
    keep = [i for i in range(len(labs)) if M[i].sum() + M[:, i].sum() > 0]
    M = M[np.ix_(keep, keep)]
    labs = [labs[i] for i in keep]
    N = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(labs) * 0.45), max(5, len(labs) * 0.4)))
    im = ax.imshow(N, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Confusion matrix (row-normalized) - {label}")
    fig.colorbar(im)
    fig.tight_layout()
    f = out / f"confusion_{label}.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    return f.name


def bar(plt, labels, series: dict, title, out: Path, fname, ylabel=""):
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.6), 4.5))
    for i, (k, v) in enumerate(series.items()):
        vals = [float(a) if a not in (None, "", NA, "미측정") else 0 for a in v]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=120)
    plt.close(fig)
    return fname


def md_table(rows: list[dict], cols: list[str] | None = None, max_rows: int = 60) -> str:
    if not rows:
        return "_데이터 없음 (미측정)_\n"
    cols = cols or list(rows[0].keys())
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows[:max_rows]:
        s += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return s


def case_images(c, out: Path):
    """탐지 성공·실패 사례: 테스트 이미지에서 FN/FP가 많은 것과 모두 맞힌 것 (정답 초록, 예측
    빨강).
    """
    from ultralytics import YOLO

    from .detection.evaluate import _label_path, read_gt

    b1 = c.res("best_b1")
    d = c.run / "eval" / "detection" / "lvis_test" / b1["label"]
    if not (d / "per_image.csv").is_file():
        return []
    rows = read_csv(d / "per_image.csv")
    fail = sorted(rows, key=lambda r: -(int(r["fn"]) + int(r["fp"])))[:3]
    ok = [r for r in rows if int(r["fn"]) == 0 and int(r["fp"]) == 0 and int(r["n_gt"]) >= 2][:3]
    data = load_yaml(c.ds / "lvis_seg" / "data.yaml")
    names = [data["names"][i] for i in sorted(data["names"])]
    models = {"B0": YOLO(str(c.pre / "yolo11n-seg.pt")), "B1": YOLO(b1["weights"])}
    files = []
    for tag, sel in (("success", ok), ("failure", fail)):
        for r in sel:
            ip = c.ds / "lvis_seg" / "images" / "test" / r["image"]
            img = imread(ip)
            h, w = img.shape[:2]
            gts = read_gt(_label_path(ip), w, h, names)
            panels = []
            for ml, m in models.items():
                cmap, _ = c.tax.model_class_to_code({int(k): v for k, v in m.names.items()})
                vis = img.copy()
                for cls, b, _ in gts:
                    cv2.rectangle(
                        vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), 2
                    )
                    cv2.putText(vis, cls, (int(b[0]), int(b[1]) + 14), 0, 0.5, (0, 200, 0), 1)
                pr = m.predict(img, conf=0.25, device=c.device, verbose=False)[0]
                for i in range(len(pr.boxes)):
                    code = cmap.get(int(pr.boxes.cls[i]))
                    if code is None:
                        continue
                    b = pr.boxes.xyxy[i].cpu().numpy()
                    cv2.rectangle(
                        vis, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 230), 2
                    )
                    cv2.putText(
                        vis,
                        f"{code} {float(pr.boxes.conf[i]):.2f}",
                        (int(b[0]), int(b[3]) - 4),
                        0,
                        0.5,
                        (0, 0, 230),
                        1,
                    )
                cv2.putText(vis, ml, (8, 24), 0, 0.8, (255, 255, 255), 3)
                cv2.putText(vis, ml, (8, 24), 0, 0.8, (0, 0, 0), 1)
                panels.append(vis)
            f = f"case_{tag}_{ip.stem}.jpg"
            imwrite(out / f, np.hstack(panels))
            files.append(f)
    return files


def video_compare(c, out: Path):
    """입력 영상 B0 vs B1 결과 영상의 같은 시점 프레임 비교."""
    files = []
    p0 = c.run / "video" / "B0" / "B0" / "annotated.mp4"
    p1 = c.run / "video" / "B1" / "B1" / "annotated.mp4"
    if not (p0.is_file() and p1.is_file()):
        return files
    c0, c1 = cv2.VideoCapture(str(p0)), cv2.VideoCapture(str(p1))
    n = int(min(c0.get(cv2.CAP_PROP_FRAME_COUNT), c1.get(cv2.CAP_PROP_FRAME_COUNT)))
    for k, fi in enumerate(np.linspace(0, max(0, n - 1), 4).astype(int)):
        c0.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        c1.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok0, f0 = c0.read()
        ok1, f1 = c1.read()
        if ok0 and ok1:
            f = f"video_B0_vs_B1_frame{fi:05d}.jpg"
            imwrite(out / f, np.hstack([f0, f1]))
            files.append(f)
    return files


def build_report(c) -> dict:
    plt = _plt()
    out = c.run / "report"
    out.mkdir(parents=True, exist_ok=True)
    figs = []
    seg = c.res("seg_models") or {}
    runs = {k: v["run_dir"] for k, v in seg.items()}
    for key, lab in (("homeobj_det", "homeobjects_det"), ("crack", "crack_seg")):
        if c.res(key):
            runs[lab] = c.res(key)["run_dir"]
    figs += plot_training(plt, runs, out)
    b1 = (c.res("best_b1") or {}).get("label")
    for lab in ("B0_yolo11n-seg_COCO", b1):
        if lab:
            f = plot_confusion(plt, c.run / "eval" / "detection" / "lvis_test" / lab, lab, out)
            if f:
                figs.append(f)
    cmp = c.run / "compare"
    pcl = (
        read_csv(cmp / "per_class_before_after.csv")
        if (cmp / "per_class_before_after.csv").is_file()
        else []
    )
    if pcl:
        models = sorted({k.split("|")[0] for r in pcl for k in r if "|" in k})
        figs.append(
            bar(
                plt,
                [r["class"] + ("*" if r["group"] == "new" else "") for r in pcl],
                {m: [r.get(f"{m}|mask_AP50_95") for r in pcl] for m in models},
                "Per-class mask AP50-95 (LVIS test, *=new class unsupported by B0)",
                out,
                "per_class_mask_ap.png",
                "AP",
            )
        )
    abl = (
        read_csv(cmp / "ablation_service_synthetic_video.csv")
        if (cmp / "ablation_service_synthetic_video.csv").is_file()
        else []
    )
    abl_ok = [r for r in abl if r.get("item_f1") not in (None, "")]
    if abl_ok:
        figs.append(
            bar(
                plt,
                [r["step"] for r in abl_ok],
                {
                    k: [r[k] for r in abl_ok]
                    for k in (
                        "item_f1",
                        "duplicate_rate",
                        "wrong_merge_rate",
                        "reappearance_link_accuracy",
                        "IDF1",
                        "HOTA",
                    )
                },
                "Ablation on synthetic camera-motion videos (service metrics)",
                out,
                "ablation_tracking.png",
            )
        )
    reid = (
        read_csv(c.run / "eval" / "reid" / "reid_comparison.csv")
        if (c.run / "eval" / "reid" / "reid_comparison.csv").is_file()
        else []
    )
    if reid:
        figs.append(
            bar(
                plt,
                [r["method"] for r in reid],
                {
                    k: [r.get(k) for r in reid]
                    for k in ("test_recall@1", "test_mAP", "test_pair_auroc")
                },
                "Re-identification (ABO test products, cross-view retrieval)",
                out,
                "reid.png",
            )
        )
    b3 = (
        read_csv(c.run / "eval" / "matching" / "b3_comparison.csv")
        if (c.run / "eval" / "matching" / "b3_comparison.csv").is_file()
        else []
    )
    if b3:
        figs.append(
            bar(
                plt,
                [r["method"] for r in b3],
                {
                    k: [r.get(k) for r in b3]
                    for k in ("missing_f1", "false_missing_rate", "match_accuracy")
                },
                "B3 before/after matching (synthetic scenarios)",
                out,
                "b3_matching.png",
            )
        )
    b4 = (
        read_json(c.run / "eval" / "damage" / "b4_eval.json")
        if (c.run / "eval" / "damage" / "b4_eval.json").is_file()
        else {}
    )
    if b4:
        cs = b4["real_data"]["crack_segmentation"]
        labs = [k for k in cs if isinstance(cs[k], dict)]
        figs.append(
            bar(
                plt,
                labs,
                {k: [cs[lb].get(k) for lb in labs] for k in ("mask_mAP50_95", "recall", "f1")},
                "B4 crack segmentation before/after fine-tuning (crack-seg test, real)",
                out,
                "b4_crack.png",
            )
        )
        ch = b4["synthetic_data"]["before_after_change"]
        ml = [k for k in ("pixel_diff", "feature_diff") if k in ch]
        figs.append(
            bar(
                plt,
                ml,
                {
                    lab: [ch[m].get(k) for m in ml]
                    for lab, k in (
                        ("new_damage_f1", "new_damage_f1"),
                        ("auroc", "auroc"),
                        ("false_alarm_unchanged", "false_alarm_rate_unchanged(조명·시점 변화만)"),
                        ("change_region_iou", "change_region_iou_mean"),
                    )
                },
                "B4 before/after change detection (synthetic)",
                out,
                "b4_change.png",
            )
        )
    gp = c.run / "logs" / "gpu_usage.csv"
    if gp.is_file():
        rows = read_csv(gp)
        if rows:
            fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            for idx in sorted({r["index"] for r in rows}):
                rr = [r for r in rows if r["index"] == idx]
                ax[0].plot([float(r["memory.used"]) for r in rr], label=f"GPU{idx}")
                ax[1].plot([float(r["utilization.gpu"]) for r in rr], label=f"GPU{idx}")
            ax[0].set_ylabel("memory MiB")
            ax[1].set_ylabel("util %")
            ax[1].set_xlabel("sample (fixed interval)")
            ax[0].legend()
            fig.tight_layout()
            fig.savefig(out / "gpu_usage.png", dpi=110)
            plt.close(fig)
            figs.append("gpu_usage.png")
    hist = c.run / "models" / "reid" / "history.csv"
    if hist.is_file():
        rows = read_csv(hist)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot([int(r["epoch"]) for r in rows], [float(r["loss"] or 0) for r in rows])
        ax[0].set_title("re-ID triplet loss")
        ax[1].plot([int(r["epoch"]) for r in rows], [float(r["val_recall@1"] or 0) for r in rows])
        ax[1].set_title("val Recall@1")
        fig.tight_layout()
        fig.savefig(out / "reid_training.png", dpi=110)
        plt.close(fig)
        figs.append("reid_training.png")
    try:
        figs += case_images(c, out)
    except Exception as e:
        log(f"  사례 이미지 생성 실패(보고서는 계속): {e!r}")
    figs += video_compare(c, out)
    write_md(c, out, figs)
    return {"figures": len(figs), "report": str(out / "REPORT.md")}


def write_md(c, out: Path, figs: list[str]):
    S = c.state["stages"]
    cmp = c.run / "compare"

    def rd(p):
        return read_csv(p) if Path(p).is_file() else []

    prof = c.p["profile"]
    L = [
        f"# 짐싹 AI 학습·평가 결과 보고서 ({c.run.name})",
        "",
        f"- 프로필: **{prof}** — {c.p.get('description', '')}",
        f"- 장치: {c.dev} / 생성: {c.state.get('created_at')}",
        "- 모든 수치는 이 실행에서 실제로 측정한 값이다. 표에 없는 지표는 '미측정'이며 사유를 "
        "함께 적었다.",
        "- **실제 공개 데이터 테스트**와 **합성 실험**은 섹션을 나눠 표시한다. 합성 결과로 실제 "
        "성능을 주장하지 않는다.",
        "",
    ]
    if prof == "smoke":
        L += [
            "> ⚠️ smoke 프로필: 최소 데이터·짧은 학습으로 파이프라인 동작만 점검한 결과다. 성능 "
            "근거로 쓰지 않는다.",
            "",
        ]
    L += [
        "## 1. 단계별 실행 결과",
        md_table(
            [
                {
                    "단계": k,
                    "상태": "진행 중(이 보고서 작성)" if k == "report" else v.get("status"),
                    "초": v.get("seconds"),
                }
                for k, v in S.items()
            ]
        ),
    ]
    L += [
        "## 2. 데이터셋",
        "출처·라이선스: `datasets/licenses.csv`, 무결성: `datasets/integrity.csv`, 클래스 확보 "
        "현황: `datasets/class_availability.csv`",
        "",
        md_table(
            rd(c.run / "datasets" / "class_availability.csv"),
            [
                "class",
                "lvis_categories",
                "train_instances",
                "val_instances",
                "test_instances",
                "selected",
                "reason",
            ],
        ),
        md_table(
            rd(c.run / "datasets" / "licenses.csv"), ["dataset", "license", "role", "domain_gap"]
        ),
    ]
    L += [
        "## 3. 물품 탐지·세그멘테이션 학습 전후 (실제 공개 데이터: LVIS 테스트 = COCO val2017 "
        "이미지)",
        "공통 클래스 = B0(COCO)도 지원하는 클래스, 신규 클래스 = B0 미지원(학습으로 새로 추가). "
        "LVIS 연합 주석 규칙 적용.",
        "",
        md_table(rd(cmp / "detection_before_after.csv")),
    ]
    if (cmp / "homeobjects_box_eval.csv").is_file():
        L += [
            "### 실내 도메인 독립 테스트 (HomeObjects-3K val, 박스만)",
            md_table(rd(cmp / "homeobjects_box_eval.csv")),
        ]
    L += [
        "## 4. 추적·재식별·중복 집계 (합성 카메라 이동 영상 — 2D 이동·확대·가림, 실제 3D 시점 "
        "변화 미포함)",
        md_table(rd(cmp / "ablation_service_synthetic_video.csv")),
        "### 재식별 임베딩 (실제 공개 데이터: ABO 테스트 상품의 다른 사진 검색)",
        md_table(rd(c.run / "eval" / "reid" / "reid_comparison.csv")),
    ]
    L += [
        "## 5. 이사 전후 매칭·분실 의심 (B3, 합성 시나리오)",
        "실제 이사 전후 영상 평가: **미측정** (영상 쌍·정답 없음)",
        "",
        md_table(
            rd(c.run / "eval" / "matching" / "b3_comparison.csv"),
            [
                "method",
                "threshold",
                "missing_precision",
                "missing_recall",
                "missing_f1",
                "false_missing_rate",
                "missed_missing_rate",
                "match_accuracy",
                "unexpected_precision",
                "unexpected_recall",
                "uncertain_items",
            ],
        ),
    ]
    b4 = c.run / "eval" / "damage" / "b4_eval.json"
    if b4.is_file():
        d = read_json(b4)
        L += [
            "## 6. 파손 탐지·상태 비교 (B4)",
            "### 실제 데이터",
            "```json",
            json.dumps(d["real_data"], ensure_ascii=False, indent=1)[:6000],
            "```",
            "### 합성 데이터 (실제 성능 근거 아님)",
            "```json",
            json.dumps(d["synthetic_data"], ensure_ascii=False, indent=1)[:6000],
            "```",
            f"- 실제 가구 손상: {d.get('real_furniture_damage')}",
        ]
    L += [
        "## 7. 입력 영상(언니방영상) 학습 전후 재분석",
        "정답 목록이 없어 정확도는 **미측정**. 자동 확정 물품 수 비교:",
        md_table(rd(c.run / "video" / "video_item_counts_B0_B1_B2.csv")),
        "검수표: `video/match_review_B0.csv`, `video/match_review_B1.csv`. 최종 API JSON: "
        "`api/items_B2_final.json`",
    ]
    L += [
        "## 8. 학습·실행 비용",
        md_table(rd(cmp / "training_cost.csv")),
        md_table(rd(cmp / "video_processing_cost.csv")),
        "- GPU 비용 환산: 미측정 (단가·과금 가정이 주어지지 않음)",
    ]
    L += ["## 9. 그림", *[f"![{f}]({f})" for f in figs]]
    L += [
        "## 10. 미측정 항목과 사유",
        "- 실제 이사 전후 영상 기반 분실 판단 정확도: 실제 전후 영상 쌍·정답 없음",
        "- 실제 가구 손상 탐지 성능: 인증 없이 받을 수 있는 가구 손상 라벨 데이터 없음",
        "- 입력 영상의 물품 목록 정확도(F1 등): 사람이 작성한 정답 목록 필요 (검수표 제공)",
        "- 실제 영상 추적의 HOTA/IDF1: 영상 추적 정답 없음 → 합성 영상으로만 측정",
        "- 실제 치수·부피, 60×40×40cm 박스 적합: Depth 모듈 미구현 → 판단 보류(unknown)",
    ]
    (out / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
