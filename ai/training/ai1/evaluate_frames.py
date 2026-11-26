#!/usr/bin/env python
"""프레임 단위 평가 (사람이 만든 정답 박스·마스크가 있는 평가 프레임 필요).

예) python -u evaluate_frames.py --label B0 --weights weights/yolo11n-seg.pt \
        --data /workspace/jimssak_data/datasets/v1/data.yaml --split test --device cuda:0
    python -u evaluate_frames.py --label B1 --weights runs/train/<name>/weights/best.pt --data ...
    --split test

출력: evaluations/frames_<label>_<시각>/ per_class.csv, summary.json
  - all_gt_classes: 정답에 있는 모든 품목 (모델 미지원 품목은 AP 0 → 새 품목 지원 효과 포함)
  - common_classes(--common-classes): B0·B1이 공통으로 지원하는 품목만 → 기존 품목 성능 유지 여부
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import (  # noqa: E402
    PROJECT_ROOT,
    environment_info,
    fail,
    imread,
    load_yaml,
    log,
    make_run_dir,
    now_iso,
    now_stamp,
    resolve_device,
    safe_id,
    set_offline_friendly_ultralytics_env,
    sha256_file,
    write_csv,
    write_json,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path_for(img_path: Path) -> Path:
    """images/<split>/x.jpg → labels/<split>/x.txt (Ultralytics 규칙: 마지막 'images' 폴더를
    'labels'로).
    """
    parts = list(img_path.parts)
    if "images" not in parts:
        fail(f"이미지 경로에 'images' 폴더가 없습니다(Ultralytics 데이터셋 구조 필요): {img_path}")
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def mean_of(rows, key, classes):
    vals = [r[key] for r in rows if r["category"] in classes and isinstance(r[key], (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="프레임 단위 box/mask mAP50-95·품목별 Recall 평가",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--label", required=True)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument(
        "--data", required=True, type=Path, help="Ultralytics data.yaml (names = taxonomy code)"
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="mAP 계산용 예측 하한 (Ultralytics val 기본과 동일)",
    )
    ap.add_argument(
        "--recall-conf",
        type=float,
        default=0.25,
        help="품목별 Recall 계산 시 운영 confidence 임계값",
    )
    ap.add_argument(
        "--common-classes",
        default=None,
        help="공통 지원 품목 (쉼표 구분). 생략 시 이 모델이 지원하는 정답 품목",
    )
    ap.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "taxonomy.yaml")
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "evaluations")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args(argv)
    set_offline_friendly_ultralytics_env()

    from jimssak_ai1.frameeval import PredInst, accumulate_image, read_yolo_seg_label, summarize
    from jimssak_ai1.taxonomy import load_taxonomy
    from ultralytics import YOLO

    data = load_yaml(args.data)
    root = Path(data.get("path") or args.data.parent)
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    split_rel = data.get(args.split)
    if not split_rel:
        fail(f"data.yaml에 '{args.split}' 분할이 없습니다: {args.data}")
    img_dir = root / split_rel
    names = data["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    tax = load_taxonomy(args.taxonomy)
    unknown = [n for n in names if n not in tax.codes]
    if unknown:
        fail(f"데이터셋 클래스명이 taxonomy code가 아닙니다: {unknown}")
    images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        fail(f"평가 이미지가 없습니다: {img_dir}")

    device = resolve_device(args.device)
    model = YOLO(str(args.weights))
    mnames = {int(k): v for k, v in model.names.items()}
    cls_to_code, mode = tax.model_class_to_code(mnames)
    supported = set(cls_to_code.values())

    stats, all_poly = {}, True
    for k, ip in enumerate(images, 1):
        lp = label_path_for(ip)
        img = imread(ip)
        h, w = img.shape[:2]
        gts, poly_ok = read_yolo_seg_label(lp, w, h, names)
        all_poly &= poly_ok
        r = model.predict(
            img, conf=args.conf, imgsz=args.imgsz, retina_masks=True, device=device, verbose=False
        )[0]
        preds = []
        if len(r.boxes):
            masks = r.masks.data.cpu().numpy() > 0.5 if r.masks is not None else None
            for i in range(len(r.boxes)):
                code = cls_to_code.get(int(r.boxes.cls[i]))
                if code is None:
                    continue
                preds.append(
                    PredInst(
                        code,
                        float(r.boxes.conf[i]),
                        r.boxes.xyxy[i].cpu().numpy(),
                        masks[i] if masks is not None else None,
                    )
                )
        accumulate_image(stats, gts, preds, w, h, with_mask=True, recall_conf=args.recall_conf)
        if k % 50 == 0:
            log(f"  {k}/{len(images)}")
    with_mask = all_poly
    if not with_mask:
        log("정답에 박스만 있는 라벨이 있어 mask mAP는 '미측정'으로 둡니다")
    rows = summarize(stats, with_mask, args.recall_conf)
    gt_classes = {r["category"] for r in rows if r["n_gt_instances"] > 0}
    common = (
        {c.strip() for c in args.common_classes.split(",")} if args.common_classes else supported
    ) & gt_classes
    for r in rows:
        r["supported_by_model"] = "yes" if r["category"] in supported else "no(미지원)"

    out = make_run_dir(
        args.output_root, args.run_name or f"frames_{now_stamp()}_{safe_id(args.label)}"
    )
    write_csv(out / "per_class.csv", rows, list(rows[0].keys()) if rows else ["category"])
    summary = {
        "label": args.label,
        "created_at": now_iso(),
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "data": str(args.data),
        "split": args.split,
        "n_images": len(images),
        "device": device,
        "class_mapping_mode": mode,
        "conf_for_map": args.conf,
        "recall_conf": args.recall_conf,
        "imgsz": args.imgsz,
        "mask_gt_available": with_mask,
        "all_gt_classes": {
            "classes": sorted(gt_classes),
            "box_mAP50_95": mean_of(rows, "box_AP50_95", gt_classes),
            "mask_mAP50_95": mean_of(rows, "mask_AP50_95", gt_classes) if with_mask else "미측정",
        },
        "common_classes": {
            "classes": sorted(common),
            "box_mAP50_95": mean_of(rows, "box_AP50_95", common),
            "mask_mAP50_95": mean_of(rows, "mask_AP50_95", common) if with_mask else "미측정",
        },
        "unsupported_gt_classes": sorted(gt_classes - supported),
        "note": "미지원 품목은 예측이 없으므로 AP=0으로 all_gt_classes 평균에 포함. YOLO "
        "confidence는 정확도가 아님.",
        "environment": environment_info(),
    }
    write_json(out / "summary.json", summary)
    log(f"완료: {out}")
    log(
        f"  이미지 {len(images)} | 전체 정답 품목 box "
        f"mAP50-95 {summary['all_gt_classes']['box_mAP50_95']} "
        f"mask {summary['all_gt_classes']['mask_mAP50_95']} | 공통 품목 "
        f"box {summary['common_classes']['box_mAP50_95']} "
        f"mask {summary['common_classes']['mask_mAP50_95']}"
    )


if __name__ == "__main__":
    main()
