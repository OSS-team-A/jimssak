#!/usr/bin/env python
"""CVAT에서 'COCO 1.0'으로 내보낸 인스턴스 분할 라벨 → Ultralytics YOLO-seg 데이터셋.

- CVAT 라벨 이름은 configs/taxonomy.yaml 의 code와 정확히 같아야 한다 (예: bed, sofa, chair, desk
  ...).
- 클래스 id는 taxonomy 순서(other 제외)로 고정 → 데이터셋 버전이 바뀌어도 id가 유지된다.
- 'other' 라벨은 학습에서 제외한다(로그에 개수 출력).
- 한 물체가 가림으로 여러 조각 다각형이면 Ultralytics의 merge_multi_segment로 한 줄로 합친다.
- 라벨이 하나도 없는 이미지는 '라벨 누락'과 구분할 수 없으므로 기본 제외한다.
  모든 대상 물품을 확인한 '대상 없음' 이미지라면 --empty-as-negative 로 포함한다.
- 분할(train/val/test)은 extract_frames.py의 frames_manifest.csv에서 파일명으로 찾는다.

예) python data_tools/cvat_coco_to_yolo.py --coco cvat_export/annotations/instances_default.json \
        --frames-root /workspace/jimssak_data/frames_v1 --out /workspace/jimssak_data/datasets/v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jimssak_ai1.common import PROJECT_ROOT, fail, log, read_csv, write_json  # noqa: E402
from jimssak_ai1.taxonomy import load_taxonomy  # noqa: E402


def rle_to_polygons(seg: dict) -> list[list[float]]:
    h, w = seg["size"]
    counts = seg["counts"]
    if isinstance(counts, str):
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            fail(
                "압축 RLE 마스크가 있습니다. CVAT에서 다각형(polygon) 도구로 라벨링하거나 "
                "pycocotools를 설치하세요."
            )
        m = mask_utils.decode(seg)
    else:
        flat = np.zeros(h * w, np.uint8)
        pos, val = 0, 0
        for c in counts:
            flat[pos : pos + c] = val
            pos += c
            val = 1 - val
        m = flat.reshape((w, h)).T  # COCO RLE는 열 우선(column-major)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]


def main(argv=None):
    ap = argparse.ArgumentParser(description="CVAT COCO 1.0 → YOLO-seg 데이터셋")
    ap.add_argument(
        "--coco",
        required=True,
        type=Path,
        nargs="+",
        help="CVAT export의 instances_*.json (여러 개 가능)",
    )
    ap.add_argument(
        "--frames-root",
        required=True,
        type=Path,
        help="extract_frames.py 출력 폴더 (frames_manifest.csv 포함)",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="데이터셋 폴더 (images/<split>, labels/<split>, data.yaml)",
    )
    ap.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "taxonomy.yaml")
    ap.add_argument("--empty-as-negative", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    from ultralytics.data.converter import merge_multi_segment

    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        fail(f"이미 있습니다: {args.out} (데이터셋 버전 폴더를 새로 만들거나 --overwrite)")
    tax = load_taxonomy(args.taxonomy)
    names = [c for c in tax.codes if c != "other"]
    cid_of = {n: i for i, n in enumerate(names)}
    man = {Path(r["file"]).name: r for r in read_csv(args.frames_root / "frames_manifest.csv")}

    stats = defaultdict(Counter)
    skipped_other = 0
    empty, missing = [], []
    for cj in args.coco:
        coco = json.loads(cj.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        bad = sorted({n for n in cats.values() if n not in tax.codes})
        if bad:
            fail(f"{cj}: taxonomy에 없는 CVAT 라벨 이름 {bad}. CVAT 라벨을 code(영문)로 맞추세요.")
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        for im in coco["images"]:
            base = Path(im["file_name"]).name
            m = man.get(base)
            if m is None:
                missing.append(base)
                continue
            split = m["split"]
            w, h = int(im["width"]), int(im["height"])
            lines = []
            for a in anns.get(im["id"], []):
                name = cats[a["category_id"]]
                if name == "other":
                    skipped_other += 1
                    continue
                seg = a.get("segmentation")
                if isinstance(seg, dict):
                    parts = rle_to_polygons(seg)
                elif isinstance(seg, list) and seg:
                    parts = [p for p in seg if len(p) >= 6]
                else:
                    fail(
                        f"{base}: 다각형/마스크가 없는 라벨(박스만?) — 분할 학습에는 윤곽이 "
                        f"필요합니다 (annotation id {a.get('id')})"
                    )
                if not parts:
                    continue
                if len(parts) > 1:
                    poly = np.concatenate(merge_multi_segment(parts), axis=0)
                else:
                    poly = np.array(parts[0], dtype=float).reshape(-1, 2)
                poly = np.clip(poly / [w, h], 0, 1)
                lines.append(f"{cid_of[name]} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
                stats[split][name] += 1
            if not lines and not args.empty_as_negative:
                empty.append(base)
                continue
            src = args.frames_root / m["file"]
            if not src.is_file():
                fail(f"프레임 이미지가 없습니다: {src}")
            (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, args.out / "images" / split / base)
            (args.out / "labels" / split / (Path(base).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            stats[split]["_images"] += 1
    data_yaml = args.out / "data.yaml"
    splits = {
        s: f"images/{s}" for s in ("train", "val", "test") if (args.out / "images" / s).is_dir()
    }
    import yaml

    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(args.out.resolve()),
                **splits,
                "names": {i: n for i, n in enumerate(names)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(
        args.out / "conversion_report.json",
        {
            "coco": [str(p) for p in args.coco],
            "per_split": {k: dict(v) for k, v in stats.items()},
            "skipped_other_annotations": skipped_other,
            "excluded_empty_images": empty,
            "images_not_in_manifest": missing,
            "names": names,
        },
    )
    log(f"완료: {args.out}  (data.yaml 생성)")
    for s, c in stats.items():
        log(
            f"  {s}: 이미지 {c['_images']} | "
            + ", ".join(f"{k} {v}" for k, v in c.items() if k != "_images")
        )
    if empty:
        log(f"  라벨 없는 이미지 {len(empty)}장 제외 (대상 없음이 확실하면 --empty-as-negative)")
    if missing:
        log(f"  frames_manifest.csv에 없는 이미지 {len(missing)}장 제외: {missing[:5]}")
    if skipped_other:
        log(f"  'other' 라벨 {skipped_other}개는 학습에서 제외")


if __name__ == "__main__":
    main()
