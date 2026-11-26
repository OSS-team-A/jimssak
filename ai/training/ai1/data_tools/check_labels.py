#!/usr/bin/env python
"""YOLO-seg 데이터셋 검수: 클래스·좌표·이미지↔라벨 매칭·분할 누수 확인 + 샘플 마스크 오버레이 저장.

예) python data_tools/check_labels.py --data /workspace/jimssak_data/datasets/v1/data.yaml \
        --frames-manifest /workspace/jimssak_data/frames_v1/frames_manifest.csv --samples 20

문제가 있으면 0이 아닌 코드로 종료한다. 오버레이(<dataset>/label_check/*.jpg)를 눈으로 열어
윤곽이 물체와 맞는지, 같은 사진의 대상 물품이 빠짐없이 라벨됐는지 확인한다.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jimssak_ai1.common import (
    PROJECT_ROOT,
    fail,
    imread,
    imwrite,
    load_yaml,
    log,
    read_csv,
    write_json,
)  # noqa: E402
from jimssak_ai1.taxonomy import load_taxonomy  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO-seg 라벨 검수")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument(
        "--frames-manifest", type=Path, default=None, help="분할 누수(같은 방이 여러 split) 검사용"
    )
    ap.add_argument("--samples", type=int, default=20, help="split별 오버레이 샘플 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "taxonomy.yaml")
    args = ap.parse_args(argv)

    data = load_yaml(args.data)
    root = Path(data.get("path") or args.data.parent)
    names = data["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    tax = load_taxonomy(args.taxonomy)
    problems = []
    bad_names = [n for n in names if n not in tax.codes]
    if bad_names:
        problems.append(f"taxonomy에 없는 클래스명: {bad_names}")
    rng = random.Random(args.seed)
    counts = defaultdict(Counter)
    out_dir = root / "label_check"
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        img_dir = root / data[split]
        lbl_dir = root / str(data[split]).replace("images", "labels", 1)
        imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        lbls = {p.stem: p for p in lbl_dir.rglob("*.txt")}
        orphan = set(lbls) - {p.stem for p in imgs}
        if orphan:
            problems.append(f"{split}: 이미지 없는 라벨 {len(orphan)}개 (예: {sorted(orphan)[:3]})")
        no_label = [p.name for p in imgs if p.stem not in lbls]
        if no_label:
            problems.append(
                f"{split}: 라벨 파일 없는 이미지 {len(no_label)}장 (예: {no_label[:3]})"
            )
        for ip in imgs:
            lp = lbls.get(ip.stem)
            if lp is None:
                continue
            for ln, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
                t = line.split()
                if not t:
                    continue
                try:
                    cid = int(t[0])
                    vals = np.array([float(x) for x in t[1:]])
                except ValueError:
                    problems.append(f"{lp.name}:{ln} 숫자가 아님")
                    continue
                if not 0 <= cid < len(names):
                    problems.append(f"{lp.name}:{ln} 클래스 id {cid} 범위 밖(0~{len(names) - 1})")
                    continue
                if len(vals) < 6 or len(vals) % 2:
                    problems.append(f"{lp.name}:{ln} 다각형 좌표 수 이상({len(vals)})")
                    continue
                if vals.min() < 0 or vals.max() > 1:
                    problems.append(f"{lp.name}:{ln} 정규화 좌표가 0~1 밖")
                pts = vals.reshape(-1, 2)
                area = 0.5 * abs(
                    np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                    - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
                )
                if area < 1e-5:
                    problems.append(f"{lp.name}:{ln} 면적이 거의 0인 다각형")
                counts[split][names[cid]] += 1
            counts[split]["_images"] += 1
        for ip in rng.sample(imgs, min(args.samples, len(imgs))):
            img = imread(ip)
            h, w = img.shape[:2]
            ov = img.copy()
            lp = lbls.get(ip.stem)
            for line in lp.read_text(encoding="utf-8").splitlines() if lp else []:
                t = line.split()
                if len(t) < 7:
                    continue
                cid = int(t[0])
                pts = (np.array([float(x) for x in t[1:]]).reshape(-1, 2) * [w, h]).astype(np.int32)
                color = tuple(int(c) for c in np.random.default_rng(cid).integers(60, 255, 3))
                cv2.fillPoly(ov, [pts], color)
                cv2.polylines(img, [pts], True, color, 3)
                cv2.putText(img, names[cid], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            vis = cv2.addWeighted(img, 0.6, ov, 0.4, 0)
            s = 960 / max(h, w)
            imwrite(
                out_dir / split / f"{ip.stem}.jpg",
                cv2.resize(vis, (int(w * s), int(h * s))),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )

    if args.frames_manifest:
        room_splits = defaultdict(set)
        for r in read_csv(args.frames_manifest):
            room_splits[r["room_id"]].add(r["split"])
        leak = {k: sorted(v) for k, v in room_splits.items() if len(v) > 1}
        if leak:
            problems.append(f"같은 방이 여러 split에 있음(누수): {leak}")

    report = {"counts": {k: dict(v) for k, v in counts.items()}, "problems": problems}
    write_json(root / "label_check_report.json", report)
    for s, c in counts.items():
        log(
            f"{s}: 이미지 {c['_images']} | "
            + ", ".join(f"{k} {v}" for k, v in sorted(c.items()) if k != "_images")
        )
    log(f"오버레이 샘플: {out_dir}")
    if problems:
        for p in problems[:40]:
            log(f"문제: {p}")
        fail(f"라벨 문제 {len(problems)}건 (전체: {root / 'label_check_report.json'})", code=1)
    log("라벨 형식 검사 통과 (윤곽 정확도·누락 여부는 오버레이를 눈으로 확인)")


if __name__ == "__main__":
    main()
