"""합성 카메라 이동 영상 (추적·재등장 평가용, 정답 ID 포함).

LVIS 테스트 이미지 한 장을 '방'으로 보고, 가상 카메라 창이 왼쪽→오른쪽으로 훑은 뒤 처음 위치로
돌아온다(루프 복귀).
- 물체가 화면 밖으로 나갔다가 다시 들어온다(재등장).
- 확대/축소가 조금씩 바뀌고, 중간에 회색 가림막이 지나간다(부분 가림).
- 정답: 원본 정답 마스크를 같은 변환으로 옮겨 계산한 '보이는 부분' 박스와 객체 ID.
한계: 평면 이미지를 옮기는 2D 이동이라 실제 3D 시점 변화(다른 각도에서 본 같은 물건)는 재현하지
못한다.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

from ..common import imread


def load_objects(img_path: Path, label_path: Path, names: list[str]):
    img = imread(img_path)
    H, W = img.shape[:2]
    objs = []
    if label_path.is_file():
        for k, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            t = line.split()
            if len(t) < 7:
                continue
            poly = (np.array([float(x) for x in t[1:]]).reshape(-1, 2) * [W, H]).astype(np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly], 1)
            if m.sum() < 0.015 * H * W:
                continue
            objs.append({"id": k, "cls": names[int(t[0])], "mask": m})
    return img, objs


def pick_images(img_dir: Path, lbl_dir: Path, names: list[str], n: int, seed: int) -> list[Path]:
    cands = []
    for ip in sorted(img_dir.glob("*.jpg")):
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.is_file():
            continue
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) >= 2:
            cands.append(ip)
    rng = random.Random(seed)
    rng.shuffle(cands)
    out = []
    for ip in cands:
        _, objs = load_objects(ip, lbl_dir / (ip.stem + ".txt"), names)
        if len(objs) >= 2:
            out.append(ip)
        if len(out) >= n:
            break
    return out


def generate(
    img: np.ndarray, objs: list[dict], n_frames: int, out_size: tuple[int, int], seed: int
):
    """프레임 생성기: (frame BGR, [(obj_id, cls, box_xyxy)])"""
    rng = random.Random(seed)
    H, W = img.shape[:2]
    ow, oh = out_size
    a = ow / oh
    ww = min(W * 0.55, H * a * 0.95)
    wh = ww / a
    yc = H / 2 + rng.uniform(-0.05, 0.05) * H
    occ_w = int(0.18 * ow)
    occ_col = rng.randint(90, 160)
    for t in range(n_frames):
        u = t / max(1, n_frames - 1)
        if u < 0.45:
            p = u / 0.45
        elif u < 0.55:
            p = 1.0
        else:
            p = 1.0 - (u - 0.55) / 0.45
        zoom = 1.0 + 0.08 * math.sin(2 * math.pi * u)
        cw, ch = ww / zoom, wh / zoom
        cx = cw / 2 + p * (W - cw)
        cy = min(max(yc + 0.04 * H * math.sin(4 * math.pi * u), ch / 2), H - ch / 2)
        s = ow / cw
        M = np.array([[s, 0, -(cx - cw / 2) * s], [0, s, -(cy - ch / 2) * s]], np.float32)
        frame = cv2.warpAffine(
            img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        occ = np.zeros((oh, ow), bool)
        if 0.2 <= u <= 0.4:
            x0 = int((u - 0.2) / 0.2 * (ow + occ_w)) - occ_w
            x1 = max(0, x0)
            x2 = min(ow, x0 + occ_w)
            if x2 > x1:
                frame[:, x1:x2] = occ_col
                occ[:, x1:x2] = True
        gts = []
        for o in objs:
            m = cv2.warpAffine(
                o["mask"], M, (ow, oh), flags=cv2.INTER_NEAREST, borderValue=0
            ).astype(bool)
            full = o["mask"].sum() * s * s
            vis = m & ~occ
            nv = int(vis.sum())
            if nv >= 150 and nv >= 0.3 * full:
                ys, xs = np.nonzero(vis)
                gts.append(
                    (
                        o["id"],
                        o["cls"],
                        np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float),
                    )
                )
        yield frame, gts
