"""합성 손상·합성 촬영 변화 (평가 전용). 합성 결과로 실제 가구 손상 탐지 성능이 입증됐다고 주장하지
않는다.

손상 유형: scratch(가는 밝은/어두운 선), crack(가지 친 꺾인 선), dent(국소 음영 타원),
discoloration(얼룩 색 변화)
촬영 변화(nuisance): 밝기·대비·색온도 변화, 작은 회전·이동·확대 — 실제 손상이 아닌 변화
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np


def foreground(img: np.ndarray) -> np.ndarray:
    """ABO 카탈로그 사진(대부분 흰 배경)의 물체 영역 추정."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg = (g < 235).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if fg.sum() < 0.05 * fg.size:
        fg[:] = 1
    return fg.astype(bool)


def _pt_in(fg, rng):
    ys, xs = np.nonzero(fg)
    i = rng.randrange(len(xs))
    return int(xs[i]), int(ys[i])


def add_damage(img: np.ndarray, kind: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    fg = foreground(img)
    out = img.copy()
    mask = np.zeros((h, w), np.uint8)
    s = max(h, w)
    if kind == "scratch":
        for _ in range(rng.randint(1, 3)):
            x, y = _pt_in(fg, rng)
            ang = rng.uniform(0, math.pi)
            L = rng.uniform(0.15, 0.4) * s
            x2, y2 = int(x + L * math.cos(ang)), int(y + L * math.sin(ang))
            col = (235, 235, 235) if rng.random() < 0.5 else (40, 40, 40)
            th = max(1, int(s / 180))
            cv2.line(out, (x, y), (x2, y2), col, th, cv2.LINE_AA)
            cv2.line(mask, (x, y), (x2, y2), 1, th + 2)
    elif kind == "crack":
        x, y = _pt_in(fg, rng)
        ang = rng.uniform(0, 2 * math.pi)
        pts = [(x, y)]
        for _ in range(rng.randint(5, 9)):
            ang += rng.uniform(-0.7, 0.7)
            step = rng.uniform(0.03, 0.07) * s
            x, y = int(x + step * math.cos(ang)), int(y + step * math.sin(ang))
            pts.append((x, y))
        th = max(1, int(s / 150))
        cv2.polylines(out, [np.array(pts)], False, (20, 20, 20), th, cv2.LINE_AA)
        cv2.polylines(mask, [np.array(pts)], False, 1, th + 2)
    elif kind == "dent":
        x, y = _pt_in(fg, rng)
        ax, ay = int(rng.uniform(0.04, 0.09) * s), int(rng.uniform(0.03, 0.07) * s)
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (x, y), (ax, ay), rng.uniform(0, 180), 0, 360, 1, -1)
        shade = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1, ax / 3))
        out = np.clip(out.astype(np.float32) * (1 - 0.45 * shade[..., None]), 0, 255).astype(
            np.uint8
        )
        mask = (shade > 0.25).astype(np.uint8)
    elif kind == "discoloration":
        x, y = _pt_in(fg, rng)
        r = int(rng.uniform(0.05, 0.1) * s)
        m = np.zeros((h, w), np.uint8)
        cv2.circle(m, (x, y), r, 1, -1)
        soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), r / 3)[..., None]
        tint = np.array(
            [rng.randint(30, 90), rng.randint(60, 120), rng.randint(90, 160)], np.float32
        )
        out = np.clip(out * (1 - 0.6 * soft) + tint * 0.6 * soft, 0, 255).astype(np.uint8)
        mask = (soft[..., 0] > 0.3).astype(np.uint8)
    mask = mask.astype(bool) & cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(
        bool
    )
    return out, mask


def nuisance(
    img: np.ndarray, rng: random.Random, strength: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """조명·색·작은 시점 변화. 반환: (변형 이미지, 이전→이후 2x3 변환행렬)"""
    h, w = img.shape[:2]
    ang = rng.uniform(-4, 4) * strength
    sc = 1 + rng.uniform(-0.05, 0.05) * strength
    tx, ty = rng.uniform(-0.03, 0.03) * w * strength, rng.uniform(-0.03, 0.03) * h * strength
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = out * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)
    out *= np.array([rng.uniform(0.9, 1.1), 1.0, rng.uniform(0.9, 1.1)], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), M
