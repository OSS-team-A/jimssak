"""간단한 외형 기술자: 마스크 영역의 HSV 색 히스토그램.

재등장 연결의 '외형 유사도' 근거로만 쓴다. 외형 유사도만으로 병합을 자동 확정하지 않는다
(link_reappearance.py 참고). 더 강한 특징(DINOv2 등)은 필요성이 확인될 때만 검토한다.
"""

from __future__ import annotations

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 16, 4, 4
DESC_DIM = H_BINS * S_BINS * V_BINS


def masked_hsv_hist(crop_bgr: np.ndarray, crop_mask: np.ndarray | None) -> np.ndarray | None:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    # 계산량을 줄이기 위해 긴 변 128px로 축소
    h, w = crop_bgr.shape[:2]
    scale = 128.0 / max(h, w)
    if scale < 1.0:
        crop_bgr = cv2.resize(
            crop_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
        )
        if crop_mask is not None:
            crop_mask = cv2.resize(
                crop_mask.astype(np.uint8),
                (crop_bgr.shape[1], crop_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
    m = None
    if crop_mask is not None:
        m = (crop_mask > 0).astype(np.uint8)
        if int(m.sum()) < 20:
            return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], m, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    s = float(hist.sum())
    if s <= 0:
        return None
    return hist / s


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Bhattacharyya 거리 (0~1, 클수록 비슷)."""
    d = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, min(1.0, 1.0 - d)))


def track_similarity(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """두 트랙의 샘플 히스토그램 집합 간 유사도: 각 샘플의 최대 유사도를 양방향 평균."""
    if len(samples_a) == 0 or len(samples_b) == 0:
        return float("nan")
    sims = np.array(
        [[hist_similarity(x, y) for y in samples_b] for x in samples_a], dtype=np.float32
    )
    return float(0.5 * (sims.max(axis=1).mean() + sims.max(axis=0).mean()))
