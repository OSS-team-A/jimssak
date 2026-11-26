"""이사 전후 같은 물품의 상태 변화 비교 (B4 상태 비교).

절차: (1) 이후 이미지를 이전 이미지에 정합(ECC affine) (2) 밝기·색 정규화 (3) 변화 지도 계산 (4)
상태 판정
- 기준선 pixel_diff: 정합 후 회색조 절대 차이
- 개선 feature_diff: 정합 후 WRN50 패치 특징으로 '이전 이미지 패치 메모리' 대비 최근접 거리
  (PatchCore 방식을 물품별로 적용)
  → 조명·색 변화에는 덜 민감하고 새로 생긴 구조(선·균열·찍힘)에 반응하도록 설계
상태: not_comparable(매칭 신뢰도 낮음/정합 실패), newly_detected_damage, uncertain,
pre_existing_damage, unchanged
"""

from __future__ import annotations

import cv2
import numpy as np

from .patchcore import FeatureExtractor, nn_dist


def align(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray | None, float]:
    """after를 before 좌표로 정합. 실패 시 (None, 0)."""
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    g1 = cv2.GaussianBlur(g1, (5, 5), 0)
    g2 = cv2.GaussianBlur(g2, (5, 5), 0)
    W = np.eye(2, 3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(
            g1,
            g2,
            W,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
            None,
            5,
        )
    except cv2.error:
        return None, 0.0
    h, w = before.shape[:2]
    out = cv2.warpAffine(
        after,
        W,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return out, float(cc)


def color_normalize(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """채널별 평균·표준편차를 ref에 맞춤 (전역 조명·색온도 변화 보정)."""
    s = src.astype(np.float32)
    r = ref.astype(np.float32)
    out = (s - s.mean((0, 1))) / (s.std((0, 1)) + 1e-6) * r.std((0, 1)) + r.mean((0, 1))
    return np.clip(out, 0, 255).astype(np.uint8)


def pixel_diff_map(before, after_aligned) -> np.ndarray:
    g1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(after_aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(np.abs(g1 - g2), (0, 0), 2) / 255.0


class FeatureDiff:
    def __init__(self, fx: FeatureExtractor):
        self.fx = fx

    def map(self, before, after_aligned) -> np.ndarray:
        P, _ = self.fx.patches([before, after_aligned])
        h, w, d = P.shape[1:]
        bank = P[0].reshape(-1, d)
        dist = nn_dist(P[1].reshape(-1, d), bank, self.fx.device).reshape(h, w)
        H, W_ = before.shape[:2]
        return cv2.GaussianBlur(
            cv2.resize(dist, (W_, H), interpolation=cv2.INTER_LINEAR), (0, 0), 3
        )


def decide(
    change_score: float,
    thr: float,
    band: float,
    match_conf: float | None,
    match_thr: float | None,
    aligned_ok: bool,
    before_damage: bool,
) -> tuple[str, str]:
    if (
        match_conf is not None and match_thr is not None and match_conf < match_thr
    ) or not aligned_ok:
        return "not_comparable", "동일 물품 매칭 신뢰도 부족 또는 영상 정합 실패"
    if change_score >= thr:
        return "newly_detected_damage", f"변화 점수 {change_score:.3f} ≥ {thr:.3f}"
    if change_score >= thr - band:
        return "uncertain", f"변화 점수 {change_score:.3f}가 경계({thr - band:.3f}~{thr:.3f}) 안"
    if before_damage:
        return "pre_existing_damage", "이사 전 이미지에서 이미 손상 검출, 새 변화 없음"
    return "unchanged", "의미 있는 변화 없음"
