"""PatchCore 이상 탐지 (Roth et al., CVPR 2022 'Towards Total Recall in Industrial Anomaly
Detection') 직접 구현.

- 백본: ImageNet 사전학습 WideResNet50-2 (고정). layer2+layer3 특징을 3x3 이웃 평균 후 결합 → 패치
  특징.
- '학습' = 정상 이미지 패치로 메모리 뱅크 구축 + greedy coreset 축소 (경사하강 학습이 아님 —
  보고서에 그대로 표기).
- 점수: 패치별 메모리 최근접 거리 → 이상 지도(가우시안 평활), 이미지 점수 = 최댓값.
기준선(global kNN): 이미지 전체 평균 특징의 최근접 정상 이미지 거리 (이상 위치 지도 없음).
"""

from __future__ import annotations

import numpy as np


class FeatureExtractor:
    def __init__(self, device: str = "cpu", imgsz: int = 256):
        import torch
        import torchvision

        self.torch = torch
        self.device = device
        self.imgsz = imgsz
        w = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        m = torchvision.models.wide_resnet50_2(weights=w).eval().to(device)
        self.m = m
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _prep(self, imgs_bgr):
        import cv2

        arr = np.stack(
            [
                cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (self.imgsz, self.imgsz))
                for im in imgs_bgr
            ]
        )
        x = self.torch.from_numpy(arr).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    def patches(self, imgs_bgr, batch: int = 16):
        """반환: (N, h, w, D) 패치 특징, 전역 특징 (N, D2)"""
        F = self.torch.nn.functional
        outs, glob = [], []
        with self.torch.inference_mode():
            for i in range(0, len(imgs_bgr), batch):
                x = self._prep(imgs_bgr[i : i + batch])
                m = self.m
                x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                l1 = m.layer1(x)
                l2 = m.layer2(l1)
                l3 = m.layer3(l2)
                glob.append(self.torch.cat([l2.mean((2, 3)), l3.mean((2, 3))], 1).cpu().numpy())
                a2 = F.avg_pool2d(l2, 3, 1, 1)
                a3 = F.interpolate(
                    F.avg_pool2d(l3, 3, 1, 1),
                    size=l2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                f = self.torch.cat([a2, a3], 1)  # (B, 1536, h, w)
                f = F.adaptive_avg_pool1d(
                    f.permute(0, 2, 3, 1).reshape(-1, 1, f.shape[1]), 1024
                ).reshape(f.shape[0], f.shape[2], f.shape[3], 1024)
                outs.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(outs), np.concatenate(glob)


def greedy_coreset(X: np.ndarray, ratio: float, device: str = "cpu", seed: int = 0) -> np.ndarray:
    import torch

    n = len(X)
    k = max(1, int(n * ratio))
    rng = np.random.default_rng(seed)
    P = torch.from_numpy(X).to(device)
    proj = torch.from_numpy(rng.normal(size=(X.shape[1], 128)).astype(np.float32)).to(device)
    Z = P @ proj
    sel = [int(rng.integers(n))]
    d = torch.cdist(Z, Z[sel]).squeeze(1)
    for _ in range(k - 1):
        i = int(torch.argmax(d))
        sel.append(i)
        d = torch.minimum(d, torch.cdist(Z, Z[i : i + 1]).squeeze(1))
    return X[sel]


def nn_dist(Q: np.ndarray, M: np.ndarray, device: str = "cpu", chunk: int = 8192) -> np.ndarray:
    import torch

    Mt = torch.from_numpy(M).to(device)
    out = []
    for i in range(0, len(Q), chunk):
        q = torch.from_numpy(Q[i : i + chunk]).to(device)
        out.append(torch.cdist(q, Mt).min(1).values.cpu().numpy())
    return np.concatenate(out)


class PatchCore:
    def __init__(self, fx: FeatureExtractor, coreset_ratio: float = 0.01, seed: int = 0):
        self.fx = fx
        self.ratio = coreset_ratio
        self.seed = seed
        self.bank = None
        self.glob = None

    def fit(self, imgs_bgr):
        P, G = self.fx.patches(imgs_bgr)
        self.hw = P.shape[1:3]
        self.bank = greedy_coreset(
            P.reshape(-1, P.shape[-1]), self.ratio, self.fx.device, self.seed
        )
        self.glob = G
        return self

    def score(self, imgs_bgr, out_size: int | None = None):
        import cv2

        P, G = self.fx.patches(imgs_bgr)
        n, h, w, d = P.shape
        dist = nn_dist(P.reshape(-1, d), self.bank, self.fx.device).reshape(n, h, w)
        size = out_size or self.fx.imgsz
        maps = np.stack(
            [
                cv2.GaussianBlur(
                    cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR), (0, 0), 4
                )
                for m in dist
            ]
        )
        img_scores = dist.reshape(n, -1).max(1)
        glob_scores = nn_dist(G, self.glob, self.fx.device)
        return img_scores, maps, glob_scores
