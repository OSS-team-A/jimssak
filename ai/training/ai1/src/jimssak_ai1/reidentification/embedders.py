"""물품 이미지 임베딩(재식별 특징).

- hist     : 마스크/이미지 HSV 색 히스토그램 (기존 방식, 학습 없음)
- dinov2   : 사전학습 DINOv2 ViT-S/14 CLS 특징 (학습 없음 — '사전학습 특징 + 유사도 매칭')
- trained  : DINOv2(고정) + ABO 다중 시점으로 triplet 학습한 투영 헤드 ('실제 재식별 학습 결과')
모든 임베딩은 L2 정규화 → 코사인 유사도 = 내적.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..appearance import masked_hsv_hist


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class HistEmbedder:
    name = "hist"

    def embed(self, imgs_bgr: list[np.ndarray], masks: list | None = None) -> np.ndarray:
        out = []
        for i, im in enumerate(imgs_bgr):
            h = masked_hsv_hist(im, masks[i] if masks else None)
            out.append(
                np.sqrt(h) if h is not None else np.zeros(256, np.float32)
            )  # Hellinger → 코사인
        return l2n(np.stack(out).astype(np.float32))


class DinoBackbone:
    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        device: str = "cpu",
        img_size: int = 224,
    ):
        import timm
        import torch

        self.torch = torch
        self.device = device
        self.model = (
            timm.create_model(model_name, pretrained=True, num_classes=0, img_size=img_size)
            .eval()
            .to(device)
        )
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.mean = np.array(cfg["mean"], np.float32).reshape(1, 1, 3)
        self.std = np.array(cfg["std"], np.float32).reshape(1, 1, 3)
        self.size = img_size
        self.dim = self.model.num_features
        self.model_name = model_name

    def preprocess(self, im_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = self.size / max(h, w)
        rgb = cv2.resize(
            rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA
        )
        pad = np.full((self.size, self.size, 3), 124, np.uint8)  # 비율 유지 + 회색 패딩
        y0, x0 = (self.size - rgb.shape[0]) // 2, (self.size - rgb.shape[1]) // 2
        pad[y0 : y0 + rgb.shape[0], x0 : x0 + rgb.shape[1]] = rgb
        return ((pad.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)

    def features(self, imgs_bgr: list[np.ndarray], batch: int = 64) -> np.ndarray:
        outs = []
        with self.torch.inference_mode():
            for i in range(0, len(imgs_bgr), batch):
                x = self.torch.from_numpy(
                    np.stack([self.preprocess(im) for im in imgs_bgr[i : i + batch]])
                ).to(self.device)
                outs.append(self.model(x).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, self.dim), np.float32)


class DinoEmbedder:
    name = "dinov2"

    def __init__(self, backbone: DinoBackbone):
        self.bb = backbone

    def embed(self, imgs_bgr, masks=None):
        return l2n(self.bb.features(imgs_bgr))


class ReidHead:
    """DINOv2 특징 → 투영 헤드 (학습 대상)."""

    def __init__(self, in_dim: int, embed_dim: int):
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )


class TrainedEmbedder:
    name = "trained"

    def __init__(self, backbone: DinoBackbone, ckpt: Path):
        import torch

        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.head = ReidHead(d["in_dim"], d["embed_dim"]).net
        self.head.load_state_dict(d["head"])
        self.head.eval().to(backbone.device)
        self.bb = backbone
        self.torch = torch

    def embed_features(self, feats: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            z = (
                self.head(self.torch.from_numpy(feats.astype(np.float32)).to(self.bb.device))
                .cpu()
                .numpy()
            )
        return l2n(z)

    def embed(self, imgs_bgr, masks=None):
        return self.embed_features(self.bb.features(imgs_bgr))
