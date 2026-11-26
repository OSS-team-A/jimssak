"""재식별 헤드 학습 (ABO 다중 시점, batch-hard triplet loss).

- 백본: 사전학습 DINOv2(고정). 학습 대상: 투영 헤드(MLP). → 학습 비용이 작아 GPU 1장으로 충분(DDP
  불필요).
- 데이터 증강: RandomResizedCrop·좌우반전·밝기/대비/채도 변화를 이미지당 n_aug개 미리 만들어 백본
  특징을 캐시.
- Loss: batch-hard triplet (margin). Optimizer: AdamW. Scheduler: cosine. Early stopping: val
  Recall@1.
- 체크포인트: 매 epoch last.pt(모델·optimizer·scheduler·epoch·난수 상태), 최고 val 점수 best.pt.
  resume 지원.
"""

from __future__ import annotations

import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..common import imread, log, write_json
from .embedders import DinoBackbone, ReidHead


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def augment(im: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = im.shape[:2]
    s = rng.uniform(0.6, 1.0)
    ch, cw = max(8, int(h * s)), max(8, int(w * s))
    y0, x0 = rng.randint(0, h - ch), rng.randint(0, w - cw)
    out = im[y0 : y0 + ch, x0 : x0 + cw]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    out = out.astype(np.float32)
    out = out * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)  # 대비·밝기
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def cache_features(
    bb: DinoBackbone, root: Path, rows: list[dict], n_aug: int, cache_path: Path, seed: int = 0
):
    if cache_path.is_file():
        d = np.load(cache_path, allow_pickle=True)
        return d["feats"], list(d["keys"])
    rng = random.Random(seed)
    feats, keys = [], []
    batch_imgs, batch_keys = [], []
    for r in rows:
        im = imread(root / r["path"])
        views = [im] + [augment(im, rng) for _ in range(n_aug)]
        for v_i, v in enumerate(views):
            batch_imgs.append(v)
            batch_keys.append(f"{r['item_id']}|{r['image_id']}|{v_i}")
        if len(batch_imgs) >= 256:
            feats.append(bb.features(batch_imgs))
            keys += batch_keys
            batch_imgs, batch_keys = [], []
    if batch_imgs:
        feats.append(bb.features(batch_imgs))
        keys += batch_keys
    F = np.concatenate(feats).astype(np.float32)
    np.savez(cache_path, feats=F, keys=np.array(keys))
    return F, keys


def retrieval_metrics(emb: np.ndarray, labels: list[str]) -> dict:
    """각 이미지를 질의로, 나머지 전체를 갤러리로. 같은 상품이 갤러리에 없는 질의는 제외."""
    lab = np.array(labels)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    r1 = r5 = 0
    aps = []
    n = 0
    for i in range(len(lab)):
        pos = lab == lab[i]
        pos[i] = False
        if not pos.any():
            continue
        n += 1
        order = np.argsort(-sim[i])
        hits = pos[order]
        r1 += hits[0]
        r5 += hits[:5].any()
        cum = np.cumsum(hits)
        prec = cum / np.arange(1, len(hits) + 1)
        aps.append(float((prec * hits).sum() / hits.sum()))
    return {
        "queries": n,
        "recall@1": round(r1 / n, 4) if n else None,
        "recall@5": round(r5 / n, 4) if n else None,
        "mAP": round(float(np.mean(aps)), 4) if aps else None,
    }


def pair_scores(emb: np.ndarray, labels: list[str], max_pairs: int = 200000, seed: int = 0):
    """같은 상품/다른 상품 쌍의 유사도 (임계값 선택·AUROC용)."""
    rng = np.random.default_rng(seed)
    lab = np.array(labels)
    n = len(lab)
    i = rng.integers(0, n, max_pairs)
    j = rng.integers(0, n, max_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    s = (emb[i] * emb[j]).sum(1)
    y = (lab[i] == lab[j]).astype(int)
    # 양성 쌍은 드물어서 전부 추가
    by = defaultdict(list)
    for k, name in enumerate(lab):
        by[name].append(k)
    pi, pj = [], []
    for ks in by.values():
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                pi.append(ks[a])
                pj.append(ks[b])
    if pi:
        s = np.concatenate([s, (emb[pi] * emb[pj]).sum(1)])
        y = np.concatenate([y, np.ones(len(pi), int)])
    return s, y


def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    if y.min() == y.max():
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # 동점 평균 순위
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def best_threshold(scores: np.ndarray, y: np.ndarray) -> dict:
    """F1 최대 임계값 (검증 데이터에서만 선택)."""
    best = {"threshold": None, "f1": -1}
    for t in np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197))):
        pred = scores >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
    return best


def train_head(
    *, bb: DinoBackbone, abo_root: Path, out_dir: Path, cfg: dict, seed: int = 0
) -> dict:
    import torch
    import torch.nn.functional as F

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(abo_root / "abo_manifest.csv")
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    t0 = time.time()
    Ftr, ktr = cache_features(bb, abo_root, tr, 3, out_dir / "feat_train.npz", seed)
    Fva, kva = cache_features(bb, abo_root, va, 0, out_dir / "feat_val.npz", seed)
    feat_sec = time.time() - t0
    items = defaultdict(lambda: defaultdict(list))  # item -> image -> [row idx of views]
    for idx, k in enumerate(ktr):
        it, im, v = k.split("|")
        items[it][im].append(idx)
    item_ids = [it for it, ims in items.items() if len(ims) >= 2]
    lab_va = [k.split("|")[0] for k in kva]
    device = bb.device
    head = ReidHead(Ftr.shape[1], cfg["embed_dim"]).net.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    start_ep, best, bad = 0, -1.0, 0
    hist = []
    last = out_dir / "last.pt"
    if last.is_file():
        d = torch.load(last, map_location="cpu", weights_only=False)
        head.load_state_dict(d["head"])
        opt.load_state_dict(d["opt"])
        sched.load_state_dict(d["sched"])
        start_ep, best, bad, hist = d["epoch"] + 1, d["best"], d["bad"], d["history"]
        random.setstate(d["py_rng"])
        np.random.set_state(d["np_rng"])
        torch.set_rng_state(d["torch_rng"])
        log(f"  재식별 헤드 재개: epoch {start_ep}")
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    Ft = torch.from_numpy(Ftr).to(device)
    P, K = cfg["batch_items"], cfg["imgs_per_item"]
    steps = max(1, len(item_ids) // P)
    for ep in range(start_ep, cfg["epochs"]):
        te = time.time()
        head.train()
        random.shuffle(item_ids)
        losses = []
        for s in range(steps):
            batch_items = item_ids[s * P : (s + 1) * P]
            if len(batch_items) < 2:
                continue
            idx, lab = [], []
            for li, it in enumerate(batch_items):
                ims = list(items[it])
                pick = (
                    random.sample(ims, min(K, len(ims)))
                    if len(ims) >= K
                    else random.choices(ims, k=K)
                )
                for im in pick:
                    idx.append(random.choice(items[it][im]))
                    lab.append(li)
            z = F.normalize(head(Ft[idx]), dim=1)
            lab_t = torch.tensor(lab, device=device)
            d = torch.cdist(z, z)
            same = lab_t[:, None] == lab_t[None, :]
            eye = torch.eye(len(lab), dtype=torch.bool, device=device)
            hardest_pos = (d * (same & ~eye)).max(1).values
            hardest_neg = d.masked_fill(same, float("inf")).min(1).values
            loss = F.relu(hardest_pos - hardest_neg + cfg["margin"]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        head.eval()
        with torch.inference_mode():
            zva = F.normalize(head(torch.from_numpy(Fva).to(device)), dim=1).cpu().numpy()
        m = retrieval_metrics(zva, lab_va)
        rec = {
            "epoch": ep + 1,
            "loss": round(float(np.mean(losses)), 5) if losses else None,
            "val_recall@1": m["recall@1"],
            "val_mAP": m["mAP"],
            "lr": opt.param_groups[0]["lr"],
            "sec": round(time.time() - te, 2),
        }
        hist.append(rec)
        log(
            f"    reid epoch {ep + 1}/{cfg['epochs']} loss={rec['loss']} val R@1={m['recall@1']} "
            f"mAP={m['mAP']}"
        )
        score = m["recall@1"] or 0.0
        ck = {
            "head": head.state_dict(),
            "in_dim": Ftr.shape[1],
            "embed_dim": cfg["embed_dim"],
            "backbone": bb.model_name,
        }
        if score > best:
            best, bad = score, 0
            torch.save({**ck, "epoch": ep + 1, "val": m}, out_dir / "best.pt")
        else:
            bad += 1
        torch.save(
            {
                **ck,
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": ep,
                "best": best,
                "bad": bad,
                "history": hist,
                "py_rng": random.getstate(),
                "np_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            last,
        )
        if bad >= cfg["patience"]:
            log(f"    조기 종료: val R@1이 {cfg['patience']} epoch 동안 개선되지 않음")
            break
    with open(out_dir / "history.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    res = {
        "status": "완료",
        "best_pt": str(out_dir / "best.pt"),
        "best_val_recall@1": best,
        "epochs_run": len(hist),
        "train_items": len(item_ids),
        "train_images": len({k.rsplit("|", 1)[0] for k in ktr}),
        "feature_cache_sec": round(feat_sec, 1),
        "train_sec": round(time.time() - t0 - feat_sec, 1),
        "loss": "batch-hard triplet",
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "backbone_frozen": True,
    }
    write_json(out_dir / "train_summary.json", res)
    return res
