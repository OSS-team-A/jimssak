"""B3 합성 평가: ABO 상품 사진으로 '이사 전/후 물품 목록'을 만들고 분실·추가를 심어 넣은 시나리오.

- 이전: 상품마다 사진 1장, 이후: 같은 상품의 '다른' 사진 1장(다른 시점) — 같은 실물의 재관측을
  흉내냄.
- 일부 상품을 이후 목록에서 빼면 정답 missing, 이전에 없던 상품을 넣으면 정답 unexpected.
- 같은 종류 상품을 여러 개 섞어 '비슷하지만 다른 물품' 혼동을 일부러 만든다.
- 클래스는 정답 product_type을 사용(검출 오류 없이 매칭 성능만 분리 측정). 실제 영상 기반 분실
  평가와 별도로 집계.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..common import imread
from .before_after import count_only_baseline, match_items, score_scenario, summarize


def make_scenarios(
    rows: list[dict],
    split: str,
    n: int,
    n_items: list[int],
    n_missing: list[int],
    n_unexp: list[int],
    seed: int,
):
    rng = random.Random(seed)
    by_item = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            by_item[r["item_id"]].append(r)
    by_item = {k: v for k, v in by_item.items() if len(v) >= 2}
    by_type = defaultdict(list)
    for k, v in by_item.items():
        by_type[v[0]["product_type"]].append(k)
    types = [t for t, ks in by_type.items() if len(ks) >= 2]
    out = []
    for s in range(n):
        k = rng.randint(*n_items)
        chosen = []
        while len(chosen) < k:
            t = rng.choice(types)
            for it in rng.sample(
                by_type[t], min(len(by_type[t]), rng.randint(1, 3))
            ):  # 같은 종류 여러 개 → 혼동
                if it not in chosen and len(chosen) < k:
                    chosen.append(it)
        miss = set(rng.sample(chosen, min(len(chosen) - 1, rng.randint(*n_missing))))
        pool = [it for it in by_item if it not in chosen]
        unexp = rng.sample(pool, min(len(pool), rng.randint(*n_unexp)))
        before, after, pairs = [], [], {}
        for it in chosen:
            ims = rng.sample(by_item[it], 2)
            before.append(
                {
                    "item_id": f"B_{it}",
                    "class": by_item[it][0]["product_type"],
                    "path": ims[0]["path"],
                }
            )
            if it not in miss:
                after.append(
                    {
                        "item_id": f"A_{it}",
                        "class": by_item[it][0]["product_type"],
                        "path": ims[1]["path"],
                    }
                )
                pairs[f"B_{it}"] = f"A_{it}"
        for it in unexp:
            im = rng.choice(by_item[it])
            after.append(
                {"item_id": f"A_{it}", "class": by_item[it][0]["product_type"], "path": im["path"]}
            )
        rng.shuffle(after)
        out.append(
            {
                "before": before,
                "after": after,
                "missing": {f"B_{i}" for i in miss},
                "unexpected": {f"A_{i}" for i in unexp},
                "pairs": pairs,
            }
        )
    return out


def embed_scenarios(scen, root: Path, embedders: dict):
    paths = sorted({x["path"] for s in scen for x in s["before"] + s["after"]})
    imgs = [imread(root / p) for p in paths]
    E = {}
    for name, emb in embedders.items():
        Z = emb.embed(imgs)
        E[name] = {p: z for p, z in zip(paths, Z)}
    return E


def run(scen, E: dict, method: str, thr: float | None, band: float = 0.05) -> list[dict]:
    scores = []
    for s in scen:
        if method == "count_only":
            res = count_only_baseline(s["before"], s["after"])
        else:
            b = [{**x, "emb": E[method][x["path"]]} for x in s["before"]]
            a = [{**x, "emb": E[method][x["path"]]} for x in s["after"]]
            res = match_items(b, a, thr, band)
        scores.append(score_scenario(res, s["missing"], s["unexpected"], s["pairs"]))
    return scores


def tune_threshold(scen, E, method, grid=None) -> dict:
    grid = grid if grid is not None else np.round(np.arange(0.0, 0.99, 0.02), 2)
    best = None
    for t in grid:
        sm = summarize(run(scen, E, method, float(t), band=0.0))
        key = (sm["missing_f1"] or 0) + sm["match_accuracy"]
        if best is None or key > best["objective"]:
            best = {
                "threshold": float(t),
                "objective": round(key, 4),
                **{k: sm[k] for k in ("missing_f1", "match_accuracy", "false_missing_rate")},
            }
    return best
