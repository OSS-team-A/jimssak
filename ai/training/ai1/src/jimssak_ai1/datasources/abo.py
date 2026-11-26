"""ABO(Amazon Berkeley Objects): 가구 상품의 다중 시점 사진 → 재식별 학습·평가용 데이터.

- 상품(item) 하나 = 실물 하나의 신원. 같은 상품의 다른 사진 = 같은 물건의 다른 시점(양성 쌍).
- 같은 모델의 색상 변형 등 '비슷하지만 다른 상품'이 많아 어려운 음성 쌍이 자연스럽게 생긴다.
- 분할 누수 방지: 사진을 공유하거나 model_number가 같은 상품들은 하나의 그룹으로 묶어 같은 분할에
  둔다.
- 카탈로그 사진에는 치수 도면·연출 사진이 섞여 있다(잡음, 한계로 기록).
"""

from __future__ import annotations

import csv
import gzip
import json
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from ..common import log, write_json
from .download import download_many

S3 = "https://amazon-berkeley-objects.s3.amazonaws.com"


def _en(vals) -> str:
    for v in vals or []:
        if str(v.get("language_tag", "")).startswith("en"):
            return v.get("value", "")
    return (vals or [{}])[0].get("value", "") if vals else ""


def load_listings(listings_tar: Path, product_types: list[str]) -> dict:
    items = {}
    want = set(product_types)
    with tarfile.open(listings_tar) as t:
        for m in t:
            if not m.name.endswith(".json.gz"):
                continue
            with gzip.open(t.extractfile(m), "rt", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    pt = (d.get("product_type") or [{}])[0].get("value")
                    if pt not in want or not d.get("main_image_id"):
                        continue
                    iid = d["item_id"]
                    imgs = [d["main_image_id"]] + list(d.get("other_image_id") or [])
                    if iid in items:
                        items[iid]["images"] = sorted(set(items[iid]["images"]) | set(imgs))
                        continue
                    items[iid] = {
                        "item_id": iid,
                        "product_type": pt,
                        "images": imgs,
                        "model_number": ((d.get("model_number") or [{}])[0].get("value") or ""),
                        "name": _en(d.get("item_name"))[:120],
                        "brand": _en(d.get("brand")),
                    }
    return items


def load_image_paths(images_csv_gz: Path) -> dict:
    with gzip.open(images_csv_gz, "rt", encoding="utf-8") as f:
        return {r["image_id"]: r["path"] for r in csv.DictReader(f)}


def build_abo(*, cache: Path, out: Path, cfg: dict, seed: int = 0) -> dict:
    rng = random.Random(seed)
    items = load_listings(cache / "abo-listings.tar", cfg["product_types"])
    paths = load_image_paths(cache / "abo_images.csv.gz")
    # 여러 상품이 공유하는 사진은 신원이 모호하므로 제거
    img_owner = Counter(i for it in items.values() for i in set(it["images"]))
    for it in items.values():
        it["images"] = [i for i in dict.fromkeys(it["images"]) if img_owner[i] == 1 and i in paths]
    items = {k: v for k, v in items.items() if len(v["images"]) >= cfg["min_images_per_item"]}
    # 그룹: 같은 model_number(+브랜드) → 같은 분할
    groups = defaultdict(list)
    for k, v in items.items():
        g = (v["brand"].lower(), v["model_number"].lower()) if v["model_number"] else ("item", k)
        groups[g].append(k)
    gkeys = sorted(groups)
    rng.shuffle(gkeys)
    chosen = []
    for g in gkeys:
        if len(chosen) >= cfg["max_items"]:
            break
        chosen.append(g)
    n = len(chosen)
    a = int(n * cfg["split"]["train"])
    b = a + int(n * cfg["split"]["val"])
    split_of = {}
    for i, g in enumerate(chosen):
        s = "train" if i < a else ("val" if i < b else "test")
        for k in groups[g]:
            split_of[k] = s
    dl = []
    manifest = []
    for k, s in split_of.items():
        it = items[k]
        for im in it["images"][:8]:  # 상품당 최대 8장
            p = out / "images" / paths[im]
            dl.append((f"{S3}/images/small/{paths[im]}", p))
            manifest.append(
                {
                    "split": s,
                    "item_id": k,
                    "group": f"{it['brand']}|{it['model_number']}" if it["model_number"] else k,
                    "product_type": it["product_type"],
                    "image_id": im,
                    "path": str(p.relative_to(out)),
                }
            )
    log(f"  ABO 이미지 다운로드 {len(dl)}장 (상품 {len(split_of)}개)")
    dls = download_many(dl, workers=cfg.get("download_workers", 16), desc="abo")
    manifest = [m for m in manifest if (out / m["path"]).is_file()]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "abo_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    st = defaultdict(lambda: {"items": set(), "images": 0, "types": Counter()})
    for m in manifest:
        st[m["split"]]["items"].add(m["item_id"])
        st[m["split"]]["images"] += 1
        st[m["split"]]["types"][m["product_type"]] += 1
    stats = {
        "dataset": "ABO",
        "listings_matched": len(items),
        "groups_used": n,
        "per_split": {
            s: {"items": len(v["items"]), "images": v["images"], "images_by_type": dict(v["types"])}
            for s, v in st.items()
        },
        "image_downloads": dls,
        "split_rule": "model_number(+brand) 그룹 단위 분할, 공유 사진 제거",
    }
    write_json(out / "dataset_stats.json", stats)
    return stats
