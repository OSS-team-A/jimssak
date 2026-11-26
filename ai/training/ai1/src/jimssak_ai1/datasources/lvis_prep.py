"""LVIS v1 → 짐싹 통합 클래스 YOLO-seg 데이터셋.

분할 원칙
- 테스트: LVIS val 중 **COCO val2017 출신 이미지만** (B0의 COCO 사전학습에 쓰이지 않은 이미지 → 학습
  전후 공정 비교).
- 학습/검증: LVIS train(= COCO train2017 이미지)에서 이미지 단위로 분리. (smoke는 val json의
  train2017 출신 이미지 사용)
- COCO 이미지는 서로 독립된 사진이라 영상·방 단위 묶음이 없다. 같은 이미지가 두 분할에 들어가지
  않게만 보장한다.

라벨
- LVIS 다각형 마스크만 사용한다(박스를 마스크로 바꾸지 않음). 여러 조각 마스크는 Ultralytics
  merge_multi_segment로 한 줄.
- LVIS는 연합(federated) 주석이라 이미지마다 '확인된' 카테고리만 정답이다. 테스트 메타에
  pos/neg/not_exhaustive를 남겨
  평가 시 확인되지 않은 클래스의 예측을 FP로 세지 않는다. 학습에서는 이를 반영할 수 없어 라벨 누락
  잡음이 남는다(한계).
"""

from __future__ import annotations

import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..common import log, write_csv, write_json
from ..taxonomy import Taxonomy
from .download import download_many


def load_lvis(zip_path: Path) -> dict:
    z = zipfile.ZipFile(zip_path)
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    log(f"  LVIS json 읽는 중: {zip_path.name}")
    return json.loads(z.read(name))


def _lvis_to_code(tax: Taxonomy, lvis_cats: list[dict]) -> dict[int, str]:
    by_name = {c["name"]: c["id"] for c in lvis_cats}
    m = {}
    missing = []
    for c in tax.categories:
        for n in c.get("lvis") or []:
            if n in by_name:
                m[by_name[n]] = c["code"]
            else:
                missing.append(n)
    if missing:
        log(f"  주의: LVIS에 없는 카테고리 이름 {missing}")
    return m


COCO_OFFSET = 1_000_000  # COCO 카테고리 id를 LVIS id와 겹치지 않게 옮기는 값


def load_coco_instances(zip_path: Path, split: str) -> dict:
    """annotations_trainval2017.zip 안의 instances_<split>.json (split = train2017 / val2017)."""
    log(f"  COCO 주석 읽는 중: instances_{split}.json")
    return json.loads(zipfile.ZipFile(zip_path).read(f"annotations/instances_{split}.json"))


def merge_coco(
    imgs: dict, anns: dict, coco: dict, tax: Taxonomy, cat2code: dict
) -> tuple[dict, set, int]:
    """COCO 원본 주석(80개 클래스는 이미지마다 빠짐없이 라벨링됨)을 공통 클래스에 사용한다.

    LVIS는 연합 주석이라 의자·테이블처럼 흔한 물건도 이미지의 약 85~88%에서 라벨이 비어 있다
    (COCO val2017 확인: 의자가 있는 572장 중 LVIS 의자 라벨은 87장). 그대로 학습하면 모델이
    그 물건을 배경으로 배우므로, COCO에 같은 개념이 있는 클래스는 COCO 주석으로 바꾸고
    나머지(신규 클래스)만 LVIS 주석을 쓴다. 같은 물체가 두 번 들어가지 않도록 LVIS 쪽 공통 클래스
    주석은 버린다.
    """
    coco_to_code = {c["coco"]: c["code"] for c in tax.categories if c.get("coco")}
    cid = {
        c["id"]: coco_to_code[c["name"]] for c in coco["categories"] if c["name"] in coco_to_code
    }
    common = set(cid.values())
    for k, code in cid.items():
        cat2code[COCO_OFFSET + k] = code
    merged = defaultdict(list)
    for i, lst in anns.items():
        keep = [a for a in lst if cat2code[a["category_id"]] not in common]
        if keep:
            merged[i] = keep
    n = 0
    for a in coco["annotations"]:
        if (
            a["category_id"] in cid
            and not a.get("iscrowd")
            and a["image_id"] in imgs
            and isinstance(a["segmentation"], list)
        ):
            merged[a["image_id"]].append({**a, "category_id": COCO_OFFSET + a["category_id"]})
            n += 1
    return merged, common, n


def _extract_from_zip(zip_path: Path, names: list[str], img_root: Path) -> dict:
    """COCO 이미지 묶음(zip)에서 필요한 사진만 꺼낸다 (예: train2017/000000000009.jpg)."""
    t0 = time.time()
    done = skipped = missing = 0
    with zipfile.ZipFile(zip_path) as z:
        members = set(z.namelist())
        for n in names:
            dst = img_root / n
            if dst.is_file() and dst.stat().st_size > 0:
                skipped += 1
                continue
            if n not in members:
                missing += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as f:
                f.write(src.read())
            done += 1
    return {
        "requested": len(names),
        "already_present": skipped,
        "downloaded": done,
        "failed": missing,
        "downloaded_bytes": 0,
        "seconds": round(time.time() - t0, 1),
        "mode": f"zip:{zip_path.name}",
    }


def _index(d: dict, cat2code: dict[int, str]):
    anns = defaultdict(list)
    for a in d["annotations"]:
        if a["category_id"] in cat2code:
            anns[a["image_id"]].append(a)
    imgs = {i["id"]: i for i in d["images"]}
    return imgs, anns


def _select(
    img_ids: list[int],
    anns: dict,
    cat2code: dict,
    max_images: int,
    per_class_cap: int,
    rng: random.Random,
):
    """희귀 클래스부터 이미지를 채워 클래스 불균형을 줄인다 (클래스별 이미지 상한)."""
    cls_imgs = defaultdict(list)
    for i in img_ids:
        for c in {cat2code[a["category_id"]] for a in anns[i]}:
            cls_imgs[c].append(i)
    order = sorted(cls_imgs, key=lambda c: len(cls_imgs[c]))
    chosen, per_cls = [], Counter()
    seen = set()
    for c in order:
        cand = cls_imgs[c][:]
        rng.shuffle(cand)
        for i in cand:
            if len(chosen) >= max_images or per_cls[c] >= per_class_cap:
                break
            if i in seen:
                # 다른 클래스로 이미 선택된 이미지 (선택 시 그 이미지의 모든 클래스 집계에 반영됨)
                continue
            seen.add(i)
            chosen.append(i)
            for cc in {cat2code[a["category_id"]] for a in anns[i]}:
                per_cls[cc] += 1
    return chosen


def _poly_lines(ann_list, cat2code, cls_index, w, h, merge_fn):
    lines = []
    for a in ann_list:
        code = cat2code[a["category_id"]]
        if code not in cls_index:
            continue
        segs = [s for s in a["segmentation"] if len(s) >= 6]
        if not segs:
            continue
        if len(segs) > 1:
            poly = np.concatenate(merge_fn(segs), axis=0)
        else:
            poly = np.array(segs[0], dtype=float).reshape(-1, 2)
        poly = np.clip(poly / [w, h], 0, 1)
        lines.append(f"{cls_index[code]} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
    return lines


def build_lvis_dataset(
    *, cache: Path, out: Path, img_root: Path, tax: Taxonomy, cfg: dict, seed: int = 0
) -> dict:
    from ultralytics.data.converter import merge_multi_segment

    rng = random.Random(seed)
    val_d = load_lvis(cache / "lvis_v1_val.json.zip")
    cat2code = _lvis_to_code(tax, val_d["categories"])
    licenses = {lic["id"]: lic["name"] for lic in val_d.get("licenses", [])}
    v_imgs, v_anns = _index(val_d, cat2code)
    use_coco = bool(cfg.get("use_coco_annotations"))
    common: set = set()
    n_coco = {}
    coco_zip = cache / "annotations_trainval2017.zip"
    if use_coco:
        cv = load_coco_instances(coco_zip, "val2017")
        if not cfg.get("use_train_json", True):  # smoke: val json 안에 train2017 출신 이미지도 있음
            cv["annotations"] += load_coco_instances(coco_zip, "train2017")["annotations"]
        v_anns, common, n_coco["val_json"] = merge_coco(v_imgs, v_anns, cv, tax, cat2code)
        del cv
    test_pool = [i for i in v_anns if "val2017" in v_imgs[i]["coco_url"]]
    if cfg.get("use_train_json", True):
        tr_d = load_lvis(cache / "lvis_v1_train.json.zip")
        t_imgs, t_anns = _index(tr_d, cat2code)
        if use_coco:
            ct = load_coco_instances(coco_zip, "train2017")
            t_anns, common, n_coco["train_json"] = merge_coco(t_imgs, t_anns, ct, tax, cat2code)
            del ct
        trainval_pool = list(t_anns)
        src_imgs, src_anns = t_imgs, t_anns
        source = "lvis_v1_train(COCO train2017)"
        del tr_d
    else:
        trainval_pool = [i for i in v_anns if "train2017" in v_imgs[i]["coco_url"]]
        src_imgs, src_anns = v_imgs, v_anns
        source = "lvis_v1_val 중 COCO train2017 출신(smoke 점검 전용)"
    if use_coco:
        log(
            f"  COCO 원본 주석으로 바꾼 공통 클래스: {sorted(common)} | 추가된 COCO "
            f"인스턴스 {n_coco}"
        )
    trainval = _select(
        sorted(trainval_pool),
        src_anns,
        cat2code,
        cfg["max_train_images"] + cfg["max_val_images"],
        cfg["per_class_image_cap"],
        rng,
    )
    rng.shuffle(trainval)
    val_ids = trainval[: cfg["max_val_images"]]
    train_ids = trainval[cfg["max_val_images"] :]
    test_ids = _select(sorted(test_pool), v_anns, cat2code, cfg["max_test_images"], 10**9, rng)

    splits = {
        "train": (train_ids, src_imgs, src_anns),
        "val": (val_ids, src_imgs, src_anns),
        "test": (test_ids, v_imgs, v_anns),
    }
    # ---- 클래스별 인스턴스 수 → 최종 클래스 확정 ----
    counts = {s: Counter() for s in splits}
    img_counts = {s: Counter() for s in splits}
    for s, (ids, _, an) in splits.items():
        for i in ids:
            for a in an[i]:
                counts[s][cat2code[a["category_id"]]] += 1
            for c in {cat2code[a["category_id"]] for a in an[i]}:
                img_counts[s][c] += 1
    avail = []
    final = []
    for c in tax.codes:
        if c == "other":
            continue
        tr, va, te = counts["train"][c], counts["val"][c], counts["test"][c]
        ok = tr >= cfg["min_train_instances"] and te >= cfg["min_test_instances"]
        reason = (
            ""
            if ok
            else (
                f"학습 인스턴스 {tr} < {cfg['min_train_instances']}"
                if tr < cfg["min_train_instances"]
                else f"테스트 인스턴스 {te} < {cfg['min_test_instances']}"
            )
        )
        if not any(cat2code.get(k) == c for k in cat2code):
            ok, reason = False, "LVIS 카테고리 매핑 없음"
        avail.append(
            {
                "class": c,
                "ko": tax.ko(c),
                "lvis_categories": ";".join(
                    next(x for x in tax.categories if x["code"] == c).get("lvis") or []
                ),
                "train_instances": tr,
                "val_instances": va,
                "test_instances": te,
                "train_images": img_counts["train"][c],
                "val_images": img_counts["val"][c],
                "test_images": img_counts["test"][c],
                "selected": "yes" if ok else "no",
                "reason": reason,
            }
        )
        if ok:
            final.append(c)
    if not final:
        raise SystemExit(
            "오류: 학습 가능한 클래스가 하나도 없습니다 (데이터 다운로드/선택 결과 확인)"
        )
    cls_index = {c: k for k, c in enumerate(final)}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "class_availability.csv", avail, list(avail[0].keys()))

    # ---- 이미지 다운로드 ----
    dl_stats = {}
    for s, (ids, imgs, _) in splits.items():
        items = []
        for i in ids:
            url = imgs[i]["coco_url"]
            sub = url.split("/")[-2]
            items.append((url, img_root / sub / url.split("/")[-1]))
        if cfg.get("coco_images") == "zip":
            log(f"  COCO 이미지 묶음에서 꺼내기 [{s}] {len(items)}장")
            by_zip = defaultdict(list)
            for url, p in items:
                by_zip[url.split("/")[-2]].append(f"{url.split('/')[-2]}/{url.split('/')[-1]}")
            parts = [
                _extract_from_zip(cache / f"{sub}.zip", names, img_root)
                for sub, names in by_zip.items()
            ]
            dl_stats[s] = {
                k: sum(p[k] for p in parts)
                for k in (
                    "requested",
                    "already_present",
                    "downloaded",
                    "failed",
                    "downloaded_bytes",
                )
            }
            dl_stats[s]["mode"] = "zip"
        else:
            log(f"  COCO 이미지 다운로드 [{s}] {len(items)}장")
            dl_stats[s] = download_many(items, workers=cfg.get("download_workers", 16), desc=s)
        if dl_stats[s]["failed"]:
            log(f"  주의: {s} 이미지 {dl_stats[s]['failed']}장 다운로드 실패 → 해당 이미지는 제외")

    # ---- 라벨 작성 ----
    test_meta = {}
    lic_counter = Counter()
    n_written = Counter()
    for s, (ids, imgs, an) in splits.items():
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for i in ids:
            im = imgs[i]
            url = im["coco_url"]
            src = img_root / url.split("/")[-2] / url.split("/")[-1]
            if not src.is_file():
                continue
            dst = out / "images" / s / src.name
            if not dst.exists():
                try:
                    dst.hardlink_to(src)
                except OSError:
                    import shutil

                    shutil.copy2(src, dst)
            lines = _poly_lines(
                an[i], cat2code, cls_index, im["width"], im["height"], merge_multi_segment
            )
            (out / "labels" / s / (src.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            n_written[s] += 1
            lic_counter[licenses.get(im.get("license"), str(im.get("license")))] += 1
            if s == "test":
                members = defaultdict(set)  # LVIS 카테고리만 (COCO로 바꾼 공통 클래스는 제외)
                for k, c in cat2code.items():
                    if k < COCO_OFFSET and c not in common:
                        members[c].add(k)
                neg = set(im.get("neg_category_ids", []))
                nex = set(im.get("not_exhaustive_category_ids", []))
                pos = {cat2code[a["category_id"]] for a in an[i]}
                test_meta[src.name] = {
                    "pos": sorted(c for c in pos if c in cls_index),
                    # 공통 클래스는 COCO가 빠짐없이 라벨링 → 없으면 '없음 확인'으로 평가
                    "neg": sorted(
                        c
                        for c in final
                        if (c in common and c not in pos)
                        or (c not in common and members[c] and members[c] <= neg)
                    ),
                    "not_exhaustive": sorted(
                        c for c in final if c not in common and members[c] & nex
                    ),
                    "coco_url": url,
                    "license": licenses.get(im.get("license")),
                }
    write_json(out / "test_federated_meta.json", test_meta)
    import yaml

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {k: c for k, c in enumerate(final)},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats = {
        "dataset": "LVIS v1",
        "source_train_val": source,
        "source_test": "lvis_v1_val 중 COCO val2017 출신",
        "images_written": dict(n_written),
        "instances": {s: dict(c) for s, c in counts.items()},
        "final_classes": final,
        "dropped_classes": [r["class"] for r in avail if r["selected"] == "no"],
        "image_downloads": dl_stats,
        "image_licenses": dict(lic_counter),
        "candidate_pool_images": {"trainval": len(trainval_pool), "test(val2017)": len(test_pool)},
        "label_format": (
            "YOLO-seg polygon (공통 클래스 = COCO 원본 다각형, 신규 클래스 = LVIS 다각형)"
            if use_coco
            else "YOLO-seg polygon (LVIS 원본 다각형)"
        ),
        "coco_annotations_used": {"common_classes": sorted(common), "instances": n_coco}
        if use_coco
        else None,
        "federated_eval_meta": "test_federated_meta.json",
    }
    write_json(out / "dataset_stats.json", stats)
    log(f"  LVIS 데이터셋 완료: 이미지 {dict(n_written)} | 최종 클래스 {len(final)}개")
    return stats
