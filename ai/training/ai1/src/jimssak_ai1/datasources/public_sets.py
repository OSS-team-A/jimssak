"""HomeObjects-3K(박스), crack-seg(균열 마스크), VisA(이상 탐지) 준비."""

from __future__ import annotations

import csv
import io
import random
import shutil
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

from ..common import log, write_json
from ..taxonomy import Taxonomy


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _extract_zip(zpath: Path, out: Path) -> Path:
    marker = out / ".extracted"
    if not marker.exists():
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        marker.write_text("ok")
    return out


def build_homeobjects(
    *, zpath: Path, work: Path, out: Path, tax: Taxonomy, val_fraction: float, seed: int = 0
) -> dict:
    """HomeObjects-3K: 원래 train → train/val(이미지 단위), 원래 val → test. 통합 클래스로 매핑,
    나머지 클래스 라벨 제거.
    """
    import yaml

    raw = _extract_zip(zpath, work / "homeobjects_raw")
    src_names = yaml.safe_load(
        io.StringIO(
            "names: {0: bed, 1: sofa, 2: chair, 3: table, 4: lamp, 5: tv, 6: laptop, 7: wardrobe, "
            "8: window, 9: door, 10: potted plant, 11: photo frame}"
        )
    )["names"]  # Ultralytics HomeObjects-3K.yaml 의 클래스 순서 (설치 패키지 파일과 대조됨)
    try:
        import ultralytics

        y = yaml.safe_load(
            (
                Path(ultralytics.__file__).parent / "cfg" / "datasets" / "HomeObjects-3K.yaml"
            ).read_text(encoding="utf-8")
        )
        src_names = y["names"]
    except Exception:
        pass
    to_code = {}
    for c in tax.categories:
        for n in c.get("homeobj") or []:
            to_code[n] = c["code"]
    final = [c for c in tax.codes if c in set(to_code.values())]
    idx = {c: i for i, c in enumerate(final)}
    rng = random.Random(seed)
    tr_imgs = sorted((raw / "images" / "train").glob("*.*"))
    rng.shuffle(tr_imgs)
    n_val = int(len(tr_imgs) * val_fraction)
    plan = {
        "val": tr_imgs[:n_val],
        "train": tr_imgs[n_val:],
        "test": sorted((raw / "images" / "val").glob("*.*")),
    }
    cnt = {s: Counter() for s in plan}
    dropped = Counter()
    for s, imgs in plan.items():
        for ip in imgs:
            lp = ip.parent.parent.parent / "labels" / ip.parent.name / (ip.stem + ".txt")
            lines = []
            if lp.is_file():
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    t = ln.split()
                    if not t:
                        continue
                    name = src_names[int(t[0])]
                    code = to_code.get(name)
                    if code is None:
                        dropped[name] += 1
                        continue
                    if len(t) != 5:
                        continue  # 박스 라벨만 사용
                    lines.append(" ".join([str(idx[code])] + t[1:]))
                    cnt[s][code] += 1
            _link(ip, out / "images" / s / ip.name)
            (out / "labels" / s).mkdir(parents=True, exist_ok=True)
            (out / "labels" / s / (ip.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {i: c for i, c in enumerate(final)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats = {
        "dataset": "HomeObjects-3K",
        "images": {s: len(v) for s, v in plan.items()},
        "instances": {s: dict(c) for s, c in cnt.items()},
        "classes": final,
        "dropped_source_classes": dict(dropped),
        "label_format": "YOLO detection 박스 (마스크 없음)",
        "split_note": "원본 val(404장)을 독립 테스트로, 원본 train에서 이미지 단위 검증 분리. "
        "촬영 공간 정보가 없어 같은 방 사진이 train/val에 섞였을 수 있음",
    }
    write_json(out / "dataset_stats.json", stats)
    return stats


def build_crackseg(
    *, zpath: Path, work: Path, out: Path, max_train_images: int | None = None, seed: int = 0
) -> dict:
    import yaml

    raw = _extract_zip(zpath, work / "crackseg_raw")
    stats = {"dataset": "crack-seg", "images": {}, "instances": {}, "unique_source_images": {}}
    rng = random.Random(seed)
    for s in ("train", "val", "test"):
        imgs = sorted((raw / "images" / s).glob("*.*"))
        if s == "train" and max_train_images:
            rng.shuffle(imgs)
            imgs = sorted(imgs[:max_train_images])
        n_inst = 0
        bases = set()
        for ip in imgs:
            lp = raw / "labels" / s / (ip.stem + ".txt")
            _link(ip, out / "images" / s / ip.name)
            if lp.is_file():
                _link(lp, out / "labels" / s / lp.name)
                n_inst += sum(1 for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip())
            bases.add(ip.name.split(".rf.")[0])
        stats["images"][s] = len(imgs)
        stats["instances"][s] = n_inst
        stats["unique_source_images"][s] = len(bases)
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "crack"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats["note"] = (
        "Roboflow 증강 내보내기: 학습 이미지 다수가 같은 원본의 증강본(.rf. 앞 번호가 같음). "
        "분할 간 원본 번호 중복은 전처리 시 확인"
    )
    tr = {p.name.split(".rf.")[0] for p in (out / "images" / "train").glob("*.*")}
    te = {p.name.split(".rf.")[0] for p in (out / "images" / "test").glob("*.*")}
    stats["source_overlap_train_test"] = len(tr & te)
    write_json(out / "dataset_stats.json", stats)
    return stats


def extract_visa(*, tar_path: Path, out: Path, categories: list[str]) -> dict:
    """VisA tar에서 필요한 범주와 split_csv만 추출 (이미 있으면 건너뜀)."""
    need = [c for c in categories if not (out / c / ".extracted").exists()]
    if need or not (out / "split_csv" / "1cls.csv").exists():
        log(f"  VisA 추출: {need}")
        with tarfile.open(tar_path) as t:
            for m in t:
                top = m.name.split("/")[0]
                if top in need or m.name.startswith("split_csv/"):
                    t.extract(m, out)
        for c in need:
            (out / c / ".extracted").write_text("ok")
    rows = list(csv.DictReader(open(out / "split_csv" / "1cls.csv", encoding="utf-8")))
    stats = {}
    for c in categories:
        r = [x for x in rows if x["object"] == c]
        stats[c] = dict(Counter(f"{x['split']}_{x['label']}" for x in r))
    write_json(out / "visa_stats.json", stats)
    return {
        "dataset": "VisA",
        "categories": stats,
        "split": "split_csv/1cls.csv (공식 1-class 분할: 학습=정상만)",
    }
