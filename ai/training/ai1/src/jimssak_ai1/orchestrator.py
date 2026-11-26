"""전체 자동 파이프라인: 단계별 실행·기록·재개.

각 단계는 성공/실패와 소요 시간을 outputs/<run>/stage_status.csv 와 state.json 에 남긴다.
실패하면 그 자리에서 멈추고(성공으로 기록하지 않음), 같은 run을 --resume 하면 성공한 단계는 건너뛰고
실패 단계부터 다시 한다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

from .common import (
    PROJECT_ROOT,
    environment_info,
    load_yaml,
    log,
    now_iso,
    now_stamp,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from .gpu import GpuMonitor, choose_devices, gpu_inventory, summarize_gpu_csv
from .taxonomy import load_taxonomy

STAGES = [
    "env_check",
    "deps_check",
    "validate_inputs",
    "pretrained_download",
    "baseline_b0_video",
    "datasets_download",
    "datasets_verify",
    "preprocess",
    "split_check",
    "train_detection",
    "train_reid",
    "matching_b3",
    "damage_b4",
    "eval_detection",
    "eval_tracking",
    "video_reanalysis",
    "compare",
    "report",
    "package",
]
STAGE_KO = {
    "env_check": "서버·GPU 환경 검사",
    "deps_check": "의존성 확인",
    "validate_inputs": "입력 영상·설정 검증",
    "pretrained_download": "사전학습 모델 다운로드",
    "baseline_b0_video": "B0 학습 전 기준선(입력 영상)",
    "datasets_download": "공개 데이터셋 다운로드",
    "datasets_verify": "데이터셋 무결성·라이선스 확인",
    "preprocess": "전처리·클래스 통합",
    "split_check": "학습·검증·테스트 분리 검증",
    "train_detection": "탐지·세그멘테이션 학습(B1)",
    "train_reid": "재식별 학습·평가(B2)",
    "matching_b3": "전후 매칭·분실 의심(B3)",
    "damage_b4": "파손 탐지·상태 비교(B4)",
    "eval_detection": "독립 테스트 탐지 평가",
    "eval_tracking": "추적·재등장 평가(합성 영상)",
    "video_reanalysis": "입력 영상 재분석·API JSON",
    "compare": "학습 전후 비교표",
    "report": "그래프·보고서",
    "package": "결과 압축",
}


class Ctx:
    def __init__(self, run_dir: Path, profile: dict, allow_cpu: bool):
        self.root = PROJECT_ROOT
        self.run = run_dir
        self.p = profile
        self.allow_cpu = allow_cpu
        self.data = self.root / "data"
        self.cache = self.data / "cache"
        self.coco_imgs = self.data / "datasets" / "coco_images"
        self.pre = self.root / "models" / "pretrained"
        self.tax = load_taxonomy()
        self.registry = load_yaml(self.root / "configs" / "datasets.yaml")
        self.state_path = run_dir / "state.json"
        self.state = (
            read_json(self.state_path)
            if self.state_path.is_file()
            else {"stages": {}, "results": {}}
        )
        # 전처리 결과 폴더는 '데이터에 영향을 주는 설정'의 해시로 구분한다. 설정을 바꾸면 새로
        # 만들고,
        # 같은 run을 재개할 때는 처음 정한 폴더를 그대로 쓴다(state.json에 기록).
        if self.state.get("dataset_dir"):
            self.ds = Path(self.state["dataset_dir"])
        else:
            key = json.dumps(
                {
                    "lvis": profile.get("lvis"),
                    "reid": {
                        k: profile["reid"].get(k)
                        for k in ("product_types", "min_images_per_item", "max_items", "split")
                    },
                    "crack": profile["damage"]["crack_model"].get("max_train_images"),
                    "homeobjects": profile.get("homeobjects", {}).get("val_fraction"),
                    "taxonomy": (self.root / "configs" / "taxonomy.yaml").read_text(
                        encoding="utf-8"
                    ),
                    "code_version": 2,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            import hashlib

            h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
            self.ds = self.data / "datasets" / f"{profile['profile']}_{h}"
            self.state["dataset_dir"] = str(self.ds)
        self.dev = self.state.get("devices")
        self.monitor = GpuMonitor(
            run_dir / "logs" / "gpu_usage.csv",
            profile.get("gpu", {}).get("monitor_interval_sec", 5),
        )

    def save(self):
        write_json(self.state_path, self.state)

    def res(self, key, default=None):
        return self.state["results"].get(key, default)

    def set(self, key, val):
        self.state["results"][key] = val
        self.save()

    @property
    def device(self):
        return self.dev["torch_device"]

    @property
    def ydev(self):
        return self.dev["yolo_device"]

    def py(self, *args):
        """프로젝트 CLI를 같은 파이썬으로 실행 (출력은 그대로 로그로)."""
        cmd = [sys.executable, "-u", *map(str, args)]
        log("  $ " + " ".join(cmd))
        subprocess.run(
            cmd, check=True, cwd=self.root, env={**os.environ, "PYTHONPATH": str(self.root / "src")}
        )


# ---------------------------------------------------------------- 단계 ----


def st_env_check(c: Ctx):
    inv = gpu_inventory()
    gcfg = c.p.get("gpu", {})
    c.dev = choose_devices(
        inv, gcfg.get("require_gpu", True), c.allow_cpu, gcfg.get("use_all_visible_gpus", True)
    )
    c.state["devices"] = c.dev
    du = shutil.disk_usage(c.root)
    mem = None
    if Path("/proc/meminfo").is_file():
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemTotal"):
                mem = round(int(ln.split()[1]) / 2**20, 1)
    smi = None
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    info = {
        "gpu": inv,
        "selected": c.dev,
        "environment": environment_info(),
        "disk_free_GiB": round(du.free / 2**30, 1),
        "ram_total_GiB": mem,
        "cpu_count": os.cpu_count(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": smi,
    }
    if c.dev["mode"] == "cuda":
        import torch

        a = torch.randn(2048, 2048, device="cuda:0")
        b = torch.randn(2048, 2048, device="cuda:0")
        err = float((a @ b).cpu().sub(a.cpu() @ b.cpu()).abs().max())
        info["cuda_matmul_check_max_abs_err"] = err
        if not err < 1.0:
            raise RuntimeError(f"GPU 연산 결과가 CPU와 다릅니다 (오차 {err})")
    (c.run / "env").mkdir(parents=True, exist_ok=True)
    write_json(c.run / "env" / "env.json", info)
    log(
        f"  장치: {c.dev} | "
        f"GPU: {[d['name'] + ' ' + str(d['total_GiB']) + 'GiB' for d in inv['devices']]} | 디스크 "
        f"여유 {info['disk_free_GiB']}GiB"
    )
    return {"devices": c.dev, "gpus": [d["name"] for d in inv["devices"]]}


def st_deps_check(c: Ctx):
    import importlib.metadata as md

    pk = {}
    for p in [
        "torch",
        "torchvision",
        "ultralytics",
        "timm",
        "scipy",
        "opencv-python",
        "opencv-python-headless",
        "numpy",
        "lap",
        "matplotlib",
        "PyYAML",
    ]:
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    before = os.environ.get("JIMSSAK_TORCH_BEFORE")
    import torch

    now = f"{torch.__version__} {torch.version.cuda}"
    if before and before.strip() != now:
        raise RuntimeError(f"설치 과정에서 torch가 바뀌었습니다: {before} → {now}")
    for m in ["ultralytics", "timm", "scipy", "cv2", "yaml", "matplotlib"]:
        __import__(m)
    write_json(
        c.run / "env" / "packages.json",
        {
            "packages": pk,
            "torch_before_install": before,
            "torch_now": now,
            "python": sys.version,
            "executable": sys.executable,
        },
    )
    return {"packages": pk}


def st_validate_inputs(c: Ctx):
    from .video_io import cleanup_temp, open_video

    v = c.root / c.p["input_video"]
    if not v.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {v}")
    cap, info, tmp = open_video(v)
    cap.release()
    cleanup_temp(tmp)
    need = {"smoke": 8, "standard": 40, "full": 120, "max": 80}.get(c.p["profile"], 40)
    free = shutil.disk_usage(c.root).free / 2**30
    if free < need:
        raise RuntimeError(f"디스크 여유 {free:.0f}GiB < 예상 필요량 {need}GiB")
    out = {
        "video": str(v),
        "decoded_wh": [info.decoded_width, info.decoded_height],
        "fps": info.fps,
        "frames": info.frame_count_reported,
        "taxonomy_classes": len(c.tax.codes),
        "disk_free_GiB": round(free, 1),
    }
    write_json(c.run / "env" / "inputs.json", out)
    return out


def _yolo_weights(c: Ctx, name: str) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset

    p = c.pre / f"{name}.pt"
    if not p.is_file():
        c.pre.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(p))
    if not p.is_file():
        raise FileNotFoundError(f"사전학습 가중치 다운로드 실패: {name}")
    return p


def st_pretrained_download(c: Ctx):
    from .common import sha256_file

    names = (
        {"yolo11n-seg"}
        | {m["name"] for m in c.p["seg_models"]}
        | {c.p["damage"]["crack_model"]["name"]}
    )
    if c.p.get("homeobjects", {}).get("enabled"):
        names.add(c.p["homeobjects"]["det_model"]["name"])
    rows = []
    for n in sorted(names):
        p = _yolo_weights(c, n)
        rows.append(
            {
                "model": n,
                "path": str(p),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "license": "AGPL-3.0 (Ultralytics)",
            }
        )
    from .reidentification.embedders import DinoBackbone

    bb = DinoBackbone(c.p["reid"]["backbone"], "cpu")
    rows.append(
        {
            "model": c.p["reid"]["backbone"],
            "path": "Hugging Face 캐시(HF_HOME)",
            "bytes": None,
            "sha256": None,
            "license": "Apache-2.0",
            "params": sum(p.numel() for p in bb.model.parameters()),
        }
    )
    from .damage.patchcore import FeatureExtractor

    fx = FeatureExtractor("cpu")
    rows.append(
        {
            "model": "wide_resnet50_2 (ImageNet)",
            "path": "torchvision 캐시(TORCH_HOME)",
            "bytes": None,
            "sha256": None,
            "license": "BSD-3 (torchvision)",
            "params": sum(p.numel() for p in fx.m.parameters()),
        }
    )
    write_csv(
        c.run / "env" / "pretrained_models.csv",
        rows,
        ["model", "path", "bytes", "sha256", "license", "params"],
    )
    return {"models": [r["model"] for r in rows]}


def st_baseline_b0_video(c: Ctx):
    vdir = c.run / "video"
    args = [
        "infer.py",
        "--video",
        c.root / c.p["input_video"],
        "--video-id",
        c.p["input_video_id"],
        "--label",
        "B0",
        "--weights",
        c.pre / "yolo11n-seg.pt",
        "--device",
        c.device,
        "--output-root",
        vdir,
        "--run-name",
        "B0",
    ]
    if c.p.get("video_max_frames"):
        args += ["--max-frames", c.p["video_max_frames"]]
    if not (vdir / "B0" / "B0" / "run_manifest.json").is_file():
        c.py(*args)
    run = vdir / "B0" / "B0"
    if not (run / "relink_hist" / "items.csv").is_file():
        c.py("link_reappearance.py", "--run", run, "--out-name", "relink_hist")
    c.py(
        "make_review_sheet.py",
        "--items",
        run / "items.csv",
        "--out",
        vdir / "match_review_B0.csv",
        "--overwrite",
    )
    m = read_json(run / "run_manifest.json")
    return {"run": str(run), "counts": m["counts"], "timing": m["timing"]}


def st_datasets_download(c: Ctx):
    from .datasources.download import append_download_log, download_file

    R = c.registry
    jobs = [
        (R["lvis_v1"]["files"]["val_ann"], c.cache / "lvis_v1_val.json.zip"),
        (R["crack_seg"]["files"]["zip"], c.cache / "crack-seg.zip"),
        (R["abo"]["files"]["listings"], c.cache / "abo-listings.tar"),
        (R["abo"]["files"]["images_meta"], c.cache / "abo_images.csv.gz"),
    ]
    if c.p["lvis"].get("use_train_json", True):
        jobs.append((R["lvis_v1"]["files"]["train_ann"], c.cache / "lvis_v1_train.json.zip"))
    if c.p["lvis"].get("use_coco_annotations"):
        jobs.append(
            (R["coco_images"]["files"]["annotations"], c.cache / "annotations_trainval2017.zip")
        )
    if c.p["lvis"].get("coco_images") == "zip":
        jobs.append((R["coco_images"]["files"]["val_zip"], c.cache / "val2017.zip"))
        if c.p["lvis"].get("use_train_json", True):
            jobs.append((R["coco_images"]["files"]["train_zip"], c.cache / "train2017.zip"))
    if c.p.get("homeobjects", {}).get("enabled"):
        jobs.append((R["homeobjects_3k"]["files"]["zip"], c.cache / "homeobjects-3K.zip"))
    if c.p["damage"].get("visa_categories"):
        jobs.append((R["visa"]["files"]["tar"], c.cache / "VisA_20220922.tar"))
    recs = []
    for f, dst in jobs:
        log(f"  다운로드: {f['url']} ({f['bytes'] / 2**20:.1f} MiB)")
        r = download_file(f["url"], dst, f["bytes"])
        append_download_log(c.run / "datasets" / "downloads.jsonl", {**r, "at": now_iso()})
        recs.append(r)
    total = sum(r["bytes"] for r in recs)
    return {
        "files": len(recs),
        "total_bytes": total,
        "downloaded_bytes_this_run": sum(r["downloaded_bytes"] for r in recs),
    }


def st_datasets_verify(c: Ctx):
    rows = []
    for f in sorted(c.cache.glob("*.zip")):
        with zipfile.ZipFile(f) as z:
            if f.stat().st_size < 2 * 2**30:
                bad = z.testzip()
                rows.append(
                    {"file": f.name, "check": "zip CRC", "ok": bad is None, "detail": bad or ""}
                )
            else:
                # 대용량(COCO train2017.zip 19GB): 목록·크기만 확인.
                # 손상 파일은 꺼낼 때 오류로 드러남
                n = len(z.namelist())
                rows.append(
                    {
                        "file": f.name,
                        "check": "zip 목록(대용량)",
                        "ok": n > 0,
                        "detail": f"{n} entries",
                    }
                )
    for f in sorted(c.cache.glob("*.tar")):
        with tarfile.open(f) as t:
            names = t.getnames()
        ok = True
        detail = f"{len(names)} entries"
        if f.name.startswith("abo-listings"):
            ok = "LICENSE-CC-BY-4.0.txt" in names
            detail += " | LICENSE-CC-BY-4.0.txt " + ("있음" if ok else "없음")
        if f.name.startswith("VisA"):
            ok = "split_csv/1cls.csv" in names
        rows.append({"file": f.name, "check": "tar 목록", "ok": ok, "detail": detail})
    import gzip

    with gzip.open(c.cache / "abo_images.csv.gz", "rt") as g:
        n = sum(1 for _ in g) - 1
    rows.append(
        {"file": "abo_images.csv.gz", "check": "gzip 읽기", "ok": n > 0, "detail": f"{n} rows"}
    )
    z = zipfile.ZipFile(c.cache / "lvis_v1_val.json.zip")
    head = z.open(z.namelist()[0]).read(400).decode("utf-8", "ignore")
    rows.append(
        {
            "file": "lvis_v1_val.json.zip",
            "check": "json 헤더",
            "ok": head.lstrip().startswith("{"),
            "detail": head[:80],
        }
    )
    lic = []
    for k, v in c.registry.items():
        if isinstance(v, dict) and "license" in v:
            lic.append(
                {
                    "dataset": v.get("name", k),
                    "license": v["license"],
                    "source": v.get("license_source", ""),
                    "role": v.get("role", ""),
                    "domain_gap": v.get("domain_gap", ""),
                }
            )
    for e in c.registry.get("excluded", []):
        lic.append(
            {
                "dataset": e["name"],
                "license": "-",
                "source": "",
                "role": "제외",
                "domain_gap": e["reason"],
            }
        )
    write_csv(c.run / "datasets" / "integrity.csv", rows, ["file", "check", "ok", "detail"])
    write_csv(
        c.run / "datasets" / "licenses.csv",
        lic,
        ["dataset", "license", "source", "role", "domain_gap"],
    )
    if not all(r["ok"] for r in rows):
        raise RuntimeError(f"데이터셋 무결성 검사 실패: {[r for r in rows if not r['ok']]}")
    return {"checked": len(rows)}


def st_preprocess(c: Ctx):
    from .datasources.abo import build_abo
    from .datasources.lvis_prep import build_lvis_dataset
    from .datasources.public_sets import build_crackseg, build_homeobjects, extract_visa

    out = {}
    if not (c.ds / "lvis_seg" / "data.yaml").is_file():
        out["lvis"] = build_lvis_dataset(
            cache=c.cache, out=c.ds / "lvis_seg", img_root=c.coco_imgs, tax=c.tax, cfg=c.p["lvis"]
        )
    else:
        out["lvis"] = read_json(c.ds / "lvis_seg" / "dataset_stats.json")
    shutil.copy2(
        c.ds / "lvis_seg" / "class_availability.csv", c.run / "datasets" / "class_availability.csv"
    )
    if not (c.ds / "crackseg" / "data.yaml").is_file():
        out["crack"] = build_crackseg(
            zpath=c.cache / "crack-seg.zip",
            work=c.ds / "_raw",
            out=c.ds / "crackseg",
            max_train_images=c.p["damage"]["crack_model"].get("max_train_images"),
        )
    else:
        out["crack"] = read_json(c.ds / "crackseg" / "dataset_stats.json")
    if c.p.get("homeobjects", {}).get("enabled"):
        if not (c.ds / "homeobjects" / "data.yaml").is_file():
            out["homeobjects"] = build_homeobjects(
                zpath=c.cache / "homeobjects-3K.zip",
                work=c.ds / "_raw",
                out=c.ds / "homeobjects",
                tax=c.tax,
                val_fraction=c.p["homeobjects"]["val_fraction"],
            )
        else:
            out["homeobjects"] = read_json(c.ds / "homeobjects" / "dataset_stats.json")
    if not (c.ds / "abo" / "abo_manifest.csv").is_file():
        out["abo"] = build_abo(cache=c.cache, out=c.ds / "abo", cfg=c.p["reid"])
    else:
        out["abo"] = read_json(c.ds / "abo" / "dataset_stats.json")
    if c.p["damage"].get("visa_categories"):
        out["visa"] = extract_visa(
            tar_path=c.cache / "VisA_20220922.tar",
            out=c.data / "datasets" / "visa",
            categories=c.p["damage"]["visa_categories"],
        )
    write_json(c.run / "datasets" / "dataset_stats.json", out)
    return {
        k: (v.get("images_written") or v.get("images") or v.get("per_split") or v.get("categories"))
        for k, v in out.items()
    }


def st_split_check(c: Ctx):
    rep = {}

    def names(d, s):
        return (
            {p.name for p in (d / "images" / s).glob("*")} if (d / "images" / s).is_dir() else set()
        )

    for key in ["lvis_seg", "crackseg", "homeobjects"]:
        d = c.ds / key
        if not d.is_dir():
            continue
        tr, va, te = names(d, "train"), names(d, "val"), names(d, "test")
        rep[key] = {
            "train": len(tr),
            "val": len(va),
            "test": len(te),
            "overlap_train_val": len(tr & va),
            "overlap_train_test": len(tr & te),
            "overlap_val_test": len(va & te),
        }
    meta = read_json(c.ds / "lvis_seg" / "test_federated_meta.json")
    rep["lvis_seg"]["test_all_from_coco_val2017"] = all(
        "val2017" in v["coco_url"] for v in meta.values()
    )
    rows = read_csv(c.ds / "abo" / "abo_manifest.csv")
    from collections import defaultdict

    it, gr = defaultdict(set), defaultdict(set)
    for r in rows:
        it[r["item_id"]].add(r["split"])
        gr[r["group"]].add(r["split"])
    rep["abo"] = {
        "items_in_multiple_splits": sum(1 for v in it.values() if len(v) > 1),
        "groups_in_multiple_splits": sum(1 for v in gr.values() if len(v) > 1),
    }
    rep["input_video_in_training"] = False
    rep["note"] = "언니방영상.mp4는 어떤 학습 데이터에도 쓰지 않음(입력 영상은 추론·검수 전용)"
    write_json(c.run / "datasets" / "split_report.json", rep)
    bad = [
        k
        for k, v in rep.items()
        if isinstance(v, dict) and any(str(x).startswith("overlap") and v[x] for x in v)
    ]
    if (
        bad
        or rep["abo"]["items_in_multiple_splits"]
        or rep["abo"]["groups_in_multiple_splits"]
        or not rep["lvis_seg"]["test_all_from_coco_val2017"]
    ):
        raise RuntimeError(f"분할 누수 발견: {rep}")
    return rep


def _best_val(run_dir: Path, key="metrics/mAP50-95(M)"):
    rows = []
    p = Path(run_dir) / "results.csv"
    if p.is_file():
        import csv

        with open(p, encoding="utf-8") as f:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return max(vals) if vals else None


def st_train_detection(c: Ctx):
    from .detection.train_yolo import train_yolo

    common = c.p["train_common"]
    res = {}
    for m in c.p["seg_models"]:
        c.monitor.stage = f"train_{m['name']}"
        r = train_yolo(
            weights=str(c.pre / f"{m['name']}.pt"),
            data_yaml=c.ds / "lvis_seg" / "data.yaml",
            project=c.run / "models" / "yolo",
            name=f"lvis_{m['name']}",
            epochs=m["epochs"],
            batch=m["batch"],
            imgsz=m["imgsz"],
            device=c.ydev,
            common=common,
        )
        r["val_mask_mAP50_95_best"] = _best_val(r["run_dir"])
        res[f"B1_{m['name']}"] = r
    best = max(res, key=lambda k: res[k]["val_mask_mAP50_95_best"] or -1)
    c.set("seg_models", res)
    c.set(
        "best_b1",
        {"label": best, "weights": res[best]["best_pt"], "rule": "검증(val) mask mAP50-95 최고"},
    )
    ho = c.p.get("homeobjects", {})
    if ho.get("enabled"):
        m = ho["det_model"]
        c.monitor.stage = "train_homeobjects_det"
        r = train_yolo(
            weights=str(c.pre / f"{m['name']}.pt"),
            data_yaml=c.ds / "homeobjects" / "data.yaml",
            project=c.run / "models" / "yolo",
            name=f"homeobj_{m['name']}",
            epochs=m["epochs"],
            batch=m["batch"],
            imgsz=m["imgsz"],
            device=c.ydev,
            common=common,
        )
        c.set("homeobj_det", r)
    est = estimate_other_profiles(c, res)
    return {
        "seg_models": {
            k: {
                kk: v[kk]
                for kk in ("epochs_completed", "batch", "best_pt", "val_mask_mAP50_95_best")
            }
            for k, v in res.items()
        },
        "best_b1": best,
        "time_estimate": est,
    }


def estimate_other_profiles(c: Ctx, res: dict) -> dict:
    """이번 실행의 epoch 시간(실측)으로 다른 프로필의 세그멘테이션 학습 시간을 선형 외삽 (해상도
    제곱 보정). 추정치임을 명시.
    """
    n_train = len(list((c.ds / "lvis_seg" / "images" / "train").glob("*")))
    out = {
        "basis": f"{c.p['profile']} 실행 실측 epoch 시간, 학습 이미지 {n_train}장, 장치 {c.dev}",
        "method": "초/이미지/epoch × (목표 imgsz/실측 imgsz)^2 × 목표 이미지 수 × epoch — "
        "추정(데이터 로딩·검증 비중 차이로 오차 큼)",
    }
    per = {}
    for m in c.p["seg_models"]:
        r = res.get(f"B1_{m['name']}", {})
        et = [x for x in (r.get("epoch_times_sec") or []) if x and x > 0]
        if et and n_train:
            per[m["name"]] = (float(np.median(et)) / n_train, m["imgsz"])
    for prof in ("smoke", "standard", "full", "max"):
        P = load_yaml(c.root / "configs" / "profiles" / f"{prof}.yaml")
        tot, det = 0.0, {}
        for m in P["seg_models"]:
            base = per.get(m["name"]) or (next(iter(per.values())) if per else None)
            if not base:
                continue
            s = base[0] * (m["imgsz"] / base[1]) ** 2 * P["lvis"]["max_train_images"] * m["epochs"]
            if m["name"] not in per:
                s *= {"yolo11s-seg": 2.0, "yolo11m-seg": 4.0}.get(
                    m["name"], 1.0
                )  # 측정 없는 크기는 대략 배수(근거 약함)
            det[m["name"]] = round(s / 3600, 2)
            tot += s
        out[prof] = {"seg_training_hours_estimate": round(tot / 3600, 2), "per_model_hours": det}
    write_json(c.run / "env" / "time_estimate.json", out)
    log(
        f"  학습 시간 추정(세그멘테이션만, {out['basis']}): "
        + ", ".join(
            f"{k} {v['seg_training_hours_estimate']}h"
            for k, v in out.items()
            if isinstance(v, dict)
        )
    )
    return out


def _embedders(c: Ctx, with_trained=True):
    from .reidentification.embedders import (
        DinoBackbone,
        DinoEmbedder,
        HistEmbedder,
        TrainedEmbedder,
    )

    bb = DinoBackbone(c.p["reid"]["backbone"], c.device)
    E = {"hist": HistEmbedder(), "dinov2": DinoEmbedder(bb)}
    ck = c.run / "models" / "reid" / "best.pt"
    if with_trained and ck.is_file():
        E["trained"] = TrainedEmbedder(bb, ck)
    return E, bb


def st_train_reid(c: Ctx):
    from .common import imread
    from .reidentification.train_reid import (
        auroc,
        best_threshold,
        pair_scores,
        read_manifest,
        retrieval_metrics,
        train_head,
    )

    E, bb = _embedders(c, with_trained=False)
    c.monitor.stage = "train_reid"
    tr = train_head(
        bb=bb, abo_root=c.ds / "abo", out_dir=c.run / "models" / "reid", cfg=c.p["reid"]
    )
    E, bb = _embedders(c)
    rows = read_manifest(c.ds / "abo" / "abo_manifest.csv")
    out = {"train": tr, "methods": {}}
    for split in ("val", "test"):
        rs = [r for r in rows if r["split"] == split]
        imgs = [imread(c.ds / "abo" / r["path"]) for r in rs]
        labels = [r["item_id"] for r in rs]
        for name, emb in E.items():
            z = emb.embed(imgs)
            s, y = pair_scores(z, labels)
            m = retrieval_metrics(z, labels)
            rec = out["methods"].setdefault(name, {})
            rec[split] = {
                **m,
                "pair_auroc": round(auroc(s, y), 4) if auroc(s, y) is not None else None,
            }
            if split == "val":
                rec["val_threshold"] = best_threshold(s, y)
    out["note"] = (
        "hist·dinov2는 학습 없는 기준선(사전학습 특징+유사도), trained는 ABO triplet 학습 결과. "
        "평가는 ABO 테스트 상품(학습과 그룹 단위로 분리)의 다른 사진 검색"
    )
    (c.run / "eval" / "reid").mkdir(parents=True, exist_ok=True)
    write_json(c.run / "eval" / "reid" / "reid_eval.json", out)
    tab = [
        {
            "method": k,
            **{f"{s}_{kk}": vv for s in ("val", "test") for kk, vv in v.get(s, {}).items()},
            "val_threshold": v.get("val_threshold", {}).get("threshold"),
        }
        for k, v in out["methods"].items()
    ]
    write_csv(c.run / "eval" / "reid" / "reid_comparison.csv", tab, list(tab[0].keys()))
    c.set("reid", out)
    return {k: v.get("test") for k, v in out["methods"].items()}


def st_matching_b3(c: Ctx):
    from .matching.before_after import summarize
    from .matching.bench import embed_scenarios, make_scenarios, run, tune_threshold
    from .reidentification.train_reid import read_manifest

    E_, _ = _embedders(c)
    rows = read_manifest(c.ds / "abo" / "abo_manifest.csv")
    mb = c.p["matching_bench"]
    val = make_scenarios(
        rows,
        "val",
        mb["n_val_scenarios"],
        mb["items_per_scenario"],
        mb["missing_per_scenario"],
        mb["unexpected_per_scenario"],
        1,
    )
    test = make_scenarios(
        rows,
        "test",
        mb["n_test_scenarios"],
        mb["items_per_scenario"],
        mb["missing_per_scenario"],
        mb["unexpected_per_scenario"],
        2,
    )
    Ev = embed_scenarios(val, c.ds / "abo", E_)
    Et = embed_scenarios(test, c.ds / "abo", E_)
    res = {"count_only": {"threshold": None, "test": summarize(run(test, Et, "count_only", None))}}
    for m in E_:
        t = tune_threshold(val, Ev, m)
        res[m] = {
            "threshold": t["threshold"],
            "val": t,
            "test": summarize(run(test, Et, m, t["threshold"])),
        }
    out = {
        "methods": res,
        "scenarios": {"val": len(val), "test": len(test)},
        "kind": "합성(ABO 상품 사진) — 실제 이사 영상 아님",
        "real_video_evaluation": "미측정 (이사 전후 실제 영상 쌍과 정답 없음)",
        "notes": "클래스는 정답 product_type 사용(검출 오류 제외). 임계값은 val 시나리오에서 선택 "
        "후 test에 고정 적용",
    }
    (c.run / "eval" / "matching").mkdir(parents=True, exist_ok=True)
    write_json(c.run / "eval" / "matching" / "b3_eval.json", out)
    tab = [{"method": k, "threshold": v["threshold"], **v["test"]} for k, v in res.items()]
    write_csv(c.run / "eval" / "matching" / "b3_comparison.csv", tab, list(tab[0].keys()))
    c.set("b3", {k: {"threshold": v["threshold"]} for k, v in res.items()})
    return {
        k: {kk: v["test"][kk] for kk in ("missing_f1", "false_missing_rate", "match_accuracy")}
        for k, v in res.items()
    }


def st_damage_b4(c: Ctx):
    from .damage_stage import run_damage

    return run_damage(c)


def st_eval_detection(c: Ctx):
    from .detection.evaluate import evaluate_model

    meta = read_json(c.ds / "lvis_seg" / "test_federated_meta.json")
    data = load_yaml(c.ds / "lvis_seg" / "data.yaml")
    final = list(data["names"].values())
    models = {"B0_yolo11n-seg_COCO": str(c.pre / "yolo11n-seg.pt")}
    for k, v in c.res("seg_models", {}).items():
        models[k] = v["best_pt"]
    imgsz = c.p["seg_models"][0]["imgsz"]
    summ = {}
    for label, w in models.items():
        c.monitor.stage = f"eval_{label}"
        summ[label] = evaluate_model(
            model_path=w,
            data_yaml=c.ds / "lvis_seg" / "data.yaml",
            split="test",
            tax=c.tax,
            device=c.device,
            out_dir=c.run / "eval" / "detection" / "lvis_test" / label,
            label=label,
            federated_meta=meta,
            imgsz=imgsz,
        )
    b0_supported = set(summ["B0_yolo11n-seg_COCO"]["supported_classes"])
    common = sorted(set(final) & b0_supported)
    new = sorted(set(final) - b0_supported)
    c.set("class_groups", {"common": common, "new": new})
    ho = {}
    if c.p.get("homeobjects", {}).get("enabled"):
        hm = dict(models)
        if c.res("homeobj_det"):
            hm["B1_homeobjects_det"] = c.res("homeobj_det")["best_pt"]
        for label, w in hm.items():
            ho[label] = evaluate_model(
                model_path=w,
                data_yaml=c.ds / "homeobjects" / "data.yaml",
                split="test",
                tax=c.tax,
                device=c.device,
                out_dir=c.run / "eval" / "detection" / "homeobjects_test" / label,
                label=label,
                imgsz=640,
                with_masks=False,
            )
    return {
        "lvis_test": {k: v["all_gt_classes"] for k, v in summ.items()},
        "common_classes": common,
        "new_classes": new,
        "homeobjects_test": {k: v["all_gt_classes"] for k, v in ho.items()},
    }


def st_eval_tracking(c: Ctx):
    from .tracking.bench import aggregate, run_bench
    from .tracking.synth_video import pick_images

    E, _ = _embedders(c)
    emb = {k: v for k, v in E.items() if k in ("dinov2", "trained")}
    data = load_yaml(c.ds / "lvis_seg" / "data.yaml")
    names = [data["names"][i] for i in sorted(data["names"])]
    tb = c.p["tracking_bench"]
    lcfg = load_yaml(c.root / "configs" / "pipeline_default.yaml")
    hist_cfg = lcfg["relink"]
    lst = lcfg["listing"]
    dets = {"B0": str(c.pre / "yolo11n-seg.pt"), "B1": c.res("best_b1")["weights"]}
    val_imgs = pick_images(
        c.ds / "lvis_seg" / "images" / "val",
        c.ds / "lvis_seg" / "labels" / "val",
        names,
        tb["n_val_sequences"],
        11,
    )
    test_imgs = pick_images(
        c.ds / "lvis_seg" / "images" / "test",
        c.ds / "lvis_seg" / "labels" / "test",
        names,
        tb["n_test_sequences"],
        12,
    )
    common = dict(
        names=names,
        n_frames=tb["frames_per_sequence"],
        out_size=tb["out_size"],
        device=c.device,
        tax=c.tax,
        confirm_frames=lst["confirm_min_frames"],
        confirm_conf=lst["confirm_min_mean_conf"],
        hist_cfg=hist_cfg,
    )
    grid = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
    vrows = run_bench(
        seq_images=val_imgs,
        lbl_dir=c.ds / "lvis_seg" / "labels" / "val",
        detectors=dets,
        trackers=["botsort"],
        methods=list(emb),
        embedders=emb,
        thresholds={},
        thr_grid=grid,
        **common,
    )
    vag = aggregate(vrows)
    thr = {}
    for d in dets:
        for m in emb:
            cand = [r for r in vag if r["detector"] == d and r["relink"] == m]
            if cand:
                thr[(d, m)] = max(cand, key=lambda r: (r["item_f1"] or 0, r["mot_IDF1"] or 0))[
                    "threshold"
                ]
    trows = []
    for d, w in dets.items():
        th = {m: thr.get((d, m)) for m in emb}
        trows += run_bench(
            seq_images=test_imgs,
            lbl_dir=c.ds / "lvis_seg" / "labels" / "test",
            detectors={d: w},
            trackers=["botsort", "bytetrack"],
            methods=["none", "hist_rule", *emb],
            embedders=emb,
            thresholds=th,
            **common,
        )
    tag = aggregate(trows)
    od = c.run / "eval" / "tracking"
    od.mkdir(parents=True, exist_ok=True)

    def flat(rs):
        return [
            {
                k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v)
                for k, v in r.items()
            }
            for r in rs
        ]

    write_csv(od / "val_rows.csv", flat(vrows), list(vrows[0].keys()))
    write_csv(od / "val_threshold_search.csv", vag, list(vag[0].keys()))
    write_csv(od / "test_rows.csv", flat(trows), list(trows[0].keys()))
    write_csv(od / "tracking_comparison.csv", tag, list(tag[0].keys()))
    thr_s = {f"{d}|{m}": v for (d, m), v in thr.items()}
    write_json(
        od / "thresholds.json",
        {
            "selected_on": "val 합성 영상(LVIS 검증 분할 이미지)",
            "rule": "물품 F1 최대, 동률이면 IDF1",
            "thresholds": thr_s,
            "val_sequences": [p.name for p in val_imgs],
            "test_sequences": [p.name for p in test_imgs],
            "kind": "합성 카메라 이동 영상 (2D 이동·확대·가림) — 실제 3D 시점 변화 미포함",
        },
    )
    c.set("tracking_thresholds", thr_s)
    return {"test_sequences": len(test_imgs), "configs": len(tag)}


def st_video_reanalysis(c: Ctx):
    from .video_items import build_item_json, embed_tracks, embedding_relink

    vdir = c.run / "video"
    b1 = c.res("best_b1")
    args = [
        "infer.py",
        "--video",
        c.root / c.p["input_video"],
        "--video-id",
        c.p["input_video_id"],
        "--label",
        "B1",
        "--weights",
        b1["weights"],
        "--device",
        c.device,
        "--output-root",
        vdir,
        "--run-name",
        "B1",
    ]
    if c.p.get("video_max_frames"):
        args += ["--max-frames", c.p["video_max_frames"]]
    run1 = vdir / "B1" / "B1"
    if not (run1 / "run_manifest.json").is_file():
        c.py(*args)
    c.py(
        "make_review_sheet.py",
        "--items",
        run1 / "items.csv",
        "--out",
        vdir / "match_review_B1.csv",
        "--overwrite",
    )
    E, _ = _embedders(c)
    method = "trained" if "trained" in E else "dinov2"
    thr = (c.res("tracking_thresholds") or {}).get(f"B1|{method}")
    if thr is None:
        thr = 0.8
    Z = embed_tracks(run1, E[method])
    np.savez(run1 / f"item_embeddings_{method}.npz", **Z)
    embedding_relink(run1, Z, thr, run1 / f"relink_{method}", method)
    vid = c.p["input_video_id"]
    api = c.run / "api"
    run0 = vdir / "B0" / "B0"
    build_item_json(
        run_dir=run0,
        items_csv=run0 / "items.csv",
        video_id=vid,
        model_label="B0 COCO yolo11n-seg + BoT-SORT",
        emb_ref=None,
        emb_keys=None,
        out_path=api / "items_B0.json",
        rel_base="video/B0/B0",
    )
    build_item_json(
        run_dir=run1,
        items_csv=run1 / "items.csv",
        video_id=vid,
        model_label=f"B1 {b1['label']} + BoT-SORT",
        emb_ref=f"video/B1/B1/item_embeddings_{method}.npz",
        emb_keys={k: k for k in Z},
        out_path=api / "items_B1.json",
        rel_base="video/B1/B1",
    )
    doc = build_item_json(
        run_dir=run1,
        items_csv=run1 / f"relink_{method}" / "items.csv",
        video_id=vid,
        model_label=f"B2 {b1['label']} + BoT-SORT + {method} 재등장 연결(임계 {thr})",
        emb_ref=f"video/B1/B1/item_embeddings_{method}.npz",
        emb_keys={k: k for k in Z},
        out_path=api / "items_B2_final.json",
        rel_base="video/B1/B1",
    )
    from collections import Counter

    def cnt(p):
        return Counter(r["category"] for r in read_csv(p) if r["status"] == "confirmed")

    comp = []
    a, b, d2 = (
        cnt(run0 / "items.csv"),
        cnt(run1 / "items.csv"),
        cnt(run1 / f"relink_{method}" / "items.csv"),
    )
    for k in sorted(set(a) | set(b) | set(d2)):
        comp.append(
            {
                "class": k,
                "B0_confirmed": a.get(k, 0),
                "B1_confirmed": b.get(k, 0),
                "B2_confirmed": d2.get(k, 0),
            }
        )
    write_csv(
        vdir / "video_item_counts_B0_B1_B2.csv",
        comp,
        ["class", "B0_confirmed", "B1_confirmed", "B2_confirmed"],
    )
    return {
        "B2_items": doc["n_items"],
        "relink_method": method,
        "relink_threshold": thr,
        "accuracy": "미측정 (정답 목록 없음 — 검수표 작성 후 evaluate_list.py로 채점)",
    }


def st_compare(c: Ctx):
    from .compare_tables import build_all

    return build_all(c)


def st_report(c: Ctx):
    from .report import build_report

    return build_report(c)


def st_package(c: Ctx):
    out = c.run / "final_models"
    out.mkdir(exist_ok=True)
    copied = []
    for k, v in (c.res("seg_models") or {}).items():
        shutil.copy2(v["best_pt"], out / f"{k}_best.pt")
        copied.append(f"{k}_best.pt")
    if c.res("homeobj_det"):
        shutil.copy2(c.res("homeobj_det")["best_pt"], out / "B1_homeobjects_det_best.pt")
        copied.append("B1_homeobjects_det_best.pt")
    if c.res("crack"):
        shutil.copy2(c.res("crack")["best_pt"], out / "B4_crack_seg_best.pt")
        copied.append("B4_crack_seg_best.pt")
    if (c.run / "models" / "reid" / "best.pt").is_file():
        shutil.copy2(c.run / "models" / "reid" / "best.pt", out / "B2_reid_head_best.pt")
        copied.append("B2_reid_head_best.pt")
    write_json(
        out / "README.json",
        {
            "files": copied,
            "pretrained_B0": "models/pretrained/yolo11n-seg.pt (Ultralytics 공식, 재다운로드 가능)",
        },
    )
    dst = c.root / "outputs" / f"jimssak_results_{c.run.name}.tar.gz"
    skip = {"feat_train.npz", "feat_val.npz"}

    def filt(ti):
        n = Path(ti.name).name
        if (
            n in skip
            or "/models/yolo/" in ti.name
            and n.endswith(".pt")
            and not n.startswith("best")
        ):
            return None
        if "/models/yolo/" in ti.name and "/weights/" in ti.name:
            return None  # 학습 폴더의 가중치는 final_models/에 best만 복사
        return ti

    with tarfile.open(dst, "w:gz") as t:
        t.add(c.run, arcname=c.run.name, filter=filt)
        lg = c.root / "logs" / c.run.name
        if lg.is_dir():
            t.add(lg, arcname=f"{c.run.name}/logs_shell")
    c.set("package", str(dst))
    log(f"  결과 압축: {dst} ({dst.stat().st_size / 2**20:.1f} MiB)")
    return {"archive": str(dst), "bytes": dst.stat().st_size}


FUNCS = {s: globals()[f"st_{s}"] for s in STAGES}


def run(
    profile_name: str,
    run_id: str | None,
    from_stage: str | None,
    allow_cpu: bool,
    only: list[str] | None = None,
) -> Path:
    prof = load_yaml(PROJECT_ROOT / "configs" / "profiles" / f"{profile_name}.yaml")
    if run_id == "latest":
        cands = sorted((PROJECT_ROOT / "outputs").glob(f"{profile_name}_*/state.json"))
        if not cands:
            raise SystemExit(f"재개할 {profile_name} 실행이 없습니다")
        run_dir = cands[-1].parent
    elif run_id:
        run_dir = PROJECT_ROOT / "outputs" / run_id
    else:
        run_dir = PROJECT_ROOT / "outputs" / f"{profile_name}_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    c = Ctx(run_dir, prof, allow_cpu)
    c.state.setdefault("profile", profile_name)
    c.state.setdefault("created_at", now_iso())
    c.state["allow_cpu"] = allow_cpu
    c.save()
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
    log(f"=== 실행 {run_dir.name} (프로필 {profile_name}) ===")
    started = from_stage is None
    c.monitor.start("init")
    try:
        for s in STAGES:
            if only and s not in only:
                continue
            if from_stage and s == from_stage:
                started = True
            prev = c.state["stages"].get(s, {})
            if not started and prev.get("status") == "ok":
                continue
            if prev.get("status") == "ok" and from_stage is None:
                log(f"[{s}] 이미 성공 → 건너뜀")
                continue
            if s != "env_check" and c.dev is None:
                st_env_check(c)
            log(f"[{s}] 시작 — {STAGE_KO[s]}")
            c.monitor.stage = s
            t0 = time.time()
            c.state["stages"][s] = {"status": "running", "start": now_iso()}
            c.save()
            try:
                summary = FUNCS[s](c)
                c.state["stages"][s] = {
                    "status": "ok",
                    "start": c.state["stages"][s]["start"],
                    "end": now_iso(),
                    "seconds": round(time.time() - t0, 1),
                    "summary": summary,
                }
                c.save()
                log(f"[{s}] 성공 ({time.time() - t0:.1f}s)")
            except BaseException as e:
                c.state["stages"][s] = {
                    "status": "failed",
                    "start": c.state["stages"][s]["start"],
                    "end": now_iso(),
                    "seconds": round(time.time() - t0, 1),
                    "error": repr(e)[:2000],
                    "traceback": traceback.format_exc()[-6000:],
                }
                c.save()
                log(f"[{s}] 실패: {e!r}")
                log(
                    f"  재개 명령: bash scripts/run_all.sh --profile {profile_name} "
                    f"--resume {run_dir.name}"
                )
                raise
            finally:
                rows = [
                    {
                        "stage": k,
                        "name_ko": STAGE_KO[k],
                        "status": v.get("status"),
                        "seconds": v.get("seconds"),
                        "start": v.get("start"),
                        "end": v.get("end"),
                        "error": (v.get("error") or "")[:300],
                    }
                    for k, v in c.state["stages"].items()
                ]
                write_csv(
                    run_dir / "stage_status.csv",
                    rows,
                    ["stage", "name_ko", "status", "seconds", "start", "end", "error"],
                )
    finally:
        c.monitor.stop()
        write_json(
            run_dir / "logs" / "gpu_usage_summary.json",
            summarize_gpu_csv(run_dir / "logs" / "gpu_usage.csv"),
        )
    log(f"=== 완료: {run_dir} ===")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="짐싹 전체 자동 학습·평가 파이프라인")
    ap.add_argument("--profile", default="standard", choices=["smoke", "standard", "full", "max"])
    ap.add_argument("--resume", default=None, help="재개할 run 이름 또는 latest")
    ap.add_argument("--from-stage", default=None, choices=STAGES, help="이 단계부터 다시 실행")
    ap.add_argument("--only", nargs="*", default=None, choices=STAGES, help="지정한 단계만 실행")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="GPU 없이 CPU로 점검 실행 (서버 실행에서는 쓰지 않음)",
    )
    a = ap.parse_args(argv)
    run(a.profile, a.resume, a.from_stage, a.allow_cpu, a.only)


if __name__ == "__main__":
    main()
