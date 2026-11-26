#!/usr/bin/env python
"""YOLO 분할 파인튜닝 (B1). GPU 한 장 기본.

예) 파이프라인 점검용 짧은 학습
    python -u train.py --data /workspace/jimssak_data/datasets/v1/data.yaml --name pilot_e10
    --epochs 10 --device cuda:0
예) 재개 (last.pt에서 이어서, 처음 지정한 --epochs '전체 목표'까지)
    python -u train.py --resume runs/train/pilot_e10/weights/last.pt

- --epochs는 '전체 목표 epoch 수'다. 재개 시 추가 epoch가 아니라 원래 목표까지 이어서 학습한다.
- best.pt는 val 분할(검증용 방)의 fitness(box·mask mAP 가중합, Ultralytics 기준)가 가장 높은 epoch.
- 결과 폴더가 이미 있으면 덮어쓰지 않고 실패한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import (  # noqa: E402
    PROJECT_ROOT,
    code_fingerprint,
    environment_info,
    fail,
    git_info,
    load_yaml,
    log,
    now_iso,
    read_json,
    resolve_device,
    set_offline_friendly_ultralytics_env,
    sha256_file,
    write_json,
)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="짐싹 YOLO 분할 파인튜닝 (체크포인트·재개·실행 기록 포함)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", type=Path, help="Ultralytics data.yaml (cvat_coco_to_yolo.py 출력)")
    ap.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "yolo11n-seg.pt",
        help="시작 가중치",
    )
    ap.add_argument("--name", help="실행 이름 (runs/train/<name>)")
    ap.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "train")
    ap.add_argument("--epochs", type=int, default=50, help="전체 목표 epoch 수")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="GPU 메모리(32GB)에 맞게 조절. -1이면 Ultralytics 자동",
    )
    ap.add_argument("--workers", type=int, default=4, help="DataLoader worker 수 (보수적 기본값)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--patience", type=int, default=30, help="val 개선이 없을 때 조기 종료까지 epoch 수"
    )
    ap.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="N epoch마다 epochN.pt 추가 저장 (-1 끄기). 디스크 쿼터 주의",
    )
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument(
        "--splits", type=Path, default=None, help="make_split.py의 splits.csv (기록용 해시)"
    )
    ap.add_argument("--resume", type=Path, default=None, help="이어서 학습할 last.pt")
    args = ap.parse_args(argv)
    set_offline_friendly_ultralytics_env()
    import ultralytics
    from ultralytics import YOLO

    device = resolve_device(args.device)
    if args.resume:
        if not args.resume.is_file():
            fail(f"재개할 체크포인트가 없습니다: {args.resume}")
        save_dir = args.resume.resolve().parent.parent
        mpath = save_dir / "train_manifest.json"
        man = read_json(mpath) if mpath.is_file() else {}
        man.setdefault("resumes", []).append(
            {
                "at": now_iso(),
                "from": str(args.resume),
                "device": device,
                "command": " ".join(sys.argv),
            }
        )
        write_json(mpath, man)
        log(f"재개: {args.resume} (원래 목표 epoch까지)")
        model = YOLO(str(args.resume))
        model.train(resume=True, device=device)
        _finish(model, save_dir, mpath)
        return

    if not args.data or not args.name:
        fail("--data 와 --name 이 필요합니다 (재개는 --resume)")
    if not args.data.is_file():
        fail(f"data.yaml이 없습니다: {args.data}")
    data = load_yaml(args.data)
    save_dir = args.project.resolve() / args.name
    if save_dir.exists():
        fail(f"이미 있는 실행 이름입니다: {save_dir} (기존 결과 보존을 위해 다른 --name을 쓰세요)")
    if not args.weights.is_file():
        from ultralytics.utils.downloads import attempt_download_asset

        args.weights.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(str(args.weights))
    ds_root = Path(data.get("path") or args.data.parent)
    label_report = ds_root / "label_check_report.json"
    manifest = {
        "created_at": now_iso(),
        "command": " ".join(sys.argv),
        "save_dir": str(save_dir),
        "data_yaml": str(args.data.resolve()),
        "data_yaml_sha256": sha256_file(args.data),
        "names": data.get("names"),
        "splits_csv": str(args.splits) if args.splits else None,
        "splits_sha256": sha256_file(args.splits)
        if args.splits and args.splits.is_file()
        else None,
        "label_check": read_json(label_report)
        if label_report.is_file()
        else "label_check_report.json 없음(check_labels.py 미실행)",
        "init_weights": str(args.weights.resolve()),
        "init_weights_sha256": sha256_file(args.weights),
        "hyper": {
            "epochs_total_target": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "patience": args.patience,
            "save_period": args.save_period,
        },
        "device": device,
        "environment": environment_info(),
        "code_sha256": code_fingerprint(),
        **git_info(),
        "ultralytics_version": ultralytics.__version__,
        "checkpoint_selection": "best.pt = val 분할 fitness 최고 epoch (Ultralytics 기준), "
        "last.pt = 마지막 epoch",
        "resume_command": f"python -u train.py --resume {save_dir / 'weights' / 'last.pt'}",
        "note": "정확히 같은 수치 재현은 검증하지 않음 (GPU 비결정성 가능).",
    }
    if "test" in data:
        manifest["warning"] = (
            "data.yaml에 test 분할이 있지만 학습·체크포인트 선택에는 val만 사용됨. test는 "
            "마지막에 한 번만 평가."
        )
    mpath = save_dir / "train_manifest.json"

    def on_start(trainer):
        Path(trainer.save_dir).mkdir(parents=True, exist_ok=True)
        if Path(trainer.save_dir).resolve() != save_dir:
            manifest["save_dir_actual"] = str(trainer.save_dir)
        write_json(Path(trainer.save_dir) / "train_manifest.json", manifest)
        log(f"학습 시작: {trainer.save_dir}")

    model = YOLO(str(args.weights))
    if model.task != "segment":
        fail(f"분할 모델이 아닙니다: {args.weights}")
    model.add_callback("on_pretrain_routine_end", on_start)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        seed=args.seed,
        deterministic=args.deterministic,
        patience=args.patience,
        save_period=args.save_period,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        plots=True,
        val=True,
    )
    _finish(
        model, Path(model.trainer.save_dir), Path(model.trainer.save_dir) / "train_manifest.json"
    )


def _finish(model, save_dir: Path, mpath: Path):
    tr = model.trainer
    man = read_json(mpath) if mpath.is_file() else {}
    w = save_dir / "weights"
    man["finished_at"] = now_iso()
    man["result"] = {
        "best_pt": str(w / "best.pt") if (w / "best.pt").is_file() else None,
        "last_pt": str(w / "last.pt") if (w / "last.pt").is_file() else None,
        "best_pt_sha256": sha256_file(w / "best.pt") if (w / "best.pt").is_file() else None,
        "epochs_completed": int(getattr(tr, "epoch", -1)) + 1,
        "best_fitness": float(tr.best_fitness or 0)
        if getattr(tr, "best_fitness", None) is not None
        else None,
        "final_val_metrics": {k: float(v) for k, v in (getattr(tr, "metrics", None) or {}).items()},
        "results_csv": str(save_dir / "results.csv"),
    }
    write_json(mpath, man)
    log(f"학습 종료: best {man['result']['best_pt']} | last {man['result']['last_pt']}")
    log(f"  재개 명령: python -u train.py --resume {w / 'last.pt'}")
    log(f"  기록: {mpath}")


if __name__ == "__main__":
    main()
