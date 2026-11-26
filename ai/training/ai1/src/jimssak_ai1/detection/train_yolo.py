"""Ultralytics YOLO 파인튜닝 래퍼: 재개·OOM 시 배치 축소 재시도·다중 GPU(DDP)·학습 기록.

- 사전학습 가중치에서 시작하는 실제 파인튜닝이다(model.train). 추론만 하는 것을 학습으로 기록하지
  않는다.
- Loss: Ultralytics 분할/탐지 기본(box: CIoU+DFL, cls: BCE, seg: mask BCE). Optimizer: 'auto'면
  Ultralytics가 SGD/AdamW 선택.
- LR 스케줄: cos_lr. Early stopping: patience. Augmentation: Ultralytics 기본(mosaic, HSV, flip,
  scale 등).
- 재개: 같은 이름의 실행에 last.pt가 있고 끝나지 않았으면 resume=True로 이어서 학습.
- OOM: 'out of memory' 오류면 batch를 절반으로 줄여 새 이름으로 처음부터 다시(재시도 횟수 제한).
  실패를 성공으로 바꾸지 않는다.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from ..common import log, read_json, sha256_file, write_json


def _finished(run_dir: Path, epochs: int) -> bool:
    return (run_dir / "jimssak_train_done.json").is_file()


def _results_rows(run_dir: Path) -> list[dict]:
    p = run_dir / "results.csv"
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def train_yolo(
    *,
    weights: str,
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    common: dict,
) -> dict:
    from ultralytics import YOLO

    project = Path(project).resolve()
    attempts = []
    cur_batch, cur_name = batch, name
    for attempt in range(int(common.get("oom_retry", 2)) + 1):
        run_dir = project / cur_name
        if _finished(run_dir, epochs):
            log(f"  [{cur_name}] 이미 학습 완료 → 건너뜀")
            return read_json(run_dir / "jimssak_train_done.json")
        last = run_dir / "weights" / "last.pt"
        t0 = time.time()
        try:
            if last.is_file():
                log(f"  [{cur_name}] last.pt에서 재개")
                model = YOLO(str(last))
                model.train(resume=True, device=device)
            else:
                log(
                    f"  [{cur_name}] 학습 시작: {weights} epochs={epochs} batch={cur_batch} "
                    f"imgsz={imgsz} device={device}"
                )
                model = YOLO(weights)
                model.train(
                    data=str(data_yaml),
                    epochs=epochs,
                    batch=cur_batch,
                    imgsz=imgsz,
                    device=device,
                    project=str(project),
                    name=cur_name,
                    exist_ok=True,
                    patience=common.get("patience", 15),
                    optimizer=common.get("optimizer", "auto"),
                    cos_lr=common.get("cos_lr", True),
                    seed=common.get("seed", 0),
                    deterministic=True,
                    workers=common.get("workers", 4),
                    save_period=common.get("save_period", -1),
                    plots=True,
                    val=True,
                    verbose=True,
                )
            secs = time.time() - t0
            rows = _results_rows(run_dir)
            best = run_dir / "weights" / "best.pt"
            n_params = None
            try:
                n_params = sum(p.numel() for p in YOLO(str(best)).model.parameters())
            except Exception:
                pass
            opt = None
            try:
                import yaml

                opt = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")).get(
                    "optimizer"
                )
            except Exception:
                pass
            ep_times = []
            prev = 0.0
            for r in rows:
                try:
                    t = float(r.get("time", "nan"))
                    ep_times.append(round(t - prev, 2))
                    prev = t
                except ValueError:
                    pass
            done = {
                "status": "완료",
                "weights_init": weights,
                "run_dir": str(run_dir),
                "best_pt": str(best),
                "last_pt": str(last),
                "best_sha256": sha256_file(best) if best.is_file() else None,
                "best_bytes": best.stat().st_size if best.is_file() else None,
                "params": n_params,
                "epochs_target": epochs,
                "epochs_completed": len(rows),
                "batch": cur_batch,
                "imgsz": imgsz,
                "device": device,
                "train_wall_sec_this_call": round(secs, 1),
                "train_time_from_results_sec": prev if rows else None,
                "epoch_times_sec": ep_times,
                "optimizer_arg": opt,
                "attempts": attempts,
                "results_csv": str(run_dir / "results.csv"),
            }
            if not best.is_file():
                raise RuntimeError("학습이 끝났지만 best.pt가 없습니다")
            write_json(run_dir / "jimssak_train_done.json", done)
            return done
        except Exception as e:
            msg = repr(e)
            oom = "out of memory" in msg.lower() or "outofmemory" in msg.lower()
            ddp = "," in str(device)
            attempts.append({"name": cur_name, "batch": cur_batch, "error": msg[:400], "oom": oom})
            if (oom or ddp) and cur_batch > 2 and attempt < int(common.get("oom_retry", 2)):
                cur_batch = max(2, cur_batch // 2)
                cur_name = f"{name}_b{cur_batch}"
                log(
                    f"  [{name}] 학습 실패({'OOM' if oom else 'DDP 오류'}) → batch {cur_batch}로 "
                    f"새로 학습: {cur_name}"
                )
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise RuntimeError(f"YOLO 학습 실패 ({name}): {msg}") from e
    raise RuntimeError(f"YOLO 학습 실패 ({name}): 재시도 소진 {attempts}")
