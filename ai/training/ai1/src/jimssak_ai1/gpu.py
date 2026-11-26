"""GPU 자원 확인·장치 선택·사용량 모니터링.

- 컨테이너에 '보이는' GPU만 사용한다(CUDA_VISIBLE_DEVICES / lab-env가 정한 범위). 물리 GPU를 임의로
  더 잡지 않는다.
- GPU가 필요한데 없으면 실패한다. CPU 실행은 명시적인 allow_cpu일 때만.
- 사용량은 nvidia-smi를 주기적으로 조회해 CSV로 남긴다(없으면 '미측정').
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from pathlib import Path


def gpu_inventory() -> dict:
    info: dict = {"cuda_available": False, "devices": []}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                free, total = torch.cuda.mem_get_info(i)
                info["devices"].append(
                    {
                        "index": i,
                        "name": p.name,
                        "capability": f"sm_{p.major}{p.minor}",
                        "total_GiB": round(total / 2**30, 2),
                        "free_GiB": round(free / 2**30, 2),
                        "multiprocessors": p.multi_processor_count,
                    }
                )
    except Exception as e:  # torch import 실패 등
        info["error"] = repr(e)
    info["nvidia_smi"] = bool(shutil.which("nvidia-smi"))
    return info


def choose_devices(inv: dict, require_gpu: bool, allow_cpu: bool, use_all: bool) -> dict:
    """반환: {'mode': 'cuda'|'cpu', 'yolo_device': '0' | '0,1' | 'cpu', 'torch_device':
    'cuda:0'|'cpu', 'n_gpus': int}
    """
    n = len(inv.get("devices", []))
    if n == 0:
        if require_gpu and not allow_cpu:
            raise SystemExit(
                "오류: 이 환경에서 CUDA GPU가 보이지 않습니다. GPU 학습을 CPU로 바꾸지 않고 "
                "중단합니다.\n"
                "  lab-env 환경에 GPU가 연결됐는지 확인하세요. (점검 목적으로만 --allow-cpu 사용 "
                "가능)"
            )
        return {"mode": "cpu", "yolo_device": "cpu", "torch_device": "cpu", "n_gpus": 0}
    k = n if use_all else 1
    return {
        "mode": "cuda",
        "yolo_device": ",".join(str(i) for i in range(k)),
        "torch_device": "cuda:0",
        "n_gpus": k,
    }


class GpuMonitor:
    """nvidia-smi를 주기적으로 조회해 CSV에 기록 (메모리·사용률). 학습 중 DDP 자식 프로세스까지
    포함한 장치 단위 값.
    """

    FIELDS = "timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw"

    def __init__(self, out_csv: Path, interval: float = 5.0):
        self.out_csv = Path(out_csv)
        self.interval = interval
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self.available = bool(shutil.which("nvidia-smi"))

    def _loop(self):
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        new = not self.out_csv.exists()
        with open(self.out_csv, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["stage"] + self.FIELDS.split(","))
            while not self._stop.is_set():
                try:
                    out = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--query-gpu={self.FIELDS}",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    for line in out.stdout.strip().splitlines():
                        w.writerow([self.stage] + [x.strip() for x in line.split(",")])
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.interval)

    stage = ""

    def start(self, stage: str = ""):
        self.stage = stage
        if not self.available or self._th is not None:
            return
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=15)


def summarize_gpu_csv(path: Path) -> dict:
    """스테이지별 최대 메모리(MiB)·평균/최대 사용률(%)·측정 시간."""
    path = Path(path)
    if not path.is_file():
        return {"status": "미측정", "reason": "nvidia-smi 기록 없음"}
    by: dict = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                s = by.setdefault(r["stage"], {"mem_max_MiB": 0.0, "util": [], "n": 0})
                s["mem_max_MiB"] = max(s["mem_max_MiB"], float(r["memory.used"]))
                s["util"].append(float(r["utilization.gpu"]))
                s["n"] += 1
            except (ValueError, KeyError):
                continue
    out = {}
    for k, s in by.items():
        u = s["util"]
        out[k] = {
            "mem_max_MiB": s["mem_max_MiB"],
            "util_mean_pct": round(sum(u) / len(u), 1) if u else None,
            "util_max_pct": max(u) if u else None,
            "samples": s["n"],
        }
    return out


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0
