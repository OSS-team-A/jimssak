"""경로·해시·CSV·실행 폴더·환경 기록 등 공통 유틸리티."""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# CSV는 Excel(Windows)에서 한글이 깨지지 않도록 BOM 포함 UTF-8로 쓴다. 읽을 때도 utf-8-sig.
CSV_ENCODING = "utf-8-sig"


def log(msg: str) -> None:
    """버퍼링 없이 즉시 출력 (lab-env run 로그용)."""
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fail(msg: str, code: int = 2) -> "None":
    print(f"오류: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def safe_id(text: str, fallback_seed: str | None = None) -> str:
    """폴더명에 쓸 ASCII 식별자. 한글 등으로 비면 해시 기반 이름을 만든다."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    if s:
        return s[:80]
    seed = fallback_seed if fallback_seed is not None else text
    return "video_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        fail(f"설정 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        fail(f"설정 파일 형식이 올바르지 않습니다(YAML 매핑이어야 함): {path}")
    return data


def make_run_dir(base: Path, name: str, overwrite: bool = False) -> Path:
    """실행별 출력 폴더를 만든다. 이미 있으면 덮어쓰지 않고 실패한다."""
    run_dir = base / name
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            fail(
                f"출력 폴더가 이미 존재합니다: {run_dir}\n"
                "  기존 결과를 보존하려면 --run-name을 바꾸고, 정말 덮어쓰려면 --overwrite를 "
                "주세요."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=CSV_ENCODING, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        fail(f"CSV 파일이 없습니다: {path}")
    with open(path, "r", encoding=CSV_ENCODING, newline="") as f:
        rows = list(csv.DictReader(f))
    # 앞뒤 공백 제거, 주석 행(#으로 시작하는 첫 칸) 무시
    out = []
    for r in rows:
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        first = next(iter(r.values()), "")
        if first.startswith("#"):
            continue
        if not any(r.values()):
            continue
        out.append(r)
    return out


def imwrite(path: Path, img, params: list[int] | None = None) -> None:
    """한글 경로에서도 동작하는 이미지 저장 (cv2.imwrite는 Windows 비ASCII 경로에서 실패할 수
    있음).
    """
    import cv2

    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def imread(path: Path):
    import cv2
    import numpy as np

    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def git_info() -> dict:
    """코드 버전 기록. Git 저장소가 아니면 그 사실을 기록한다."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode != 0:
            return {"git": "not_a_git_repo_or_no_commit"}
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {"git_commit": commit.stdout.strip(), "git_dirty": bool(dirty.stdout.strip())}
    except Exception as e:  # git 미설치 등
        return {"git": f"unavailable: {type(e).__name__}"}


def code_fingerprint() -> str:
    """src/ 와 진입점 스크립트의 내용 해시 (Git이 없을 때도 코드 버전을 구분하기 위함)."""
    h = hashlib.sha256()
    files = sorted((PROJECT_ROOT / "src").rglob("*.py")) + sorted(PROJECT_ROOT.glob("*.py"))
    for p in files:
        h.update(p.relative_to(PROJECT_ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def package_versions(names: Iterable[str]) -> dict:
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def environment_info() -> dict:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "torch",
                "torchvision",
                "ultralytics",
                "opencv-python",
                "opencv-python-headless",
                "numpy",
                "lap",
                "scipy",
            ]
        ),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            info["cuda_capability"] = [
                list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
            ]
        info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def resolve_device(device: str) -> str:
    """장치 문자열 검증. GPU를 요청했는데 없으면 CPU로 숨겨 바꾸지 않고 실패한다."""
    import torch

    d = device.strip().lower()
    if d == "cpu":
        return "cpu"
    if d in ("cuda", "gpu"):
        d = "cuda:0"
    if d.isdigit():
        d = f"cuda:{d}"
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            fail(
                f"--device {device} 를 요청했지만 이 환경에서 CUDA GPU가 보이지 않습니다.\n"
                "  컨테이너에 GPU가 연결됐는지(nvidia-smi, scripts/check_env.py) 확인하세요.\n"
                "  CPU로 실행하려면 명시적으로 --device cpu 를 주세요."
            )
        idx = int(d.split(":")[1]) if ":" in d else 0
        if idx >= torch.cuda.device_count():
            fail(
                f"cuda:{idx} 가 없습니다. 보이는 GPU 수: {torch.cuda.device_count()} (컨테이너 안 "
                f"번호는 0부터)"
            )
        return f"cuda:{idx}"
    fail(f"알 수 없는 장치: {device} (예: cpu, cuda:0)")
    return ""


def set_offline_friendly_ultralytics_env() -> None:
    """Ultralytics가 실행 중 패키지를 자동 설치하지 않도록 한다 (의존성은 requirements로만 관리)."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    # 서버 컨테이너에서는 Ultralytics 설정·폰트 캐시를 영속 저장소(/workspace)에 둔다
    if (
        "YOLO_CONFIG_DIR" not in os.environ
        and Path("/workspace").is_dir()
        and os.access("/workspace", os.W_OK)
    ):
        cache = Path("/workspace/.cache/ultralytics")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(cache)
