#!/usr/bin/env python
"""서버 담당자에게 넘길 프로젝트 압축(dist/jimssak_project.tar.gz) 생성.

포함: 코드·설정·스크립트·문서·템플릿·입력 영상(data/input_videos).
제외: 가상환경, 다운로드 캐시·데이터셋, 사전학습 가중치(서버에서 자동 다운로드), 실행 결과, 로그,
개인 검수표 사본.
압축 안 최상위 폴더 이름은 jimssak/ (서버에서 ~/workspace 에 풀면 ~/workspace/jimssak).
"""

import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {
    ".venv",
    ".cache",
    "outputs",
    "logs",
    "runs",
    "evaluations",
    "weights",
    "dist",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".claude",
    "results_from_server",
    ".ruff_cache",
}
EXCLUDE_SUB = {("data", "cache"), ("data", "datasets"), ("models", "pretrained")}
EXCLUDE_FILES = {"logs_smoke_local.txt"}
EXCLUDE_PREFIX = ("annotations/match_review_B0_local_cpu_",)


def keep(rel: Path) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_DIRS for p in parts):
        return False
    if len(parts) >= 2 and (parts[0], parts[1]) in EXCLUDE_SUB:
        return False
    if rel.name in EXCLUDE_FILES or rel.as_posix().startswith(EXCLUDE_PREFIX):
        return False
    if rel.suffix in {".pt", ".pth", ".npz", ".pyc", ".tgz", ".part"} or rel.name.endswith(
        ".tar.gz"
    ):
        return False
    return True


def main():
    out = ROOT / "dist" / "jimssak_project.tar.gz"
    out.parent.mkdir(exist_ok=True)
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and keep(p.relative_to(ROOT)))
    with tarfile.open(out, "w:gz") as t:
        for p in files:
            rel = p.relative_to(ROOT)
            ti = t.gettarinfo(str(p), arcname=f"jimssak/{rel.as_posix()}")
            if rel.suffix == ".sh":
                ti.mode = 0o755
            with open(p, "rb") as f:
                t.addfile(ti, f)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}  {out.stat().st_size / 2**20:.1f} MiB  파일 {len(files)}개  sha256 {h}")
    vid = [f for f in files if f.suffix.lower() == ".mp4"]
    print("포함된 영상:", [str(v.relative_to(ROOT)) for v in vid])


if __name__ == "__main__":
    main()
