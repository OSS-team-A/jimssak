"""다운로드 유틸: 재시도·이어받기·크기 검증·병렬 이미지 받기. 받은 바이트 수를 기록한다."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..common import log

UA = "jimssak-research-downloader/1.0"


class DownloadError(RuntimeError):
    pass


def _head_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def download_file(
    url: str,
    dst: Path,
    expected_bytes: int | None = None,
    retries: int = 5,
    chunk: int = 1 << 20,
    quiet: bool = False,
) -> dict:
    """파일 하나를 받는다. 이미 같은 크기로 있으면 건너뛴다. 중간에 끊기면 Range로 이어받는다."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = expected_bytes or _head_size(url)
    if dst.is_file() and size and dst.stat().st_size == size:
        return {"url": url, "path": str(dst), "bytes": size, "downloaded_bytes": 0, "skipped": True}
    part = dst.with_suffix(dst.suffix + ".part")
    got = 0
    last_err = None
    t0 = time.time()
    for attempt in range(1, retries + 1):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": UA}
            if start:
                headers["Range"] = f"bytes={start}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if (start and r.status == 206) else "wb"
                if mode == "wb":
                    start = 0
                with open(part, mode) as f:
                    last_log = time.time()
                    while True:
                        b = r.read(chunk)
                        if not b:
                            break
                        f.write(b)
                        got += len(b)
                        if not quiet and size and time.time() - last_log > 30:
                            log(
                                f"    {dst.name}: {(start + got) / 2**20:.0f}/{size / 2**20:.0f} "
                                f"MiB"
                            )
                            last_log = time.time()
            if size and part.stat().st_size != size:
                raise DownloadError(f"크기 불일치 {part.stat().st_size} != {size}")
            os.replace(part, dst)
            return {
                "url": url,
                "path": str(dst),
                "bytes": dst.stat().st_size,
                "downloaded_bytes": got,
                "skipped": False,
                "seconds": round(time.time() - t0, 1),
            }
        except Exception as e:  # 네트워크 오류 → 재시도
            last_err = e
            if not quiet:
                log(f"    다운로드 재시도 {attempt}/{retries}: {url} ({e!r})")
            time.sleep(min(30, 2**attempt))
    raise DownloadError(f"다운로드 실패: {url} → {last_err!r}")


def download_many(items: list[tuple[str, Path]], workers: int = 16, desc: str = "") -> dict:
    """작은 파일(이미지) 여러 개 병렬 다운로드. 실패 목록을 돌려준다(실패를 성공으로 치지 않음)."""
    ok, failed, nbytes, skipped = 0, [], 0, 0
    t0 = time.time()
    todo = [
        (u, Path(p)) for u, p in items if not (Path(p).is_file() and Path(p).stat().st_size > 0)
    ]
    skipped = len(items) - len(todo)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(download_file, u, p, None, 4, 1 << 16, True): (u, p) for u, p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            u, p = futs[f]
            try:
                r = f.result()
                ok += 1
                nbytes += r.get("downloaded_bytes", 0)
            except Exception as e:
                failed.append({"url": u, "error": repr(e)[:200]})
            if i % 500 == 0:
                log(f"    {desc} {i}/{len(todo)} (실패 {len(failed)})")
    return {
        "requested": len(items),
        "already_present": skipped,
        "downloaded": ok,
        "failed": len(failed),
        "failed_examples": failed[:20],
        "downloaded_bytes": nbytes,
        "seconds": round(time.time() - t0, 1),
    }


def append_download_log(log_path: Path, record: dict) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
