"""영상 열기·메타데이터 확인.

좌표계 약속: 이 프로젝트의 '원본 좌표계'는 회전 메타데이터를 적용한 표시 방향(display orientation)의
원본 해상도 픽셀 좌표다. 예) 저장 1920x1080 + rotation -90 → 표시 1080x1920, 좌표도 1080x1920 기준.
모델 입력 크기(imgsz)로 축소된 좌표가 아니다. 저장 해상도·회전값은 manifest에 함께 기록한다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .common import fail, log


@dataclass
class VideoInfo:
    path: str
    opened_via: str
    fps: float
    frame_count_reported: int
    decoded_width: int
    decoded_height: int
    orientation_meta_deg: float | None
    orientation_auto: bool | None
    ffprobe: dict | None = None
    notes: list[str] = field(default_factory=list)


def _ffprobe(path: Path) -> dict | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:"
                "stream_side_data=rotation:stream_tags=rotate:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            timeout=60,
        )
        if out.returncode != 0:
            return {"error": out.stderr.decode("utf-8", "replace")[:500]}
        return json.loads(out.stdout.decode("utf-8", "replace"))
    except Exception as e:
        return {"error": repr(e)}


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo, Path | None]:
    """영상을 연다. Windows에서 비ASCII(한글) 경로를 OpenCV가 못 열면 임시 하드링크/복사본을
    사용한다.

    반환: (capture, info, 정리해야 할 임시 파일 경로 또는 None)
    """
    path = Path(path)
    if not path.is_file():
        fail(f"영상 파일이 없습니다: {path}")
    temp_path: Path | None = None
    cap = cv2.VideoCapture(str(path))
    opened_via = "direct"
    if not cap.isOpened() and os.name == "nt" and not str(path).isascii():
        tmpdir = Path(tempfile.mkdtemp(prefix="jimssak_vid_"))
        temp_path = tmpdir / ("input" + path.suffix.lower())
        try:
            os.link(path, temp_path)
            opened_via = "windows_ascii_hardlink"
        except OSError:
            shutil.copy2(path, temp_path)
            opened_via = "windows_ascii_copy"
        cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        fail(
            f"영상을 열 수 없습니다: {path}\n"
            "  코덱(HEVC 등) 디코딩을 OpenCV가 지원하는지 scripts/check_env.py --video 로 "
            "확인하세요."
        )
    # 회전 메타데이터 자동 적용 (OpenCV 4.5+ FFmpeg 백엔드). 명시적으로 켠다.
    orientation_auto = None
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        orientation_auto = bool(cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))
    orient_meta = None
    if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        orient_meta = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))

    ok, frame = cap.read()
    if not ok or frame is None:
        fail(f"첫 프레임을 디코딩하지 못했습니다: {path}")
    h, w = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # 일부 백엔드는 POS_FRAMES 되감기가 부정확하므로 다시 연다
    cap.release()
    cap = cv2.VideoCapture(str(temp_path or path))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    info = VideoInfo(
        path=str(path),
        opened_via=opened_via,
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        frame_count_reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        decoded_width=int(w),
        decoded_height=int(h),
        orientation_meta_deg=orient_meta,
        orientation_auto=orientation_auto,
        ffprobe=_ffprobe(path),
    )
    if info.fps <= 0:
        info.notes.append("FPS를 읽지 못함: 타임스탬프는 CAP_PROP_POS_MSEC만 사용")
    # ffprobe가 있으면 회전 적용 여부를 교차 확인
    try:
        st = (info.ffprobe or {}).get("streams", [{}])[0]
        rot = None
        for sd in st.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = float(sd["rotation"])
        if rot is None and "rotate" in (st.get("tags") or {}):
            rot = -float(st["tags"]["rotate"])
        if rot is not None and int(round(abs(rot))) % 180 == 90:
            expect = (int(st["height"]), int(st["width"]))
            if (w, h) != expect:
                info.notes.append(
                    f"경고: 회전 메타데이터 {rot}°인데 디코딩 프레임이 {w}x{h} "
                    f"(기대 {expect[0]}x{expect[1]}). "
                    "OpenCV가 회전을 적용하지 않았을 수 있음"
                )
    except Exception:
        pass
    for n in info.notes:
        log(n)
    return cap, info, temp_path


def cleanup_temp(temp_path: Path | None) -> None:
    if temp_path is None:
        return
    try:
        temp_path.unlink(missing_ok=True)
        temp_path.parent.rmdir()
    except OSError:
        pass
