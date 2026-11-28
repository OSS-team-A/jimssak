"""Item photos cut from the room video at the times the AI analyzer reports.

The analyzer sends only `time_ms` (and optionally `bbox`) for each snapshot; the backend
captures that frame from the stored video with ffmpeg, so the analyzer never uploads images.
ffmpeg turns phone videos upright before filters run, so `bbox` is in upright pixels.
"""

import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

# Photos are shown on a phone; this keeps each JPG small.
MAX_WIDTH = 640
TIMEOUT_SECONDS = 15


class SnapshotFailed(Exception):
    """The frame could not be captured: a time past the end, or a box outside the frame."""


class SnapshotCapture:
    def __init__(self, ffmpeg: str | None = None):
        self._ffmpeg = ffmpeg

    @property
    def ffmpeg(self) -> str:
        # A system ffmpeg if there is one, otherwise the binary bundled with imageio-ffmpeg.
        if self._ffmpeg is None:
            self._ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        return self._ffmpeg

    def capture(
        self, video: Path, time_ms: int, target: Path, bbox: list[int] | None = None,
    ) -> None:
        filters = []
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            filters.append(f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}")
        filters.append(f"scale='min({MAX_WIDTH},iw)':-2")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            done = subprocess.run(
                [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-ss", f"{time_ms / 1000:.3f}", "-i", str(video),
                 "-frames:v", "1", "-vf", ",".join(filters), "-q:v", "4", str(target)],
                capture_output=True, timeout=TIMEOUT_SECONDS, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SnapshotFailed(f"ffmpeg did not run: {exc}") from exc
        # ffmpeg writes nothing for a time past the end or a box outside the frame.
        if done.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            raise SnapshotFailed(done.stderr.decode("utf-8", "replace")[-300:])
