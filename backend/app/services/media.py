import shutil
import time
import uuid
from pathlib import Path, PurePath
from typing import BinaryIO

from .moves import Media

# Extension → accepted MIME types. Browsers may send octet-stream for .mov files.
VIDEO_TYPES = {
    ".mp4": {"video/mp4", "application/octet-stream"},
    ".mov": {"video/quicktime", "application/octet-stream"},
}
CANONICAL_TYPE = {".mp4": "video/mp4", ".mov": "video/quicktime"}
# MP4 and QuickTime files start with a box whose type sits at bytes 4-8.
BOX_TYPES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}
CHUNK = 1024 * 1024


class UnsupportedMedia(ValueError):
    pass


class MediaTooLarge(ValueError):
    pass


def clean_filename(filename: str | None) -> str:
    """Keep only the display name; it is never used to build a storage path."""
    name = PurePath((filename or "").replace("\\", "/")).name.strip()
    return name[-255:] or "video"


def video_extension(filename: str, content_type: str | None, head: bytes) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in VIDEO_TYPES:
        raise UnsupportedMedia("mp4 또는 mov 영상만 올릴 수 있습니다.")
    if (content_type or "application/octet-stream") not in VIDEO_TYPES[extension]:
        raise UnsupportedMedia("파일 형식과 확장자가 맞지 않습니다.")
    if head[4:8] not in BOX_TYPES:
        raise UnsupportedMedia("영상 파일로 읽을 수 없습니다.")
    return extension


class MediaStorage:
    def __init__(self, root: Path, max_bytes: int):
        self.root = root
        self.max_bytes = max_bytes

    def path(self, media: Media) -> Path:
        return self.root / media.storage_path

    def save_video(
        self, move_id: str, source: BinaryIO, filename: str | None, content_type: str | None
    ) -> Media:
        """Write the upload under its own ID. A failure leaves no file behind."""
        name = clean_filename(filename)
        extension = video_extension(name, content_type, source.read(8))
        source.seek(0)

        media_id = str(uuid.uuid4())
        relative = f"{move_id}/{media_id}{extension}"
        target = self.root / relative
        partial = target.with_name(f"{media_id}.uploading")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            size = 0
            with partial.open("xb") as out:
                while chunk := source.read(CHUNK):
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise MediaTooLarge(
                            f"영상은 {self.max_bytes // (1024 * 1024)}MB 이하만 올릴 수 있습니다."
                        )
                    out.write(chunk)
            partial.replace(target)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return Media(
            id=media_id, move_id=move_id, filename=name,
            content_type=CANONICAL_TYPE[extension], size_bytes=size,
            storage_path=relative, created_at=int(time.time()),
        )

    def discard(self, media: Media) -> None:
        self.path(media).unlink(missing_ok=True)

    def snapshot_dir(self, move_id: str, analysis_id: str) -> Path:
        """Item photos captured for one analysis, kept next to the move's video."""
        return self.root / move_id / "snapshots" / analysis_id

    def discard_snapshots(self, move_id: str, analysis_id: str | None = None) -> None:
        """Remove one analysis's photos, or every analysis's photos of the move."""
        target = self.root / move_id / "snapshots"
        if analysis_id is not None:
            target = target / analysis_id
        shutil.rmtree(target, ignore_errors=True)

    def snapshot_staging(self, move_id: str, job_id: str) -> Path:
        """A private folder for one result request's photos, published only if it is saved.

        Two requests reporting the same job never share a folder, so the one that loses
        cannot delete the winner's photos.
        """
        return self.root / move_id / "snapshots" / f"{job_id}.{uuid.uuid4().hex}.staging"

    def publish_snapshots(self, move_id: str, job_id: str, staging: Path) -> None:
        """Make a saved result's photos the job's photos."""
        target = self.snapshot_dir(move_id, job_id)
        shutil.rmtree(target, ignore_errors=True)
        if staging.is_dir():
            staging.replace(target)
