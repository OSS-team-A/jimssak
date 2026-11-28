"""Internal API for the AI analyzer: take a job, fetch its video, report back.

Protected by a shared key (X-Worker-Key). Users never call these routes.
"""

import json
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from ..schemas.analysis import AnalysisProgress, AnalysisResultV2, AnalysisSection, WorkerJob
from ..schemas.comparison import ComparisonJob, ComparisonJobItem, ComparisonResult
from ..services.analysis import AnalysisJob, AnalysisStore, StaleAnalysis, snapshot_url
from ..services.comparison import ComparisonStore, ComparisonTask
from ..services.snapshots import SnapshotFailed

router = APIRouter(prefix="/api/v1/internal/analyses", tags=["internal: AI 분석기"])


def require_worker(
    request: Request,
    x_worker_key: Annotated[str | None, Header(description="AI 분석기 전용 키")] = None,
) -> None:
    expected = request.app.state.analysis_worker_key
    if expected is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "분석기 연동이 꺼져 있습니다.")
    if not x_worker_key or not secrets.compare_digest(x_worker_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "분석기 키가 올바르지 않습니다.")


def get_analyses(request: Request) -> AnalysisStore:
    return request.app.state.analysis_store


Analyses = Annotated[AnalysisStore, Depends(get_analyses)]
Worker = Depends(require_worker)
STALE = {409: {"description": "현재 작업이 아님(영상이 교체됨) 또는 이미 끝난 작업"}}


def _section(job: AnalysisJob) -> AnalysisSection:
    return AnalysisSection(status=job.status, analysis_id=job.analysis_id,
                           progress=job.progress, demo=job.demo)


@router.post(
    "/claim",
    dependencies=[Worker],
    summary="대기 중인 분석 작업 가져가기",
    description=(
        "가장 오래 기다린 작업 하나를 `running`으로 바꾸고 돌려준다. 대기 작업이 없으면 204.\n\n"
        "- `kind: analysis`(첫 분석): 결과의 `analysis_id`와 `video.video_id`에 받은 값을 "
        "그대로 넣어 `/internal/analyses/{analysis_id}/result`로 보낸다.\n"
        "- `kind: comparison`(이사 후 비교): 이사 전·후 영상과 확정 목록(`items`)을 받는다. "
        "결과는 `/internal/comparisons/{comparison_id}/result`로 보낸다.\n\n"
        "첫 분석 작업을 먼저 내준다."
    ),
    response_model=WorkerJob | ComparisonJob,
    responses={204: {"description": "대기 중인 작업 없음"}},
)
def claim(request: Request, analyses: Analyses) -> WorkerJob | ComparisonJob | Response:
    job = analyses.claim_next()
    if job is not None:
        media = request.app.state.move_store.get_video(job.move_id)
        return WorkerJob(
            analysis_id=job.analysis_id, move_id=job.move_id, video_id=job.media_id,
            video_url=f"/api/v1/internal/analyses/{job.analysis_id}/video",
            content_type=media.content_type if media else "video/mp4",
        )
    task = request.app.state.comparison_store.claim_next()
    if task is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    base = f"/api/v1/internal/comparisons/{task.comparison_id}"
    return ComparisonJob(
        comparison_id=task.comparison_id, move_id=task.move_id,
        before_video_id=task.before_media_id, after_video_id=task.after_media_id,
        before_video_url=f"{base}/before-video", after_video_url=f"{base}/after-video",
        items=[ComparisonJobItem(
            item_id=item["item_id"], label=item.get("label"), label_ko=item["label_ko"],
            quantity=item["quantity"], dimensions_cm=item.get("dimensions_cm") or {},
        ) for item in task.inventory],
    )


@router.get(
    "/{analysis_id}/video",
    dependencies=[Worker],
    summary="분석할 영상 내려받기",
    responses={404: {"description": "작업이나 영상이 없음"}},
)
def video(analysis_id: str, request: Request, analyses: Analyses) -> FileResponse:
    job = analyses.get(analysis_id)
    media = request.app.state.move_store.get_video(job.move_id) if job else None
    if job is None or media is None or media.id != job.media_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "영상을 찾을 수 없습니다.")
    path = request.app.state.media_storage.path(media)
    return FileResponse(path, media_type=media.content_type, filename=media.filename)


@router.post(
    "/{analysis_id}/progress",
    dependencies=[Worker],
    summary="진행률 보고",
    responses=STALE,
)
def progress(analysis_id: str, body: AnalysisProgress, analyses: Analyses) -> AnalysisSection:
    try:
        return _section(analyses.report_progress(analysis_id, body.progress))
    except StaleAnalysis as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 진행 중인 작업이 아닙니다.") from exc


@router.post(
    "/{analysis_id}/result",
    dependencies=[Worker],
    summary="분석 결과 또는 실패 보내기",
    description=(
        "분석 결과 v2.0을 보낸다. 완료면 `status: completed`와 `items`, 실패면 "
        "`status: failed`와 `error`(code, message, retryable). 원본 그대로 저장한다.\n\n"
        "물품 사진은 `snapshots`에 `time_ms`(영상 속 시각)만 보내면 백엔드가 그 시각의 화면을 "
        "캡처해 `image_url`을 채운다. `bbox: [x1, y1, x2, y2]`를 함께 보내면 그 영역만 "
        "잘라낸다. 물품당 최대 3장, 모두 합쳐 200장. 영상 길이를 넘는 시각이나 화면 밖 "
        "영역이면 422."
    ),
    responses={**STALE, 422: {"description": "v2.0 형식이 아니거나 ID가 맞지 않음"}},
)
def result(
    analysis_id: str, body: AnalysisResultV2, analyses: Analyses, request: Request,
) -> AnalysisSection:
    job = analyses.get(analysis_id)
    if body.analysis_id != analysis_id or (job and body.video.video_id != job.media_id):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "analysis_id와 video.video_id는 작업을 가져갈 때 받은 값을 그대로 보내주세요.",
        )
    raw = body.model_dump(mode="json")
    storage = request.app.state.media_storage
    captured = False
    try:
        if body.status == "completed":
            job = analyses.active(analysis_id)
            _capture_snapshots(request, job, raw)
            captured = True
            done = analyses.submit(analysis_id, "completed", json.dumps(raw, ensure_ascii=False),
                                   None)
        else:
            error = json.dumps(raw["error"], ensure_ascii=False)
            done = analyses.submit(analysis_id, "failed", None, error)
    except StaleAnalysis as exc:
        if captured:
            storage.discard_snapshots(job.move_id, analysis_id)
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 진행 중인 작업이 아닙니다.") from exc
    return _section(done)


def _capture_snapshots(request: Request, job: AnalysisJob, raw: dict) -> None:
    """Cut each requested photo from the video and point `image_url` at it."""
    media = request.app.state.move_store.get_video(job.move_id)
    if media is None or media.id != job.media_id:
        raise StaleAnalysis(job.analysis_id)
    video = request.app.state.media_storage.path(media)
    capture_photos(request, job.move_id, job.analysis_id, [
        (video, item["item_id"], snapshot)
        for item in raw["items"] for snapshot in item["snapshots"]
    ])


def capture_photos(
    request: Request, move_id: str, job_id: str, photos: list[tuple[Path, str, dict]],
) -> None:
    """Capture (video, item_id, snapshot) photos into the job's folder and fill `image_url`.

    Any failure removes the photos already taken, so a refused result leaves none behind.
    """
    storage = request.app.state.media_storage
    capture = request.app.state.snapshot_capture
    folder = storage.snapshot_dir(move_id, job_id)
    storage.discard_snapshots(move_id, job_id)
    try:
        for number, (video, item_id, snapshot) in enumerate(photos, start=1):
            try:
                capture.capture(video, snapshot["time_ms"], folder / f"{number}.jpg",
                                snapshot.get("bbox"))
            except SnapshotFailed as exc:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"{item_id}의 {snapshot['time_ms']}ms 사진을 만들 수 없습니다. "
                    "time_ms가 영상 길이 안인지, bbox가 화면 안인지 확인해주세요.",
                ) from exc
            snapshot["image_url"] = snapshot_url(move_id, job_id, number)
    except BaseException:
        storage.discard_snapshots(move_id, job_id)
        raise


# --- after-move comparisons (SC-14~16) ---

comparison_router = APIRouter(prefix="/api/v1/internal/comparisons",
                              tags=["internal: AI 분석기"])


def get_comparisons(request: Request) -> ComparisonStore:
    return request.app.state.comparison_store


Comparisons = Annotated[ComparisonStore, Depends(get_comparisons)]


def _comparison_section(task: ComparisonTask) -> AnalysisSection:
    return AnalysisSection(status=task.status, analysis_id=task.comparison_id,
                           progress=task.progress, demo=task.demo)


def _comparison_video(request: Request, comparisons: ComparisonStore, comparison_id: str,
                      after: bool) -> FileResponse:
    task = comparisons.get(comparison_id)
    moves = request.app.state.move_store
    media = None
    if task is not None:
        media = moves.get_after_video(task.move_id) if after else moves.get_video(task.move_id)
    expected = (task.after_media_id if after else task.before_media_id) if task else None
    if media is None or media.id != expected:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "영상을 찾을 수 없습니다.")
    path = request.app.state.media_storage.path(media)
    return FileResponse(path, media_type=media.content_type, filename=media.filename)


@comparison_router.get(
    "/{comparison_id}/before-video", dependencies=[Worker], summary="이사 전 영상 내려받기",
    responses={404: {"description": "작업이나 영상이 없음"}},
)
def before_video(comparison_id: str, request: Request,
                 comparisons: Comparisons) -> FileResponse:
    return _comparison_video(request, comparisons, comparison_id, after=False)


@comparison_router.get(
    "/{comparison_id}/after-video", dependencies=[Worker], summary="이사 후 영상 내려받기",
    responses={404: {"description": "작업이나 영상이 없음"}},
)
def after_video(comparison_id: str, request: Request,
                comparisons: Comparisons) -> FileResponse:
    return _comparison_video(request, comparisons, comparison_id, after=True)


@comparison_router.post(
    "/{comparison_id}/progress", dependencies=[Worker], summary="비교 진행률 보고",
    responses=STALE,
)
def comparison_progress(comparison_id: str, body: AnalysisProgress,
                        comparisons: Comparisons) -> AnalysisSection:
    try:
        return _comparison_section(comparisons.report_progress(comparison_id, body.progress))
    except StaleAnalysis as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 진행 중인 작업이 아닙니다.") from exc


@comparison_router.post(
    "/{comparison_id}/result",
    dependencies=[Worker],
    summary="비교 결과 또는 실패 보내기",
    description=(
        "확정 목록(`items`)의 물품마다 정확히 한 번씩 `result`(ok / changed / missing)와 "
        "`reasons`를 보낸다. 사진은 `before_snapshots`(이사 전 영상)·`after_snapshots`"
        "(이사 후 영상)에 `time_ms`(선택 `bbox`)만 넣으면 백엔드가 캡처한다. 각 3장까지, "
        "모두 합쳐 200장. 실패면 `status: failed`와 `error`."
    ),
    responses={**STALE, 422: {"description": "형식, ID, 물품 목록이 맞지 않거나 캡처할 수 없음"}},
)
def comparison_result(
    comparison_id: str, body: ComparisonResult, comparisons: Comparisons, request: Request,
) -> AnalysisSection:
    task = comparisons.get(comparison_id)
    if body.comparison_id != comparison_id or (
        task and body.video.video_id != task.after_media_id
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "comparison_id와 video.video_id(이사 후 영상)는 작업에서 받은 값을 그대로 보내주세요.",
        )
    raw = body.model_dump(mode="json")
    storage = request.app.state.media_storage
    captured = False
    try:
        if body.status == "completed":
            task = comparisons.active(comparison_id)
            _check_items(task, raw["items"])
            _capture_comparison(request, task, raw)
            captured = True
            done = comparisons.submit(comparison_id, "completed",
                                      json.dumps(raw, ensure_ascii=False), None)
        else:
            error = json.dumps(raw["error"], ensure_ascii=False)
            done = comparisons.submit(comparison_id, "failed", None, error)
    except StaleAnalysis as exc:
        if captured:
            storage.discard_snapshots(task.move_id, comparison_id)
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 진행 중인 작업이 아닙니다.") from exc
    return _comparison_section(done)


def _check_items(task: ComparisonTask, items: list[dict]) -> None:
    """Every item of the compared list exactly once, and nothing else."""
    expected = [item["item_id"] for item in task.inventory]
    sent = [item["item_id"] for item in items]
    if sorted(sent) != sorted(expected):
        missing = sorted(set(expected) - set(sent))
        extra = sorted(set(sent) - set(expected))
        repeated = sorted({i for i in sent if sent.count(i) > 1})
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "확정 목록의 물품마다 정확히 한 번씩 보내주세요. "
            f"빠짐: {missing}, 목록에 없음: {extra}, 중복: {repeated}",
        )


def _capture_comparison(request: Request, task: ComparisonTask, raw: dict) -> None:
    moves = request.app.state.move_store
    before, after = moves.get_video(task.move_id), moves.get_after_video(task.move_id)
    if (before is None or before.id != task.before_media_id
            or after is None or after.id != task.after_media_id):
        raise StaleAnalysis(task.comparison_id)
    storage = request.app.state.media_storage
    before_path, after_path = storage.path(before), storage.path(after)
    capture_photos(request, task.move_id, task.comparison_id, [
        *((before_path, item["item_id"], shot)
          for item in raw["items"] for shot in item["before_snapshots"]),
        *((after_path, item["item_id"], shot)
          for item in raw["items"] for shot in item["after_snapshots"]),
    ])
