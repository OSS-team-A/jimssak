"""Internal API for the AI analyzer: take a job, fetch its video, report back.

Protected by a shared key (X-Worker-Key). Users never call these routes.
"""

import json
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from ..schemas.analysis import AnalysisProgress, AnalysisResultV2, AnalysisSection, WorkerJob
from ..services.analysis import AnalysisJob, AnalysisStore, StaleAnalysis

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
        "가장 오래 기다린 작업 하나를 `running`으로 바꾸고 돌려준다. 결과의 `analysis_id`와 "
        "`video.video_id`에는 받은 값을 그대로 넣는다. 대기 작업이 없으면 204."
    ),
    response_model=WorkerJob,
    responses={204: {"description": "대기 중인 작업 없음"}},
)
def claim(request: Request, analyses: Analyses) -> WorkerJob | Response:
    job = analyses.claim_next()
    if job is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    media = request.app.state.move_store.get_video(job.move_id)
    return WorkerJob(
        analysis_id=job.analysis_id, move_id=job.move_id, video_id=job.media_id,
        video_url=f"/api/v1/internal/analyses/{job.analysis_id}/video",
        content_type=media.content_type if media else "video/mp4",
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
        "`status: failed`와 `error`(code, message, retryable). 원본 그대로 저장한다."
    ),
    responses={**STALE, 422: {"description": "v2.0 형식이 아니거나 ID가 맞지 않음"}},
)
def result(analysis_id: str, body: AnalysisResultV2, analyses: Analyses) -> AnalysisSection:
    job = analyses.get(analysis_id)
    if body.analysis_id != analysis_id or (job and body.video.video_id != job.media_id):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "analysis_id와 video.video_id는 작업을 가져갈 때 받은 값을 그대로 보내주세요.",
        )
    raw = body.model_dump(mode="json")
    try:
        if body.status == "completed":
            done = analyses.submit(analysis_id, "completed", json.dumps(raw, ensure_ascii=False),
                                   None)
        else:
            error = json.dumps(raw["error"], ensure_ascii=False)
            done = analyses.submit(analysis_id, "failed", None, error)
    except StaleAnalysis as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "현재 진행 중인 작업이 아닙니다.") from exc
    return _section(done)
