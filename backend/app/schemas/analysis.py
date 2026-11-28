"""Video analysis result format v2.0, as agreed with the AI side.

The backend issues `analysis_id`, sends the upload's `media_id` as `video.video_id`,
and expects both back unchanged. Extra fields from the AI are kept, not rejected.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AnalysisError",
    "AnalysisItem",
    "AnalysisOut",
    "AnalysisProgress",
    "AnalysisResultV2",
    "AnalysisSection",
    "WorkerJob",
]

AnalysisStatus = Literal["queued", "running", "completed", "failed"]
MAX_INVENTORY_ITEMS = 200
MAX_ITEM_NAME_LENGTH = 50
MAX_ITEM_NOTES_LENGTH = 500
MAX_SNAPSHOTS_PER_ITEM = 3
MAX_SNAPSHOTS = 200
DimensionsStatus = Literal["estimated", "partial", "unavailable"]


class V2(BaseModel):
    model_config = ConfigDict(extra="allow")


class Dimensions(V2):
    width: float | None = Field(default=None, ge=0, description="가로(cm). 못 구하면 null")
    depth: float | None = Field(default=None, ge=0, description="깊이(cm). 못 구하면 null")
    height: float | None = Field(default=None, ge=0, description="높이(cm). 못 구하면 null")


class Snapshot(V2):
    time_ms: int = Field(ge=0, description="영상에서 이 사진의 시각(ms)")
    image_url: str | None = Field(default=None, description="물품 사진 주소")


class AnalysisItem(V2):
    """물품 하나(실제 물건 하나 = 한 줄)."""

    item_id: str = Field(min_length=1, description="분석 결과 안에서 고정된 물품 ID")
    source: Literal["ai", "demo", "user"] = Field(
        description="ai: AI 분석, demo: 데모 분석기(가짜), user: 사용자가 추가"
    )
    label: str = Field(min_length=1, description="물품 종류 코드. 예: desk, wardrobe")
    label_ko: str = Field(min_length=1, description="화면 표시 이름. 예: 책상")
    quantity: int = Field(ge=1, description="수량")
    confidence: float = Field(ge=0, le=1, description="인식 신뢰도(0~1)")
    dimensions_cm: Dimensions = Field(description="가로·깊이·높이(cm)")
    dimensions_status: DimensionsStatus = Field(
        description="estimated: 세 치수 모두, partial: 일부만, unavailable: 못 구함"
    )
    disassembly_required: bool | None = Field(
        default=None, description="분해 필요 여부. 모르면 null(사용자에게 물어봄)"
    )
    needs_review: bool = Field(description="사용자 확인이 필요하면 true")
    review_reasons: list[str] = Field(default_factory=list, description="확인이 필요한 이유")
    snapshots: list[Snapshot] = Field(default_factory=list, description="물품 사진")


class SubmittedSnapshot(Snapshot):
    """분석기가 보내는 사진 요청. 백엔드가 `time_ms` 시각의 화면을 영상에서 캡처한다."""

    image_url: str | None = Field(
        default=None, description="비워 보낸다. 백엔드가 캡처한 사진 주소로 채운다"
    )
    bbox: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, min_length=4, max_length=4,
        description=(
            "물품 영역 [x1, y1, x2, y2]. 똑바로 세운 화면의 픽셀 기준. "
            "보내면 이 영역만 잘라내고, 없으면 전체 화면을 쓴다"
        ),
    )

    @model_validator(mode="after")
    def box_has_area(self):
        if self.bbox is not None and not (
            self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]
        ):
            raise ValueError("bbox는 x1 < x2, y1 < y2여야 합니다.")
        return self


class SubmittedAnalysisItem(AnalysisItem):
    label_ko: str = Field(min_length=1, max_length=MAX_ITEM_NAME_LENGTH)
    notes: str | None = Field(default=None, max_length=MAX_ITEM_NOTES_LENGTH)
    snapshots: list[SubmittedSnapshot] = Field(
        default_factory=list, max_length=MAX_SNAPSHOTS_PER_ITEM,
        description=f"물품 사진 요청. 물품당 최대 {MAX_SNAPSHOTS_PER_ITEM}장",
    )


class AnalysisError(V2):
    code: str = Field(min_length=1, description="실패 코드. 예: video_too_dark, timeout")
    message: str = Field(description="사용자에게 보여 줄 문장")
    retryable: bool = Field(description="true면 다시 촬영해서 해결할 수 있음(SC-06B)")


class VideoInfo(V2):
    video_id: str = Field(min_length=1, description="백엔드가 넘긴 영상 ID(media_id)")
    duration_s: float | None = Field(default=None, ge=0)
    capture_source: str | None = None


class AnalysisResultV2(V2):
    """AI 분석기가 보내는 최종 결과(완료 또는 실패)."""

    schema_version: Literal["2.0"]
    analysis_id: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    progress: float = Field(ge=0, le=1)
    video: VideoInfo
    items: list[SubmittedAnalysisItem] = Field(default_factory=list, max_length=MAX_INVENTORY_ITEMS)
    warnings: list[str] = Field(default_factory=list)
    error: AnalysisError | None = None

    @model_validator(mode="after")
    def failure_has_error(self):
        if self.status == "failed" and self.error is None:
            raise ValueError("status가 failed이면 error가 필요합니다.")
        if self.status == "completed" and self.error is not None:
            raise ValueError("status가 completed이면 error는 null이어야 합니다.")
        if sum(len(item.snapshots) for item in self.items) > MAX_SNAPSHOTS:
            raise ValueError(f"사진은 모두 합쳐 {MAX_SNAPSHOTS}장까지 보낼 수 있습니다.")
        return self


class AnalysisProgress(BaseModel):
    progress: float = Field(ge=0, le=1, description="0~1")


class WorkerJob(BaseModel):
    """분석기가 가져간 작업."""

    kind: Literal["analysis"] = "analysis"
    analysis_id: str
    move_id: str
    video_id: str = Field(description="결과의 video.video_id에 그대로 돌려준다.")
    video_url: str = Field(description="영상 파일을 내려받는 내부 API 경로")
    content_type: str


class AnalysisSection(BaseModel):
    """영상 분석 상태(SC-06 분석 대기·완료·실패)."""

    status: Literal["not_started", "queued", "running", "completed", "failed"] = Field(
        description=(
            "not_started: 영상 없음, queued: 분석 대기, running: 분석 중, "
            "completed: 완료(SC-07 목록으로), failed: 실패(SC-06B)"
        )
    )
    analysis_id: str | None = Field(default=None, description="분석 작업 ID")
    progress: float | None = Field(default=None, description="진행률 0~1")
    item_count: int | None = Field(default=None, description="완료 시 찾은 물품 줄 수")
    demo: bool = Field(
        default=False, description="데모 분석기의 가짜 결과면 true(ANALYSIS_DEMO 설정)"
    )
    error: AnalysisError | None = Field(default=None, description="실패 사유")
    updated_at: datetime | None = Field(default=None, description="상태가 바뀐 시각(UTC)")


class AnalysisOut(V2):
    """분석 결과 조회. AI와 합의한 분석 결과 v2.0 형식 그대로이며 `demo`만 덧붙인다.

    완료 전(`queued`, `running`)이나 실패(`failed`)에도 같은 모양이고 `items`는 빈 목록이다.
    영상이 없으면 `status`가 `not_started`이고 `analysis_id`, `video`는 null이다.
    """

    schema_version: Literal["2.0"] = "2.0"
    analysis_id: str | None = Field(default=None, description="분석 작업 ID")
    status: Literal["not_started", "queued", "running", "completed", "failed"]
    progress: float = Field(default=0, description="진행률 0~1")
    video: VideoInfo | None = Field(default=None, description="분석한 영상. video_id는 media_id")
    items: list[AnalysisItem] = Field(
        default_factory=list, description="완료 시 물품 목록. 그 전에는 빈 목록"
    )
    warnings: list[str] = Field(default_factory=list)
    error: AnalysisError | None = Field(default=None, description="실패 사유")
    demo: bool = Field(
        default=False, description="데모 분석기의 가짜 결과면 true(ANALYSIS_DEMO 설정)"
    )
