"""After-move comparison (SC-14~16): the after-move video checked against the confirmed list.

Each item of the confirmed list gets one result: `ok` (looks the same), `changed` (its state
looks different) or `missing` (not found in the after-move video). Items that are not `ok`
need the user's answer before the review can be finished.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import MAX_SNAPSHOTS, MAX_SNAPSHOTS_PER_ITEM, AnalysisError, SubmittedSnapshot

ComparisonResultType = Literal["ok", "changed", "missing"]
UserCheck = Literal["confirmed", "not_found"]
RESULT_DOC = (
    "| 값 | 뜻 | 화면 |\n"
    "|---|---|---|\n"
    "| `ok` | 이사 전과 같아 보임 | 이상 없음 |\n"
    "| `changed` | 상태가 달라 보임 | 상태 변화 의심, 전후 사진 비교 |\n"
    "| `missing` | 이사 후 영상에서 찾지 못함 | 직접 확인했는지, 아직 못 찾았는지 묻기 |"
)
USER_CHECK_DOC = (
    "확인이 필요한 물품(`changed`·`missing`)에 대한 사용자 답. "
    "`confirmed`: 직접 확인했어요, `not_found`: 아직 못 찾았어요, 아직 답하지 않았으면 null."
)


class ComparisonPhoto(BaseModel):
    time_ms: int = Field(description="영상에서 이 사진의 시각(ms)")
    image_url: str | None = Field(
        description="사진 주소. 인증 헤더를 붙여 받는다(`GET .../analysis/photos/{name}`)"
    )


class ComparisonItem(BaseModel):
    item_id: str = Field(description="확정 목록의 물품 ID")
    label_ko: str = Field(description="물품 이름")
    quantity: int = Field(description="수량")
    result: ComparisonResultType = Field(description=RESULT_DOC)
    reasons: list[str] = Field(description="그렇게 판단한 이유. 화면에 그대로 보여 준다")
    before_photos: list[ComparisonPhoto] = Field(description="이사 전 영상의 사진")
    after_photos: list[ComparisonPhoto] = Field(description="이사 후 영상의 사진")
    needs_check: bool = Field(description="사용자 확인이 필요하면 true(`changed`·`missing`)")
    user_check: UserCheck | None = Field(description=USER_CHECK_DOC)


class ComparisonSummary(BaseModel):
    ok: int = Field(description="이상 없음 물품 수")
    changed: int = Field(description="상태 변화 의심 물품 수")
    missing: int = Field(description="찾지 못한 물품 수")
    unchecked: int = Field(description="확인이 필요한데 아직 답하지 않은 물품 수")


class ComparisonOut(BaseModel):
    """이사 후 비교 결과. 완료 전에는 `items`가 빈 목록이다."""

    status: Literal["not_started", "queued", "running", "completed", "failed"] = Field(
        description=(
            "not_started: 이사 후 영상 전, queued·running: 비교 중(2~3초마다 다시 조회), "
            "completed: 결과 있음, failed: 실패(`error`, 이사 후 영상을 다시 올리면 새로 비교)"
        )
    )
    comparison_id: str | None = Field(description="비교 작업 ID. 시작 전이면 null")
    progress: float = Field(description="진행률 0~1")
    demo: bool = Field(description="true면 데모 분석기가 만든 가짜 결과")
    error: AnalysisError | None = Field(description="실패 사유. 실패가 아니면 null")
    items: list[ComparisonItem] = Field(description="확정 목록 순서의 물품별 결과")
    summary: ComparisonSummary
    reviewed_at: datetime | None = Field(
        description="사용자가 검토를 마친 시각(UTC). 마치기 전이면 null"
    )


class ComparisonCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_check: UserCheck = Field(description=USER_CHECK_DOC)


# --- the AI analyzer's side ---

class SubmittedComparisonItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: str = Field(min_length=1, description="작업에서 받은 확정 목록의 item_id")
    result: ComparisonResultType = Field(description=RESULT_DOC)
    reasons: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=5, description="판단 이유. 사용자에게 그대로 보인다"
    )
    before_snapshots: list[SubmittedSnapshot] = Field(
        default_factory=list, max_length=MAX_SNAPSHOTS_PER_ITEM,
        description="이사 전 영상에서 캡처할 시각(`time_ms`, 선택 `bbox`)",
    )
    after_snapshots: list[SubmittedSnapshot] = Field(
        default_factory=list, max_length=MAX_SNAPSHOTS_PER_ITEM,
        description="이사 후 영상에서 캡처할 시각(`time_ms`, 선택 `bbox`)",
    )


class ComparisonVideo(BaseModel):
    model_config = ConfigDict(extra="allow")

    video_id: str = Field(min_length=1, description="작업에서 받은 이사 후 영상 ID")


class ComparisonResult(BaseModel):
    """AI 분석기가 보내는 비교 결과. 확정 목록의 물품마다 정확히 한 번씩 보낸다."""

    model_config = ConfigDict(extra="allow")

    schema_version: Literal["1.0"]
    comparison_id: str = Field(min_length=1)
    status: Literal["completed", "failed"]
    progress: float = Field(ge=0, le=1)
    video: ComparisonVideo
    items: list[SubmittedComparisonItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: AnalysisError | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "failed" and self.error is None:
            raise ValueError("status가 failed이면 error가 필요합니다.")
        if self.status == "completed" and self.error is not None:
            raise ValueError("status가 completed이면 error는 null이어야 합니다.")
        photos = sum(len(i.before_snapshots) + len(i.after_snapshots) for i in self.items)
        if photos > MAX_SNAPSHOTS:
            raise ValueError(f"사진은 모두 합쳐 {MAX_SNAPSHOTS}장까지 보낼 수 있습니다.")
        return self


class ComparisonJobItem(BaseModel):
    item_id: str
    label: str | None = Field(description="물품 종류 코드. 사용자가 추가한 물품이면 null")
    label_ko: str
    quantity: int
    dimensions_cm: dict = Field(description="가로·깊이·높이(cm). 모르는 치수는 null")


class ComparisonJob(BaseModel):
    """분석기가 가져간 비교 작업."""

    kind: Literal["comparison"] = "comparison"
    comparison_id: str
    move_id: str
    before_video_id: str
    after_video_id: str = Field(description="결과의 `video.video_id`에 그대로 넣는다")
    before_video_url: str
    after_video_url: str
    items: list[ComparisonJobItem] = Field(description="비교할 확정 목록")
