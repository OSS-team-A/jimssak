from pydantic import BaseModel, ConfigDict, Field, model_validator

from .comparison import ComparisonSelection, UserCheck
from .move_chat import ChatInput, ChatText


class ComparisonChatInput(ChatInput, ComparisonSelection):
    """이사 후 확인 채팅. 화면에 열린 비교의 ID와 메시지를 보낸다."""


class SuggestedComparisonCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1, description="현재 비교 결과에서 확인이 필요한 물품 ID")
    user_check: UserCheck


class ComparisonChatInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: ChatText = Field(description="제안 또는 추가 질문")
    checks: list[SuggestedComparisonCheck] = Field(
        default_factory=list, max_length=20, description="사용자가 명시한 물품 확인 답변 제안",
    )

    @model_validator(mode="after")
    def unique_items(self) -> "ComparisonChatInterpretation":
        if len({check.item_id for check in self.checks}) != len(self.checks):
            raise ValueError("같은 물품에는 답변을 한 번만 제안할 수 있어요.")
        return self


class ComparisonChatReply(ComparisonChatInterpretation):
    comparison_id: str = Field(description="제안의 기준이 된 비교 ID. 답 저장에도 사용한다.")
