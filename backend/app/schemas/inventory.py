from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .analysis import (
    MAX_ITEM_NAME_LENGTH,
    MAX_ITEM_NOTES_LENGTH,
    Dimensions,
    DimensionsStatus,
    Snapshot,
)

ItemName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ITEM_NAME_LENGTH)
]
ItemNotes = Annotated[str, StringConstraints(max_length=MAX_ITEM_NOTES_LENGTH)]
DisassemblyPossible = Annotated[
    bool | None, Field(description="분리 가능 여부. null이면 확인되지 않음")
]


class ItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_ko: ItemName
    quantity: int = Field(default=1, ge=1)
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    notes: ItemNotes | None = None
    disassembly_required: bool | None = None
    disassembly_possible: DisassemblyPossible = None


class ItemPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_ko: ItemName | None = None
    quantity: int | None = Field(default=None, ge=1)
    dimensions_cm: Dimensions | None = None
    notes: ItemNotes | None = None
    disassembly_required: bool | None = None
    disassembly_possible: DisassemblyPossible = None

    @model_validator(mode="after")
    def validate_changes(self):
        if not self.model_fields_set:
            raise ValueError("수정할 항목을 보내주세요.")
        for field in self.model_fields_set & {"label_ko", "quantity", "dimensions_cm"}:
            if getattr(self, field) is None:
                raise ValueError(f"{field} 값이 필요합니다.")
        return self


class InventoryItem(BaseModel):
    item_id: str
    source: Literal["demo", "ai", "user"]
    label_ko: str
    quantity: int
    dimensions_cm: Dimensions
    dimensions_status: DimensionsStatus | None = Field(
        default=None, description="분석 당시 치수 추정 상태"
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None
    disassembly_required: bool | None = None
    disassembly_possible: DisassemblyPossible = None
    snapshots: list[Snapshot] = Field(default_factory=list)


class InventorySection(BaseModel):
    status: Literal["empty", "draft", "confirmed"] = "empty"
    item_count: int = 0
    total_quantity: int = 0


class Inventory(InventorySection):
    items: list[InventoryItem] = Field(default_factory=list)
