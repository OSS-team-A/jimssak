from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .analysis import Dimensions, Snapshot

ItemName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_ko: ItemName
    quantity: int = Field(default=1, ge=1)
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    notes: str | None = None
    disassembly_required: bool | None = None


class ItemPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_ko: ItemName | None = None
    quantity: int | None = Field(default=None, ge=1)
    dimensions_cm: Dimensions | None = None
    notes: str | None = None
    disassembly_required: bool | None = None

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
    notes: str | None = None
    disassembly_required: bool | None = None
    snapshots: list[Snapshot] = Field(default_factory=list)


class InventorySection(BaseModel):
    status: Literal["empty", "draft", "confirmed"] = "empty"
    item_count: int = 0
    total_quantity: int = 0


class Inventory(InventorySection):
    items: list[InventoryItem] = Field(default_factory=list)
