from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Owner = Literal["member", "guest"]
# Item keys in the order the chat asks for them.
Step = Literal["room_video", "origin", "destination", "schedule"]


class MoveCreated(BaseModel):
    move_id: str
    owner: Owner
    guest_token: str | None = Field(
        default=None,
        description="Guests only. Send it as X-Guest-Token on later requests for this move.",
    )
    expires_at: datetime | None = Field(
        default=None, description="Guests only. Extended on every request for this move."
    )


class Section(BaseModel):
    status: Literal["empty", "not_started"]


class MediaInfo(BaseModel):
    media_id: str
    filename: str
    content_type: Literal["video/mp4", "video/quicktime"]
    size_bytes: int
    space: str | None = Field(description="Room label such as 방1 or 거실, if given.")
    created_at: datetime


class RoomVideos(BaseModel):
    status: Literal["empty", "uploaded"]
    items: list[MediaInfo] = Field(default_factory=list)


class MoveData(BaseModel):
    room_video: RoomVideos
    analysis: Section
    inventory: Section
    origin: Section
    destination: Section
    schedule: Section


class MoveSnapshot(BaseModel):
    move_id: str
    owner: Owner
    created_at: datetime
    expires_at: datetime | None
    stage: Literal["collecting"] = "collecting"
    data: MoveData
    missing: list[Step] = Field(description="User inputs still empty, in asking order.")
    next_step: Step | None = Field(description="First item in missing, or null when complete.")
