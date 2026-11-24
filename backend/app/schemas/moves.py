from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

Owner = Literal["member", "guest"]
# Item keys in the order the chat asks for them.
Step = Literal["room_video", "origin", "destination", "schedule"]


class MoveCreated(BaseModel):
    move_id: UUID
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
    media_id: UUID
    filename: str
    content_type: Literal["video/mp4", "video/quicktime"]
    size_bytes: int
    created_at: datetime


class UploadedVideo(MediaInfo):
    replaced: bool = Field(description="True when this upload replaced an earlier video.")


class RoomVideo(BaseModel):
    """One video per move: small one-room moves are filmed in a single take."""

    status: Literal["empty", "uploaded"]
    video: MediaInfo | None = None


class MoveData(BaseModel):
    room_video: RoomVideo
    analysis: Section
    inventory: Section
    origin: Section
    destination: Section
    schedule: Section


class MoveSnapshot(BaseModel):
    move_id: UUID
    owner: Owner
    created_at: datetime
    expires_at: datetime | None
    stage: Literal["collecting"] = "collecting"
    data: MoveData
    missing: list[Step] = Field(description="User inputs still empty, in asking order.")
    next_step: Step | None = Field(description="First item in missing, or null when complete.")
