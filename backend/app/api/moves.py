from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials

from ..schemas.auth import User
from ..schemas.moves import (
    MediaInfo,
    MoveCreated,
    MoveData,
    MoveSnapshot,
    RoomVideos,
    Section,
)
from ..services.media import MediaStorage, MediaTooLarge, UnsupportedMedia
from ..services.moves import Media, Move, MoveStore
from .auth import Store, bearer, unauthorized

# Room for multipart boundaries and small form fields around the video.
FORM_OVERHEAD_BYTES = 64 * 1024


class BodyLimitRoute(APIRoute):
    """Refuse an oversized upload from its Content-Length before the body is read.

    FastAPI parses the whole form before running dependencies, so this check
    cannot live in the endpoint. Bodies without a length are still capped while
    the file is written.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            limit = request.app.state.media_storage.max_bytes + FORM_OVERHEAD_BYTES
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > limit:
                return JSONResponse(
                    {"detail": "업로드 용량이 너무 큽니다."},
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )
            return await handler(request)

        return handle


router = APIRouter(prefix="/api/v1/moves", tags=["moves"], route_class=BodyLimitRoute)
guest_header = APIKeyHeader(
    name="X-Guest-Token", auto_error=False, description="POST /moves가 비회원에게 발급한 토큰"
)


def get_move_store(request: Request) -> MoveStore:
    return request.app.state.move_store


def get_media_storage(request: Request) -> MediaStorage:
    return request.app.state.media_storage


MoveStoreDep = Annotated[MoveStore, Depends(get_move_store)]
MediaStorageDep = Annotated[MediaStorage, Depends(get_media_storage)]
Bearer = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
GuestToken = Annotated[str | None, Depends(guest_header)]


def optional_member(credentials: Bearer, store: Store) -> User | None:
    """No header means guest; a header that does not verify is an error, not a guest."""
    if credentials is None:
        return None
    user = store.current_user(credentials.credentials)
    if user is None:
        raise unauthorized("세션이 만료되었거나 유효하지 않습니다.")
    return user


Member = Annotated[User | None, Depends(optional_member)]


def get_accessible_move(
    move_id: str, member: Member, guest_token: GuestToken, moves: MoveStoreDep
) -> Move:
    """Shared ownership check for every /moves/{move_id} route."""
    if member is not None:
        move = moves.get_for_member(move_id, member.id)
    elif guest_token:
        move = moves.get_for_guest(move_id, guest_token)
    else:
        raise unauthorized("로그인하거나 비회원 토큰을 보내주세요.")
    if move is None:
        # Same answer for missing, expired and someone else's move.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "이사를 찾을 수 없습니다.")
    return move


AccessibleMove = Annotated[Move, Depends(get_accessible_move)]


def _time(epoch: int | None) -> datetime | None:
    return None if epoch is None else datetime.fromtimestamp(epoch, UTC)


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="이사 생성",
    description=(
        "Bearer 토큰이 있으면 계정에 저장되는 회원 이사를 만든다. 없으면 비회원 이사를 만들고 "
        "`guest_token`을 발급한다. 비회원 이사는 마지막 사용 후 정해진 시간이 지나면 열 수 없다."
    ),
    responses={401: {"description": "Bearer 토큰이 만료되었거나 유효하지 않음"}},
)
def create_move(member: Member, moves: MoveStoreDep) -> MoveCreated:
    if member is not None:
        move = moves.create_for_member(member.id)
        return MoveCreated(move_id=move.id, owner="member")
    move, guest_token = moves.create_for_guest()
    return MoveCreated(
        move_id=move.id, owner="guest", guest_token=guest_token,
        expires_at=_time(move.expires_at),
    )


@router.get(
    "/{move_id}",
    summary="이사 상태 조회",
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
    },
)
def get_move(move: AccessibleMove, moves: MoveStoreDep) -> MoveSnapshot:
    videos = [_media_info(item) for item in moves.list_media(move.id)]
    data = MoveData(
        room_video=RoomVideos(status="uploaded" if videos else "empty", items=videos),
        analysis=Section(status="not_started"),
        inventory=Section(status="empty"),
        origin=Section(status="empty"),
        destination=Section(status="empty"),
        schedule=Section(status="empty"),
    )
    missing = [
        step for step in ("room_video", "origin", "destination", "schedule")
        if getattr(data, step).status == "empty"
    ]
    return MoveSnapshot(
        move_id=move.id,
        owner="guest" if move.is_guest else "member",
        created_at=_time(move.created_at),
        expires_at=_time(move.expires_at),
        data=data,
        missing=missing,
        next_step=missing[0] if missing else None,
    )


@router.post(
    "/{move_id}/media",
    status_code=status.HTTP_201_CREATED,
    summary="방 영상 업로드",
    description=(
        "multipart/form-data로 `file`(mp4·mov)과 선택 항목 `space`(예: 방1, 거실)를 보낸다. "
        "한 이사에 공간별로 여러 영상을 올릴 수 있다. AI 분석은 요청하지 않는다."
    ),
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
        413: {"description": "최대 용량 초과"},
        415: {"description": "mp4·mov가 아니거나 영상으로 읽을 수 없는 파일"},
    },
)
def upload_media(
    move: AccessibleMove,
    file: Annotated[UploadFile, File(description="방 촬영 영상 (mp4, mov)")],
    moves: MoveStoreDep,
    storage: MediaStorageDep,
    space: Annotated[str | None, Form(max_length=30)] = None,
) -> MediaInfo:
    try:
        media = storage.save_video(
            move.id, file.file, file.filename, file.content_type,
            (space or "").strip() or None,
        )
    except UnsupportedMedia as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except MediaTooLarge as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    try:
        moves.add_media(media)
    except BaseException:
        storage.discard(media)
        raise
    return _media_info(media)


def _media_info(media: Media) -> MediaInfo:
    return MediaInfo(
        media_id=media.id, filename=media.filename, content_type=media.content_type,
        size_bytes=media.size_bytes, space=media.space, created_at=_time(media.created_at),
    )
