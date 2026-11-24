from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials

from ..schemas.auth import User
from ..schemas.moves import (
    MediaInfo,
    MoveCreated,
    MoveData,
    MoveSnapshot,
    MoveUpdate,
    Place,
    PlaceInput,
    PlaceSection,
    RoomVideo,
    Schedule,
    ScheduleSection,
    Section,
    UploadedVideo,
)
from ..services.auth import AuthStore
from ..services.media import MediaStorage, MediaTooLarge, UnsupportedMedia
from ..services.moves import Media, Move, MoveStore, PlaceRecord, ScheduleRecord
from .auth import Store, bearer, unauthorized

# Room for multipart boundaries and small form fields around the video.
FORM_OVERHEAD_BYTES = 64 * 1024


guest_header = APIKeyHeader(
    name="X-Guest-Token", auto_error=False, description="POST /moves가 비회원에게 발급한 토큰"
)


def resolve_member(
    store: AuthStore, credentials: HTTPAuthorizationCredentials | None
) -> User | None:
    """No header means guest; a header that does not verify is an error, not a guest."""
    if credentials is None:
        return None
    user = store.current_user(credentials.credentials)
    if user is None:
        raise unauthorized("세션이 만료되었거나 유효하지 않습니다.")
    return user


def resolve_move(
    moves: MoveStore, move_id: str, member: User | None, guest_token: str | None
) -> Move:
    """The one ownership rule for every /moves/{move_id} route."""
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


class GuardedRoute(APIRoute):
    """Run size and access checks before the request body is read.

    FastAPI parses the whole form before running dependencies, so an upload
    would otherwise be received in full before a wrong token is refused. Both
    checks only need headers and the path. Dependencies still run the same
    access rule afterwards. Bodies without a length are capped while written.
    """

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            state = request.app.state
            limit = state.media_storage.max_bytes + FORM_OVERHEAD_BYTES
            length = request.headers.get("content-length", "")
            if length.isdigit() and int(length) > limit:
                return JSONResponse(
                    {"detail": "업로드 용량이 너무 큽니다."},
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )
            raw_move_id = request.path_params.get("move_id")
            if raw_move_id is not None and request.method != "GET":
                try:
                    move_id = str(UUID(raw_move_id))
                except ValueError:
                    return JSONResponse(
                        {"detail": "이사 ID 형식이 올바르지 않습니다."},
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    )
                member = resolve_member(state.auth_store, await bearer(request))
                resolve_move(state.move_store, move_id, member, await guest_header(request))
            return await handler(request)

        return handle


router = APIRouter(prefix="/api/v1/moves", tags=["moves"], route_class=GuardedRoute)


def get_move_store(request: Request) -> MoveStore:
    return request.app.state.move_store


def get_media_storage(request: Request) -> MediaStorage:
    return request.app.state.media_storage


MoveStoreDep = Annotated[MoveStore, Depends(get_move_store)]
MediaStorageDep = Annotated[MediaStorage, Depends(get_media_storage)]
Bearer = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
GuestToken = Annotated[str | None, Depends(guest_header)]


def optional_member(credentials: Bearer, store: Store) -> User | None:
    return resolve_member(store, credentials)


Member = Annotated[User | None, Depends(optional_member)]


def get_accessible_move(
    move_id: UUID, member: Member, guest_token: GuestToken, moves: MoveStoreDep
) -> Move:
    # IDs are stored in canonical lowercase form, whatever form the path used.
    return resolve_move(moves, str(move_id), member, guest_token)


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
    return _snapshot(move, moves)


@router.patch(
    "/{move_id}",
    summary="이사 기본정보 저장",
    description=(
        "SC-06의 출발지·도착지(도로명주소, 상세주소, 층, 승강기)와 이사 일정(날짜, 시작 시간)을 "
        "저장한다. 보낸 항목(`origin`, `destination`, `schedule`)만 바뀌고, 보내지 않은 "
        "항목은 그대로 둔다. 항목을 `null`로 보내면 지운다. 승강기(`elevator`)는 사용자가 "
        "고칠 때만 보내며, 보내지 않으면 같은 주소에서는 기존 값을 유지하고 주소가 바뀌면 "
        "확인 필요(unknown)로 돌아간다. 저장 후의 이사 상태를 반환한다."
    ),
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
        422: {"description": "주소 누락, 층 범위(-5~80, 0 제외), 지난 날짜, 정시가 아닌 시간"},
    },
)
def update_move(update: MoveUpdate, move: AccessibleMove, moves: MoveStoreDep) -> MoveSnapshot:
    sent = update.model_fields_set
    current, _ = moves.get_details(move.id)
    places = {
        role: _place_record(getattr(update, role), current.get(role))
        for role in ("origin", "destination") if role in sent
    }
    schedule = None
    if update.schedule is not None:
        schedule = ScheduleRecord(
            update.schedule.move_date.isoformat(), update.schedule.start_time.isoformat()
        )
    moves.update_details(move.id, places, schedule, schedule_sent="schedule" in sent)
    return _snapshot(move, moves)


def _place_record(place: PlaceInput | None, current: PlaceRecord | None) -> PlaceRecord | None:
    if place is None:
        return None
    if place.elevator is not None:
        # The user corrected it on the confirmation screen.
        elevator = place.elevator
        source = None if elevator == "unknown" else "user"
    elif current is not None and current.road_address == place.road_address:
        # Same building: keep what we had, whether looked up or entered.
        elevator, source = current.elevator, current.elevator_source
    else:
        # New building: the old answer no longer applies.
        elevator, source = "unknown", None
    return PlaceRecord(
        road_address=place.road_address,
        detail_address=place.detail_address,
        floor=place.floor,
        elevator=elevator,
        elevator_source=source,
    )


def _place_section(record: PlaceRecord | None) -> PlaceSection:
    if record is None:
        return PlaceSection(status="empty")
    return PlaceSection(
        status="filled" if record.floor is not None else "partial",
        value=Place(**vars(record)),
    )


def _snapshot(move: Move, moves: MoveStore) -> MoveSnapshot:
    video = moves.get_video(move.id)
    places, schedule = moves.get_details(move.id)
    data = MoveData(
        room_video=RoomVideo(
            status="uploaded" if video else "empty",
            video=_media_info(video) if video else None,
        ),
        analysis=Section(status="not_started"),
        inventory=Section(status="empty"),
        origin=_place_section(places.get("origin")),
        destination=_place_section(places.get("destination")),
        schedule=ScheduleSection(
            status="filled" if schedule else "empty",
            value=Schedule(move_date=schedule.move_date, start_time=schedule.start_time)
            if schedule else None,
        ),
    )
    # Anything not yet complete still needs asking; partial places count as missing.
    missing = [
        step for step in ("room_video", "origin", "destination", "schedule")
        if getattr(data, step).status not in ("uploaded", "filled")
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
        "multipart/form-data로 `file`(mp4·mov)을 보낸다. 소형 원룸 이사라 이사 한 건에 영상은 "
        "하나다. 이미 영상이 있으면 새 영상으로 교체하고(재촬영) `replaced`가 true다. "
        "AI 분석은 요청하지 않는다."
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
) -> UploadedVideo:
    try:
        media = storage.save_video(move.id, file.file, file.filename, file.content_type)
    except UnsupportedMedia as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except MediaTooLarge as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    try:
        replaced = moves.replace_video(media)
    except BaseException:
        storage.discard(media)
        raise
    # The database no longer points at the old file, so it can go.
    if replaced is not None:
        storage.discard(replaced)
    return UploadedVideo(**_media_info(media).model_dump(), replaced=replaced is not None)


def _media_info(media: Media) -> MediaInfo:
    return MediaInfo(
        media_id=media.id, filename=media.filename, content_type=media.content_type,
        size_bytes=media.size_bytes, created_at=_time(media.created_at),
    )
