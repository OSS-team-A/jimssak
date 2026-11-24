from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials

from ..schemas.auth import User
from ..schemas.moves import MoveCreated, MoveData, MoveSnapshot, RoomVideos, Section
from ..services.moves import Move, MoveStore
from .auth import Store, bearer, unauthorized

router = APIRouter(prefix="/api/v1/moves", tags=["moves"])
guest_header = APIKeyHeader(
    name="X-Guest-Token", auto_error=False, description="POST /moves가 비회원에게 발급한 토큰"
)


def get_move_store(request: Request) -> MoveStore:
    return request.app.state.move_store


MoveStoreDep = Annotated[MoveStore, Depends(get_move_store)]
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
def get_move(move: AccessibleMove) -> MoveSnapshot:
    data = MoveData(
        room_video=RoomVideos(status="empty"),
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
