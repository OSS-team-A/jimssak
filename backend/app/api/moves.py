import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials
from psycopg import Connection

from ..schemas.analysis import AnalysisOut, AnalysisSection
from ..schemas.auth import User
from ..schemas.moves import (
    KST,
    MAX_DAYS_AHEAD,
    BuildingInfo,
    DayInfoOut,
    MediaInfo,
    MoveCreated,
    MoveData,
    MoveList,
    MoveSnapshot,
    MoveSummary,
    MoveUpdate,
    Place,
    PlaceInput,
    PlaceSection,
    ReviewDetails,
    RoomVideo,
    RouteSection,
    Schedule,
    ScheduleSection,
    UploadedVideo,
)
from ..services.analysis import DEMO_PHOTOS, SNAPSHOT_NAME, AnalysisJob, AnalysisStore
from ..services.auth import AuthStore
from ..services.building_register import (
    BuildingLookupFailed,
    BuildingRegister,
    LotCode,
    count,
    decide_elevator,
    select_building,
)
from ..services.calendar import Calendar
from ..services.comparison import ComparisonStore
from ..services.date_info import is_complete, judge_move_date
from ..services.inventory import load_inventory
from ..services.media import MediaStorage, MediaTooLarge, UnsupportedMedia
from ..services.moves import (
    DateInfoRecord,
    LookupRecord,
    Media,
    Move,
    MoveStore,
    PlaceRecord,
    RouteRecord,
    ScheduleRecord,
    place_ladder,
)
from ..services.route import AddressNotLocated, RouteLookupFailed, Router
from .auth import Store, bearer, unauthorized

DEMO_PHOTO_DIR = Path(__file__).resolve().parents[1] / "demo"
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
        # Claiming checks the guest token against a member login itself, and before it
        # the move is not the member's yet, so the ownership pre-check would refuse it.
        check_access = not self.path.endswith("/claim")

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
            if raw_move_id is not None and request.method != "GET" and check_access:
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


def get_building_register(request: Request) -> BuildingRegister:
    return request.app.state.building_register


RegisterDep = Annotated[BuildingRegister, Depends(get_building_register)]


def get_router(request: Request) -> Router:
    return request.app.state.router


RouterDep = Annotated[Router, Depends(get_router)]


def get_analysis_store(request: Request) -> AnalysisStore:
    return request.app.state.analysis_store


AnalysisDep = Annotated[AnalysisStore, Depends(get_analysis_store)]


def get_comparison_store(request: Request) -> ComparisonStore:
    return request.app.state.comparison_store


ComparisonDep = Annotated[ComparisonStore, Depends(get_comparison_store)]


def get_calendar(request: Request) -> Calendar:
    return request.app.state.calendar


CalendarDep = Annotated[Calendar, Depends(get_calendar)]
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
    description=(
        "이사 한 건의 현재 상태를 반환한다. 새로고침이나 다시 열었을 때 화면을 복원하는 데 "
        "쓴다. 회원 이사는 `Authorization: Bearer`, 비회원 이사는 `X-Guest-Token`으로 연다. "
        "비회원 이사는 조회할 때마다 만료 시각이 다시 늘어난다.\n\n"
        "**`data`의 항목**\n\n"
        "| 항목 | 화면 | status 값 |\n"
        "|---|---|---|\n"
        "| `room_video` | SC-05 영상 | empty / uploaded |\n"
        "| `analysis` | SC-06 분석 | not_started / queued / running / completed / failed |\n"
        "| `inventory` | SC-07 이삿짐 목록 | empty / draft / confirmed |\n"
        "| `origin`, `destination` | SC-06 주소·층, 최종 확인 | empty / partial / filled |\n"
        "| `route` | 견적 거리 | not_ready / found / not_found / failed |\n"
        "| `schedule` | SC-06 일정 | empty / filled (`day`에 날짜 할증 조건) |\n"
        "| `review` | SC-07 짐 받는 날짜·이사 특이사항 | 저장된 정보 또는 null |\n\n"
        "`missing`은 아직 입력하지 않은 항목을 물어볼 순서대로, `next_step`은 그 첫 항목을 "
        "준다. 둘 다 비면(`[]`, null) SC-06까지 입력이 끝난 것이다."
    ),
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
    },
)
def get_move(move: AccessibleMove, moves: MoveStoreDep, analyses: AnalysisDep) -> MoveSnapshot:
    return _snapshot(move, moves, analyses)


@router.get(
    "",
    summary="나의 이사 기록",
    description="로그인한 회원의 이사를 최신순으로 반환한다(SC-13). 비회원은 사용할 수 없다.",
    responses={401: {"description": "로그인 필요"}},
)
def list_moves(
    member: Member,
    moves: MoveStoreDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MoveList:
    if member is None:
        raise unauthorized("로그인이 필요합니다.")
    found, total = moves.list_for_member(member.id, limit, offset)
    return MoveList(
        moves=[
            MoveSummary(
                move_id=item.id, created_at=_time(item.created_at), move_date=item.move_date,
                start_time=item.start_time, origin_address=item.origin_address,
                destination_address=item.destination_address,
            )
            for item in found
        ],
        total=total, limit=limit, offset=offset,
    )


@router.post(
    "/{move_id}/claim",
    summary="비회원 이사를 계정에 연결",
    description=(
        "SC-12. 로그인 토큰(Bearer)과 이 이사의 비회원 토큰(`X-Guest-Token`)을 함께 보내면 "
        "비회원 이사를 로그인한 회원의 이사로 바꾼다. 영상·주소·건물·거리·날짜 정보는 그대로 "
        "남고 만료 기한이 없어진다. 이후에는 회원 토큰으로만 조회한다. 이미 같은 회원에게 "
        "연결된 이사를 다시 요청하면 그대로 성공한다."
    ),
    responses={
        401: {"description": "로그인 토큰 또는 비회원 토큰 없음, 로그인 만료"},
        404: {"description": "비회원 토큰이 틀림, 만료된 비회원 이사, 다른 회원의 이사"},
    },
)
def claim_move(
    move_id: UUID, member: Member, guest_token: GuestToken, moves: MoveStoreDep,
    analyses: AnalysisDep,
) -> MoveSnapshot:
    if member is None:
        raise unauthorized("로그인이 필요합니다.")
    if not guest_token:
        raise unauthorized("연결할 이사의 비회원 토큰을 보내주세요.")
    move = moves.claim(str(move_id), guest_token, member.id)
    if move is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "이사를 찾을 수 없습니다.")
    return _snapshot(move, moves, analyses)


@router.patch(
    "/{move_id}",
    summary="이사 기본정보 저장",
    description=(
        "SC-06의 출발지·도착지(도로명주소, 상세주소, 층, 승강기)와 이사 일정(날짜, 시작 시간)을 "
        "저장한다. SC-07에서 검토한 짐 받는 날짜와 이사 특이사항은 `review`로 저장한다. "
        "보낸 항목(`origin`, `destination`, `schedule`, `review`)을 교체하며, "
        "항목을 `null`로 보내면 지운다.\n\n"
        "주소 검색(`GET /addresses`)에서 고른 후보의 `building_name`, `adm_cd`, `mountain`, "
        "`lot_main`, `lot_sub`를 함께 보내면 건축물대장을 조회해 건물 정보(`building`)와 승강기를 "
        "채운다. 같은 지번을 다시 보내면 외부 조회 없이 저장된 결과를 쓰고, 조회가 실패했던 "
        "주소만 다시 조회한다. 조회가 실패해도 저장은 성공한다. 외부 API가 간헐적으로 "
        "실패하므로, 응답의 `building.status`가 `failed`이면 같은 본문으로 한 번 더 저장한다.\n\n"
        "출발지·도착지가 모두 있으면 카카오 자동차 길찾기로 도로거리를 계산해 `route`에 "
        "저장한다. 두 도로명주소 중 하나가 바뀔 때만 다시 계산하며, `route.status`가 "
        "`failed`이면 같은 본문으로 다시 저장해 재계산한다.\n\n"
        "승강기(`elevator`)는 사용자가 고칠 때만 보낸다. 사용자가 고친 값은 같은 주소인 동안 "
        "조회 결과로 덮어쓰지 않는다. 저장 후의 이사 상태를 반환한다."
    ),
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
        422: {"description": "주소 누락, 층 범위(-5~80, 0 제외), 지난 날짜, 정시가 아닌 시간"},
    },
)
def update_move(
    update: MoveUpdate, move: AccessibleMove, moves: MoveStoreDep, register: RegisterDep,
    router: RouterDep, calendar: CalendarDep, analyses: AnalysisDep,
) -> MoveSnapshot:
    prepare_move_update(update, move, moves, register, router, calendar)(None)
    return _snapshot(move, moves, analyses)


def validate_move_review(
    update: MoveUpdate, current_schedule: ScheduleRecord | None,
    current_review: ReviewDetails | None,
) -> None:
    sent = update.model_fields_set
    if not sent & {"review", "schedule"}:
        return
    review = update.review if "review" in sent else current_review
    delivery = review.delivery_date if review else None
    pickup = (
        update.schedule.move_date if update.schedule else None
    ) if "schedule" in sent else (
        date.fromisoformat(current_schedule.move_date) if current_schedule else None
    )
    if delivery and pickup and delivery < pickup:
        raise HTTPException(422, "짐 받는 날짜는 짐 빼는 날짜와 같거나 이후여야 합니다.")
    previous = current_review.delivery_date if current_review else None
    if delivery and delivery != previous:
        days = (delivery - datetime.now(KST).date()).days
        if not 0 <= days <= MAX_DAYS_AHEAD:
            raise HTTPException(422, "짐 받는 날짜는 오늘부터 365일 이내로 입력해주세요.")


def prepare_move_update(
    update: MoveUpdate, move: Move, moves: MoveStore, register: BuildingRegister,
    router: Router, calendar: Calendar,
) -> Callable[[Connection | None], None]:
    sent = update.model_fields_set
    current, current_schedule = moves.get_details(move.id)
    if sent & {"review", "schedule"}:
        validate_move_review(update, current_schedule, moves.get_review(move.id))
    known = moves.get_lookups(move.id)
    places: dict[str, PlaceRecord | None] = {}
    lookups: dict[str, LookupRecord | None] = {}
    sent_places = {
        role: getattr(update, role) for role in ("origin", "destination") if role in sent
    }
    # The two addresses the move will have once this save is done.
    addresses = {}
    for role in ("origin", "destination"):
        if role in sent_places:
            place = sent_places[role]
            addresses[role] = place.road_address if place else None
        else:
            addresses[role] = current[role].road_address if role in current else None
    route_sent, route_job = _route_plan(addresses, moves.get_route(move.id))
    date_sent, date_job = _date_plan(update, sent, moves.get_date_info(move.id))
    # External calls can be slow and are retried, so they all run at the same time.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {
            role: pool.submit(_look_up, register, place, known.get(role))
            for role, place in sent_places.items() if place is not None
        }
        route_future = pool.submit(_find_route, router, *route_job) if route_job else None
        date_future = (
            pool.submit(judge_move_date, moves, calendar, date_job) if date_job else None
        )
    for role, place in sent_places.items():
        if place is None:
            places[role] = lookups[role] = None
            continue
        lookups[role] = pending[role].result()
        places[role] = _place_record(place, current.get(role), lookups[role])
    schedule = None
    if update.schedule is not None:
        schedule = ScheduleRecord(
            update.schedule.move_date.isoformat(), update.schedule.start_time.isoformat()
        )

    def save(conn: Connection | None) -> None:
        with moves.db.connect() if conn is None else nullcontext(conn) as active:
            if sent & {"review", "schedule"}:
                active.execute("SELECT 1 FROM moves WHERE id = %s FOR NO KEY UPDATE", (move.id,))
                _, latest_schedule = moves.get_details(move.id, conn=active)
                latest_review = moves.get_review(move.id, conn=active)
                validate_move_review(update, latest_schedule, latest_review)
            moves.update_details(
                move.id, places, schedule, schedule_sent="schedule" in sent, lookups=lookups,
                route=route_future.result() if route_future else None, route_sent=route_sent,
                date_info=date_future.result() if date_future else None, date_info_sent=date_sent,
                review=update.review, review_sent="review" in sent, conn=active,
            )

    return save


def _date_plan(
    update: MoveUpdate, sent: set[str], known: DateInfoRecord | None
) -> tuple[bool, date | None]:
    """Whether the stored date info changes, and which date to judge if any."""
    if "schedule" not in sent:
        return False, None
    if update.schedule is None:
        return known is not None, None
    day = update.schedule.move_date
    if known is not None and known.move_date == day.isoformat() and is_complete(known):
        return False, None
    # A new date, or one whose holidays or lunar date could not be read before.
    return True, day


def _day_info(record: DateInfoRecord | None) -> DayInfoOut | None:
    if record is None:
        return None
    info = json.loads(record.info_json)
    return DayInfoOut(
        **info, holiday_status=record.holiday_status, lunar_status=record.lunar_status,
        checked_at=_time(record.checked_at),
    )


def _route_plan(
    addresses: dict[str, str | None], known: RouteRecord | None
) -> tuple[bool, tuple[str, str] | None]:
    """Whether the stored route changes, and which address pair to compute if any."""
    origin, destination = addresses["origin"], addresses["destination"]
    if origin is None or destination is None:
        # Nothing to measure yet; drop a route left from earlier addresses.
        return known is not None, None
    if known is not None and known.status != "failed" and (
        (known.origin_address, known.destination_address) == (origin, destination)
    ):
        return False, None
    return True, (origin, destination)


def _find_route(router: Router, origin: str, destination: str) -> RouteRecord:
    checked_at = int(time.time())
    base = {"origin_address": origin, "destination_address": destination,
            "checked_at": checked_at}
    try:
        found = router.route(origin, destination)
    except AddressNotLocated:
        return RouteRecord(status="not_found", **base)
    except RouteLookupFailed:
        return RouteRecord(status="failed", **base)
    return RouteRecord(
        status="found", distance_m=found.distance_m, duration_s=found.duration_s,
        origin_x=found.origin.x, origin_y=found.origin.y,
        destination_x=found.destination.x, destination_y=found.destination.y, **base,
    )


def _route_section(route: RouteRecord | None) -> RouteSection:
    if route is None:
        return RouteSection(status="not_ready")
    return RouteSection(
        status=route.status, distance_m=route.distance_m, duration_s=route.duration_s,
        basis="kakao_car_recommend" if route.status == "found" else None,
        checked_at=_time(route.checked_at),
    )


def _look_up(
    register: BuildingRegister, place: PlaceInput, known: LookupRecord | None
) -> LookupRecord:
    """Find the user's building, calling the register only for a lot not read before."""
    if place.adm_cd is None or place.lot_main is None:
        return LookupRecord(status="not_requested")
    lot = LotCode(place.adm_cd, place.mountain, place.lot_main, place.lot_sub)
    codes = {"adm_cd": lot.adm_cd, "mountain": lot.mountain,
             "lot_main": lot.lot_main, "lot_sub": lot.lot_sub}
    same_lot = known is not None and known.rows_json is not None and (
        (known.adm_cd, known.mountain, known.lot_main, known.lot_sub)
        == (lot.adm_cd, lot.mountain, lot.lot_main, lot.lot_sub)
    )
    if same_lot:
        rows, checked_at = json.loads(known.rows_json), known.checked_at
    else:
        checked_at = int(time.time())
        try:
            rows = register.fetch(lot)
        except BuildingLookupFailed:
            return LookupRecord(status="failed", checked_at=checked_at, **codes)
    # Matching runs on the kept rows, so a changed 동 in the detail address needs no new call.
    match = select_building(rows, place.road_address, place.building_name, place.detail_address)
    return LookupRecord(
        status=match.status,
        rows_json=json.dumps(rows, ensure_ascii=False),
        building_json=json.dumps(match.row, ensure_ascii=False) if match.row else None,
        checked_at=checked_at,
        **codes,
    )


def _place_record(
    place: PlaceInput, current: PlaceRecord | None, lookup: LookupRecord
) -> PlaceRecord:
    same_address = current is not None and current.road_address == place.road_address
    if place.elevator is not None:
        # The user corrected it on the confirmation screen.
        elevator = place.elevator
        source = None if elevator == "unknown" else "user"
    elif same_address and current.elevator_source == "user":
        # A user's answer for this building outranks the register.
        elevator, source = current.elevator, current.elevator_source
    elif lookup.status == "found":
        elevator = decide_elevator(json.loads(lookup.building_json))
        source = None if elevator == "unknown" else "building_register"
    elif lookup.status != "not_requested":
        # Looked up but could not tell which building, or the lookup failed.
        elevator, source = "unknown", None
    elif same_address:
        # No lookup codes this time: keep what we had for this building.
        elevator, source = current.elevator, current.elevator_source
    else:
        # New building: the old answer no longer applies.
        elevator, source = "unknown", None
    if place.ladder is not None:
        ladder = place.ladder
    else:
        # Like the elevator, a choice made for one building does not carry over to another.
        ladder = current.ladder if same_address else None
    return PlaceRecord(
        road_address=place.road_address,
        detail_address=place.detail_address,
        floor=place.floor,
        elevator=elevator,
        elevator_source=source,
        ladder=ladder,
    )


def _building_info(lookup: LookupRecord | None) -> BuildingInfo:
    if lookup is None:
        return BuildingInfo(status="not_requested")
    info = BuildingInfo(status=lookup.status, checked_at=_time(lookup.checked_at))
    if lookup.building_json:
        row = json.loads(lookup.building_json)
        info.building_name = (row.get("bldNm") or "").strip() or None
        info.dong_name = (row.get("dongNm") or "").strip() or None
        info.above_ground_floors = count(row.get("grndFlrCnt"))
        info.ride_elevators = count(row.get("rideUseElvtCnt"))
        info.emergency_elevators = count(row.get("emgenUseElvtCnt"))
    return info


def _place_section(record: PlaceRecord | None, lookup: LookupRecord | None) -> PlaceSection:
    if record is None:
        return PlaceSection(status="empty")
    ladder, ladder_source = place_ladder(record, lookup)
    fields = {**vars(record), "ladder": ladder}
    return PlaceSection(
        status="filled" if record.floor is not None else "partial",
        value=Place(**fields, ladder_source=ladder_source, building=_building_info(lookup)),
    )


def _snapshot(move: Move, moves: MoveStore, analyses: AnalysisStore) -> MoveSnapshot:
    video = moves.get_video(move.id)
    places, schedule = moves.get_details(move.id)
    lookups = moves.get_lookups(move.id)
    route = moves.get_route(move.id)
    date_info = moves.get_date_info(move.id)
    job = analyses.current(move.id)
    data = MoveData(
        review=moves.get_review(move.id),
        room_video=RoomVideo(
            status="uploaded" if video else "empty",
            video=_media_info(video) if video else None,
        ),
        analysis=_analysis_section(job),
        inventory=load_inventory(analyses.db, job),
        origin=_place_section(places.get("origin"), lookups.get("origin")),
        destination=_place_section(places.get("destination"), lookups.get("destination")),
        route=_route_section(route),
        schedule=ScheduleSection(
            status="filled" if schedule else "empty",
            value=Schedule(move_date=schedule.move_date, start_time=schedule.start_time)
            if schedule else None,
            day=_day_info(date_info) if schedule else None,
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
    analyses: AnalysisDep,
    request: Request,
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
    # The database no longer points at the old file, so it can go, with its item photos.
    if replaced is not None:
        storage.discard(replaced)
    storage.discard_snapshots(move.id)
    # Every new video gets a fresh analysis; the previous one's late result is refused.
    analyses.start(
        move.id, media.id, demo=request.app.state.analysis_demo,
        demo_fail=media.filename.lower().startswith("fail"),
    )
    return UploadedVideo(**_media_info(media).model_dump(), replaced=replaced is not None)


@router.post(
    "/{move_id}/after-media",
    status_code=status.HTTP_201_CREATED,
    summary="이사 후 방 영상 업로드",
    description=(
        "SC-15 촬영 종료 시 이사 후 영상을 저장하고, 확정한 이삿짐 목록과 비교를 시작한다. "
        "이사 전 영상은 유지된다. 이사 후 영상을 다시 올리면 이전 이사 후 영상·비교 결과·"
        "확인 답을 버리고 새로 비교한다. 결과는 `GET /moves/{move_id}/comparison`으로 "
        "2~3초마다 확인한다."
    ),
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
        409: {"description": "이삿짐 목록을 아직 확정하지 않음"},
        413: {"description": "최대 용량 초과"},
        415: {"description": "mp4·mov가 아니거나 영상으로 읽을 수 없는 파일"},
    },
)
def upload_after_media(
    move: AccessibleMove,
    file: Annotated[UploadFile, File(description="이사 후 방 촬영 영상 (mp4, mov)")],
    moves: MoveStoreDep,
    storage: MediaStorageDep,
    analyses: AnalysisDep,
    comparisons: ComparisonDep,
    request: Request,
) -> UploadedVideo:
    job = analyses.current(move.id)
    inventory = load_inventory(analyses.db, job)
    if job is None or inventory.status != "confirmed":
        raise HTTPException(
            status.HTTP_409_CONFLICT, "이사 후 비교는 이삿짐 목록을 확정한 뒤에 할 수 있어요."
        )
    try:
        media = storage.save_video(move.id, file.file, file.filename, file.content_type)
    except UnsupportedMedia as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except MediaTooLarge as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    try:
        replaced = moves.replace_after_video(media)
    except BaseException:
        storage.discard(media)
        raise
    if replaced is not None:
        storage.discard(replaced)
    storage.discard_comparison_snapshots(move.id)
    comparisons.start(
        move.id, job.analysis_id, job.media_id, media.id,
        [item.model_dump(mode="json") for item in inventory.items],
        demo=request.app.state.analysis_demo,
        demo_fail=media.filename.lower().startswith("fail"),
    )
    return UploadedVideo(**_media_info(media).model_dump(), replaced=replaced is not None)


def _media_info(media: Media) -> MediaInfo:
    return MediaInfo(
        media_id=media.id, filename=media.filename, content_type=media.content_type,
        size_bytes=media.size_bytes, created_at=_time(media.created_at),
    )


def _analysis_section(job: AnalysisJob | None) -> AnalysisSection:
    if job is None:
        return AnalysisSection(status="not_started")
    items = json.loads(job.result_json)["items"] if job.result_json else None
    return AnalysisSection(
        status=job.status, analysis_id=job.analysis_id, progress=job.progress,
        item_count=len(items) if items is not None else None, demo=job.demo,
        error=json.loads(job.error_json) if job.error_json else None,
        updated_at=_time(job.updated_at),
    )


@router.get(
    "/{move_id}/analysis",
    summary="영상 분석 결과",
    description=(
        "AI와 합의한 분석 결과 v2.0 형식 그대로 반환하고 `demo`만 덧붙인다. 완료 전에는 "
        "`items`가 빈 목록이다. SC-06에서 `status`가 `completed`가 될 때까지 몇 초마다 "
        "`GET /moves/{move_id}`로 확인한 뒤 이 API로 목록을 받는다. `demo`가 true면 "
        "데모 분석기의 가짜 결과다."
    ),
    responses={
        401: {"description": "인증 정보 없음"},
        404: {"description": "없거나, 만료되었거나, 다른 사용자의 이사"},
    },
)
def get_analysis(move: AccessibleMove, analyses: AnalysisDep) -> AnalysisOut:
    job = analyses.current(move.id)
    if job is None:
        return AnalysisOut(status="not_started")
    if job.result_json:
        # Completed: the stored v2.0 result as the analyzer sent it.
        return AnalysisOut(**json.loads(job.result_json), demo=job.demo)
    return AnalysisOut(
        analysis_id=job.analysis_id, status=job.status, progress=job.progress,
        video={"video_id": job.media_id},
        error=json.loads(job.error_json) if job.error_json else None, demo=job.demo,
    )


@router.get(
    "/{move_id}/analysis/photos/{name}",
    summary="분석 물품 사진",
    description=(
        "분석 결과·이삿짐 목록의 `snapshots[].image_url`이 가리키는 물품 사진(JPG). AI 분석 "
        "결과의 사진은 영상에서 캡처한 것이고, 데모 결과는 샘플 사진이다. 이 이사를 열 수 "
        "있는 사용자만 받을 수 있으므로 `<img src>`에 바로 넣지 말고 인증 헤더를 붙여 받는다."
    ),
    response_class=FileResponse,
    responses={404: {"description": "없는 사진이거나 다른 사용자의 이사"}},
)
def get_analysis_photo(
    name: str, move: AccessibleMove, storage: MediaStorageDep,
) -> FileResponse:
    if name in DEMO_PHOTOS.values():
        return FileResponse(DEMO_PHOTO_DIR / name, media_type="image/jpeg")
    match = SNAPSHOT_NAME.match(name)
    # The folder is under this move's own directory, so another move's photos are unreachable.
    path = storage.snapshot_dir(move.id, match[1]) / f"{match[2]}.jpg" if match else None
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "사진을 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/jpeg")

