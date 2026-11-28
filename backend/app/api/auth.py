from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..schemas.auth import LoginRequest, LoginResponse, RefreshRequest, SignupRequest, User
from ..services.auth import AuthStore


class AuthRoute(APIRoute):
    """Keep submitted credentials out of validation error responses."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                detail = [
                    {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()
                ]
                raise HTTPException(422, detail=detail) from exc

        return handle


router = APIRouter(prefix="/api/v1/auth", tags=["auth"], route_class=AuthRoute)
bearer = HTTPBearer(auto_error=False)


def get_store(request: Request) -> AuthStore:
    return request.app.state.auth_store


Store = Annotated[AuthStore, Depends(get_store)]


def unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> str:
    if credentials is None:
        raise unauthorized("로그인이 필요합니다.")
    return credentials.credentials


Token = Annotated[str, Depends(get_token)]


@router.post(
    "/signup",
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    summary="회원가입",
    description=(
        "이메일과 비밀번호(8~128자)로 가입한다(SC-02B). 이메일은 대소문자를 구분하지 않는다. "
        "가입만으로는 로그인되지 않으므로 이어서 `POST /auth/login`을 호출한다."
    ),
    responses={409: {"description": "이미 가입한 이메일"}},
)
def signup(payload: SignupRequest, store: Store) -> User:
    user = store.signup(
        payload.email, payload.password.get_secret_value(), payload.name, payload.phone_number
    )
    if user is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 가입한 이메일입니다.")
    return user


@router.post(
    "/login",
    summary="로그인",
    description=(
        "이메일과 비밀번호로 로그인해 액세스 토큰과 리프레시 토큰을 받는다(SC-02A). "
        "비회원으로 진행하던 이사가 있으면 로그인 후 `POST /moves/{move_id}/claim`으로 연결한다."
    ),
    responses={401: {"description": "이메일 또는 비밀번호 불일치"}},
)
def login(payload: LoginRequest, store: Store, response: Response) -> LoginResponse:
    session = store.login(payload.email, payload.password.get_secret_value())
    if session is None:
        raise unauthorized("이메일 또는 비밀번호가 올바르지 않습니다.")
    response.headers["Cache-Control"] = "no-store"
    return session


@router.post(
    "/refresh",
    summary="토큰 갱신",
    description=(
        "액세스 토큰이 만료되면(401) 리프레시 토큰으로 새 토큰 쌍을 받는다. 이전 액세스·리프레시 "
        "토큰은 폐기되므로 두 값을 함께 바꾼다. 같은 리프레시 토큰으로 동시에 요청하면 하나만 "
        "성공한다. 최초 로그인부터 14일이 지나면 다시 로그인해야 한다."
    ),
    responses={401: {"description": "만료되었거나 유효하지 않은 리프레시 토큰"}},
)
def refresh(payload: RefreshRequest, store: Store, response: Response) -> LoginResponse:
    session = store.refresh(payload.refresh_token.get_secret_value())
    if session is None:
        raise unauthorized("리프레시 토큰이 만료되었거나 유효하지 않습니다.")
    response.headers["Cache-Control"] = "no-store"
    return session


def get_current_user(token: Token, store: Store) -> User:
    user = store.current_user(token)
    if user is None:
        raise unauthorized("세션이 만료되었거나 유효하지 않습니다.")
    return user


@router.get(
    "/me",
    response_model_exclude_none=True,
    summary="내 정보",
    description="회원 ID와 이메일, 가입 시 입력한 이름과 전화번호를 조회한다.",
    responses={401: {"description": "Bearer 토큰 누락, 만료 또는 무효"}},
)
def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="로그아웃",
    description=(
        "현재 로그인의 액세스 토큰과 리프레시 토큰을 함께 폐기한다. 다른 기기의 로그인은 "
        "유지된다. 이후 같은 토큰으로 요청하면 401이다."
    ),
    responses={401: {"description": "Bearer 인증 헤더가 없거나 형식이 올바르지 않음"}},
)
def logout(token: Token, store: Store) -> None:
    store.logout(token)
