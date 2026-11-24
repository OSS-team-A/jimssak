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
    status_code=status.HTTP_201_CREATED,
    responses={409: {"description": "이미 가입한 이메일"}},
)
def signup(payload: SignupRequest, store: Store) -> User:
    user = store.signup(payload.email, payload.password.get_secret_value())
    if user is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "이미 가입한 이메일입니다.")
    return user


@router.post("/login", responses={401: {"description": "이메일 또는 비밀번호 불일치"}})
def login(payload: LoginRequest, store: Store, response: Response) -> LoginResponse:
    session = store.login(payload.email, payload.password.get_secret_value())
    if session is None:
        raise unauthorized("이메일 또는 비밀번호가 올바르지 않습니다.")
    response.headers["Cache-Control"] = "no-store"
    return session


@router.post(
    "/refresh",
    summary="토큰 갱신",
    description="리프레시 토큰으로 새 토큰 쌍을 발급하고 이전 토큰 쌍을 폐기합니다.",
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


@router.get("/me", responses={401: {"description": "Bearer 토큰 누락, 만료 또는 무효"}})
def me(user: Annotated[User, Depends(get_current_user)]) -> User:
    return user


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    responses={401: {"description": "Bearer 인증 헤더가 없거나 형식이 올바르지 않음"}},
)
def logout(token: Token, store: Store) -> None:
    store.logout(token)
