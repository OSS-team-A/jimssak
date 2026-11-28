from typing import Literal

from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator


class LoginRequest(BaseModel):
    """로그인 본문."""

    email: EmailStr = Field(description="가입한 이메일. 대소문자를 구분하지 않는다.")
    password: SecretStr = Field(min_length=1, max_length=128, description="비밀번호")

    @field_validator("email")
    @classmethod
    def normalize_email(cls, email: str) -> str:
        return email.lower()


class SignupRequest(LoginRequest):
    """회원가입 본문."""

    name: str | None = Field(default=None, min_length=1, max_length=50, description="회원 이름")
    phone_number: str | None = Field(
        default=None, pattern=r"^0[0-9]{1,2}-?[0-9]{3,4}-?[0-9]{4}$",
        description="국내 전화번호. 하이픈 포함 입력 가능, 숫자만 저장",
    )

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("phone_number")
    @classmethod
    def normalize_phone(cls, value: str | None) -> str | None:
        return value.replace("-", "") if value else None

    password: SecretStr = Field(
        min_length=8, max_length=128, description="비밀번호(8~128자). 서버에는 해시만 저장된다."
    )


class RefreshRequest(BaseModel):
    """토큰 갱신 본문."""

    refresh_token: SecretStr = Field(
        min_length=1, max_length=128, description="로그인 또는 이전 갱신에서 받은 리프레시 토큰"
    )


class User(BaseModel):
    """회원 정보."""

    id: int = Field(description="회원 ID")
    name: str | None = None
    phone_number: str | None = None
    email: EmailStr = Field(description="이메일(소문자로 저장)")


class LoginResponse(BaseModel):
    """로그인·토큰 갱신 결과. 두 토큰을 함께 저장하고, 갱신하면 둘 다 바꾼다."""

    access_token: str = Field(
        description="API 요청 시 `Authorization: Bearer <access_token>` 헤더로 보낸다."
    )
    refresh_token: str = Field(
        description=(
            "액세스 토큰이 만료되면 `POST /auth/refresh`로 새 토큰 쌍을 받을 때 쓴다. "
            "한 번 쓰면 폐기된다."
        )
    )
    token_type: Literal["bearer"] = Field(default="bearer", description="항상 bearer")
    expires_in: int = Field(description="액세스 토큰 유효기간(초). 기본 24시간")
    refresh_expires_in: int = Field(
        description="다시 로그인해야 할 때까지 남은 시간(초). 최초 로그인부터 최대 14일"
    )
