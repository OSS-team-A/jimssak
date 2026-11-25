from typing import Literal

from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator


class LoginRequest(BaseModel):
    email: EmailStr
    password: SecretStr = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, email: str) -> str:
        return email.lower()


class SignupRequest(LoginRequest):
    password: SecretStr = Field(min_length=8, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: SecretStr = Field(min_length=1, max_length=128)


class User(BaseModel):
    id: int
    email: EmailStr


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access token lifetime in seconds")
    refresh_expires_in: int = Field(description="Seconds remaining until login is required")
