"""Pydantic request/response schemas for the auth API."""

from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """Request body for creating a new user account."""

    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    """Request body for logging in."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response returned after successful registration or login."""

    access_token: str
    token_type: str = "bearer"
