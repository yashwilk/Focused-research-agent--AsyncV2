"""
Auth API endpoints: register and login.

Kept deliberately minimal — this is a research-agent backend, not an
identity provider. No email verification, password reset, or OAuth flows;
those are integration points a real deployment would add via a proper
identity provider (Auth0, Cognito, etc.) rather than hand-rolled here.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.auth.schema import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from focused_research_agent.auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from focused_research_agent.database.database import get_db
from focused_research_agent.database.model import User

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@auth_router.post(
    "/register", status_code=status.HTTP_201_CREATED, response_model=TokenResponse
)
async def register(
    request: UserRegisterRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> TokenResponse:
    """Create a new user account and return an access token."""
    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        created_at=datetime.now(UTC),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    logger.info("User registered. email=%s", request.email)
    return TokenResponse(access_token=create_access_token(request.email))


@auth_router.post("/login", response_model=TokenResponse)
async def login(
    request: UserLoginRequest, db: Annotated[AsyncSession, Depends(get_db)]
) -> TokenResponse:
    """Authenticate an existing user and return an access token."""
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    logger.info("User logged in. email=%s", request.email)
    return TokenResponse(access_token=create_access_token(request.email))
