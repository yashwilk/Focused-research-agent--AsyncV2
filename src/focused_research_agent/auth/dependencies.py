"""
FastAPI dependency for resolving the current authenticated user from a
bearer token. Applied to every research/chat/report/conversations route.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from focused_research_agent.auth.security import decode_access_token
from focused_research_agent.database.database import get_db
from focused_research_agent.database.models import User

_bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Resolve the bearer token to a User row, or raise 401.

    Args:
        credentials: Bearer token extracted from the Authorization header.
        db: Injected async database session.

    Returns:
        User: The authenticated user row.

    Raises:
        HTTPException: 401 if the token is missing, invalid, expired, or
            does not correspond to a known user.
    """
    email = decode_access_token(credentials.credentials)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
