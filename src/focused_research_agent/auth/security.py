"""
Password hashing and JWT issuance/verification for the Focused Research
Agent.

Directly addresses the README's stated gap: "No authentication on API
endpoints." Every write-heavy, quota-consuming endpoint (research, chat,
report) requires a valid bearer token after this module is wired in via
auth/dependencies.py.
"""

import logging
from datetime import UTC, datetime, timedelta

import bcrypt
from jose import JWTError, jwt

from focused_research_agent.auth.config import get_auth_settings

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def create_access_token(subject: str) -> str:
    """Mint a JWT whose `sub` claim is the given user identifier (email).

    Args:
        subject: Identifier to embed as the token's subject — this
            project uses the user's email.

    Returns:
        str: Encoded JWT access token.
    """
    settings = get_auth_settings()
    expire = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(UTC)}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> str | None:
    """Decode a JWT and return its `sub` claim, or None if invalid/expired.

    Args:
        token: Encoded JWT access token.

    Returns:
        str | None: The subject (email) if the token is valid, else None.
    """
    settings = get_auth_settings()
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload.get("sub")
    except JWTError as e:
        logger.info("token_verification_failed error=%s", e)
        return None
