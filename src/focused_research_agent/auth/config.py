"""
Auth configuration for the Focused Research Agent.

Follows the same pattern as every other config module in this project
(api_config.py, database_config.py, llm_config.py): a frozen dataclass
built from environment variables with sensible defaults, loaded once.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AuthSettings:
    """Structured auth settings used by the JWT issuance/verification layer.

    Attributes:
        secret_key: HMAC signing key for JWTs. MUST be overridden in any
            real deployment — the default is a clearly-marked dev value.
        algorithm: JWT signing algorithm.
        access_token_expire_minutes: Token lifetime in minutes.
    """

    secret_key: str
    algorithm: str
    access_token_expire_minutes: int


def get_auth_settings() -> AuthSettings:
    """Load auth settings from environment variables with sensible defaults."""
    secret_key = os.getenv("AUTH_SECRET_KEY", "dev-only-insecure-key-change-me")
    algorithm = os.getenv("AUTH_ALGORITHM", "HS256")
    expire_minutes = int(os.getenv("AUTH_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    return AuthSettings(
        secret_key=secret_key,
        algorithm=algorithm,
        access_token_expire_minutes=expire_minutes,
    )
