"""
Secrets provider abstraction for the Focused Research Agent.

Reads secrets from environment variables (via os.getenv + python-dotenv),
matching how the project already reads config today.
"""

import os
from abc import ABC, abstractmethod


class SecretsProvider(ABC):
    """Abstract contract for secrets providers."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Return the secret value for a key, or None if not found."""
        ...


class EnvSecretsProvider(SecretsProvider):
    """Reads secrets from environment variables."""

    def get(self, key: str) -> str | None:
        return os.getenv(key)


def get_secrets_provider() -> SecretsProvider:
    """Return the configured secrets provider."""
    return EnvSecretsProvider()
