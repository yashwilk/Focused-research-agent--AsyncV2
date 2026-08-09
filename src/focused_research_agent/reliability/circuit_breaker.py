"""A minimal async circuit breaker for external provider calls.

Directly addresses the README's stated gap: "No circuit breaker for
provider outages." When a provider (Groq, Tavily, Ollama) starts failing
repeatedly, this stops hammering it for a cooldown period instead of
piling up slow timeouts on every single request — both protecting the
provider and giving the caller a fast, clear failure instead of a slow one.

States:
    CLOSED     — normal operation, calls pass through.
    OPEN       — failure threshold exceeded; calls fail immediately
                 without hitting the provider, until the recovery timeout
                 elapses.
    HALF_OPEN  — recovery timeout elapsed; the next call is allowed through
                 as a trial. Success closes the breaker; failure re-opens it.

This is intentionally simple (in-process, no shared state across
workers) — appropriate for a single-instance deployment. A multi-instance
deployment would back this with Redis (SET failure counters with TTL)
instead of local memory; the interface below would not need to change.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, TypeVar

from focused_research_agent.core.metrics import (
    circuit_breaker_state,
    provider_call_duration_seconds,
)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""

    def __init__(self, provider_name: str, retry_after_seconds: float):
        self.provider_name = provider_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Circuit breaker open for '{provider_name}'. "
            f"Retry after {retry_after_seconds:.1f}s."
        )


@dataclass
class CircuitBreaker:
    """Per-provider async circuit breaker.

    Attributes:
        name: Identifies the protected provider in logs/errors.
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout_seconds: How long to stay OPEN before trying a
            HALF_OPEN trial call.
    """

    name: str
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            was_already_open = self._state == CircuitState.OPEN
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            if not was_already_open:
                circuit_breaker_state.labels(provider=self.name).inc()

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Run an async callable through the circuit breaker.

        Args:
            fn: A zero-argument async callable to execute.

        Returns:
            The callable's return value.

        Raises:
            CircuitBreakerOpenError: If the circuit is open and the
                recovery timeout hasn't elapsed yet.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            remaining = self.recovery_timeout_seconds - (
                time.monotonic() - self._opened_at
            )
            raise CircuitBreakerOpenError(self.name, max(remaining, 0.0))

        start = time.monotonic()
        try:
            result = await fn()
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result
        finally:
            provider_call_duration_seconds.labels(provider=self.name).observe(
                time.monotonic() - start
            )


# One breaker per provider, shared across requests within a process.
_breakers: dict[str, CircuitBreaker] = {}


# One breaker per provider, shared across requests within a process.
_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str, failure_threshold: int = 5, recovery_timeout_seconds: float = 30.0
) -> CircuitBreaker:
    """Return the process-wide circuit breaker for a named provider."""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout_seconds=recovery_timeout_seconds,
        )
    return _breakers[name]
