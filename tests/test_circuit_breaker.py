"""Tests for the async circuit breaker."""

import asyncio

import pytest

from focused_research_agent.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
)


async def test_circuit_stays_closed_on_success():
    breaker = CircuitBreaker(
        name="test", failure_threshold=3, recovery_timeout_seconds=1
    )

    async def ok():
        return "success"

    result = await breaker.call(ok)
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


async def test_circuit_opens_after_threshold_failures():
    breaker = CircuitBreaker(
        name="test", failure_threshold=2, recovery_timeout_seconds=5
    )

    async def fail():
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(fail)

    assert breaker.state == CircuitState.OPEN


async def test_open_circuit_rejects_calls_immediately():
    breaker = CircuitBreaker(
        name="test", failure_threshold=1, recovery_timeout_seconds=5
    )

    async def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    async def should_not_run():
        raise AssertionError("this should never execute while circuit is open")

    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(should_not_run)


async def test_circuit_half_opens_after_recovery_timeout():
    breaker = CircuitBreaker(
        name="test", failure_threshold=1, recovery_timeout_seconds=0.1
    )

    async def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await breaker.call(fail)
    assert breaker.state == CircuitState.OPEN

    await asyncio.sleep(0.2)
    assert breaker.state == CircuitState.HALF_OPEN


async def test_successful_half_open_call_closes_circuit():
    breaker = CircuitBreaker(
        name="test", failure_threshold=1, recovery_timeout_seconds=0.1
    )

    async def fail():
        raise RuntimeError("boom")

    async def ok():
        return "recovered"

    with pytest.raises(RuntimeError):
        await breaker.call(fail)

    await asyncio.sleep(0.2)
    result = await breaker.call(ok)

    assert result == "recovered"
    assert breaker.state == CircuitState.CLOSED
