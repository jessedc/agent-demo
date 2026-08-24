"""Simple, dependency-free tools for the agent loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from langchain_core.tools import BaseTool, tool

# A clock is any zero-arg callable returning a datetime. Injecting one makes
# `current_time` deterministic in tests without touching the loop or the tool
# contract.
Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    """Return the real current UTC time."""
    return datetime.now(UTC)


def make_current_time(now: Clock | None = None) -> BaseTool:
    """Build a ``current_time`` tool backed by ``now`` (defaults to real UTC time).

    Inject a fixed clock for deterministic tests::

        frozen = make_current_time(lambda: datetime(2024, 1, 1, tzinfo=UTC))
        frozen.invoke({})  # -> "2024-01-01T00:00:00+00:00"
    """
    clock = now if now is not None else _default_clock

    @tool
    def current_time() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return clock().isoformat()

    return current_time


@tool
def add(a: int, b: int) -> int:
    """Add two integers together and return the sum."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together and return the product."""
    return a * b


DEFAULT_TOOLS: tuple[BaseTool, ...] = (add, multiply, make_current_time())
