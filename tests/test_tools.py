"""Tests for the built-in tools."""

from __future__ import annotations

from datetime import UTC, datetime

from agent.tools import DEFAULT_TOOLS, make_current_time


def test_current_time_uses_injected_clock() -> None:
    fixed = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    tool = make_current_time(lambda: fixed)
    assert tool.invoke({}) == "2024-01-02T03:04:05+00:00"


def test_current_time_default_returns_valid_iso_utc() -> None:
    tool = make_current_time()
    parsed = datetime.fromisoformat(tool.invoke({}))
    assert parsed.tzinfo is not None


def test_default_tools_include_current_time() -> None:
    assert "current_time" in {tool.name for tool in DEFAULT_TOOLS}
