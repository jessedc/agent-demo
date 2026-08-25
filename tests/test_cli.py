"""Tests for the interactive command-line session."""

from __future__ import annotations

import io

from langchain_core.messages import AIMessage

from agent import cli
from tests.fake_llm import ScriptedChatModel


def test_repl_history_and_clear_commands(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedChatModel(replies=[AIMessage(content="hello")])
    monkeypatch.setattr(cli, "build_llm", lambda: llm)
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO("Hi\n/history\n/clear\n/history\n"),
    )

    assert cli.main([]) == 0

    output = capsys.readouterr().out
    assert "hello" in output
    assert "Human: Hi" in output
    assert "Assistant: hello" in output
    assert "History cleared." in output
    assert "(history is empty)" in output


def test_ping_prints_pong_without_starting_agent(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    """A query of "ping" prints "pong" without building or running the agent."""

    def fail(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("the agent must not be built for a 'ping' query")

    monkeypatch.setattr(cli, "build_llm", fail)
    monkeypatch.setattr(cli, "AgentLoop", fail)

    assert cli.main(["ping"]) == 0

    output = capsys.readouterr().out
    assert output == "pong\n"


def test_repl_quit_command_exits_loop(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedChatModel(
        replies=[AIMessage(content="before-quit"), AIMessage(content="after-quit")]
    )
    monkeypatch.setattr(cli, "build_llm", lambda: llm)
    monkeypatch.setattr(
        cli.sys,
        "stdin",
        io.StringIO("answer me\n/quit\nshould not run\n"),
    )

    assert cli.main([]) == 0

    output = capsys.readouterr().out
    assert "before-quit" in output
    assert "after-quit" not in output
    assert "should not run" not in output
