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


def test_repl_back_removes_last_turn(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedChatModel(
        replies=[
            AIMessage(
                content="",
                tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call_1"}],
            ),
            AIMessage(content="5"),
            AIMessage(content="hi"),
        ]
    )
    monkeypatch.setattr(cli, "build_llm", lambda: llm)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("What is 2 + 3?\n/back\n/history\n"))

    assert cli.main([]) == 0

    output = capsys.readouterr().out
    assert "5" in output
    assert "4 messages removed" in output
    assert "(history is empty)" in output


def test_repl_back_with_no_history(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    llm = ScriptedChatModel(replies=[AIMessage(content="hi")])
    monkeypatch.setattr(cli, "build_llm", lambda: llm)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("/back\n"))

    assert cli.main([]) == 0

    output = capsys.readouterr().out
    assert "0 messages removed" in output


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
