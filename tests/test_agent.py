"""Tests for the native tool-calling agent loop."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.loop import AgentError, AgentLoop, AgentSession
from agent.tools import add, multiply
from tests.fake_llm import ScriptedChatModel


def _tool_call(name: str, args: dict[str, object], call_id: str) -> dict[str, object]:
    return {"name": name, "args": args, "id": call_id}


def test_agent_returns_final_answer_directly() -> None:
    llm = ScriptedChatModel(replies=[AIMessage(content="42")])
    agent = AgentLoop(llm, [add])
    assert agent.run("meaning of life") == "42"


def test_agent_uses_tool_then_returns_final_answer() -> None:
    llm = ScriptedChatModel(
        replies=[
            AIMessage(content="", tool_calls=[_tool_call("add", {"a": 2, "b": 3}, "call_1")]),
            AIMessage(content="5"),
        ]
    )
    agent = AgentLoop(llm, [add])
    assert agent.run("What is 2 + 3?") == "5"


def test_agent_chains_multiple_tool_calls() -> None:
    llm = ScriptedChatModel(
        replies=[
            AIMessage(content="", tool_calls=[_tool_call("multiply", {"a": 6, "b": 7}, "call_1")]),
            AIMessage(content="", tool_calls=[_tool_call("add", {"a": 42, "b": 1}, "call_2")]),
            AIMessage(content="43"),
        ]
    )
    agent = AgentLoop(llm, [add, multiply])
    assert agent.run("compute") == "43"


def test_agent_handles_unknown_tool() -> None:
    llm = ScriptedChatModel(
        replies=[
            AIMessage(content="", tool_calls=[_tool_call("nope", {"input": "x"}, "call_1")]),
            AIMessage(content="recovered"),
        ]
    )
    agent = AgentLoop(llm, [add])
    assert agent.run("hi") == "recovered"


def test_agent_raises_when_iterations_exhausted() -> None:
    llm = ScriptedChatModel(
        replies=[
            AIMessage(content="", tool_calls=[_tool_call("add", {"a": 1, "b": 1}, "call_1")]),
            AIMessage(content="", tool_calls=[_tool_call("add", {"a": 1, "b": 1}, "call_2")]),
        ]
    )
    agent = AgentLoop(llm, [add], max_iterations=2)
    with pytest.raises(AgentError):
        agent.run("hi")


def test_session_retains_tool_result_for_later_turn() -> None:
    llm = ScriptedChatModel(
        replies=[
            AIMessage(
                content="",
                tool_calls=[_tool_call("multiply", {"a": 6, "b": 7}, "call_1")],
            ),
            AIMessage(content="42"),
            AIMessage(content="43"),
        ]
    )
    session = AgentSession(AgentLoop(llm, [multiply]))

    assert session.run("What is 6 times 7?") == "42"
    assert session.run("Now add one to that.") == "43"

    second_turn_input = llm.invocations[-1]
    assert [message.content for message in second_turn_input[1:]] == [
        "What is 6 times 7?",
        "",
        "42",
        "42",
        "Now add one to that.",
    ]
    assert isinstance(second_turn_input[3], ToolMessage)


def test_session_clear_discards_history() -> None:
    llm = ScriptedChatModel(replies=[AIMessage(content="first"), AIMessage(content="second")])
    session = AgentSession(AgentLoop(llm, [add]))
    session.run("one")

    session.clear()
    assert session.history == ()
    session.run("two")

    assert [message.content for message in llm.invocations[-1][1:]] == ["two"]


def test_session_message_limit_drops_oldest_complete_turn() -> None:
    llm = ScriptedChatModel(
        replies=[AIMessage(content="a1"), AIMessage(content="a2"), AIMessage(content="a3")]
    )
    session = AgentSession(AgentLoop(llm, [add]), max_messages=4)

    session.run("q1")
    session.run("q2")
    session.run("q3")

    assert [message.content for message in session.history] == ["q2", "a2", "q3", "a3"]
    assert all(
        isinstance(message, (HumanMessage, AIMessage, ToolMessage)) for message in session.history
    )


def test_session_rejects_non_positive_message_limit() -> None:
    llm = ScriptedChatModel(replies=[])
    with pytest.raises(ValueError, match="max_messages"):
        AgentSession(AgentLoop(llm, [add]), max_messages=0)


def test_loop_rejects_non_positive_iteration_limit() -> None:
    llm = ScriptedChatModel(replies=[])
    with pytest.raises(ValueError, match="max_iterations"):
        AgentLoop(llm, [add], max_iterations=0)
