r"""A minimal, self-contained agent loop using native tool calling.

The loop is deliberately explicit so it can serve as a reference
implementation of the modern agent pattern:

1. Bind the tools to the chat model (``bind_tools``).
2. Build the conversation so far (system prompt + retained session history).
3. Ask the model for the next step.
4. If the reply contains tool calls, run each tool and append the
   observations as ``ToolMessage``\ s, then repeat.
5. Stop when the model replies with no tool calls (a final answer) or the
   iteration budget runs out.

Unlike a text-protocol (ReAct) loop, the model decides to call tools natively,
so the loop never has to parse ``Action``/``Action Input`` text.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool


class AgentError(RuntimeError):
    """Raised when the agent cannot produce a final answer."""


def default_system_prompt() -> str:
    """Build the system prompt for the agent."""
    return (
        "You are a helpful assistant that answers questions. "
        "Use the available tools when they help, and answer directly otherwise."
    )


def _content_to_text(content: str | list[Any]) -> str:
    """Normalize a message's ``content`` (string or list of parts) to text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


class AgentLoop:
    """Run a native-tool-calling agent loop against a chat model and tools."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        *,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ) -> None:
        """Bind the tools to the model and store the loop's configuration."""
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.llm: Runnable[Any, AIMessage] = llm.bind_tools(list(tools))
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt or default_system_prompt()
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, query: str) -> str:
        """Answer one independent ``query`` without retaining its messages."""
        return self._run(query, [])

    def _run(self, query: str, history: list[BaseMessage]) -> str:
        """Run one turn, appending every message to caller-owned ``history``."""
        history.append(HumanMessage(content=query))
        for _ in range(self.max_iterations):
            messages = [SystemMessage(content=self.system_prompt), *history]
            response = self.llm.invoke(messages)
            history.append(response)
            if not response.tool_calls:
                final = _content_to_text(response.content)
                self._log(f"Final answer: {final}")
                return final
            for call in response.tool_calls:
                self._log(f"Tool call: {call['name']}({call['args']})")
                observation = self._run_tool(call["name"], call["args"])
                self._log(f"  -> {observation}")
                history.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=call.get("id") or f"call_{len(history)}",
                    )
                )

        raise AgentError(
            f"Agent did not produce a final answer within {self.max_iterations} iterations."
        )

    def _log(self, message: str) -> None:
        """Emit a trace line to stderr when verbose mode is enabled."""
        if self.verbose:
            print(message, file=sys.stderr)

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name, returning a string observation (never raising)."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available tools: {', '.join(self.tools)}"
        try:
            result = tool.invoke(args)
        except Exception as exc:  # noqa: BLE001 - observations must never crash the loop
            return f"Error: {type(exc).__name__}: {exc}"
        return str(result)


class AgentSession:
    """Own conversation history across multiple turns of an :class:`AgentLoop`.

    The message limit is applied by dropping the oldest complete turns. A single
    turn that uses many tools is kept intact even if it alone exceeds the limit,
    because splitting a tool-call exchange produces invalid model input.
    """

    def __init__(self, agent: AgentLoop, *, max_messages: int | None = 100) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be positive or None")
        self.agent = agent
        self.max_messages = max_messages
        self._messages: list[BaseMessage] = []

    @property
    def history(self) -> tuple[BaseMessage, ...]:
        """Return an immutable view of the retained conversation messages."""
        return tuple(self._messages)

    def run(self, query: str) -> str:
        """Answer ``query`` using prior turns and retain the completed turn."""
        try:
            return self.agent._run(query, self._messages)
        finally:
            self._trim_history()

    def clear(self) -> None:
        """Discard all retained conversation messages."""
        self._messages.clear()

    def back(self) -> int:
        """Remove the last complete turn and return how many messages were dropped.

        A turn is the last user message together with every message that followed
        it (the assistant's tool calls, tool observations, and final answer). If
        no user message has been sent yet, nothing is removed and zero is returned.
        """
        last_human = -1
        for index, message in enumerate(self._messages):
            if isinstance(message, HumanMessage):
                last_human = index
        if last_human == -1:
            return 0
        removed = len(self._messages) - last_human
        del self._messages[last_human:]
        return removed

    def _trim_history(self) -> None:
        """Drop oldest complete turns until history fits the configured limit."""
        if self.max_messages is None:
            return

        while len(self._messages) > self.max_messages:
            next_turn = next(
                (
                    index
                    for index, message in enumerate(self._messages[1:], start=1)
                    if isinstance(message, HumanMessage)
                ),
                None,
            )
            if next_turn is None:
                return
            del self._messages[:next_turn]
