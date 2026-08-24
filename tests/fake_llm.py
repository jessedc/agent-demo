"""A scripted chat model used to test the agent loop without a real endpoint."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """A fake chat model that returns pre-scripted replies, one per call.

    ``replies`` is a list of :class:`AIMessage` objects. A message with
    ``tool_calls`` makes the loop run a tool; a message without them is the
    final answer.
    """

    replies: list[AIMessage]
    invocations: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # The scripted model ignores tool schemas; it just needs to satisfy the
        # loop's expectation that bind_tools returns a runnable yielding AIMessages.
        return cast(Runnable[LanguageModelInput, AIMessage], self)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.invocations.append(list(messages))
        reply = self.replies.pop(0)
        return ChatResult(generations=[ChatGeneration(message=reply)])
