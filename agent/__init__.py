"""A minimal, self-contained agent loop using native tool calling, built on LangChain.

The public surface is intentionally small:

- :class:`agent.loop.AgentLoop` — the loop itself.
- :class:`agent.loop.AgentSession` — retained multi-turn history.
- :func:`agent.config.build_llm` — build a chat model from environment config.
- :data:`agent.tools.DEFAULT_TOOLS` — the built-in, dependency-free tools.
"""

from agent.config import LLMConfig, build_llm
from agent.loop import AgentError, AgentLoop, AgentSession
from agent.tools import DEFAULT_TOOLS

__all__ = [
    "AgentError",
    "AgentLoop",
    "AgentSession",
    "DEFAULT_TOOLS",
    "LLMConfig",
    "build_llm",
]
