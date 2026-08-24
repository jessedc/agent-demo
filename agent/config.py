"""Configuration for the chat model backing the agent loop.

The model is an OpenAI-compatible endpoint, so it is configured through
environment variables and built with :class:`langchain_openai.ChatOpenAI`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

# Defaults point at the user's OpenAI-compatible endpoint. Override any of
# these with the corresponding environment variable.
DEFAULT_BASE_URL = "http://host.tail9001.ts.net:8080"
DEFAULT_MODEL = "DeepSeek-V4-Flash-0731-UD-IQ2_M"
# Local OpenAI-compatible servers usually ignore the key; keep a placeholder
# so the OpenAI client doesn't reject the request.
DEFAULT_API_KEY = "not-needed"
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class LLMConfig:
    """Resolved settings for the OpenAI-compatible chat model."""

    base_url: str
    model: str
    api_key: str
    temperature: float

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Read configuration from ``LLM_*`` environment variables."""
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            api_key=os.environ.get("LLM_API_KEY", DEFAULT_API_KEY),
            temperature=float(os.environ.get("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))),
        )


def build_llm(config: LLMConfig | None = None) -> BaseChatModel:
    """Build a chat model from :class:`LLMConfig` (defaults to the environment)."""
    from langchain_openai import ChatOpenAI

    resolved = config or LLMConfig.from_env()
    return ChatOpenAI(
        model=resolved.model,
        base_url=resolved.base_url,
        api_key=SecretStr(resolved.api_key),
        temperature=resolved.temperature,
    )
