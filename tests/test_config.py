"""Tests for agent configuration."""

from __future__ import annotations

import pytest

from agent.config import LLMConfig


def test_config_defaults() -> None:
    config = LLMConfig(
        base_url="http://example.test:8080",
        model="some-model",
        api_key="key",
        temperature=0.0,
    )
    assert config.base_url == "http://example.test:8080"
    assert config.model == "some-model"
    assert config.api_key == "key"
    assert config.temperature == 0.0


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://env.test:1234")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    config = LLMConfig.from_env()
    assert config.base_url == "http://env.test:1234"
    assert config.model == "env-model"
    assert config.api_key == "env-key"
    assert config.temperature == 0.5
