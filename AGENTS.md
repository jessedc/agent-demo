# AGENTS

Guidance for coding agents working in this repository.

## What this project is

A minimal, self-contained **native tool-calling agent loop** built on LangChain, intended as a reference implementation of the modern agent pattern. The loop is deliberately explicit and readable in one sitting — that readability is a feature, not an accident. Do not hide the loop behind a framework or add abstraction layers that obscure the data flow.

See `FEATURE_IDEAS.md` for the roadmap and the intended scope of each planned feature before starting work that might overlap with it.

## Toolchain

- **Python 3.14** (pinned in `.python-version` and `pyproject.toml`).
- **uv** is the only package manager. Do not use `pip`, `poetry`, or `pipenv` directly. Sync with `uv sync`; run anything with `uv run ...`.
- Runtime deps: `langchain-core`, `langchain-openai`. Dev deps (ruff, mypy, pyright, pytest) are in the `dev` group and installed by `uv sync`.

## Layout

```
agent/        the package (shipped by the wheel)
  __init__.py public re-exports — keep this surface small and intentional
  loop.py     AgentLoop + AgentSession — the core loop, do not obfuscate
  tools.py    dependency-free tools (add, multiply, current_time)
  config.py   LLMConfig + build_llm (OpenAI-compatible endpoint via env vars)
  cli.py      the `agent` entry point and REPL
tests/        loop/parser/config tests; uses a scripted fake LLM, no network
scripts/      check.sh (validation gate) and fix.sh (auto-fixes)
```

Root-level scripts stay run-as-scripts; only `agent/` is shipped in the wheel (`[tool.hatch.build.targets.wheel] packages`).

## Validation gate

Before declaring a task done, run the full gate and confirm it is green:

```bash
./scripts/check.sh
```

This runs, in order: `ruff format --check`, `ruff check`, `pytest`, `mypy`, `pyright`. All five must pass. To apply safe auto-fixes first:

```bash
./scripts/fix.sh        # ruff format + ruff check --fix
# or
./scripts/check.sh --fix
```

Run a single check during iteration if you want fast feedback, but always finish with the full gate. Note that `mypy` and `pyright` both type-check `agent` and `tests` — keep their include lists in sync if you add new top-level directories.

## Code style

- **Full type annotations are required.** `mypy` runs with `disallow_untyped_defs = true`; every definition must be annotated. Add `from __future__ import annotations` at the top of every module (already the convention).
- **Line length 100.** `E501` is ignored because line length is the formatter's job — run `ruff format` rather than hand-wrapping to a column.
- **Ruff lint rule set:** `E, F, W, I, UP, B, C4, SIM`. Don't add `noqa` directives unless you have a concrete reason; the existing `# noqa: BLE001` on the tool exception handler is intentional (observations must never crash the loop).
- Match the existing module-docstring-first, section-comment style in `pyproject.toml` and the docstring tone in `agent/loop.py`.

## Testing conventions

- Tests use **`ScriptedChatModel`** (`tests/fake_llm.py`) — a fake chat model that returns pre-scripted `AIMessage` replies. **No test should hit the network.** The real LLM endpoint is never reached from tests.
- A reply with `tool_calls` drives the loop to run a tool; a reply without them is the final answer. `llm.invocations` captures the message lists sent to the model for assertions.
- When adding a feature, add a scripted test alongside it. The deterministic fake-model tests are a load-bearing constraint of this project — do not replace them with network-dependent tests.
- `pytest` is configured with `testpaths = ["tests"]` and `pythonpath = ["."]`, so imports like `from agent.loop import ...` and `from tests.fake_llm import ...` work without extra setup.

## Design constraints worth respecting

- **Keep the loop readable in one sitting.** `AgentLoop._run` is the spine of the project. Prefer extending behavior with small, visible steps over introducing indirection.
- **History ownership stays outside the core loop.** `AgentLoop._run` accepts a caller-owned `history` list; `AgentSession` is the only thing that retains and trims it. Don't move retention into the loop.
- **Tools return string observations and never raise into the loop.** `_run_tool` catches all exceptions and converts them to `Error: ...` strings. Preserve this contract when adding tools or execution policy.
- **The public API is `agent/__init__.py`.** Only re-export intentionally-named symbols; update `__all__` when the surface changes.

## Configuration and secrets

- The LLM is an OpenAI-compatible endpoint configured via `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TEMPERATURE` env vars (see `agent/config.py`). Defaults point at a local server.
- **Never commit secrets.** `.gitignore` already excludes `secrets/`, `*.pem`, and `.env`. Do not put API keys, tokens, or endpoint credentials in source files — read them from the environment.

## Things to avoid

- Don't add a framework on top of the loop (LangGraph, auto-agents, etc.) — that defeats the reference-implementation purpose.
- Don't introduce network calls in tests.
- Don't add new top-level packages without updating the `mypy`/`pyright` include lists.
- Don't commit changes to `.venv`, `uv.lock` should only change when deps actually change.