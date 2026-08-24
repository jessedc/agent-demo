# Agent-demo

A minimal, self-contained **agent loop using native tool calling**, built on LangChain, intended as a reference implementation of the modern agent pattern.

Quick start:

```bash
uv sync
uv run agent "What is 2 + 3?"
```

The loop is explicit and easy to follow:

1. Bind the tools to the chat model (`bind_tools`).
2. Build the conversation (system prompt + user query + history).
3. Ask the model for the next step.
4. If the reply contains tool calls, run each tool, append the observations, and repeat.
5. Stop when the model replies with no tool calls (a final answer) or the iteration budget runs out.

See [`FEATURE_IDEAS.md`](FEATURE_IDEAS.md) for planned next steps, likely by an agent.

## Layout

- `agent/loop.py` — the `AgentLoop` (native tool-calling loop).
- `AgentSession` in `agent/loop.py` — multi-turn history and retention limits.
- `agent/tools.py` — simple, dependency-free tools (`add`, `multiply`, `current_time`).
- `agent/config.py` — `LLMConfig` + `build_llm` for the OpenAI-compatible endpoint.
- `agent/cli.py` — the `agent` entry point.
- `tests/` — loop, parser, and config tests using a scripted fake LLM (no network).

## Configuration

The model is an OpenAI-compatible endpoint configured via environment variables (defaults shown):

| Variable          | Default                                                          |
| ----------------- | ---------------------------------------------------------------- |
| `LLM_BASE_URL`    | `http://host.your-tail-net.ts.net:8080`                    |
| `LLM_MODEL`       | `DeepSeek-V4-Flash-0731-UD-IQ2_M`                        |
| `LLM_API_KEY`     | `not-needed` (placeholder for local servers)                     |
| `LLM_TEMPERATURE` | `0.0`                                                            |

## Usage

Run a single query:

```bash
uv run agent "What is 2 + 3?"
```

Or start an interactive REPL (omit the query):

```bash
uv run agent
```

The REPL retains human, assistant, and tool messages between questions. Use `/history` to inspect the retained messages and `/clear` to start a fresh session. The default history limit is 100 messages; change it with `--max-history`:

```bash
uv run agent --max-history 20
```

See more details about how things working with `--verbose`:

```bash
uv run agent --verbose
```

## Develop

Core commands:

```bash
uv sync               # install deps + create the venv
uv run agent          # run the entry point
./scripts/check.sh    # full validation gate (format + lint + tests + types)
./scripts/fix.sh      # auto-fix format + lint
```

Use the library directly:

```python
from agent import AgentLoop, AgentSession, build_llm
from agent.tools import DEFAULT_TOOLS

agent = AgentLoop(build_llm(), DEFAULT_TOOLS)
session = AgentSession(agent, max_messages=20)
print(session.run("What is 6 times 7?"))
print(session.run("Now add one to that."))
```

Call `agent.run(...)` directly when each query should remain independent. A session drops the oldest complete turns when its limit is exceeded so that an assistant tool call is never separated from its tool result. If one complete turn alone exceeds the limit, that turn is retained intact.
