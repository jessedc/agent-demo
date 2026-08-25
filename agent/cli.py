"""Command-line entry point for the agent loop.

Run a single query::

    uv run agent "What is 2 + 3?"

Or start an interactive REPL by omitting the query.
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent import AgentError, AgentLoop, AgentSession, build_llm
from agent.loop import _content_to_text
from agent.tools import DEFAULT_TOOLS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LangChain agent loop.")
    parser.add_argument("query", nargs="?", help="The question to ask the agent.")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum tool-calling iterations before giving up (default: 10).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each tool call and observation to stderr as the loop runs.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=100,
        help="Maximum retained messages in an interactive session (default: 100).",
    )
    args = parser.parse_args(argv)
    if args.max_history < 1:
        parser.error("--max-history must be positive")

    if args.query == "ping":
        print("pong")
        return 0

    agent = AgentLoop(
        build_llm(), DEFAULT_TOOLS, max_iterations=args.max_iterations, verbose=args.verbose
    )

    if args.query:
        try:
            print(agent.run(args.query))
        except AgentError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    session = AgentSession(agent, max_messages=args.max_history)
    print("Agent ready. Type a question, /history, /clear, or /quit (Ctrl-D to exit).")
    run_repl(session)
    return 0


def run_repl(session: AgentSession) -> None:
    """Run the interactive read-eval-print loop for an ``AgentSession``."""
    for line in sys.stdin:
        query = line.strip()
        if not query:
            continue
        if query == "/quit":
            return
        if query == "/clear":
            session.clear()
            print("History cleared.")
            continue
        if query == "/history":
            _print_history(session.history)
            continue
        try:
            print(session.run(query))
        except AgentError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def _print_history(history: tuple[BaseMessage, ...]) -> None:
    """Print retained session messages in a compact, readable form."""
    if not history:
        print("(history is empty)")
        return

    for message in history:
        content = _content_to_text(message.content)
        if isinstance(message, HumanMessage):
            print(f"Human: {content}")
        elif isinstance(message, ToolMessage):
            print(f"Tool [{message.tool_call_id}]: {content}")
        elif isinstance(message, AIMessage):
            if content:
                print(f"Assistant: {content}")
            for call in message.tool_calls:
                print(f"Assistant tool call: {call['name']}({call['args']})")


if __name__ == "__main__":
    raise SystemExit(main())
