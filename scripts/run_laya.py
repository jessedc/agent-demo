# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "laya>=0.3.22",
#     "torch>=2.14",
# ]
#
# # GB10 (DGX Spark) is aarch64 + Blackwell (sm_121) with a CUDA 13 driver. PyPI's
# # aarch64 torch wheel is CPU-only, so pull torch from PyTorch's CUDA 13.0 index there.
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#     { index = "pytorch-cu130", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
# ]
# ///
"""Run the Laya decision model (huggingface.co/convaiinnovations/laya) on an NVIDIA GB10.

Self-contained: `uv run` builds an isolated environment from the inline metadata above,
separate from the `agent` project, so torch/transformers never touch the project lockfile.

    uv run scripts/run_laya.py                        # demo ticket, routed checkpoint
    uv run scripts/run_laya.py --state "text" --model multilingual
    uv run scripts/run_laya.py --state-file ticket.json --questions-file q.json
    uv run scripts/run_laya.py --bench 50             # latency benchmark after warmup

Checkpoints download to the Hugging Face cache (~/.cache/huggingface) on first use.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# transformers probes for TensorFlow at import; if TF is installed that can deadlock
# model construction (see the Laya model card). Must be set before laya is imported.
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from laya import Router  # noqa: E402

DEMO_STATE: dict[str, str] = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": (
        "Hi, we were billed twice for March. Please refund the duplicate today "
        "or we will cancel our plan."
    ),
}

DEMO_QUESTIONS: dict[str, Any] = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--state", help="state text to evaluate (default: demo ticket)")
    src.add_argument("--state-file", type=Path, help="JSON or text file holding the state")
    p.add_argument("--questions-file", type=Path, help="JSON file of typed questions")
    p.add_argument(
        "--model",
        choices=["english", "multilingual", "typed-decisions"],
        help="force a checkpoint instead of automatic language routing",
    )
    p.add_argument("--max-len", type=int, help="token budget override (multilingual: up to 8192)")
    p.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    p.add_argument("--preload", action="store_true", help="load all three checkpoints up front")
    p.add_argument("--bench", type=int, default=0, metavar="N", help="time N predictions")
    return p.parse_args()


def load_state(args: argparse.Namespace) -> str | dict[str, Any] | list[Any]:
    if args.state is not None:
        return str(args.state)
    if args.state_file is not None:
        text = args.state_file.read_text()
        try:
            loaded: str | dict[str, Any] | list[Any] = json.loads(text)
        except json.JSONDecodeError:
            return text
        return loaded
    return DEMO_STATE


def check_device(device: str) -> None:
    """Fail early with a useful message rather than silently falling back to CPU."""
    print(f"torch {torch.__version__}  (CUDA build: {torch.version.cuda})", file=sys.stderr)
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is not available to torch. On GB10 check `nvidia-smi`, and that torch came "
            "from the cu130 index (a '+cpu' or missing CUDA version above means it did not)."
        )
    major, minor = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    free, total = torch.cuda.mem_get_info(device)
    print(
        f"device: {name}  sm_{major}{minor}  "
        f"memory {free / 2**30:.1f}/{total / 2**30:.1f} GiB free (unified on GB10)",
        file=sys.stderr,
    )
    arch = f"sm_{major}{minor}"
    archs = torch.cuda.get_arch_list()
    if arch not in archs and f"sm_{major}0" not in archs:
        print(
            f"warning: torch built for {archs}, not {arch}; kernels may JIT or fail",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()
    check_device(args.device)

    state = load_state(args)
    questions: dict[str, Any] = (
        json.loads(args.questions_file.read_text()) if args.questions_file else DEMO_QUESTIONS
    )

    t0 = time.perf_counter()
    router = Router(device=args.device, preload=args.preload)
    if not args.preload:
        # Load the checkpoint now so the first timed prediction excludes download/build.
        router.predict(state, questions, model=args.model, max_len=args.max_len)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    result = router.predict(state, questions, model=args.model, max_len=args.max_len)
    print(json.dumps(result, indent=2, default=str))

    if args.bench > 0:
        timings: list[float] = []
        for _ in range(args.bench):
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            router.predict(state, questions, model=args.model, max_len=args.max_len)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        timings.sort()
        p95 = timings[max(0, int(len(timings) * 0.95) - 1)]
        print(
            f"bench: {args.bench} runs, {len(questions)} questions/call  "
            f"p50 {statistics.median(timings):.1f} ms  p95 {p95:.1f} ms  "
            f"min {timings[0]:.1f} ms",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
