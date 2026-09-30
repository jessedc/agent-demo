# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "laya[serve]>=0.3.22",
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
"""Serve the Laya decision model (huggingface.co/convaiinnovations/laya) on an NVIDIA GB10.

Runs laya's own HTTP server (`POST /v1/systemone`, `GET /health`) with a single checkpoint
pinned: every request is answered by it, whatever language it is in or `model` it names,
so no other checkpoint is ever downloaded or loaded.

Self-contained: `uv run` builds an isolated environment from the inline metadata above,
separate from the `agent` project, so torch/transformers never touch the project lockfile.

    uv run scripts/run_laya.py                        # English only, 127.0.0.1:8000
    uv run scripts/run_laya.py --host 0.0.0.0         # reachable from other machines
    uv run scripts/run_laya.py --model multilingual   # pin a different checkpoint
    uv run scripts/run_laya.py --model auto           # laya's language routing

Set LAYA_API_KEY in the environment to require `Authorization: Bearer <key>`; without it
the server accepts any request that can reach it. Checkpoints download to the Hugging Face
cache (~/.cache/huggingface) on first run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Sequence
from typing import Any

# transformers probes for TensorFlow at import; if TF is installed that can deadlock
# model construction (see the Laya model card). Must be set before laya is imported.
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from laya import Router  # noqa: E402
from laya import serve as laya_serve  # noqa: E402

CHECKPOINTS = ["english", "multilingual", "typed-decisions"]


class PinnedRouter(Router):  # type: ignore[misc]
    """A Router that answers every request with one checkpoint.

    laya-serve passes each request's `model` field through to `predict` and
    `predict_batch`; overriding it here is what keeps a non-English request from
    loading the multilingual checkpoint.
    """

    def __init__(self, pinned: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pinned = pinned

    def predict(
        self, state: Any, questions: dict[str, Any], model: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        result: dict[str, Any] = super().predict(state, questions, model=self.pinned, **kwargs)
        return result

    def predict_batch(
        self, requests: Sequence[dict[str, Any]], *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        pinned = [{**r, "model": self.pinned} for r in requests]
        results: list[dict[str, Any]] = super().predict_batch(pinned, *args, **kwargs)
        return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model",
        choices=[*CHECKPOINTS, "auto"],
        default="english",
        help="checkpoint to serve (default: english); 'auto' keeps laya's language routing",
    )
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    return p.parse_args()


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


def build_router(model: str, device: str) -> Router:
    """Load the served checkpoint(s) and run one warmup prediction."""
    t0 = time.perf_counter()
    router: Router
    if model == "auto":
        router = Router(device=device)
        router.preload(["english", "multilingual"])
    else:
        router = PinnedRouter(model, device=device, max_loaded=1)
        router.preload([model])
    warmup = {"ok": {"type": "noul", "instructions": "Is this a test?"}}
    router.predict("warmup", warmup)
    print(f"loaded {model} in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return router


def main() -> None:
    args = parse_args()
    check_device(args.device)
    router = build_router(args.model, args.device)

    # laya-serve reads its bind address from the environment and builds its own Router;
    # hand it ours instead so the pinned checkpoint is the only one it can use.
    os.environ["LAYA_HOST"] = args.host
    os.environ["LAYA_PORT"] = str(args.port)
    laya_serve.build_router = lambda: router
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("LAYA_API_KEY"):
        print(f"warning: listening on {args.host} with no LAYA_API_KEY set", file=sys.stderr)
    print(f"serving on http://{args.host}:{args.port}/v1/systemone", file=sys.stderr)
    laya_serve.main()


if __name__ == "__main__":
    main()
