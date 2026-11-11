"""Run a roborun-bench suite from the repository root.

    python -m benchmarks safety --ablate all --out examples/results/safety.json
    python -m benchmarks disturbances

The suites are not part of the roborun package; they need a clone of the repository.
"""

from __future__ import annotations

import argparse
import importlib
import sys

SUITES = ("safety", "crash", "stopping", "concurrency", "stale", "trace", "ablation",
          "model-safety", "disturbances")
MODULES = {"trace": "diagnosis", "model-safety": "model_safety"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__.splitlines()[0])
    ap.add_argument("suite", choices=SUITES)
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--workers", type=int, help="simulation processes at once (default: half the CPUs)")
    ap.add_argument("--ablate", help="also run roborun with these parts off: gate,task,world or all")
    ap.add_argument("--trace-results", default="examples/results/trace.json",
                    help="ablation: the trace suite's results, for its row")
    ap.add_argument("--model", help="trace and model-safety: the model, default openai/gpt-5.4")
    ap.add_argument("--samples", type=int, help="model-safety: samples per instruction (default 10)")
    ap.add_argument("--tasks", help="model-safety: instruction ids, e.g. 1,4,10 (default all 12)")
    ap.add_argument("--modes", help="model-safety: code,tool (default both)")
    args = ap.parse_args(argv)
    module = importlib.import_module(f"benchmarks.{MODULES.get(args.suite, args.suite)}")
    return module.main(args)


if __name__ == "__main__":
    sys.exit(main())
