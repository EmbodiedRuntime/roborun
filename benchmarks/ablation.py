"""Ablation: every suite on roborun with one part switched off at a time (see ``ablate``).

Prints one matrix: a row per suite's main metric, a column per arm. A part is needed if some
row breaks when only that part is off. Without a trace nothing changes while programs run,
so the trace column copies roborun's numbers except in the trace suite's row, which is read
from a trace results file (``python -m benchmarks trace --out``; the trace suite calls a model).
"""

from __future__ import annotations

import json
import os
from typing import Any

from benchmarks import ablate, concurrency, crash, safety, stale, stopping
from roborun.backends import mujoco as mj

RUNTIME_PARTS = ("gate", "task", "world")  # trace: see above
ARMS = ["baseline", "roborun", *(ablate.PREFIX + p for p in ablate.PARTS)]


def _sel(results: list[dict], arm: str, **match: Any) -> list[dict]:
    out = []
    for r in results:
        if r["arm"] != arm:
            continue
        m = {**r, **r.get("metrics", {})}
        if all(m.get(k) == v for k, v in match.items()):
            out.append(m)
    return out


def _frac(rs: list[dict], key: str) -> str:
    return f"{sum(bool(r[key]) for r in rs)}/{len(rs)}"


def _med(rs: list[dict], key: str, fmt: str = "{}") -> str:
    xs = sorted(r[key] for r in rs)
    return fmt.format(xs[len(xs) // 2]) if xs else "—"


ROWS: list[tuple[str, str, str, Any]] = [
    # (suite, part it exercises, metric, how to compute it for one arm's results)
    ("safety", "Gate", "unsafe runs that caused harm",
     lambda R, a: _frac([r for r in _sel(R, a) if r["unsafe"]], "harm")),
    ("safety", "Gate", "safe runs stopped by mistake",
     lambda R, a: f"{sum(not r['completed'] for r in _sel(R, a) if not r['unsafe'])}"
                  f"/{len([r for r in _sel(R, a) if not r['unsafe']])}"),
    ("stopping", "Task", "arm speed when cancel returned",
     lambda R, a: _med(_sel(R, a, scenario="cancel"), "speed_when_cancel_returned", "{} m/s")),
    ("stopping", "Gate, Task", "setpoints from cancelled code reaching the arm",
     lambda R, a: _med(_sel(R, a, scenario="zombie"), "commands_reaching_arm_after_cancel")),
    ("stopping", "Gate", "target switches between two agents",
     lambda R, a: _med(_sel(R, a, scenario="conflict"), "target_switches")),
    ("stopping", "Gate", "operator stop overridden by retries",
     lambda R, a: _frac(_sel(R, a, scenario="operator"), "reached_goal_anyway")),
    ("crash", "Task", "agent killed: arm kept going to the goal",
     lambda R, a: _frac(_sel(R, a, case="kill"), "reached_goal")),
    ("crash", "Task", "agent frozen: arm kept going to the goal",
     lambda R, a: _frac(_sel(R, a, case="freeze"), "reached_goal")),
    ("concurrency", "Task", "total time with 3 s decisions",
     lambda R, a: _med(_sel(R, a, latency=3.0), "total", "{:.1f} s")),
    ("stale", "World", "occluded cup: grasps closed on nothing",
     lambda R, a: str(sum(r["closes_on_nothing"] for r in _sel(R, a, scenario="occluded")))),
    ("stale", "World", "tracker died: cup placed",
     lambda R, a: _frac(_sel(R, a, scenario="tracker_died"), "placed")),
    ("stale", "World, Gate", "sensor silent, hand in path: arm moved into it",
     lambda R, a: _frac(_sel(R, a, scenario="sensor_dropout", hand_moves=True), "harm")),
    ("stale", "World", "sensor silent, hand away: arm stopped anyway",
     lambda R, a: _frac(_sel(R, a, scenario="sensor_dropout", hand_moves=False), "stopped")),
]


def run(workers: int | None = None) -> dict[str, list[dict]]:
    out = {}
    for name, mod in (("safety", safety), ("stopping", stopping), ("crash", crash),
                      ("concurrency", concurrency), ("stale", stale)):
        print(f"running {name}...", flush=True)
        out[name] = mod.benchmark(workers, ablate_parts=RUNTIME_PARTS)
    return out


def matrix(results: dict[str, list[dict]], trace_results: dict | None = None) -> str:
    def col(arm: str) -> str:
        return "roborun" if arm == ablate.PREFIX + "trace" else arm

    head = ["suite", "part", "metric", "baseline", "roborun",
            *(f"no {p}" for p in ablate.PARTS)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for suite, part, metric, f in ROWS:
        R = results[suite]
        lines.append(f"| {suite} | {part} | {metric} | "
                     + " | ".join(f(R, col(a)) for a in ARMS) + " |")
    if trace_results is not None:
        rs = trace_results["results"]

        def found(fb: str) -> str:
            sel = [r for r in rs if r["feedback"] == fb]
            return f"{sum(r['correct'] for r in sel)}/{len(sel)}"

        # Every runtime arm keeps the trace except no-trace; the baseline has none.
        cells = [found("error line"), found("trace"), found("trace"), found("trace"),
                 found("trace"), found("error line")]
        lines.append("| trace | Trace | root cause found by the model | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(args: Any) -> int:
    if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
        print("Fetching the Franka Panda model from MuJoCo Menagerie (about 36 MB)...")
        mj.fetch_menagerie()
    results = run()
    trace_results = None
    path = getattr(args, "trace_results", None)
    if path and os.path.exists(path):
        with open(path) as f:
            trace_results = json.load(f)
    print()
    print(matrix(results, trace_results))
    if getattr(args, "out", None):
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0
