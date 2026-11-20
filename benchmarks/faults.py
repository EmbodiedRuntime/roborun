"""The sensor-fault benchmark: what happens when a sensor is wrong, not late.

A stale value is caught by its age. A sensor that keeps reporting, wrongly, is not: its
values are fresh. It can only be caught by comparing it with another source.

Task: put the red cup on the tray. The cup slips out of the gripper while it is carried (as
in the disturbance benchmark), and the gripper's contact sensor is stuck: it goes on
reporting that it holds the cup. Eight variants.

Programs:

- ``task_based``: the program of the disturbance benchmark. Its skills notice a slip by
  what the gripper holds, so with this sensor they notice nothing;
- ``task_based_consistent``: the same, plus a relation between sources (what the gripper
  says it holds must be where the gripper is, and the fingers must be apart around it), and
  the expectation that it stays consistent while the cup is carried. When the sources
  disagree the place fails, and the program picks the cup up again where the tracker sees it.
  The sensor stays stuck; once the cup is in the gripper again, the sources agree again.

    python -m benchmarks faults
"""

from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

from benchmarks import disturbances as d
from roborun import ExpectationFailed, TaskFailed, start
from roborun.backends import mujoco as mj
from roborun.expect import consistent, disagree, relation

VARIANTS = 8
REACH = 0.06  # a held cup's centre is within this of the point between the fingertips


def held_is_in_the_gripper(v: Any) -> Any:
    """The contact sensor, the finger opening, the tracker and the arm must tell one story."""
    name = v.value("gripper.holding")
    if name is None:
        return True
    width = v.value("gripper.width")
    if width < mj.EMPTY_WIDTH:
        return disagree(f"gripper.holding says {name}, but the fingers are closed "
                        f"({width * 1000:.0f} mm apart)")
    gap = mj.dist(v.value(name), v.value("arm.ee"))
    return gap <= REACH or disagree(
        f"gripper.holding says {name}, but {name} is tracked {gap:.2f} m from the gripper")


HELD = relation("held", held_is_in_the_gripper, inputs={
    "gripper.holding": 0.1, "gripper.width": 0.1, "arm.ee": 0.1, "cup_red": mj.MAX_AGE})


async def task_based_consistent() -> None:
    await d._perceive()
    start(HELD, detached=True)
    for attempt in range(3):
        try:
            await mj.grasp("cup_red")
            await mj.place("tray").expect(consistent("held", grace=0.1, phase=("approach", "lower")))
            return
        except TaskFailed as e:
            if attempt == 2 or not isinstance(e.root, ExpectationFailed):
                raise


PROGRAMS = {"task_based": d.task_based, "task_based_consistent": task_based_consistent}


def slip_with_a_stuck_sensor(b: Any, v: int) -> dict:
    params = d.slip(b, v)
    real, sim, dropped = b._holding, b.sim, []

    def sim_and_note(action: str, **kwargs: Any) -> dict:
        out = sim(action, **kwargs)
        if action == "drop":
            dropped.append(b.now())
        return out

    b.sim = sim_and_note
    b._holding = lambda: "cup_red" if dropped else real()
    return {**params, "fault": "gripper.holding stuck at cup_red after the slip"}


SCENARIOS = {"slip_stuck_sensor": slip_with_a_stuck_sensor}


def _job(args: tuple) -> dict:
    return asdict(d.run_one(*args, programs=PROGRAMS, scenarios=SCENARIOS))


def benchmark(workers: int | None = None) -> list[dict]:
    jobs = [(s, v, p) for s in SCENARIOS for v in range(VARIANTS) for p in PROGRAMS]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    lines = ["The cup slips while carried, and the contact sensor keeps reporting it held:", "",
             "| program | cup in the tray | false success | median time | ended as |",
             "|---|---|---|---|---|"]
    for p in PROGRAMS:
        rs = [r for r in results if r["program"] == p]
        ok = [r["completion"] for r in rs if r["success"]]
        false = sum(r["outcome"] == "false success" for r in rs)
        med = f"{statistics.median(ok):.1f} s" if ok else "—"
        ends = ", ".join(sorted({r["outcome"] for r in rs}))
        lines.append(f"| `{p}` | {len(ok)}/{len(rs)} | {false}/{len(rs)} | {med} | {ends} |")
    return "\n".join(lines)


def main(args: Any) -> int:
    if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
        print("Fetching the Franka Panda model from MuJoCo Menagerie (about 36 MB)...")
        mj.fetch_menagerie()
    results = benchmark(getattr(args, "workers", None))
    print(summarize(results))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0
