"""Concurrency: a slow planner and a moving arm, one after the other or at the same time.

A four-step task (red cup to the tray, then the blue cup) is driven by a planner whose every
decision takes ``D`` seconds of simulated time, standing in for a model call. The plan does not
depend on how each step turns out, so the next decision can be made while the current step
runs, if the steps let it:

- ``baseline``: blocking calls. Decide, run the step to the end, decide again.
- ``roborun``: each step starts as a task and returns at once; the planner decides the next
  step while the arm moves, and the step starts as soon as the arm is free.

Main metric: total time; also how long the arm stood idle waiting for a decision.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from benchmarks import ablate
from roborun import now, run_sim, sleep, start, world
from roborun.backends import mujoco as mj

SCENE = {
    "cup_red": {"pos": [0.5, -0.1], "color": "red"},
    "cup_blue": {"pos": [0.42, 0.12], "color": "blue"},
    "tray": {"kind": "tray", "pos": [0.3, 0.38]},
}
SPOTS = {"cup_red": (0.25, 0.38, 0.053), "cup_blue": (0.35, 0.38, 0.053)}
STEPS = [("grasp", "cup_red"), ("place", "cup_red"), ("grasp", "cup_blue"), ("place", "cup_blue")]
LATENCIES = (1.0, 2.0, 3.0, 5.0)


@dataclass
class Result:
    arm: str
    latency: float
    total: float
    idle: float  # seconds the arm stood still waiting for the next step
    placed: int


async def decide(k: int, latency: float) -> tuple[str, str]:
    await sleep(latency)  # a model call
    return STEPS[k]


def body(step: tuple[str, str]) -> Any:
    verb, cup = step
    return mj.grasp(cup) if verb == "grasp" else mj.place(SPOTS[cup])


async def blocking(latency: float) -> None:
    for k in range(len(STEPS)):
        step = await decide(k, latency)
        await start(body(step))


async def overlapped(latency: float) -> None:
    step = await decide(0, latency)
    for k in range(len(STEPS)):
        running = start(body(step))
        nxt = asyncio.ensure_future(decide(k + 1, latency)) if k + 1 < len(STEPS) else None
        await running
        if nxt is not None:
            step = await nxt


def run_one(arm: str, latency: float) -> Result:
    backend = mj.MujocoBackend(objects=SCENE)
    idle = [0.0]
    started = [None]

    def watch(b: mj.MujocoBackend) -> None:
        if b.moving or b._ee_speed > 0.002:
            started[0] = started[0] if started[0] is not None else b.now()
        elif started[0] is not None:
            idle[0] += 0.01  # the run's step; counted once the arm has started working

    backend.watchers.append(watch)

    async def main() -> float:
        for cup in ("cup_red", "cup_blue", "tray"):
            start(mj.track(cup), detached=True)
        await world.next("cup_blue")
        t0 = now()
        await (blocking(latency) if arm == "baseline" else overlapped(latency))
        return now() - t0

    try:
        with ablate.without(ablate.part_of(arm)):
            task = asyncio.run(run_sim(main, backend=backend, until=120))
    finally:
        backend.close()
    placed = sum(abs(backend.truth(c)[0] - SPOTS[c][0]) < 0.03 and abs(backend.truth(c)[1] - SPOTS[c][1]) < 0.03
                 for c in SPOTS)
    # idle counted after the last motion too; take off the final settling tail
    return Result(arm, latency, round(task.result, 2), round(idle[0], 2), placed)


def _job(args: tuple) -> dict:
    return asdict(run_one(*args))


def benchmark(workers: int | None = None, ablate_parts: tuple[str, ...] = ()) -> list[dict]:
    jobs = [(arm, d) for d in LATENCIES for arm in ablate.arms(ablate_parts)]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in results))
    lines = [("Four steps (two cups onto the tray); each decision takes D seconds. Total time "
             "(arm idle waiting for a decision):"), "",
             "| decision time D | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for d in LATENCIES:
        cells = []
        for a in arms:
            r = next(r for r in results if r["arm"] == a and r["latency"] == d)
            cells.append(f"{r['total']:.1f} s ({r['idle']:.1f} s)")
        lines.append(f"| {d:.0f} s | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(args: Any) -> int:
    if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
        print("Fetching the Franka Panda model from MuJoCo Menagerie (about 36 MB)...")
        mj.fetch_menagerie()
    results = benchmark(ablate_parts=ablate.parse(getattr(args, "ablate", None)))
    print(summarize(results))
    if getattr(args, "out", None):
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0
