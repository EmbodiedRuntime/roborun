"""The disturbance benchmark: what happens to a pick-and-place when the world misbehaves.

Task: put the red cup on the tray. Five kinds of disturbance, eight variants each:

- ``repeated_push``: the cup is pushed three times, 2 s apart;
- ``moving``: the cup slides steadily (2 or 4 cm/s) for 4 s, like a conveyor;
- ``occluded_push``: the tracker loses the cup for 2 s, and the cup is pushed meanwhile;
- ``slip``: the cup slips out of the gripper while it is carried;
- ``removed``: someone takes the cup away. Nothing can succeed; what matters is giving up
  soon, for the right reason, without closing the gripper on nothing.

Four programs, built from the same skills:

- ``blocking``: perceive once, ``grasp_blocking``, ``place_blocking``;
- ``blocking_retry``: retry the grasp after a miss, perceiving again (up to three tries);
- ``verify_retry``: the strongest usual fix: after each pick-and-place, check that the cup is in
  the tray, and redo everything if not (up to three rounds);
- ``task_based``: ``grasp("cup_red")`` follows the tracked cup and holds still while its
  data is stale; ``place`` notices a slip on the way; retry after a miss or a slip.

    python -m benchmarks disturbances [--out results.json]
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from roborun import TaskFailed, run_sim, world
from roborun.backends import mujoco as mj
from roborun.demos.cup_nudge import SCENE, _perceive, in_tray

UNTIL = 40.0


# -- programs --------------------------------------------------------------------------------


async def blocking() -> None:
    await _perceive()
    cup = world.get("cup_red").value
    await mj.grasp_blocking(cup)
    await mj.place_blocking("tray")


async def blocking_retry() -> None:
    await _perceive()
    for attempt in range(3):
        try:
            await mj.grasp_blocking(world.get("cup_red").value)
            break
        except TaskFailed:
            if attempt == 2:
                raise
    await mj.place_blocking("tray")


class NotInTray(Exception):
    pass


async def verify_retry() -> None:
    await _perceive()
    for _ in range(3):
        try:
            await mj.grasp_blocking(world.get("cup_red").value)
            await mj.place_blocking("tray")
        except TaskFailed:
            continue
        if in_tray(world.get("cup_red").value):
            return
    raise NotInTray("the cup is not in the tray after three rounds")


async def task_based() -> None:
    await _perceive()
    for attempt in range(3):
        try:
            await mj.grasp("cup_red")
            await mj.place("tray")
            return
        except TaskFailed as e:
            if attempt == 2 or not isinstance(e.root, (mj.GraspMissed, mj.Dropped,
                                                       mj.NothingHeld)):
                raise


PROGRAMS: dict[str, Callable] = {
    "blocking": blocking, "blocking_retry": blocking_retry,
    "verify_retry": verify_retry, "task_based": task_based,
}


# -- disturbances ------------------------------------------------------------------------------

CUP = (0.5, -0.1)


def _quietly(fn: Callable[[], Any]) -> Callable[[], None]:
    def run() -> None:
        try:
            fn()
        except ValueError:
            pass  # e.g. a push while the cup is in the gripper: nobody pushes it out

    return run


def repeated_push(b: Any, v: int) -> dict:
    t0 = (0.5, 1.0, 1.5, 2.0)[v % 4]
    sign = 1 if v < 4 else -1
    spots = [(0.45, -0.1 + 0.08 * sign), (0.55, -0.1), (0.48, -0.1 - 0.06 * sign)]
    for i, xy in enumerate(spots):
        b.at(t0 + 2.0 * i, _quietly(lambda xy=xy: b.slide("cup_red", xy, 0.4)))
    return {"first_push": t0, "pushes": 3}


def moving(b: Any, v: int) -> dict:
    speed = (0.02, 0.04)[v % 2]
    sign = 1 if (v // 2) % 2 == 0 else -1
    start = (0.3, 1.0)[v // 4]
    b.at(start, _quietly(lambda: b.sim("move", object="cup_red", velocity=(0.0, sign * speed),
                                       duration=4.0)))
    return {"speed": speed, "start": start}


def occluded_push(b: Any, v: int) -> dict:
    t0 = (0.5, 1.0, 1.5, 2.0)[v % 4]
    xy = (0.45, -0.02) if v < 4 else (0.55, -0.18)
    b.at(t0, _quietly(lambda: b.sim("occlude", object="cup_red", duration=2.0)))
    b.at(t0 + 0.5, _quietly(lambda: b.slide("cup_red", xy, 0.4)))
    return {"occluded_at": t0}


def slip(b: Any, v: int) -> dict:
    after = (0.3, 0.6, 1.0, 1.5)[v % 4]  # seconds of carrying before it slips
    side = 0.07 if v < 4 else -0.07
    state = {"held_since": None, "done": False}

    def watch(backend: Any) -> None:
        if state["done"]:
            return
        if backend._holding() == "cup_red" and not backend.moving:
            state["held_since"] = None  # still closing the gripper, not yet carrying
        elif backend._holding() == "cup_red":
            state["held_since"] = state["held_since"] or backend.now()
            if backend.now() - state["held_since"] >= after:
                backend.sim("drop", object="cup_red", side=side)
                state["done"] = True

    b.watchers.append(watch)
    return {"slips_after": after}


def removed(b: Any, v: int) -> dict:
    t = (0.5, 1.0, 1.5, 2.0)[v % 4]
    b.at(t, _quietly(lambda: b.sim("remove", object="cup_red")))
    b.removed_at = t
    return {"removed_at": t}


SCENARIOS: dict[str, Callable] = {
    "repeated_push": repeated_push, "moving": moving, "occluded_push": occluded_push,
    "slip": slip, "removed": removed,
}
VARIANTS = 8


# -- one run -------------------------------------------------------------------------------------


@dataclass
class Result:
    scenario: str
    variant: int
    program: str
    success: bool
    completion: float | None
    outcome: str
    closes_on_nothing: int  # gripper closings that caught nothing
    params: dict


def run_one(scenario: str, variant: int, program: str, *, view: bool = False) -> Result:
    backend = mj.MujocoBackend(objects=SCENE, view=view)
    backend.removed_at = None
    params = SCENARIOS[scenario](backend, variant)
    holding: list[tuple[float, str | None]] = []
    step = backend.step

    def stepped(dt: float) -> None:
        step(dt)
        holding.append((backend.now(), backend._holding()))

    backend.step = stepped
    try:
        task = asyncio.run(run_sim(PROGRAMS[program], backend=backend, until=UNTIL))
    finally:
        backend.close()
    placed = task.state.value == "succeeded" and in_tray(backend.truth("cup_red"))
    closes = [t for t, c in backend.log if c.kind == "effector" and c.payload.get("close")]

    def held_at(t: float) -> str | None:
        later = [h for ts, h in holding if ts >= t]
        return later[0] if later else None

    on_nothing = sum(1 for t in closes if held_at(t + 0.5) is None)
    if placed:
        outcome = "placed"
    elif task.error is not None:
        root = task.error.root if isinstance(task.error, TaskFailed) else task.error
        outcome = type(root).__name__
    else:
        outcome = ("false success" if task.state.value == "succeeded"  # it said done; it was not
                   else f"not in tray ({task.state.value})")
    return Result(scenario, variant, program, placed,
                  round(task.ended_at, 2) if task.ended_at is not None else None,
                  outcome, on_nothing, params)


def _job(args: tuple) -> dict:
    return asdict(run_one(*args))


def benchmark(workers: int | None = None) -> list[dict]:
    jobs = [(s, v, p) for s in SCENARIOS for v in range(VARIANTS) for p in PROGRAMS]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    def med(xs: list[float]) -> str:
        return f"{statistics.median(xs):.1f} s" if xs else "—"

    progs = list(PROGRAMS)
    lines = [("Success (cup in the tray) and median completion time of the successes; "
             "false successes (the program reported success, but the cup is not in the tray) "
             "in brackets:"), "",
             "| disturbance | " + " | ".join(f"`{p}`" for p in progs) + " |",
             "|---" * (len(progs) + 1) + "|"]
    for s in SCENARIOS:
        if s == "removed":
            continue
        cells = []
        for p in progs:
            rs = [r for r in results if r["scenario"] == s and r["program"] == p]
            ok = [r["completion"] for r in rs if r["success"]]
            false = sum(r["outcome"] == "false success" for r in rs)
            cells.append(f"{len(ok)}/{len(rs)}, {med(ok)}" + (f" [{false} false]" if false else ""))
        lines.append(f"| {s.replace('_', ' ')} | " + " | ".join(cells) + " |")
    lines += ["", ("Cup removed: nothing can succeed. When each program gave up (after the "
              "removal), why, and how often it closed the gripper on nothing:"), "",
              "| program | median time to give up | reported | closings on nothing |",
              "|---|---|---|---|"]
    for p in progs:
        rs = [r for r in results if r["scenario"] == "removed" and r["program"] == p]
        waits = [r["completion"] - r["params"]["removed_at"] for r in rs if r["completion"]]
        reasons = sorted({r["outcome"] for r in rs})
        lines.append(f"| `{p}` | {med(waits)} | {', '.join(reasons)} | "
                     f"{sum(r['closes_on_nothing'] for r in rs)} in {len(rs)} runs |")
    return "\n".join(lines)


def main(args: Any) -> int:
    if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
        print("Fetching the Franka Panda model from MuJoCo Menagerie (about 36 MB)...")
        mj.fetch_menagerie()
    results = benchmark()
    print(summarize(results))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0
