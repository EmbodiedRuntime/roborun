"""Stopping: four ways hand-rolled stopping goes wrong, on a baseline and on roborun.

The baseline sends commands straight to the robot driver (as a program using the driver's API
does) and cancels with ``asyncio.Task.cancel``, sending a stop where a careful program would.
roborun sends the same commands through the Gate. Every scenario runs on the same simulated
arm, in lockstep, so results are deterministic.

1. ``cancel``: cancel a move partway. When cancel returns, is the arm still moving?
2. ``zombie``: the cancelled program swallows the cancellation and keeps sending setpoints, as
   code with a broad ``except`` does. How many reach the arm?
3. ``conflict``: a second agent starts commanding the arm while the first is mid-motion. How
   often does the arm's target switch between them?
4. ``operator``: the operator stops the arm while a program with retry logic runs. Does the
   program set the arm moving again?
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from benchmarks import ablate
from roborun import action, now, run_sim, runtime, send, sleep, start, world
from roborun.backends import mujoco as mj
from roborun.checks import Command

SCENE = {"cup_red": {"pos": [0.48, -0.12], "color": "red"}}
P0, P1 = (0.30, -0.30, 0.20), (0.62, 0.30, 0.15)  # a 0.68 m move, about 3 s
PA, PB = (0.55, -0.25, 0.20), (0.35, 0.30, 0.20)
VARIANTS = 6
UNTIL = 15.0


# -- the two ways of commanding the arm ----------------------------------------------------


async def command(arm: str, kind: str, **payload: Any) -> None:
    if arm == "baseline":
        runtime().backend.execute(Command(kind, "arm", payload))  # the driver's API
    else:
        await send(kind, "arm", **payload)


class Stalled(Exception):
    pass


async def arrive(target: tuple, stall: float = 0.5) -> None:
    """Wait until the arm is at ``target``; raise ``Stalled`` if it stops short for ``stall`` s."""
    still_since = None
    while world.get("arm.moving").value or math.dist(world.get("arm.ee").value, target) > mj.REACH_TOL:
        if world.get("arm.moving").value:
            still_since = None
        else:
            still_since = still_since if still_since is not None else now()
            if now() - still_since > stall:
                raise Stalled(f"stopped short of {target}")
        await sleep(mj.CYCLE)


async def move(arm: str, target: tuple) -> None:
    try:
        await command(arm, "goal", target=target)
        await arrive(target)
    except asyncio.CancelledError:
        await command(arm, "stop")  # the careful thing: stop on cancel
        if arm != "baseline":
            await mj.wait_stopped()  # roborun's cancel ends when the arm is at rest
        raise


roborun_move = action(resources={"arm"})(move)


def spawn(arm: str, body: Any) -> Any:
    """Start ``body``: as an asyncio task on the baseline, as a roborun task on roborun."""
    return asyncio.ensure_future(body) if arm == "baseline" else start(body)


async def cancel(arm: str, task: Any) -> None:
    if arm == "baseline":
        task.cancel()  # returns at once, like a goal-cancel request
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.sleep(0)
    else:
        await task.cancel_and_wait()  # returns when the arm is at rest


# -- scenarios ------------------------------------------------------------------------------


@dataclass
class Result:
    scenario: str
    variant: int
    arm: str
    metrics: dict = field(default_factory=dict)


async def _to_start(arm: str) -> None:
    t = spawn(arm, roborun_move(arm, P0) if arm != "baseline" else move(arm, P0))
    await t


def _moved(track: list, t0: float) -> float:
    pts = [p for t, p in track if t >= t0]
    return sum(math.dist(a, b) for a, b in itertools.pairwise(pts))


async def s_cancel(arm: str, v: int, track: list, out: dict) -> None:
    await _to_start(arm)
    t = spawn(arm, roborun_move(arm, P1) if arm != "baseline" else move(arm, P1))
    await sleep(0.6 + 0.3 * v)
    await cancel(arm, t)
    out["speed_when_cancel_returned"] = round(runtime().backend._ee_speed, 3)
    t_ret = now()
    await sleep(2.0)
    out["moved_after_return"] = round(_moved(track, t_ret), 3)


async def s_zombie(arm: str, v: int, track: list, out: dict) -> None:
    await _to_start(arm)

    async def zombie() -> None:
        p, n = list(P0), math.dist(P0, P1)
        step = [0.012 * (b - a) / n for a, b in zip(P0, P1)]
        for _ in range(60):  # a setpoint loop that swallows its cancellation
            try:
                p = [a + s for a, s in zip(p, step)]
                await command(arm, "setpoint", target=tuple(p))
                await sleep(0.05)
            except BaseException:  # noqa: BLE001 - the bug under test
                await asyncio.sleep(0)
                continue

    body = action(resources={"arm"})(zombie)() if arm != "baseline" else zombie()
    t = spawn(arm, body)
    await sleep(0.5 + 0.2 * v)
    t_cancel = now()
    log0 = len(runtime().backend.log)
    t.cancel()
    await sleep(3.0)
    after = runtime().backend.log[log0:]
    out["commands_reaching_arm_after_cancel"] = sum(1 for _, c in after if c.kind == "setpoint")
    out["moved_after_cancel"] = round(_moved(track, t_cancel + 0.3), 3)  # beyond braking


async def s_conflict(arm: str, v: int, track: list, out: dict) -> None:
    await _to_start(arm)

    async def stream(who: str, goal: tuple, seconds: float) -> None:
        end = now() + seconds
        while now() < end:
            ee = world.get("arm.ee").value
            d = math.dist(ee, goal)
            k = min(1.0, 0.02 / d) if d > 0 else 1.0
            await command(arm, "setpoint", target=tuple(e + k * (g - e) for e, g in zip(ee, goal)),
                          who=who)
            await sleep(0.05)

    def body(who: str, goal: tuple, seconds: float) -> Any:
        if arm != "baseline":
            return action(resources={"arm"})(stream)(who, goal, seconds)
        return stream(who, goal, seconds)

    log0 = len(runtime().backend.log)
    a = spawn(arm, body("A", PA, 2.5))
    await sleep(0.5 + 0.2 * v)
    b = spawn(arm, body("B", PB, 2.5))
    await a
    await b
    whos = [c.payload.get("who") for _, c in runtime().backend.log[log0:] if c.kind == "setpoint"]
    out["target_switches"] = sum(1 for x, y in itertools.pairwise(whos) if x != y)


async def s_operator(arm: str, v: int, track: list, out: dict) -> None:
    await _to_start(arm)

    async def stubborn() -> None:  # retries whatever stops it, until it gets there
        while math.dist(world.get("arm.ee").value, P1) > mj.REACH_TOL:
            try:
                if arm != "baseline":
                    await start(roborun_move(arm, P1))
                else:
                    await move(arm, P1)
            except Exception:  # noqa: BLE001 - the program's retry logic
                await sleep(0.2)

    t = spawn(arm, stubborn())
    await sleep(0.6 + 0.3 * v)
    t_stop = now()
    if arm == "baseline":
        runtime().backend.execute(Command("stop", "arm", {}))  # the e-stop button on the driver
    else:
        runtime().gate.operator_stop(["arm"])
    await sleep(4.0)
    out["moved_after_stop"] = round(_moved(track, t_stop), 3)
    out["reached_goal_anyway"] = math.dist(world.get("arm.ee").value, P1) < 0.01
    t.cancel()


SCENARIOS = {"cancel": s_cancel, "zombie": s_zombie, "conflict": s_conflict, "operator": s_operator}


def run_one(scenario: str, variant: int, arm: str) -> Result:
    backend = mj.MujocoBackend(objects=SCENE)
    track: list = []
    backend.watchers.append(lambda b: track.append((b.now(), b.ee)))
    out: dict = {}

    async def main() -> None:
        await SCENARIOS[scenario](arm, variant, track, out)

    try:
        with ablate.without(ablate.part_of(arm)):
            asyncio.run(run_sim(main, backend=backend, until=UNTIL))
    finally:
        backend.close()
    return Result(scenario, variant, arm, out)


def _job(args: tuple) -> dict:
    return asdict(run_one(*args))


def benchmark(workers: int | None = None, ablate_parts: tuple[str, ...] = ()) -> list[dict]:
    jobs = [(s, v, arm) for s in SCENARIOS for v in range(VARIANTS) for arm in ablate.arms(ablate_parts)]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in results))

    def med(scenario: str, arm: str, key: str) -> Any:
        xs = sorted(r["metrics"].get(key) for r in results
                    if r["scenario"] == scenario and r["arm"] == arm)
        return xs[len(xs) // 2]

    def count(scenario: str, arm: str, key: str) -> str:
        rs = [r for r in results if r["scenario"] == scenario and r["arm"] == arm]
        return f"{sum(bool(r['metrics'].get(key)) for r in rs)}/{len(rs)}"

    rows = [
        ("cancel a move", "arm speed when cancel returned (median)",
         *(f"{med('cancel', a, 'speed_when_cancel_returned')} m/s" for a in arms)),
        ("cancelled code keeps sending", "setpoints reaching the arm after cancel (median)",
         *(str(med("zombie", a, "commands_reaching_arm_after_cancel")) for a in arms)),
        ("", "distance moved after cancel (median)",
         *(f"{med('zombie', a, 'moved_after_cancel')} m" for a in arms)),
        ("a second agent joins", "times the arm's target switched agent (median)",
         *(str(med("conflict", a, "target_switches")) for a in arms)),
        ("operator stop, program retries", "arm reached the program's goal anyway",
         *(count("operator", a, "reached_goal_anyway") for a in arms)),
    ]
    lines = ["| scenario | metric | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    lines += [f"| {s} | {m} | " + " | ".join(cells) + " |" for s, m, *cells in rows]
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
