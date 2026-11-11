"""Two rates: the model sets a constraint now and then; a fast loop keeps it every cycle.

The blog's "fixed loop with fixed tools" and "two-level hierarchy" (VoxPoser, ReKep,
HAMSTER): the model fills in a declarative target, here "keep the gripper at this offset from
that tracked object", and a program re-solves it at 20 Hz. The model is not on the reaction
path. It is asked again only when the constraint cannot be kept: the object is lost, or the
Gate rejects where it leads.

The fast loop streams ``setpoint`` commands, so the Gate's stream protections are in play: a
``max_jump`` check rejects a setpoint far from the arm (so the loop limits its step), and
the watchdog stops the arm if the stream goes silent (so the loop sends ``stop`` before it
pauses).

    python examples/patterns/two_rate.py                     # scripted constraints
    python examples/patterns/two_rate.py --model claude-code # ask a model for them
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path

from roborun import StaleData, TaskFailed, action, now, run_sim, runtime, send, sleep, start, world
from roborun.backends import mujoco as mj
from roborun.checks import max_jump

SCENE = {"cup_red": {"pos": [0.5, -0.1], "color": "red"},
         "tray": {"kind": "tray", "pos": [0.3, 0.38]}}


@action(resources={"arm"})
async def keep(target: str, offset: tuple, duration: float = 5.0, rate: float = 20.0,
               max_step: float = 0.04) -> float:
    """Keep the gripper at ``offset`` from the tracked ``target`` for ``duration`` seconds.

    Returns the median distance, over the second half of the time, from where the
    constraint says the gripper should be.
    """
    errors, end = [], now() + duration
    while now() < end:
        p = await mj.position(target, on_stale=mj._stop)  # holds still while it is stale
        goal = [p[i] + offset[i] for i in range(3)]
        ee = world.get("arm.ee").value
        d = math.dist(goal, ee)
        errors.append(d)
        k = min(1.0, max_step / d) if d > 0 else 1.0
        await send("setpoint", "arm", target=tuple(ee[i] + k * (goal[i] - ee[i]) for i in range(3)))
        await sleep(1.0 / rate)
    await mj._stop()
    settled = sorted(errors[len(errors) // 2:])  # the second half: after getting there
    return settled[len(settled) // 2]


SYSTEM = """You give constraints to a robot arm controller. Reply with one JSON object:
{"target": "<tracked object name>", "offset": [dx, dy, dz], "duration": seconds}.
The controller keeps the gripper at target position + offset (metres) for that long."""


def scripted_planner():
    replies = iter([
        {"target": "cup_red", "offset": [0, 0, 0.12], "duration": 8.0},
        {"target": "tray", "offset": [0, 0, 0.15], "duration": 4.0},
    ])

    async def plan(request: str) -> dict:
        return next(replies)

    return plan


def model_planner(name: str):
    sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))
    from models import pick

    model = pick(name)

    async def plan(request: str) -> dict:
        reply = await model(SYSTEM, [{"role": "user", "content": request}])
        return json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))

    return plan


async def main(plan) -> list:
    runtime().gate.add_check(max_jump(0.06))  # a stream must move in small steps
    for name in ("cup_red", "tray"):
        start(mj.track(name), detached=True)
    await world.next("cup_red")
    log = []
    request = ("Hover above the red cup and follow it. Objects: "
               f"{world.get('scene.objects').value}")
    for _ in range(3):
        c = await plan(request)
        log.append(("constraint", round(now(), 2), c))
        try:
            median_error = await keep(c["target"], tuple(c["offset"]), c.get("duration", 5.0))
            log.append(("kept", round(now(), 2), round(median_error, 4)))
            if c["target"] != "cup_red":
                return log
        except TaskFailed as e:
            if not isinstance(e.root, StaleData):
                raise
            log.append(("lost", round(now(), 2), str(e.root)))
        request = (f"The constraint could not be kept: {log[-1]}. "
                   "Give a constraint that parks the arm over the tray.")
    return log


def run(planner, view: bool = False) -> tuple:
    backend = mj.MujocoBackend(objects=SCENE, view=view)
    # The cup rides a conveyor for 4 s, then someone takes it away.
    backend.at(1.0, lambda: backend.sim("move", object="cup_red", velocity=(0, 0.03),
                                        duration=4.0))
    backend.at(6.0, lambda: backend.sim("remove", object="cup_red"))
    lag: list[float] = []  # how far the gripper is from 12 cm above the moving cup

    def measure(b: mj.MujocoBackend) -> None:
        if 3.0 <= b.now() <= 5.0:
            cup = b.truth("cup_red")
            lag.append(math.dist(b.ee, (cup[0], cup[1], cup[2] + 0.12)))

    backend.watchers.append(measure)
    try:
        task = asyncio.run(run_sim(lambda: main(planner), backend=backend, until=40))
    finally:
        backend.close()
    backend.lag = sorted(lag)[len(lag) // 2] if lag else None
    return task, backend


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="claude-code or api; default: scripted constraints")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    planner = model_planner(args.model) if args.model else scripted_planner()
    task, backend = run(planner, args.view)
    for entry in task.result or []:
        print(*entry)
    setpoints = sum(1 for _, c in backend.log if c.kind == "setpoint")
    print(f"{task.state.value}; {setpoints} setpoints streamed and checked; while the cup "
          f"moved at 3 cm/s the gripper stayed {backend.lag * 1000:.0f} mm (median) from "
          "12 cm above it")
