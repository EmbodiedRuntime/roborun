"""Event-driven, decided by a program: monitors run alongside the actions they watch.

The blog's example: "a slip monitor has to run alongside the grasp and interrupt it when the
cup starts to slide". No model is involved; this is the event-driven, programmatic cell of
the blog's 2×2 table. Here the skills are deliberately plain (``place_blocking`` does not
check for slips); the program adds the reactions itself:

- a handler that redirects the grasp whenever the tracked cup moves (``on_change``);
- a monitor that watches the gripper while the cup is carried, cancels the motion when the
  cup slips (``changes`` + ``cancel_and_wait``), and goes back for it.

    python examples/patterns/parallel_monitor.py            # someone pushes the cup, then it slips
    python examples/patterns/parallel_monitor.py --view
"""

from __future__ import annotations

import argparse
import asyncio
import math

from roborun import changes, on_change, run_sim, start, world
from roborun.backends import mujoco as mj

SCENE = {"cup_red": {"pos": [0.5, -0.1], "color": "red"},
         "tray": {"kind": "tray", "pos": [0.3, 0.38]}}


async def put_on_tray(cup: str = "cup_red") -> int:
    start(mj.track(cup), detached=True)
    start(mj.track("tray"), detached=True)
    pose = (await world.next(cup)).value
    for attempt in range(1, 4):
        g = start(mj.grasp(pose))  # a fixed target; the handler below keeps it current

        @on_change(cup, while_running=g)
        async def follow(v: object, g: object = g) -> None:  # each new pose, while g runs
            if math.dist(v.value, g.target) > 0.005:
                g.update(target=v.value)

        await g
        p = start(mj.place_blocking("tray"))
        slipped = False
        async for held in changes("gripper.holding", while_running=p):
            if held.value is None and world.get("gripper.width").value < mj.EMPTY_WIDTH:
                slipped = True  # the fingers closed on nothing: the cup is gone
                await p.cancel_and_wait()  # the arm is at rest before we go on
                break
        if not slipped:
            await p
            return attempt
        pose = world.get(cup, max_age=0.3).value  # where it fell; pick it up again
    raise RuntimeError("the cup kept slipping")


def disturb(backend: mj.MujocoBackend) -> None:
    """Someone pushes the cup during the reach; later it slips while carried."""
    backend.at(1.0, lambda: backend.slide("cup_red", (0.45, -0.02), 0.4))
    state = {"since": None, "done": False}

    def slip(b: mj.MujocoBackend) -> None:
        if not state["done"] and b._holding() == "cup_red" and b.moving:
            state["since"] = state["since"] or b.now()
            if b.now() - state["since"] > 1.0:
                b.sim("drop", object="cup_red", side=0.07)
                state["done"] = True

    backend.watchers.append(slip)


def run(view: bool = False) -> tuple:
    backend = mj.MujocoBackend(objects=SCENE, view=view)
    disturb(backend)
    try:
        task = asyncio.run(run_sim(put_on_tray, backend=backend, until=40))
    finally:
        backend.close()
    return task, backend.truth("cup_red")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    task, cup = run(ap.parse_args().view)
    print(f"{task.state.value} after {task.result} attempt(s), {task.ended_at:.1f} s; "
          f"cup at {tuple(round(v, 3) for v in cup)}")
