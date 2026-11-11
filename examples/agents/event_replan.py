"""Event-driven, decided by the model: a new instruction mid-task makes the model replan.

The cell of the blog's 2×2 table that current systems leave empty: an event (here, a new
language instruction) arrives while the robot is working, and a model decides what to do
about it. Instructions are stamped messages on a queue key, so none is skipped, and each
one:

1. cancels the plan in progress and waits until the robot is at rest
   (``cancel_and_wait``: cancelling is not cancelled);
2. asks the model for a new plan, given what the robot actually holds now;
3. starts the new plan as a task, and goes back to listening.

    python examples/agents/event_replan.py               # scripted model
    python examples/agents/event_replan.py --model claude-code
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from roborun import QUEUE, TaskCancelled, changes, now, run_sim, sleep, start, world
from roborun.backends import mujoco as mj

sys.path.insert(0, str(Path(__file__).parent))
from models import Model, pick

SCENE = {"cup_red": {"pos": [0.5, -0.1], "color": "red"},
         "cup_blue": {"pos": [0.42, 0.12], "color": "blue"},
         "tray": {"kind": "tray", "pos": [0.3, 0.38]}}
SKILLS = {"grasp": mj.grasp, "place": mj.place, "move_to": mj.move_to}

SYSTEM = """You plan for a robot arm over a table (metres; table top z = 0). Reply with only a
JSON list of steps, each {"skill": "grasp" | "place" | "move_to", "args": {...}}:
- grasp(target): an object name; picks it up
- place(target): a surface name ("tray") or a position [x, y, z] for the held object's centre
- move_to(target): [x, y, z]
Cups are 0.08 m tall with radius 0.02; the tray surface is at z = 0.01, size 0.2 x 0.15.
Put two cups on the tray at different spots so they do not collide."""


async def run_plan(plan: list[dict]) -> None:
    for step in plan:
        await SKILLS[step["skill"]](**step["args"])


async def ask(model: Model, instruction: str) -> list[dict]:
    state = {name: world.get(name).value for name in ("cup_red", "cup_blue", "tray")}
    state["gripper holds"] = world.get("gripper.holding").value
    reply = await model(SYSTEM, [{"role": "user", "content":
                                  f"Instruction: {instruction}\nState now: {json.dumps(state)}"}])
    return json.loads(re.search(r"\[.*\]", reply, re.DOTALL).group(0))


async def assistant(model: Model, log: list) -> None:
    current = None
    async for msg in changes("instructions"):  # every instruction, in order
        log.append(("heard", round(now(), 2), msg.value))
        if current is not None and not current.state.terminal:
            await current.cancel_and_wait()  # the robot is at rest before we replan
            log.append(("cancelled", round(now(), 2), current.name))
        plan = await ask(model, msg.value)
        log.append(("plan", round(now(), 2), [(s["skill"], s["args"]) for s in plan]))
        current = start(run_plan(plan), name=f"plan for {msg.value!r}")
        log.append(("current", current))


async def scene(model: Model, instructions: list[tuple[float, str]]) -> list:
    world.declare("instructions", QUEUE, maxlen=20)
    for name in ("cup_red", "cup_blue", "tray"):
        start(mj.track(name), detached=True)
    await world.next("cup_red")
    log: list = []
    listener = start(assistant(model, log), name="assistant")
    for at, text in instructions:  # an operator speaking
        await sleep(max(0.0, at - now()))
        world.write("instructions", text, source="operator")
    while sum(e[0] == "current" for e in log) < len(instructions):  # one plan per instruction
        await sleep(0.05)
    plan = next(e[1] for e in reversed(log) if e[0] == "current")
    try:
        await plan
        log.append(("done", round(now(), 2), plan.name))
    except TaskCancelled:
        pass
    listener.cancel()
    return [e for e in log if e[0] != "current"]


def scripted() -> Model:
    plans = [
        [{"skill": "grasp", "args": {"target": "cup_red"}},
         {"skill": "place", "args": {"target": "tray"}}],
        None,  # filled from the state: put down what it holds first
    ]

    async def model(system: str, messages: list[dict]) -> str:
        if plans[0] is not None:
            p, plans[0] = plans[0], None
            return json.dumps(p)
        holds = json.loads(messages[-1]["content"].split("State now: ")[1])["gripper holds"]
        p = [{"skill": "place", "args": {"target": [0.5, -0.1, 0.053]}}] if holds else []
        p += [{"skill": "grasp", "args": {"target": "cup_blue"}},
              {"skill": "place", "args": {"target": [0.26, 0.38, 0.053]}},
              {"skill": "grasp", "args": {"target": "cup_red"}},
              {"skill": "place", "args": {"target": [0.34, 0.38, 0.053]}}]
        return json.dumps(p)

    return model


INSTRUCTIONS = [(0.5, "put the red cup on the tray"),
                (3.0, "wait, put the blue cup on the tray first, then the red one")]


def run(model: Model, view: bool = False, pace: float | None = None) -> tuple:
    backend = mj.MujocoBackend(objects=SCENE, view=view)
    try:
        task = asyncio.run(run_sim(lambda: scene(model, INSTRUCTIONS), backend=backend,
                                   until=120, pace=pace))
    finally:
        backend.close()
    return task, backend


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="claude-code or api; default: scripted")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    # A real model thinks in wall time, so the simulation runs at wall speed with it.
    task, backend = run(pick(args.model) if args.model else scripted(), args.view,
                        pace=1.0 if args.model else None)
    for entry in task.result or [repr(task.error)]:
        print(*entry) if isinstance(entry, tuple) else print(entry)
    print("cups:", {n: tuple(round(v, 3) for v in backend.truth(n)) for n in ("cup_red", "cup_blue")})
