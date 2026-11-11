"""Graph as policy (the blog's GaP): the model writes a graph; an interpreter runs it without the model.

Nodes are skills or checks; edges say where to go on each outcome (``ok``, an error name such
as ``Dropped``, or ``*``), and a node may ``watch`` a World key, leaving early when it
changes. The interpreter is about thirty lines because each node is just a task:

- running a node is ``start(skill(...))``; its outcome is the task's result or root error;
- leaving a node early is ``cancel_and_wait``: the arm is at rest before the next node;
- the run is in the trace like any program, node by node.

    python examples/agents/graph_policy.py                 # a scripted graph; the cup slips once
    python examples/agents/graph_policy.py --model claude-code
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from roborun import TaskFailed, changes, run_sim, start, world
from roborun.backends import mujoco as mj
from roborun.demos.cup_nudge import in_tray

sys.path.insert(0, str(Path(__file__).parent))
from models import Model, pick

SCENE = {"cup_red": {"pos": [0.5, -0.1], "color": "red"},
         "tray": {"kind": "tray", "pos": [0.3, 0.38]}}
SKILLS = {"grasp": mj.grasp, "place": mj.place, "move_to": mj.move_to}


async def in_tray_check(object: str) -> bool:
    return in_tray(world.get(object, max_age=0.5).value)


CHECKS = {"in_tray": in_tray_check}


async def run_graph(graph: dict, max_steps: int = 20) -> list:
    """Run ``graph`` from its ``start`` node until an ``end`` node; return the path taken."""
    node_name, path = graph["start"], []
    for _ in range(max_steps):
        node = graph["nodes"][node_name]
        if "end" in node:
            path.append((node_name, node["end"]))
            return path
        if "skill" in node:
            body = SKILLS[node["skill"]](**node.get("args", {}))
        else:
            body = CHECKS[node["check"]](**node.get("args", {}))
        task = start(body, name=node_name)
        outcome = None
        if "watch" in node:  # leave early if the watched key changes while the node runs
            before = world.latest(node["watch"]["key"])
            async for v in changes(node["watch"]["key"], while_running=task):
                if before is None or v.value != before.value:
                    await task.cancel_and_wait()
                    outcome = "changed"
                    break
        if outcome is None:
            try:
                result = await task
                outcome = "ok" if result is not False else "false"
            except TaskFailed as e:
                outcome = type(e.root).__name__
        path.append((node_name, outcome))
        edges = graph["edges"].get(node_name, {})
        node_name = edges.get(outcome) or edges.get("*") or graph["edges"].get("*", {}).get(outcome)
        if node_name is None:
            path.append(("no edge", outcome))
            return path
    path.append(("step limit", None))
    return path


GRAPH = {
    "start": "grasp",
    "nodes": {
        "grasp": {"skill": "grasp", "args": {"target": "cup_red"}},
        "place": {"skill": "place", "args": {"target": "tray"}},
        "check": {"check": "in_tray", "args": {"object": "cup_red"}},
        "done": {"end": "success"},
        "give_up": {"end": "failure"},
    },
    "edges": {
        "grasp": {"ok": "place", "GraspMissed": "grasp"},
        "place": {"ok": "check", "Dropped": "grasp", "NothingHeld": "grasp"},
        "check": {"ok": "done", "false": "grasp"},
        "*": {"StaleData": "give_up"},
    },
}

SYSTEM = """You write a graph that a robot runs without you. Reply with only a JSON object:
{"start": node, "nodes": {name: node}, "edges": {name: {outcome: next_node}, "*": {outcome: node}}}
A node is {"skill": "grasp"|"place"|"move_to", "args": {...}}, {"check": "in_tray",
"args": {"object": name}} (outcome "ok" or "false"), or {"end": "success"|"failure"}.
Skill outcomes are "ok" or an error name: GraspMissed, Dropped, NothingHeld, StaleData.
grasp(target=name) follows a tracked object; place(target="tray") puts the held object on it.
Edges under "*" apply to every node. Recover from misses and slips; give up if the object is
lost (StaleData)."""


async def main(model: Model | None) -> tuple:
    for name in ("cup_red", "tray"):
        start(mj.track(name), detached=True)
    await world.next("cup_red")
    if model is None:
        graph = GRAPH
    else:
        reply = await model(SYSTEM, [{"role": "user", "content":
                                      f"Task: put the red cup on the tray. Objects: "
                                      f"{world.get('scene.objects').value}"}])
        graph = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))
    return graph, await run_graph(graph)


def slip_once(backend: mj.MujocoBackend) -> None:
    state = {"since": None, "done": False}

    def watch(b: mj.MujocoBackend) -> None:
        if not state["done"] and b._holding() == "cup_red" and b.moving:
            state["since"] = state["since"] or b.now()
            if b.now() - state["since"] > 1.2:
                b.sim("drop", object="cup_red", side=0.07)
                state["done"] = True

    backend.watchers.append(watch)


def run(model: Model | None = None, view: bool = False) -> tuple:
    backend = mj.MujocoBackend(objects=SCENE, view=view)
    slip_once(backend)
    try:
        task = asyncio.run(run_sim(lambda: main(model), backend=backend, until=60,
                                   pace=1.0 if model is not None else None))
    finally:
        backend.close()
    return task, backend


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="claude-code or api; default: a scripted graph")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    task, backend = run(pick(args.model) if args.model else None, args.view)
    if task.result is None:
        print(task.state.value, repr(task.error))
    else:
        graph, path = task.result
        print("graph:", json.dumps(graph)[:400])
        print("path: ", " -> ".join(f"{n} ({o})" for n, o in path))
    print("cup:", tuple(round(v, 3) for v in backend.truth("cup_red")))
