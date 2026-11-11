"""ReAct on roborun: the model calls one tool per turn and watches the world between turns.

    python examples/agents/react.py "put the red cup on the tray"
    python examples/agents/react.py "put the red cup on the tray" --push cup_red 0.45 0.05 6

What the harness needs, and where it comes from:

- act without blocking: a skill call returns a task id at once, and the robot keeps moving
  while the model thinks about the next step;
- observe while acting: ``wait(task, timeout, wake_on=[keys])`` returns when the task ends,
  the time is up, or a watched key changes, and says which;
- correct course mid-action: ``update`` redirects a running task, ``cancel`` stops it, and
  the task passes through ``cancelling`` until the robot is at rest;
- know why a step failed: the observation carries the reason (a Gate rejection code or the
  root error), and ``trace`` explains it;
- clean up: closing the session cancels everything the agent started.

Claude Code driving the robot through ``roborun mcp`` is the same loop with native tool
calls; this file shows it explicitly, with any text model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import Model, pick
from robot import push_later, robot

from roborun.harness import Client, HarnessError

SYSTEM = """\
You control a robot arm over a table by calling one tool per turn. Coordinates are metres in
the robot's frame; the table top is z = 0.

Reply with exactly one JSON object and nothing else:
  {{"thought": "...", "tool": "<name>", "args": {{...}}}}   to act, or
  {{"thought": "...", "done": "<what happened>"}}          when the task is finished.

Skills start a task and return its id at once; the robot keeps moving while you think:
{skills}

Task tools:
- wait(task, timeout=5, wake_on=[]): returns when the task ends, `timeout` seconds pass, or a
  key in wake_on (e.g. "cup_red") changes; "woke" says which, "value" has the new value.
- status(task), cancel(task), update(task, params): inspect, stop, or redirect a task.
- world(keys): read stamped values, e.g. ["scene.objects", "cup_red", "gripper.holding"].
- trace(task): why a task failed.
Objects are referred to by name once tracked: track(name="cup_red")."""


def skills_text(client: Client, only: list[str] | None = None) -> str:
    return "\n".join(f"- {t['name']}({', '.join(t['schema']['properties'])}): "
                     + " ".join(t["description"].split())
                     for t in client.tools if only is None or t["name"] in only)


def parse(reply: str) -> dict:
    m = re.search(r"\{.*\}", reply, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in the reply")
    return json.loads(m.group(0))


TASK_TOOLS = {"wait", "status", "cancel", "update", "world", "trace"}


async def act(client: Client, tool: str, args: dict, only: list[str] | None = None) -> object:
    """One tool call; failures become observations, as in any ReAct loop."""
    try:
        if tool == "wait":
            return await client.wait(args["task"], args.get("timeout", 5),
                                     args.get("wake_on", ()))
        if tool == "status":
            return await client.status(args["task"])
        if tool == "cancel":
            return await client.cancel(args["task"])
        if tool == "update":
            return await client.update(args["task"], **args["params"])
        if tool == "world":
            return await client.world(*args["keys"])
        if tool == "trace":
            return (await client.trace(args["task"]))["text"]
        if only is not None and tool not in only:
            return {"error": f"no tool {tool!r}; tools: {', '.join(only)}"}
        return await client.start(tool, **args)
    except (HarnessError, KeyError, TypeError) as e:
        return {"error": str(e)}


async def react(client: Client, model: Model, instruction: str, steps: int = 30,
                only: list[str] | None = None, notes: str = "") -> str:
    """``only`` limits the skills offered (e.g. thin tools); ``notes`` is added to the prompt."""
    system = SYSTEM.format(skills=skills_text(client, only)) + (f"\n\n{notes}" if notes else "")
    messages = [{"role": "user", "content": f"Task: {instruction}"}]
    for step in range(1, steps + 1):
        reply = await model(system, messages)
        try:
            action = parse(reply)
        except ValueError as e:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"Invalid reply ({e}); reply with one JSON object."}]
            continue
        print(f"[{step}] {action.get('thought', '')}")
        if "done" in action:
            return action["done"]
        observation = await act(client, action.get("tool", ""), action.get("args") or {}, only)
        text = json.dumps(observation, default=str)
        print(f"    {action.get('tool')}({json.dumps(action.get('args'))}) -> {text[:160]}")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"Observation: {text}"}]
    return "gave up after the step limit"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction")
    ap.add_argument("--model", help="claude-code (default) or api")
    ap.add_argument("--socket", help="a running `roborun serve`; default: simulate here")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--push", nargs=4, metavar=("OBJECT", "X", "Y", "AFTER"))
    args = ap.parse_args()
    async with robot(args.socket, view=args.view) as client:
        if args.push:
            o, x, y, after = args.push
            push_later(client, o, (float(x), float(y)), float(after))
        print("done:", await react(client, pick(args.model), args.instruction))


if __name__ == "__main__":
    asyncio.run(main())
