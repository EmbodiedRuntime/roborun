"""Thin tools (the blog's Agent as Policy): the model gets observations and low-level commands.

Instead of skills such as ``grasp`` and ``place``, the model has only ``move_to`` (the
gripper's position) and ``gripper`` (open or close), plus the World. It works out every
approach, descent, and lift itself. It is the same ReAct loop (``react.py``) with fewer,
smaller tools; the runtime's part does not change:

- each motion is a task that returns at once, so the model can think while the arm moves;
- every command passes the Gate, so a wrong height is rejected (``check:workspace``), not
  driven into the table, and the observation says why;
- the World says what the gripper holds, with stamps.

The blog's trade-off table in practice: more model decisions and steps, simpler tools,
recovery left to the model.

    python examples/agents/thin_tools.py "put the red cup on the tray"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import pick
from react import react
from robot import push_later, robot

THIN = ["move_to", "gripper", "track"]
NOTES = """You have only low-level commands. To pick something up: open the gripper, move above
it, move down to its centre height, close the gripper, check what it holds, lift. To put it
down: move above the spot, lower it, open the gripper, lift. Cup centres are 0.04 m above the
surface they stand on; the tray surface is at z = 0.01. Read positions with world(...) after
tracking them, and always wait for a motion to finish before the next one."""


async def thin_agent(client, model, instruction: str) -> str:
    return await react(client, model, instruction, steps=40, only=THIN, notes=NOTES)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instruction")
    ap.add_argument("--model", help="claude-code (default) or api")
    ap.add_argument("--socket")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--push", nargs=4, metavar=("OBJECT", "X", "Y", "AFTER"))
    args = ap.parse_args()
    async with robot(args.socket, view=args.view) as client:
        if args.push:
            o, x, y, after = args.push
            push_later(client, o, (float(x), float(y)), float(after))
        print("done:", await thin_agent(client, pick(args.model), args.instruction))
        print("cup:", (await client.world("cup_red"))["cup_red"]["value"])


if __name__ == "__main__":
    asyncio.run(main())
