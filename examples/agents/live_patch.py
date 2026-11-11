"""Changing a program while it runs (the blog's pattern D): a fix takes effect at a safe point.

A program is running; research mode (a person or a model) reads its source and the live state,
finds a bug, and submits a corrected version. The runtime applies it at the program's next
task boundary (SPEC §6):

- the step running now finishes under the old version: a grasp in progress is never cut off;
- the old version then ends ``cancelled`` with reason ``replaced``, and the new one starts;
- the new version picks up where the old one left off by reading the World (what the gripper
  holds, which cups are already on the tray); provenance records which version it replaces.

Here version 1 puts both cups at the tray's centre, so the second would land on the first.

    python examples/agents/live_patch.py                     # a scripted reviewer
    python examples/agents/live_patch.py --model claude-code   # simulation at 0.3x speed

A fix that arrives after the bug has acted cannot undo it: if version 1 ends (say, failed)
before its next step, version 2 still starts when it ends and continues from the World as it
is. With a real model the review takes about 15 s, so the example slows the simulation down.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import Model, pick
from robot import robot

from roborun.harness import Client

V1 = '''
async def main():
    for name in ("cup_red", "cup_blue", "tray"):
        start(track(name), detached=True)
    await sleep(0.2)
    for cup in ("cup_red", "cup_blue"):
        await grasp(cup)
        await place("tray")
'''

V2 = '''
def on_tray(p):
    return abs(p[0] - 0.3) < 0.08 and abs(p[1] - 0.38) < 0.055 and p[2] > 0.045

async def main():
    for name in ("cup_red", "cup_blue", "tray"):
        start(track(name), detached=True)
    await sleep(0.2)
    spots = {"cup_red": (0.25, 0.38, 0.053), "cup_blue": (0.35, 0.38, 0.053)}
    held = world.get("gripper.holding").value
    if held in spots:                       # continue where the old version left off
        await place(spots[held])
    for cup, spot in spots.items():
        if on_tray(world.get(cup).value):
            continue
        await grasp(cup)
        await place(spot)
'''

SYSTEM = """You review a robot program that is running now. The robot arm works over a table
(metres; table top z = 0). Cups are 0.08 m tall, radius 0.02; the tray surface is at z = 0.01,
centre (0.3, 0.38), size 0.2 x 0.15. If the program has a bug, reply with a corrected full
program in a ```python block, defining `async def main():`. It will replace the running one at
its next step, so it must start from the current state: world.get(key).value is the latest
value of a key (world.get returns a stamped value), e.g. world.get("gripper.holding").value is
the name of the held object or None, world.get("cup_red").value is (x, y, z). Skip what is
already done.
Available: grasp(target), place(target: name or [x, y, z] for the held object's centre),
move_to, track(name), start, sleep, world."""


async def review(model: Model, client: Client, code: str) -> str:
    state = await client.world("gripper.holding", "cup_red", "cup_blue", "tray")
    reply = await model(SYSTEM, [{"role": "user", "content":
                                  f"Program:\n```python\n{code}\n```\n\nState now: "
                                  f"{ {k: v.get('value') for k, v in state.items()} }"}])
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    return m.group(1) if m else reply


async def scripted(system: str, messages: list[dict]) -> str:
    return f"The second cup goes to the same spot as the first.\n```python\n{V2}\n```"


async def live_patch(client: Client, model: Model, after: float = 1.0) -> dict:
    v1 = await client.program(V1, generated_by="author")
    await client.wait(v1["id"], timeout=after)  # it is running; research mode looks at it
    v2_code = await review(model, client, V1)
    swap = await client.replace(v1["id"], v2_code, generated_by="reviewer")
    v1_done = await client.wait(v1["id"], timeout=30)
    v2_done = await client.wait(swap["new"]["id"], timeout=60)
    cups = await client.world("cup_red", "cup_blue")
    return {"v1": v1_done, "v2": v2_done, "swap": swap,
            "cups": {k: v["value"] for k, v in cups.items()}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="claude-code or api; default: a scripted reviewer")
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()
    # A model's review takes about 15 s. At full speed version 1 reaches its bug first (the
    # second cup lands on the first, and a fix cannot undo that), so with a real model the
    # simulation runs at 0.3x and the review starts as soon as version 1 does.
    async with robot(view=args.view, pace=0.3 if args.model else None) as client:
        r = await live_patch(client, pick(args.model) if args.model else scripted,
                             after=0.2 if args.model else 1.0)
    print(f"v1 {r['v1']['id']}: {r['v1']['state']} ({r['v1'].get('reason')}) "
          f"at {r['v1']['ended']:.2f} s")
    print(f"v2 {r['v2']['id']}: {r['v2']['state']} at {r['v2']['ended']:.2f} s")
    print("cups:", {k: tuple(round(x, 3) for x in v) for k, v in r["cups"].items()})


if __name__ == "__main__":
    asyncio.run(main())
