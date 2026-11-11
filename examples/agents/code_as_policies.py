"""Code as Policies on roborun: the model writes a program; the runtime runs it as a task.

    python examples/agents/code_as_policies.py "put both cups on the tray"
    python examples/agents/code_as_policies.py "put the red cup on the tray" --push cup_red 0.45 0.05 4

What the harness needs, and where it comes from:

- run generated code without trusting it: ``program(code)`` runs it as one task; every
  command it sends passes the Gate (ownership, checks), so a bad program is rejected, not
  obeyed;
- skills the code can call by name, with their documentation: the host's tools;
- code that keeps working while the world changes: the skills are tasks, so
  ``grasp("cup_red")`` follows a pushed cup without the program doing anything;
- know why a program failed, to fix it: ``trace(id)`` gives the chain of tasks, the
  rejection or error, and the state at the end; that text is the next user message;
- provenance: each attempt records its source and which model turn wrote it.

The loop itself is ordinary code, about twenty-five lines.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models import Model, pick
from robot import push_later, robot

from roborun.harness import Client

SYSTEM = """\
You write Python programs that control a robot arm over a table. Coordinates are metres in
the robot's frame; the table top is z = 0. Write one program that defines `async def main():`
and reply with only that program in a ```python block.

Skills (each returns a task; `await` it to wait for the result, or `start(...)` it to run it
in the background and get a handle):
{skills}

Also available: start(body, detached=False) -> Task; await task; sleep(seconds);
world.get(key).value; changes(key, while_running=task); TaskFailed, TaskCancelled.
To refer to an object by name, track it first: start(track("cup_red"), detached=True).
`await` raises TaskFailed if a skill fails and TaskCancelled if it is rejected or cancelled."""


def skills_text(client: Client) -> str:
    out = []
    for t in client.tools:
        args = ", ".join(t["schema"]["properties"])
        doc = " ".join(t["description"].split())
        out.append(f"- {t['name']}({args}): {doc}")
    return "\n".join(out)


def extract_code(reply: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    return m.group(1) if m else reply


async def code_as_policies(client: Client, model: Model, instruction: str,
                           attempts: int = 3) -> dict:
    scene = (await client.world("scene.objects"))["scene.objects"]["value"]
    system = SYSTEM.format(skills=skills_text(client))
    messages = [{"role": "user", "content": f"Objects: {scene}\n\nTask: {instruction}"}]
    for attempt in range(1, attempts + 1):
        reply = await model(system, messages)
        code = extract_code(reply)
        print(f"--- attempt {attempt}: program ---\n{code.strip()}\n")
        try:
            prog = await client.program(code, generated_by=f"code-as-policies:{attempt}")
        except Exception as e:  # noqa: BLE001 - it did not compile; tell the model
            feedback = f"The program was not accepted: {e}"
        else:
            done = await client.wait(prog["id"], timeout=120)
            print(f"--- attempt {attempt}: {done['state']} ---")
            if done["state"] == "succeeded":
                return done
            feedback = (await client.trace(prog["id"]))["text"]
        print(feedback, "\n")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"The program failed:\n{feedback}\n\n"
                                                 "Write a corrected program."}]
    return done


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
        result = await code_as_policies(client, pick(args.model), args.instruction)
        cups = await client.world("cup_red", "cup_blue")
        print("result:", result["state"], {k: v and v.get("value") for k, v in cups.items()})


if __name__ == "__main__":
    asyncio.run(main())
