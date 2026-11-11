"""Trace: five failures, explained by one error line or by roborun's trace. Which finds the cause?

Each failure is produced by running a short program in the simulated scene. A model reads the
program and either the error the program ended with (one line) or the failure chain from
roborun's trace (``roborun trace show --failed``), and picks the root cause from five options.
Main metric: how often it picks the right one.

    OPENROUTER_API_KEY=... python -m benchmarks trace          # GPT-5.4 by default
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import textwrap
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

import roborun
from roborun import run_sim, runtime, start, world
from roborun.backends import mujoco as mj
from roborun.checks import workspace_box
from roborun.trace import Trace, show_failed_spans

SCENE = {"cup_red": {"pos": [0.5, -0.1], "color": "red"}, "tray": {"kind": "tray", "pos": [0.3, 0.38]}}

PICK_PLACE = '''
async def main():
    start(track("cup_red"), detached=True)
    start(track("tray"), detached=True)
    await grasp("cup_red")
    await place("tray")
'''

PROGRAMS = {
    # the cup moves after the program read its position, and the program grasps that position
    "moved": '''
async def main():
    start(track("cup_red"), detached=True)
    start(track("tray"), detached=True)
    pose = (await world.next("cup_red")).value
    await grasp(pose)
    await place("tray")
''',
    # perception loses the cup during the reach
    "stale": PICK_PLACE,
    # a target below the allowed height
    "rejected": '''
async def main():
    start(track("cup_red"), detached=True)
    cup = (await world.next("cup_red")).value
    await move_to((cup[0], cup[1] - 0.15, 0.15))
    await move_to((cup[0], cup[1] - 0.15, 0.0))
    await grasp("cup_red")
''',
    # the cup slips out of the gripper while carried
    "slipped": PICK_PLACE,
    # a helper started earlier still holds the arm
    "busy": '''
@action(resources={"arm"})
async def survey():
    for p in [(0.3, -0.3, 0.3), (0.55, 0.0, 0.3)] * 20:
        await move_to(p)

async def main():
    start(track("cup_red"), detached=True)
    start(track("tray"), detached=True)
    start(survey(), detached=True)
    await start(grasp("cup_red"), wait_timeout=4.0)
    await place("tray")
''',
}

OPTIONS = {
    "A": "The cup was moved after the program read its position, so the grasp went to the old spot.",
    "B": "The program could not see the cup for a while (its position data went stale), so the "
         "skill held still and gave up.",
    "C": "A safety check rejected a command because its target was outside the allowed workspace.",
    "D": "The cup slipped out of the gripper while it was being carried.",
    "E": "Another task was holding the arm, so the step could not start in time.",
}
ANSWER = {"moved": "A", "stale": "B", "rejected": "C", "slipped": "D", "busy": "E"}
VARIANTS = 3


def _namespace() -> dict:
    ns = {"start": start, "world": world, "action": roborun.action, "sleep": roborun.sleep}
    for name in ("grasp", "place", "move_to", "track"):
        ns[name] = getattr(mj, name)
    return ns


def disturb(kind: str, b: mj.MujocoBackend, v: int) -> None:
    rng = random.Random(f"{kind}-{v}")
    if kind == "moved":
        to = (0.5 + rng.choice([-1, 1]) * 0.08, -0.1 + rng.uniform(-0.04, 0.04))
        b.at(1.0 + 0.3 * v, lambda: b.sim("push", object="cup_red", to=to, duration=0.4))
    elif kind == "stale":
        b.at(0.8 + 0.3 * v, lambda: b.sim("occlude", object="cup_red", duration=8.0))
    elif kind == "slipped":
        state = {"since": None, "done": False}

        def slip(bb: mj.MujocoBackend) -> None:
            if not state["done"] and bb._holding() == "cup_red" and bb.moving:
                state["since"] = state["since"] or bb.now()
                if bb.now() - state["since"] > 0.8 + 0.3 * v:
                    bb.sim("drop", object="cup_red", side=0.07)
                    state["done"] = True

        b.watchers.append(slip)


@dataclass
class Case:
    kind: str
    variant: int
    program: str
    error_line: str
    trace: str


def produce(kind: str, v: int) -> Case:
    backend = mj.MujocoBackend(objects=SCENE)
    disturb(kind, backend, v)
    ns = _namespace()
    code = PROGRAMS[kind]
    exec(compile(code, f"<{kind}>", "exec"), ns)  # noqa: S102 - our own fixed program text
    tr = Trace()

    async def main() -> None:
        runtime().gate.add_check(workspace_box((0.15, -0.6, 0.02), (0.8, 0.6, 0.8)))
        await ns["main"]()

    try:
        task = asyncio.run(run_sim(main, backend=backend, until=40, trace=tr))
    finally:
        backend.close()
    err = task.error
    error_line = f"{type(err).__name__}: {err}" if err is not None else f"ended {task.state.value}"
    text, _ = show_failed_spans(tr.spans, title="run")
    return Case(kind, v, code.strip(), error_line, text)


PROMPT = """A robot program failed in simulation. Read the program and the {what} below, then
pick the most likely root cause. Answer with one letter on the last line.

Program:
```python
{program}
```

{what_cap}:
```
{evidence}
```

Root causes:
{options}
"""


def ask(model: str, prompt: str) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "reasoning": {"effort": "medium"}, "max_completion_tokens": 4000}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def grade(case: Case, feedback: str, model: str) -> dict:
    evidence = case.error_line if feedback == "error line" else case.trace
    prompt = PROMPT.format(what=feedback, what_cap=feedback.capitalize(), program=case.program,
                           evidence=evidence,
                           options="\n".join(f"{k}. {v}" for k, v in OPTIONS.items()))
    reply = ask(model, prompt)
    letters = re.findall(r"\b([A-E])\b", reply.strip().splitlines()[-1] if reply.strip() else "")
    pick = letters[-1] if letters else None
    return {"kind": case.kind, "variant": case.variant, "feedback": feedback, "pick": pick,
            "correct": pick == ANSWER[case.kind]}


def main(args: Any) -> int:
    model = getattr(args, "model", None) or "openai/gpt-5.4"
    cases = [produce(k, v) for k in PROGRAMS for v in range(VARIANTS)]
    results = [grade(c, fb, model) for c in cases for fb in ("error line", "trace")]
    lines = [f"Root cause found by {model}, out of {VARIANTS} runs of each failure:", "",
             "| failure | from the error line | from the trace |", "|---|---|---|"]
    for k in PROGRAMS:
        cells = [f"{sum(r['correct'] for r in results if r['kind'] == k and r['feedback'] == fb)}/{VARIANTS}"
                 for fb in ("error line", "trace")]
        lines.append(f"| {k} | {cells[0]} | {cells[1]} |")
    tot = [sum(r["correct"] for r in results if r["feedback"] == fb) for fb in ("error line", "trace")]
    lines.append(f"| **all** | **{tot[0]}/{len(cases)}** | **{tot[1]}/{len(cases)}** |")
    print("\n".join(lines))
    if getattr(args, "out", None):
        with open(args.out, "w") as f:
            json.dump({"cases": [asdict(c) for c in cases], "results": results}, f, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # print the evidence for each failure, without a model
    for k in PROGRAMS:
        c = produce(k, 0)
        print("=" * 20, k, "\nERROR LINE:", c.error_line, "\nTRACE:\n" + textwrap.indent(c.trace, "  "))
