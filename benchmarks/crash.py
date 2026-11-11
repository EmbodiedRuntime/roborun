"""Crash: the agent process dies mid-motion. Does the arm keep going?

An agent (a separate process, connected to the harness socket) starts a long move across the
table and is then killed (SIGKILL) partway through. On roborun the move belongs to the agent's
session, so a closed connection cancels it and the arm comes to rest. The baseline starts the
same move detached from the session, which is what a robot driver does with a goal whose client
has gone (a ROS 2 action server, a trajectory executor): it keeps executing.

A second case freezes the agent instead (SIGSTOP, like a process stuck in a blocking call): its
connection stays open, so only the missing heartbeats tell the runtime it is gone. The harness
client pings every 0.25 s; after four missed pings the host stops what the session started.

Simulated time follows wall time here (the kill is a wall-clock event), so runs are not
bit-for-bit deterministic.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import signal
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks import ablate
from roborun.backends import mujoco as mj
from roborun.harness import Host

SCENE = {"cup_red": {"pos": [0.48, -0.12], "color": "red"}, "tray": {"kind": "tray", "pos": [0.3, 0.38]}}
START, GOAL = (0.30, -0.30, 0.20), (0.62, 0.30, 0.15)  # about 0.68 m, 3 s at full speed
VARIANTS = 6
AFTER = 4.0  # watch this long after the kill

AGENT = """
import asyncio, sys
from roborun.harness import Client

async def main(path, detached):
    c = await Client.connect(path, session="agent")
    t = await c.start("move_to", target=list({start}))
    await c.wait(t["id"], timeout=30)
    await c.call("start", tool="move_to", args={{"target": list({goal})}}, detached=detached)
    print("moving", flush=True)
    await asyncio.sleep(3600)

asyncio.run(main(sys.argv[1], sys.argv[2] == "1"))
"""


@dataclass
class Result:
    case: str  # kill or freeze
    variant: int
    arm: str
    after: float  # seconds into the move when the agent was killed or frozen
    moved_after: float  # metres the gripper travelled after that
    rest_after: float | None  # seconds until it was at rest (None: still moving or at goal)
    reached_goal: bool


async def _run(case: str, variant: int, arm: str) -> Result:
    backend = mj.MujocoBackend(objects=SCENE)
    host = await Host.local(backend, tools=["roborun.backends.mujoco"], pace=1.0)
    path = str(Path(tempfile.mkdtemp(prefix="roborun-crash-")) / "host.sock")
    await host.start(path)
    track: list[tuple[float, tuple]] = []
    backend.watchers.append(lambda b: track.append((b.now(), b.ee)))
    code = AGENT.format(start=START, goal=GOAL)
    agent = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, path, "1" if arm == "baseline" else "0",
        stdout=asyncio.subprocess.PIPE)
    try:
        line = (await asyncio.wait_for(agent.stdout.readline(), 30)).decode()
        assert line.strip() == "moving", line
        delay = 0.4 + 0.3 * variant
        await asyncio.sleep(delay)
        t_event = backend.now()
        os.kill(agent.pid, signal.SIGKILL if case == "kill" else signal.SIGSTOP)
        await asyncio.sleep(AFTER)
    finally:
        if case == "freeze":
            os.kill(agent.pid, signal.SIGKILL)
        await agent.wait()
        await host.close()
        backend.close()
    after = [(t, p) for t, p in track if t >= t_event]
    moved = sum(math.dist(a[1], b[1]) for a, b in itertools.pairwise(after))
    rest = None
    for i in range(len(after) - 10):
        window = after[i:i + 11]
        if math.dist(window[0][1], window[-1][1]) < 0.001:  # under 1 mm in 0.1 s
            rest = window[0][0] - t_event
            break
    reached = math.dist(after[-1][1], GOAL) < 0.01
    return Result(case, variant, arm, round(delay, 2), round(moved, 3),
                  None if rest is None or reached else round(rest, 2), reached)


def run_one(case: str, variant: int, arm: str) -> Result:
    with ablate.without(ablate.part_of(arm)):
        return asyncio.run(_run(case, variant, arm))


def _job(args: tuple) -> dict:
    return asdict(run_one(*args))


def benchmark(workers: int | None = None, ablate_parts: tuple[str, ...] = ()) -> list[dict]:
    jobs = [(c, v, arm) for c in ("kill", "freeze") for v in range(VARIANTS)
            for arm in ablate.arms(ablate_parts)]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    def med(xs: list[float]) -> str:
        xs = sorted(xs)
        return f"{xs[len(xs) // 2]:.2f}" if xs else "—"

    lines = [("The agent starts a 0.68 m move and is killed (or frozen) partway through. "
              "After that moment:"), "",
             "| agent | arm | kept moving to the goal | median distance moved | median time to rest |",
             "|---|---|---|---|---|"]
    arms = list(dict.fromkeys(r["arm"] for r in results))
    for case in ("kill", "freeze"):
        for arm in arms:
            rs = [r for r in results if r["case"] == case and r["arm"] == arm]
            rests = [r["rest_after"] for r in rs if r["rest_after"] is not None]
            lines.append(f"| {'killed' if case == 'kill' else 'frozen'} | {arm} | "
                         f"{sum(r['reached_goal'] for r in rs)}/{len(rs)} | "
                         f"{med([r['moved_after'] for r in rs])} m | "
                         f"{med(rests) + ' s' if rests else '—'} |")
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
