"""The cup-nudge demo (SPEC Appendix B): a person pushes the cup while the robot reaches for it.

Three programs do the same job, putting the red cup on the tray:

- ``blocking``: perceive once, then ``grasp_blocking`` that position. It is how a robot
  program is usually written when every call blocks until the motion is done.
- ``blocking_retry``: the same, but if the grasp misses, perceive again and retry (up to
  two retries). It is the usual fix: closed loop between actions.
- ``task_based``: ``grasp("cup_red")`` as a task that follows the tracked cup. It is closed
  loop *within* an action, which the runtime makes ordinary.

The benchmark pushes the cup 8 cm in four directions at six moments of the approach and
measures, for each program, success (the cup ends upright in the tray), completion time,
and reaction latency: the time from the start of the push to the first arm goal aimed at
the cup's new place.

    roborun demo cup-nudge                 # watch one push (the MuJoCo viewer if possible)
    roborun demo cup-nudge --benchmark     # the sweep; prints a table, writes JSON
    roborun demo cup-nudge --gif out.gif   # the two programs side by side
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from roborun import TaskFailed, run_sim, start, world
from roborun.backends import mujoco as mj

CUP = (0.5, -0.1)
TRAY = (0.3, 0.38)
SCENE = {"cup_red": {"pos": list(CUP), "color": "red"},
         "tray": {"kind": "tray", "pos": list(TRAY)}}
PUSH_TIMES = (0.25, 0.5, 1.0, 1.5, 2.0, 2.4)  # seconds; the gripper closes at about 2.7 s
PUSH_DISTANCE = 0.08
DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
PUSH_DURATION = 0.4
UNTIL = 25.0


# -- the three programs ------------------------------------------------------------------


async def _perceive() -> None:
    start(mj.track("cup_red"), detached=True)
    start(mj.track("tray"), detached=True)
    await world.next("cup_red")


async def blocking() -> None:
    await _perceive()
    cup = world.get("cup_red").value  # perceive once
    await mj.grasp_blocking(cup)
    await mj.place("tray")


async def blocking_retry() -> None:
    await _perceive()
    for attempt in range(3):
        cup = world.get("cup_red").value  # perceive again before each attempt
        try:
            await mj.grasp_blocking(cup)
            break
        except TaskFailed:
            if attempt == 2:
                raise
    await mj.place("tray")


async def task_based() -> None:
    await _perceive()
    await mj.grasp("cup_red")  # a task that follows the tracked cup
    await mj.place("tray")


PROGRAMS: dict[str, Callable] = {
    "blocking": blocking, "blocking_retry": blocking_retry, "task_based": task_based,
}


# -- one run ---------------------------------------------------------------------------------


@dataclass
class Result:
    program: str
    push_time: float | None
    direction: str | None
    success: bool
    completion: float | None  # seconds, when the program ended
    latency: float | None  # seconds from the push to the first goal aimed at the new place
    outcome: str


def in_tray(cup: Any) -> bool:
    return (abs(cup[0] - TRAY[0]) < 0.1 - mj.CUP_RADIUS
            and abs(cup[1] - TRAY[1]) < 0.075 - mj.CUP_RADIUS
            and abs(cup[2] - (0.01 + mj.CUP_HALF_HEIGHT)) < 0.008)


def pushed_to(direction: str) -> tuple[float, float]:
    dx, dy = DIRECTIONS[direction]
    return (CUP[0] + dx * PUSH_DISTANCE, CUP[1] + dy * PUSH_DISTANCE)


def run_one(program: str, push_time: float | None = None, direction: str | None = None,
            *, view: bool = False, on_step: Callable | None = None) -> Result:
    backend = mj.MujocoBackend(objects=SCENE, view=view)
    target = pushed_to(direction) if push_time is not None else None
    if target is not None:
        backend.at(push_time, lambda: backend.slide("cup_red", target, PUSH_DURATION))
    if on_step is not None:
        step = backend.step

        def stepped(dt: float) -> None:
            step(dt)
            on_step(backend)

        backend.step = stepped
    try:
        task = asyncio.run(run_sim(PROGRAMS[program], backend=backend, until=UNTIL))
    finally:
        backend.close()
    cup = backend.truth("cup_red")
    success = task.state.value == "succeeded" and in_tray(cup)
    latency = None
    if target is not None:
        aimed = [t for t, c in backend.log if c.kind == "goal" and t >= push_time - 1e-9
                 and math.dist(c.payload["target"][:2], target) < 0.02]
        latency = round(min(aimed) - push_time, 3) if aimed else None
    if success:
        outcome = "placed"
    elif task.error is not None:
        root = task.error.root if isinstance(task.error, TaskFailed) else task.error
        outcome = type(root).__name__
    else:
        outcome = f"cup not in tray ({task.state.value})"
    return Result(program, push_time, direction, success,
                  round(task.ended_at, 2) if task.ended_at is not None else None,
                  latency, outcome)


# -- the benchmark ---------------------------------------------------------------------------


def _job(args: tuple) -> dict:
    return asdict(run_one(*args))


def benchmark(workers: int | None = None) -> list[dict]:
    jobs = [(p, None, None) for p in PROGRAMS]
    jobs += [(p, t, d) for p in PROGRAMS for t in PUSH_TIMES for d in DIRECTIONS]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    def fmt(xs: list[float]) -> str:
        return f"{statistics.median(xs):.2f} s" if xs else "never"

    lines = [("| program | success, no push | success, pushed | median completion "
             "(successes) | median reaction latency |"),
             "|---|---|---|---|---|"]
    for p in PROGRAMS:
        base = [r for r in results if r["program"] == p and r["push_time"] is None]
        pushed = [r for r in results if r["program"] == p and r["push_time"] is not None]
        ok = [r for r in pushed if r["success"]]
        lat = [r["latency"] for r in pushed if r["latency"] is not None]
        reacted = f"{fmt(lat)} ({len(lat)}/{len(pushed)} reacted)" if lat else "never"
        lines.append(f"| `{p}` | {sum(r['success'] for r in base)}/{len(base)} | "
                     f"{len(ok)}/{len(pushed)} | {fmt([r['completion'] for r in ok])} | "
                     f"{reacted} |")
    by_time = ["", "Success by push time (all four directions):", "",
               "| push at | " + " | ".join(f"`{p}`" for p in PROGRAMS) + " |",
               "|---" * (len(PROGRAMS) + 1) + "|"]
    for t in PUSH_TIMES:
        cells = []
        for p in PROGRAMS:
            rs = [r for r in results if r["program"] == p and r["push_time"] == t]
            cells.append(f"{sum(r['success'] for r in rs)}/{len(rs)}")
        by_time.append(f"| {t:.2f} s | " + " | ".join(cells) + " |")
    return "\n".join(lines + by_time)


# -- the GIF ---------------------------------------------------------------------------------


def record(program: str, push_time: float, direction: str, *, fps: int = 15,
           size: tuple[int, int] = (360, 270)) -> tuple[list, Result]:
    import mujoco

    frames: list = []
    state: dict[str, Any] = {"next": 0.0}

    def grab(backend: Any) -> None:
        t = backend.now() + 0.01
        if t + 1e-9 < state["next"]:
            return
        state["next"] += 1.0 / fps
        if "r" not in state:
            state["r"] = mujoco.Renderer(backend.model, size[1], size[0])
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.36, 0.12, 0.2]
            cam.distance, cam.azimuth, cam.elevation = 1.3, 215.0, -24.0
            state["cam"] = cam
        state["r"].update_scene(backend.data, state["cam"])
        frames.append((t, state["r"].render().copy()))

    result = run_one(program, push_time, direction, on_step=grab)
    return frames, result


def gif(path: str, push_time: float = 1.0, direction: str = "+y") -> None:
    """The blocking and the task-based program side by side, with the same push."""
    from PIL import Image, ImageDraw

    left, rl = record("blocking", push_time, direction)
    right, rr = record("task_based", push_time, direction)
    n = max(len(left), len(right))
    clock = [t for t, _ in (left if len(left) >= len(right) else right)]
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    images = []
    for i, (t, (_, a), (_, b)) in enumerate(zip(clock, left, right)):
        w, h = a.shape[1], a.shape[0]
        img = Image.new("RGB", (2 * w + 6, h + 56), (24, 26, 30))
        img.paste(Image.fromarray(a), (0, 56))
        img.paste(Image.fromarray(b), (w + 6, 56))
        d = ImageDraw.Draw(img)
        d.text((10, 6), "blocking: grasp_blocking(pose seen at start)", fill=(235, 235, 235))
        d.text((w + 16, 6), 'task: grasp("cup_red") follows the cup', fill=(235, 235, 235))
        status = f"t = {t:4.1f} s"
        if push_time <= t < push_time + 1.2:
            status += "   someone pushes the cup"
        d.text((10, 22), status, fill=(250, 200, 90))
        d.text((w + 16, 22), status, fill=(250, 200, 90))
        for x, r in ((10, rl), (w + 16, rr)):
            if t >= (r.completion or 99) - 0.1 or i == n - 1:
                ok = r.success
                d.text((x, 38), ("done: " if ok else "failed: ") + r.outcome,
                       fill=(130, 220, 140) if ok else (240, 120, 110))
        images.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    hold = [images[-1]] * 20
    images[0].save(path, save_all=True, append_images=images[1:] + hold,
                   duration=int(1000 / 15), loop=0, optimize=True)


# -- entry point -------------------------------------------------------------------------------


def main(args: Any) -> int:
    if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
        print("Fetching the Franka Panda model from MuJoCo Menagerie (about 36 MB)...")
        mj.fetch_menagerie()
    if args.gif:
        gif(args.gif)
        print(f"wrote {args.gif}")
        return 0
    if args.benchmark:
        results = benchmark()
        print(summarize(results))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
            print(f"\nwrote {args.out}")
        return 0
    view = args.view if args.view is not None else bool(os.environ.get("DISPLAY"))
    for program in ("blocking", "task_based"):
        r = run_one(program, 1.0, "+y", view=view)
        print(f"{program:>10}: {'success' if r.success else 'failure'} ({r.outcome}), "
              f"{r.completion} s simulated; reacted to the push after "
              f"{'never' if r.latency is None else f'{r.latency} s'}")
    return 0
