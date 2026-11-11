"""Stale: the data stops, the world keeps changing. Does the robot act on the old values?

Every value in the World carries the time of the observation behind it, so a skill or a
check can tell a current value from an old one. Without that, the last value of a key looks
the same whether it was measured 10 ms or 10 s ago. Three ways the data stops:

1. ``occluded``: the camera loses the cup for 2.5 s while the arm reaches for it, and the cup
   is pushed meanwhile.
2. ``tracker_died``: the tracking process crashes mid-reach, and the cup is moved afterwards.
3. ``sensor_dropout``: the sensor that reports where a person's hand is goes silent for 3 s.
   In half the variants the hand moves into the arm's path meanwhile; in the other half it
   stays where it was (there, stopping is a cost, not a save).
4. ``sensor_glitch``: the hand sensor drops out briefly (0.3 or 0.5 s, once a second) while
   the hand rests 0.3, 0.5 or 0.8 m from the arm's path. Nothing is in danger; every stop is
   a cost.

The hand monitor comes in two policies. ``roborun`` uses a fixed freshness limit: the hand's
position must be at most 0.2 s old, or the arm stops. ``roborun-growing`` accepts data of any
age but grows the margin around the last seen position by 1.6 m/s (the approach speed of a
hand in ISO 13855) times its age: it stops only where the hand could have got to.

The programs are the same on every arm. The pick-and-place retries after a missed grasp and,
told that its data is stale, restarts the tracker; the sweep retries a move that was stopped.

Arms: ``baseline`` (no runtime: no Gate, plain tasks, no timestamps), ``roborun``, and
``roborun-no-world``, roborun with its Gate and tasks but values without usable timestamps.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from benchmarks import ablate
from roborun import TaskCancelled, TaskFailed, now, provider, run_sim, runtime, sleep, start, world
from roborun.backends import mujoco as mj
from roborun.checks import Accept, Monitor, Reject, View
from roborun.demos.cup_nudge import in_tray
from roborun.errors import StaleData

SCENE = {"cup_red": {"pos": [0.48, -0.12], "color": "red"}, "tray": {"kind": "tray", "pos": [0.3, 0.38]}}
CUP = (0.48, -0.12)
PUSH = 0.09
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (0.7, 0.7), (0.7, -0.7)]
VARIANTS = 6
UNTIL = 30.0

# sensor_dropout: the arm sweeps between A and B; the hand rests at H0, or moves to H1 on the path
A, B = (0.40, 0.10, 0.12), (0.70, 0.10, 0.12)
H0, H1 = (0.55, 0.36), (0.55, 0.11)
HAND_R = 0.08  # the hand itself: the gripper moving inside this is harm
MARGIN = 0.05  # the monitor keeps this much further away
HAND_MAX_AGE = 0.2  # the monitor's freshness limit for the hand's position
V_HAND = 1.6  # m/s: how fast a hand may approach (the ISO 13855 value), for the growing margin
SWEEP_UNTIL = 9.0
GLITCH_DISTANCES = (0.3, 0.5, 0.8)  # sensor_glitch: the resting hand's distance from the path
GLITCH_GAPS = (0.3, 0.5)  # sensor_glitch: each silence lasts this long, once a second
POLICIES = {"roborun": "fixed", "roborun-growing": "growing"}


# -- the pick-and-place (occluded, tracker_died) ---------------------------------------------


async def pick_place(trackers: dict) -> None:
    for name in ("cup_red", "tray"):
        trackers[name] = start(mj.track(name), detached=True)
    await world.next("cup_red")
    for attempt in range(3):
        try:
            await mj.grasp("cup_red")
            await mj.place("tray")
            return
        except TaskFailed as e:
            if attempt == 2:
                raise
            if isinstance(e.root, StaleData):  # perception stopped: restart it
                trackers["cup_red"].cancel()
                trackers["cup_red"] = start(mj.track("cup_red"), detached=True)
            elif not isinstance(e.root, (mj.GraspMissed, mj.Dropped, mj.NothingHeld)):
                raise


def occluded(b: mj.MujocoBackend, v: int, trackers: dict) -> dict:
    t0 = 0.4 + 0.2 * v
    dx, dy = DIRECTIONS[v]
    to = (CUP[0] + PUSH * dx, CUP[1] + PUSH * dy)
    b.at(t0, lambda: b.sim("occlude", object="cup_red", duration=2.5))
    b.at(t0 + 0.2, lambda: b.sim("push", object="cup_red", to=to))
    return {"occluded_at": t0, "pushed_to": to}


def tracker_died(b: mj.MujocoBackend, v: int, trackers: dict) -> dict:
    t0 = 0.4 + 0.2 * v
    dx, dy = DIRECTIONS[v]
    to = (CUP[0] + PUSH * dx, CUP[1] + PUSH * dy)
    b.at(t0, lambda: trackers["cup_red"].cancel("crashed"))
    b.at(t0 + 0.3, lambda: b.sim("push", object="cup_red", to=to))
    return {"died_at": t0, "pushed_to": to}


# -- the sweep and the hand (sensor_dropout) --------------------------------------------------


def hand_at(t: float, moves: bool, t_move: float) -> tuple[float, float]:
    if not moves or t < t_move or t > t_move + 3.0:
        return H0
    a = min(1.0, (t - t_move) / 0.5) if t < t_move + 2.5 else 1 - min(1.0, (t - t_move - 2.5) / 0.5)
    a = a * a * (3 - 2 * a)
    return (H0[0] + a * (H1[0] - H0[0]), H0[1] + a * (H1[1] - H0[1]))


def hand_monitor(view: View) -> Any:
    """Fixed: the hand's position must be at most ``HAND_MAX_AGE`` old (declared as the input's
    limit, so older data fails closed)."""
    ee, hand = view.value("arm.ee"), view.value("person.hand")
    d = math.hypot(ee[0] - hand[0], ee[1] - hand[1])
    if d < HAND_R + MARGIN and ee[2] < 0.5:
        return Reject(f"the gripper is {d:.3f} m from the person's hand")
    return Accept()


def growing_hand_monitor(view: View) -> Any:
    """Growing: any age is accepted, but the hand may have moved ``V_HAND`` m/s since it was
    seen, so the margin grows with the data's age. Stops only where the hand could be."""
    ee, sv = view.value("arm.ee"), view["person.hand"]
    age = max(0.0, sv.age(view.now))
    d = math.hypot(ee[0] - sv.value[0], ee[1] - sv.value[1])
    reach = HAND_R + MARGIN + V_HAND * age
    if d < reach and ee[2] < 0.5:
        return Reject(f"the hand, seen {age:.2f} s ago, could be within {reach:.2f} m; gripper at {d:.3f} m")
    return Accept()


def sweep_program(hand: Any, silent: Any, policy: str, out: dict) -> Any:
    """Sweep between A and B until ``SWEEP_UNTIL``; ``hand(t)`` is where the hand is, and the
    sensor reports it 30 times a second except while ``silent(t)``."""

    @provider
    async def hand_sensor() -> None:
        while True:
            t = now()
            if not silent(t):
                world.write("person.hand", hand(t), source="hand_sensor")
            await sleep(1 / 30)

    async def main() -> None:
        start(hand_sensor(), detached=True)
        await world.next("person.hand")
        if policy == "growing":
            monitor = Monitor(growing_hand_monitor, "hand",
                              inputs={"arm.ee": 0.1, "person.hand": math.inf}, rate=50, budget=0.05)
        else:
            monitor = Monitor(hand_monitor, "hand",
                              inputs={"arm.ee": 0.1, "person.hand": HAND_MAX_AGE}, rate=50, budget=0.05)
        runtime().gate.add_monitor(monitor)
        out["moves"] = 0
        while now() < SWEEP_UNTIL:
            for p in (A, B):
                try:
                    await mj.move_to(p)
                    out["moves"] += 1
                except (TaskFailed, TaskCancelled):
                    await sleep(0.3)  # stopped: try again
    return main


# -- running and measuring ---------------------------------------------------------------------


@dataclass
class Result:
    scenario: str
    variant: int
    arm: str
    metrics: dict = field(default_factory=dict)


def _arm_parts(arm: str) -> tuple[str, ...]:
    if ablate.is_baseline(arm):
        return ("gate", "task", "world")
    p = ablate.part_of(arm)
    return (p,) if p else ()


def run_one(scenario: str, variant: int, arm: str) -> Result:
    with contextlib.ExitStack() as stack:
        for p in _arm_parts(arm):
            stack.enter_context(ablate.without(p))
        if scenario in SWEEPS:
            return _run_sweep(scenario, variant, arm)
        return _run_pick(scenario, variant, arm)


def _run_pick(scenario: str, variant: int, arm: str) -> Result:
    backend = mj.MujocoBackend(objects=SCENE)
    trackers: dict = {}
    params = {"occluded": occluded, "tracker_died": tracker_died}[scenario](backend, variant, trackers)
    holding: list[tuple[float, Any]] = []
    backend.watchers.append(lambda b: holding.append((b.now(), b._holding())))
    try:
        task = asyncio.run(run_sim(lambda: pick_place(trackers), backend=backend, until=UNTIL))
    finally:
        backend.close()
    closes = [t for t, c in backend.log if c.kind == "effector" and c.payload.get("close")]

    def held_at(t: float) -> Any:
        later = [h for ts, h in holding if ts >= t]
        return later[0] if later else None

    placed = task.state.value == "succeeded" and in_tray(backend.truth("cup_red"))
    stale_seen = any(isinstance(getattr(t, "error", None), StaleData)
                     or isinstance(getattr(getattr(t, "error", None), "root", None), StaleData)
                     for t in task.runtime.all_tasks())
    if placed:
        outcome = "placed"
    elif task.error is not None:
        root = task.error.root if isinstance(task.error, TaskFailed) else task.error
        outcome = type(root).__name__
    else:
        outcome = task.state.value
    return Result(scenario, variant, arm, {
        "placed": placed, "outcome": outcome,
        "closes_on_nothing": sum(1 for t in closes if held_at(t + 0.5) is None),
        "noticed_stale": stale_seen,
        "time": round(task.ended_at, 2) if task.ended_at is not None else None, **params})


def _dropout(v: int) -> tuple[Any, Any, dict]:
    moves = v % 2 == 0  # the hand moves into the path in half the variants
    t_move = 2.0 + 0.5 * (v // 2)
    t_drop = t_move - 0.4
    return ((lambda t: hand_at(t, moves, t_move)), (lambda t: t_drop <= t < t_drop + 3.0),
            {"hand_moves": moves, "t_move": t_move})


def _glitch(v: int) -> tuple[Any, Any, dict]:
    d, gap = GLITCH_DISTANCES[v % 3], GLITCH_GAPS[v // 3]
    rest = (H0[0], A[1] + d)  # the hand rests d metres from the path, and stays
    return ((lambda t: rest), (lambda t: t >= 1.0 and (t - 1.0) % 1.0 < gap),
            {"hand_moves": False, "distance": d, "gap": gap})


SWEEPS = {"sensor_dropout": _dropout, "sensor_glitch": _glitch}


def _locked_time(events: list, end: float) -> float:
    total, since = 0.0, None
    for e in events:
        if e.cause != "monitor:hand":
            continue
        if e.kind == "lock" and since is None:
            since = e.t
        elif e.kind == "unlock" and since is not None:
            total, since = total + e.t - since, None
    return total + (end - since if since is not None else 0.0)


def _run_sweep(scenario: str, variant: int, arm: str) -> Result:
    hand, silent, params = SWEEPS[scenario](variant)
    backend = mj.MujocoBackend(objects=SCENE)
    harm = [0.0]  # seconds the gripper moved inside the hand
    last = [None]

    def watch(b: mj.MujocoBackend) -> None:
        ee, t = b.ee, b.now()
        h = hand(t)
        speed = 0.0 if last[0] is None else math.dist(ee, last[0]) / 0.01
        last[0] = ee
        if math.hypot(ee[0] - h[0], ee[1] - h[1]) < HAND_R and ee[2] < 0.5 and speed > 0.02:
            harm[0] += 0.01

    backend.watchers.append(watch)
    out: dict = {}
    program = sweep_program(hand, silent, POLICIES.get(arm, "fixed"), out)
    try:
        task = asyncio.run(run_sim(program, backend=backend, until=UNTIL))
        events = list(task.runtime.gate.events)
    finally:
        backend.close()
    stops = [e for e in events if e.kind == "lock" and e.cause == "monitor:hand"]
    return Result(scenario, variant, arm, {
        **params, "harm": harm[0] > 0, "harm_time": round(harm[0], 2),
        "stopped": bool(stops), "stops": len(stops),
        "locked_time": round(_locked_time(events, SWEEP_UNTIL), 2), "moves": out.get("moves", 0),
        "stopped_for": sorted({"stale data" if "old (limit" in e.detail or "missing" in e.detail
                               else "too close" for e in stops})})


SCENARIOS = ("occluded", "tracker_died", "sensor_dropout", "sensor_glitch")


def _job(args: tuple) -> dict:
    return asdict(run_one(*args))


def benchmark(workers: int | None = None, ablate_parts: tuple[str, ...] = ()) -> list[dict]:
    extra = tuple(p for p in ablate_parts if p != "world")
    jobs = [(s, v, arm) for s in SCENARIOS for v in range(VARIANTS)
            for arm in ablate.arms(("world", *extra))]
    # the growing-margin monitor matters only where a monitor reads the hand sensor
    jobs += [(s, v, "roborun-growing") for s in SWEEPS for v in range(VARIANTS)]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs))


def summarize(results: list[dict]) -> str:
    arms = list(dict.fromkeys(r["arm"] for r in results))

    def rows(scenario: str, pred: Any = None) -> dict[str, list[dict]]:
        return {a: [r["metrics"] for r in results if r["scenario"] == scenario and r["arm"] == a
                    and (pred is None or pred(r["metrics"]))] for a in arms}

    def cell(ms: list[dict], key: str, kind: str = "count") -> str:
        if not ms:
            return "—"
        if kind == "mean":
            return f"{sum(m[key] for m in ms) / len(ms):.1f}"
        if kind == "sum":
            return str(sum(m[key] for m in ms))
        return f"{sum(bool(m[key]) for m in ms)}/{len(ms)}"

    lines = ["| scenario | metric | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]

    def add(scenario: str, metric: str, key: str, kind: str = "count", pred: Any = None) -> None:
        rs = rows(scenario, pred)
        lines.append(f"| {scenario.replace('_', ' ')} | {metric} | "
                     + " | ".join(cell(rs[a], key, kind) for a in arms) + " |")

    for s in ("occluded", "tracker_died"):
        add(s, "gripper closed on nothing (total)", "closes_on_nothing", "sum")
        add(s, "cup placed in the tray", "placed")
    add("sensor_dropout", "hand moved into the path: arm moved into the hand", "harm",
        pred=lambda m: m["hand_moves"])
    add("sensor_dropout", "hand stayed away: arm stopped anyway", "stopped",
        pred=lambda m: not m["hand_moves"])
    for d in GLITCH_DISTANCES:
        add("sensor_glitch", f"hand {d} m from the path: moves completed (mean)", "moves", "mean",
            pred=lambda m, d=d: m["distance"] == d)
        add("sensor_glitch", f"hand {d} m from the path: seconds stopped (mean, of 9)",
            "locked_time", "mean", pred=lambda m, d=d: m["distance"] == d)
    add("sensor_glitch", "arm moved into the hand", "harm")
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
