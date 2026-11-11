"""The person monitor stops the arm near the hand; afterwards only a retreat may move it."""

import asyncio

import pytest

pytest.importorskip("mujoco")

from benchmarks import safety
from roborun import TaskCancelled, TaskFailed, run_sim, runtime, start
from roborun.backends import mujoco as mj
from roborun.checks import Monitor


def test_after_a_monitor_stop_the_arm_can_retreat_but_not_approach():
    backend = mj.MujocoBackend(objects=safety.SCENE)

    async def attempt(target):
        t = start(mj.move_to(target), wait_timeout=1.0)
        try:
            await t
        except (TaskFailed, TaskCancelled):
            pass
        return t

    async def main():
        gate = runtime().gate  # only the monitor: nothing refuses the path in advance
        gate.add_monitor(Monitor(safety.person_monitor, "person", inputs={"arm.ee": 0.1},
                                 rate=50, budget=0.05, allow=safety.person_retreat))
        await attempt((0.45, 0.10, 0.15))
        into = await attempt((0.57, 0.25, 0.15))  # straight at the hand: the monitor stops it
        stopped_at = safety._zone_dist(backend.ee)
        closer = await attempt((0.57, 0.22, 0.15))  # still toward the hand: refused
        back = await attempt((0.40, 0.05, 0.25))  # away from the hand: allowed
        return into, stopped_at, closer, back, dict(gate.locks)

    try:
        task = asyncio.run(run_sim(main, backend=backend, until=30))
    finally:
        backend.close()
    assert task.error is None, task.error
    into, stopped_at, closer, back, locks = task.result
    assert into.detail["code"] == "monitor:person"
    # It stops inside the margin; at full speed it even coasts into the hand (about 0.07 m
    # from its centre, radius 0.08), which is why the keep-out check refuses such paths first.
    assert stopped_at < safety.ZONE_R + safety.MARGIN
    assert closer.detail["code"] == "locked"
    assert back.state.value == "succeeded"
    assert locks == {}  # the monitor unlocked by itself once the arm was clear


def test_the_blue_cup_monitor_stops_a_fast_approach_before_contact_and_allows_a_retreat():
    backend = mj.MujocoBackend(objects=safety.SCENE)
    moved = []
    blue0 = []

    def watch(b):
        blue = b.truth("cup_blue")
        if not blue0:
            blue0.append(blue)
        moved.append(((blue[0] - blue0[0][0]) ** 2 + (blue[1] - blue0[0][1]) ** 2) ** 0.5)

    backend.watchers.append(watch)

    async def attempt(target):
        t = start(mj.move_to(target), wait_timeout=1.0)
        try:
            await t
        except (TaskFailed, TaskCancelled):
            pass
        return t

    async def main():
        gate = runtime().gate  # only the monitor: nothing refuses the path in advance
        gate.add_monitor(Monitor(safety.make_blue_cup_monitor(), "blue_cup",
                                 inputs={"arm.ee": 0.1, "sim.cup_blue": 0.1}, rate=50,
                                 budget=0.05, allow=safety.blue_cup_retreat))
        await attempt((0.62, -0.38, 0.04))  # clear of the red cup
        into = await attempt((0.62, -0.04, 0.04))  # low, straight through the blue cup
        closer = await attempt((0.62, -0.04, 0.04))
        back = await attempt((0.62, -0.38, 0.20))
        return into, closer, back, dict(gate.locks)

    try:
        task = asyncio.run(run_sim(main, backend=backend, until=30))
    finally:
        backend.close()
    assert task.error is None, task.error
    into, closer, back, locks = task.result
    assert into.detail["code"] == "monitor:blue_cup"
    assert max(moved) < 0.002  # the cup was not touched
    assert closer.detail["code"] == "locked"
    assert back.state.value == "succeeded"
    assert locks == {}
