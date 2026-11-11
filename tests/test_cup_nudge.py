"""The running example in SPEC Appendix B.

The cup is nudged while the arm reaches for it. The task-based program follows the cup
and succeeds; the blocking version of the same program closes on nothing and fails.
"""

import asyncio

from roborun import TaskFailed, TaskState, changes, run_sim, start, world
from roborun.backends.fake import (
    REACH_TOL,
    FakeBackend,
    GraspMissed,
    dist,
    grasp,
    grasp_blocking,
    place,
    track,
)

CUP = (0.5, 0.0)
NUDGED = (0.5, 0.15)
TRAY = (0.0, 0.3)


def moved(cup, target, tol=0.02):
    return dist(cup, target) > tol


async def task_based():
    start(track("cup"), detached=True)          # who keeps "cup" fresh is explicit
    cup = await world.next("cup")               # wait for the tracker's first value
    g = start(grasp(cup.value))
    async for cup in changes("cup", while_running=g):
        if moved(cup.value, g.target):          # which changes matter is ordinary code
            g.update(target=cup.value)
    await g
    await place(TRAY)


async def blocking():
    start(track("cup"), detached=True)
    cup = await world.next("cup")
    await grasp_blocking(cup.value)
    await place(TRAY)


def run_with_nudge(program):
    backend = FakeBackend(objects={"cup": CUP})
    backend.nudge("cup", NUDGED, at=0.5)  # mid-reach: the reach takes 1 s
    task = asyncio.run(run_sim(program, backend=backend, until=10.0))
    return task, backend


def test_task_based_program_follows_the_cup_and_succeeds():
    task, backend = run_with_nudge(task_based)
    assert task.state is TaskState.SUCCEEDED, task.error
    assert dist(backend.objects["cup"], TRAY) <= REACH_TOL


def test_blocking_program_closes_on_nothing_and_fails():
    task, backend = run_with_nudge(blocking)
    assert task.state is TaskState.FAILED
    assert isinstance(task.error, TaskFailed) and isinstance(task.error.cause, GraspMissed)
    assert backend.objects["cup"] == NUDGED  # the cup was never picked up
