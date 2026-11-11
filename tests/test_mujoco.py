"""The runtime's semantics on a physics simulator: a MuJoCo Franka Panda (SPEC §1, §1.8, §2).

Skipped when MuJoCo or the Menagerie model is not installed:
    pip install 'roborun[mujoco]'
    python -c "from roborun.backends.mujoco import fetch_menagerie; fetch_menagerie()"
"""

import asyncio

import pytest

mujoco_backend = pytest.importorskip("roborun.backends.mujoco")
if not (mujoco_backend.menagerie_path() / "franka_emika_panda/panda.xml").exists():
    pytest.skip("MuJoCo Menagerie Panda model not found", allow_module_level=True)

from roborun import (
    BackendError,
    TaskFailed,
    TaskState,
    changes,
    now,
    run_sim,
    sleep,
    start,
    world,
)
from roborun.backends.mujoco import (
    REACH_TOL,
    GraspMissed,
    MujocoBackend,
    dist,
    grasp,
    grasp_blocking,
    move_to,
    place,
    track,
)

S = TaskState
CUP = (0.5, 0.0, 0.04)
NUDGED = (0.5, 0.15)
TRAY = (0.3, 0.35, 0.045)


def run(main, *, until=20.0, nudge=False, **kwargs):
    backend = MujocoBackend(**kwargs)
    if nudge:
        backend.nudge("cup", NUDGED, at=1.0)  # while the arm is still approaching
    task = asyncio.run(run_sim(main, backend=backend, until=until))
    return task, backend


def test_move_to_ends_at_the_target_and_at_rest():
    target = (0.5, 0.1, 0.3)

    async def main():
        await move_to(target)
        return world.get("arm.ee").value, world.get("arm.moving").value

    task, _ = run(main)
    ee, moving = task.result
    assert task.state is S.SUCCEEDED
    assert dist(ee, target) <= REACH_TOL and moving is False


def test_cancel_brakes_physically_before_the_task_is_cancelled():
    async def main():
        m = start(move_to((0.4, -0.3, 0.2)))
        await sleep(0.5)
        m.cancel()
        at_cancel = (m.state, world.get("arm.moving").value, world.get("arm.ee").value, now())
        await m.stopped()
        return at_cancel, m.state, m.ended_at, world.get("arm.ee").value

    task, _ = run(main)
    (state, moving, ee_at_cancel, t_cancel), final, ended_at, ee_final = task.result
    assert (state, moving) == (S.CANCELLING, True)
    assert final is S.CANCELLED
    assert ended_at - t_cancel >= 0.2  # braking at 1 m/s^2 from 0.25 m/s takes time
    assert dist(ee_at_cancel, ee_final) > 0.01  # and the arm kept moving meanwhile


def test_unreachable_goal_fails_with_a_backend_error():
    async def main():
        try:
            await move_to((1.5, 0.0, 0.3))
        except TaskFailed as e:
            return e.cause

    task, _ = run(main)
    err = task.result
    assert isinstance(err, BackendError)
    assert err.code == "unreachable" and err.source == "mujoco.panda"


async def task_based():
    start(track("cup"), detached=True)
    cup = await world.next("cup")
    g = start(grasp(cup.value))
    async for cup in changes("cup", while_running=g):
        if dist(cup.value, g.target) > 0.02:
            g.update(target=cup.value)
    await g
    await place(TRAY)


async def blocking():
    await grasp_blocking(CUP)
    await place(TRAY)


def test_task_based_program_follows_the_nudged_cup_and_places_it():
    task, backend = run(task_based, nudge=True, objects={"cup": CUP[:2]})
    assert task.state is S.SUCCEEDED, task.error
    cup = backend.truth("cup")
    assert dist(cup[:2], TRAY[:2]) <= 0.015
    assert abs(cup[2] - 0.04) < 0.005  # standing upright on the floor


def test_blocking_program_closes_on_nothing_and_fails():
    task, backend = run(blocking, nudge=True, objects={"cup": CUP[:2]})
    assert task.state is S.FAILED
    assert isinstance(task.error, TaskFailed) and isinstance(task.error.cause, GraspMissed)
    assert dist(backend.truth("cup")[:2], NUDGED) < 0.005  # never touched


def test_runs_are_deterministic():
    a = run(task_based, nudge=True, objects={"cup": CUP[:2]})
    b = run(task_based, nudge=True, objects={"cup": CUP[:2]})
    assert a[1].truth("cup") == b[1].truth("cup")
    assert a[0].ended_at == b[0].ended_at
