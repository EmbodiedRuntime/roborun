"""Task semantics on the fake backend (SPEC §1)."""

import pytest

from roborun import (
    InvalidState,
    TaskCancelled,
    TaskFailed,
    TaskState,
    now,
    on_change,
    sleep,
    start,
    timeout,
    world,
)
from roborun.backends.fake import dist, move_to

S = TaskState


def test_await_body_runs_it_as_a_task_and_returns_its_result(sim):
    async def main():
        return await move_to((0.1, 0.0))

    task, backend = sim(main)
    assert task.state is S.SUCCEEDED
    assert task.result == (0.1, 0.0)
    assert dist(backend.ee, (0.1, 0.0)) < 1e-6


def test_cancel_is_a_request_and_the_arm_keeps_moving_until_stopped(sim):
    async def main():
        m = start(move_to((1.0, 0.0)))
        await sleep(0.2)
        m.cancel()  # no await: the request is still sent
        at_cancel = (m.state, world.get("arm.moving").value)
        final = await m.stopped()
        return at_cancel, final, m.reason, world.get("arm.moving").value, m.ended_at

    task, backend = sim(main)
    at_cancel, final, reason, moving_after, ended_at = task.result
    assert at_cancel == (S.CANCELLING, True)
    assert final is S.CANCELLED and reason == "requested"
    assert moving_after is False
    assert ended_at >= 0.2 + backend.stop_time - 1e-6  # stopping took time


def test_motion_that_finishes_while_stopping_ends_succeeded(sim):
    async def main():
        m = start(move_to((0.2, 0.0)))
        # 0.2 m at 0.5 m/s takes 0.4 s; cancel 10 ms before arrival.
        await sleep(0.39)
        m.cancel()
        return await m.stopped(), m.result

    task, _ = sim(main)
    assert task.result == (S.SUCCEEDED, (0.2, 0.0))


def test_stopped_never_raises_and_await_raises_task_cancelled(sim):
    async def main():
        m = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        m.cancel()
        assert await m.stopped() is S.CANCELLED
        try:
            await m
        except TaskCancelled as e:
            return e.reason

    task, _ = sim(main)
    assert task.result == "requested"


def test_failure_is_raised_as_task_failed_with_the_cause(sim):
    async def boom():
        raise ValueError("bad")

    async def main():
        try:
            await start(boom())
        except TaskFailed as e:
            return type(e.cause)

    task, _ = sim(main)
    assert task.result is ValueError


def test_no_orphans_parent_ends_only_after_its_children(sim):
    child = {}

    async def parent():
        child["t"] = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        return "done"  # returns while the child is still moving

    async def main():
        p = start(parent())
        await p.stopped()
        return p

    task, backend = sim(main)
    p, c = task.result, child["t"]
    assert p.state is S.SUCCEEDED
    assert c.state is S.CANCELLED and c.reason == "parent"
    assert p.ended_at >= c.ended_at
    assert not backend.moving


def test_cancelling_a_parent_cancels_its_children(sim):
    child = {}

    async def parent():
        child["t"] = start(move_to((1.0, 0.0)))
        await child["t"]

    async def main():
        p = start(parent())
        await sleep(0.1)
        p.cancel()
        await p.stopped()
        return p

    task, _ = sim(main)
    p, c = task.result, child["t"]
    assert p.state is S.CANCELLED
    assert c.state is S.CANCELLED and c.reason == "parent"


def test_update_is_valid_only_while_running(sim):
    async def main():
        m = start(move_to((0.05, 0.0)))
        await m
        with pytest.raises(InvalidState):
            m.update(target=(0.0, 0.0))
        return True

    task, _ = sim(main)
    assert task.state is S.SUCCEEDED


def test_timeout_passes_through_cancelling_and_exits_after_the_robot_stops(sim):
    async def main():
        tasks = []
        try:
            async with timeout(0.2):
                m = start(move_to((1.0, 0.0)))
                tasks.append(m)
                await m
        except TimeoutError:
            return now(), tasks[0].state, tasks[0].reason, world.get("arm.moving").value

    task, backend = sim(main)
    t, state, reason, moving = task.result
    assert state is S.CANCELLED and reason == "timeout"
    assert moving is False
    assert t >= 0.2 + backend.stop_time - 1e-6


def test_provenance_records_the_body(sim):
    async def main():
        m = start(move_to((0.01, 0.0)))
        await m
        return m.provenance["code"]

    task, _ = sim(main)
    assert task.result.endswith("move_to")


def test_on_change_runs_for_each_new_value_and_stops_with_its_parent(sim):
    seen = []

    async def program():
        @on_change("x")
        async def record(v):
            seen.append(v.value)

        await sleep(0.05)
        for i in range(3):
            world.write("x", i)
            await sleep(0.02)
        return record

    task, _ = sim(program)
    assert seen == [0, 1, 2]
    assert task.result.state is S.CANCELLED and task.result.reason == "parent"


def test_cancel_and_wait_returns_after_the_robot_stops(sim):
    async def main():
        m = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        state = await m.cancel_and_wait()
        return state, world.get("arm.moving").value

    task, _ = sim(main)
    assert task.result == (S.CANCELLED, False)
