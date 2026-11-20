"""Behaviour that the TLA+ model of the Gate (spec/Gate.tla) requires.

Each test is a counterexample TLC found against an earlier implementation.
"""

import asyncio
import time

from roborun import TaskState, hold, run_sim, runtime, send, sleep, start, world
from roborun.api import current
from roborun.backends.fake import FakeBackend, move_to
from roborun.checks import Accept
from roborun.errors import BackendError, ResourceBusy

S = TaskState


def test_a_command_whose_grant_was_released_while_its_checks_ran_is_not_executed():
    # ExecutedForTheOwner. A child sends through its parent's `hold`; the parent leaves
    # the hold while the check still runs, and the arm goes to another task.
    def slow(cmd, view):
        time.sleep(0.05)
        return Accept()

    async def main():
        runtime().gate.add_check(slow, kinds={"goal"}, budget=1.0)
        seen = {}

        async def child():
            try:
                await send("goal", "arm", target=(1.0, 0.0))
                seen["child"] = "accepted"
            except asyncio.CancelledError:
                seen["child"] = current().detail["code"]
                raise

        async def other():
            await sleep(0.5)
            seen["moving_while_other_owns"] = world.get("arm.moving").value

        async with hold("arm"):
            c = start(child())
            for _ in range(3):
                await asyncio.sleep(0)  # the child's command is with the checks now
        await start(other(), resources={"arm"})
        return await c.stopped(), seen

    backend = FakeBackend()
    state, seen = asyncio.run(run_sim(main, backend=backend)).result
    assert seen == {"child": "unauthorized", "moving_while_other_owns": False}
    assert state is S.CANCELLED and backend.log == []


class Faulty(FakeBackend):
    """A backend whose controller reports a fault for one target."""

    def execute(self, cmd):
        if cmd.kind == "goal" and tuple(cmd.payload["target"]) == (9.0, 9.0):
            raise BackendError("fault", "joint 4 over torque", source="fake.arm", stamp=self.now())
        super().execute(cmd)


def test_a_fault_ends_the_requests_waiting_for_the_actuator():
    # NoWaiterUntilReset: as after an operator stop, nothing waits for a reset.
    async def main():
        async def faults():
            await sleep(0.2)
            await send("goal", "arm", target=(9.0, 9.0))

        owner = start(faults(), resources={"arm"})
        await sleep(0.1)
        waiter = start(move_to((0.1, 0.0)))  # waits for the arm
        await owner.stopped()
        await sleep(0.1)
        state, error = waiter.state, waiter.error
        runtime().gate.reset(["arm"])
        again = await move_to((0.1, 0.0))
        return state, error, again

    state, error, again = asyncio.run(run_sim(main, backend=Faulty())).result
    assert state is S.FAILED and isinstance(error, ResourceBusy)
    assert "locked until reset: fault" in str(error)
    assert again == (0.1, 0.0)


# -- MotionHasOwner: an actuator changes hands only at rest -------------------------------------


async def fire_and_forget(target=(3.0, 0.0)):
    await send("goal", "arm", target=target)  # six seconds away, and it does not wait


def test_a_task_that_ends_with_its_arm_moving_has_it_stopped_before_the_next_owner(sim):
    async def main():
        rt = runtime()
        t = start(fire_and_forget(), resources={"arm"})
        await sleep(0.05)
        nxt = start(sleep(0.0), resources={"arm"})  # waits for the arm
        busy = start(sleep(0.0), resources={"arm"}, on_busy="fail")
        state, moving = await t.stopped(), rt.backend.moving
        await busy.stopped()
        await nxt.stopped()
        return (state, moving, nxt.started_at - t.ended_at, rt.backend.moving, str(busy.error),
                [(e.kind, e.actuators, e.cause, e.detail) for e in rt.gate.events])

    task, backend = sim(main)
    state, moving, waited, moving_then, busy, events = task.result
    assert state is TaskState.SUCCEEDED and moving  # it ended, and the arm was on its way
    assert [c.kind for _, c in backend.log] == ["goal", "stop"]  # the Gate stopped it
    assert waited >= backend.stop_time - 1e-9 and moving_then is False  # granted at rest
    assert "waited for by" in busy or "still stopping after fire_and_forget" in busy
    assert events == [("fallback", ("arm",), "released", "left moving by fire_and_forget")]


def test_a_task_that_ends_at_rest_is_not_stopped(sim):
    async def main():
        await start(move_to((0.2, 0.0)))
        return [e.kind for e in runtime().gate.events]

    task, backend = sim(main)
    assert task.result == [] and [c.kind for _, c in backend.log] == ["goal"]


def test_leaving_a_hold_with_the_arm_moving_stops_it_and_a_take_does_not_cancel_the_task(sim):
    async def body():
        async with hold("arm"):
            await send("goal", "arm", target=(3.0, 0.0))
        await sleep(0.5)  # it goes on, without the arm
        return "went on"

    async def main():
        rt = runtime()
        t = start(body())
        await sleep(0.02)
        refused = start(sleep(0.0), resources={"arm"}, on_busy="fail")
        taker = start(sleep(0.0), resources={"arm"}, on_busy="take")
        await refused.stopped()
        await taker.stopped()
        return (str(refused.error), taker.state, rt.backend.moving,
                taker.started_at, await t.stopped(), t.result)

    task, backend = sim(main)
    refused, taker, moving, started, state, result = task.result
    assert "['arm'] still stopping after" in refused and refused.endswith("body")
    assert taker is TaskState.SUCCEEDED and moving is False
    assert started >= 0.0 + backend.stop_time - 1e-9
    assert (state, result) == (TaskState.SUCCEEDED, "went on")  # the take cancelled nobody


# -- Ownership.tla with deeper trees and `hold` ---------------------------------------------------


def test_two_tasks_that_each_hold_what_the_other_owns_wait_until_a_wait_timeout(sim):
    """ownership_limit_hold_deadlock: a limit of `hold`, which asks while owning."""
    async def one(mine, other, **kwargs):
        await sleep(0.1)
        async with hold(other, **kwargs):
            pass
        return mine

    async def main():
        a = start(one("arm", "gripper"), resources={"arm"})
        b = start(one("gripper", "arm"), resources={"gripper"})
        await sleep(5.0)
        stuck = (a.state, b.state)
        a.cancel()
        states = (await a.stopped(), await b.stopped())
        # With a wait_timeout the one that gives up fails, and the other goes on.
        a = start(one("arm", "gripper", wait_timeout=0.5), resources={"arm"})
        b = start(one("gripper", "arm"), resources={"gripper"})
        return stuck, states, await a.stopped(), type(a.error).__name__, await b.stopped()

    stuck, states, a, error, b = sim(main)[0].result
    assert stuck == (TaskState.RUNNING, TaskState.RUNNING)
    assert states == (TaskState.CANCELLED, TaskState.SUCCEEDED)  # cancelling one frees the other
    assert (a, error, b) == (TaskState.FAILED, "ResourceBusy", TaskState.SUCCEEDED)


def test_a_task_cannot_take_what_the_hold_of_its_grandparent_is_taking(sim):
    """OwnLine: it would cancel the grandparent, and so itself."""
    async def grandchild():
        await sleep(0.0)

    async def child():
        await sleep(0.1)  # by now the grandparent waits in its hold
        t = start(grandchild(), resources={"gripper"}, on_busy="take")
        await t.stopped()
        return t.state, str(t.error)

    async def top():
        c = start(child())
        await sleep(0.05)
        async with hold("gripper", on_busy="take"):
            pass
        return await c

    async def slow_to_stop():
        try:
            await sleep(10.0)
        except asyncio.CancelledError:
            await sleep(0.5)  # the hold above waits meanwhile
            raise

    async def main():
        other = start(slow_to_stop(), resources={"gripper"})
        await sleep(0.01)
        t = start(top(), resources={"arm"})
        return await t.stopped(), t.result, await other.stopped()

    state, (taken, error), other = sim(main)[0].result
    assert state is TaskState.SUCCEEDED and other is TaskState.CANCELLED
    assert taken is TaskState.FAILED and "wait for it instead of taking" in error


def test_a_cancellation_reaches_three_levels_and_their_holds(sim):
    """CancelReachesSubtree: every token below is revoked, every request withdrawn."""
    async def leaf():
        async with hold("gripper"):  # waits: the gripper is owned elsewhere
            pass

    async def middle():
        start(leaf())
        await send("goal", "arm", target=(3.0, 0.0))  # with the token two levels up
        await sleep(10.0)

    async def top():
        async with hold("arm"):
            await start(middle())

    async def main():
        rt = runtime()
        owner = start(sleep(10.0), resources={"gripper"})
        t = start(top())
        await sleep(0.2)
        waiting = len(rt.gate._queue)
        t.cancel()
        state = await t.stopped()
        tree = [x.state for x in rt.all_tasks() if x.name.endswith(("middle", "leaf"))]
        nxt = start(sleep(0.0), resources={"arm"}, on_busy="fail")
        owner.cancel()
        return waiting, state, tree, rt.backend.moving, len(rt.gate._queue), await nxt.stopped()

    waiting, state, tree, moving, left, nxt = sim(main)[0].result
    assert waiting == 1 and left == 0
    assert state is TaskState.CANCELLED and tree == [TaskState.CANCELLED] * 2
    assert moving is False and nxt is TaskState.SUCCEEDED  # the arm is free, and at rest
