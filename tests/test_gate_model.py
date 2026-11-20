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
