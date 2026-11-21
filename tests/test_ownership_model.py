"""Behaviour that the TLA+ model of ownership (spec/Ownership.tla) requires of the Gate.

Each test is a counterexample TLC found against an earlier implementation, or a rule the
model was used to check. They run with the Gate in-process and across a process boundary.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from roborun import TaskFailed, TaskState, hold, run_sim, runtime, send, sleep, start, world
from roborun.backends.fake import FakeBackend
from roborun.errors import ResourceBusy
from roborun.gateproc import GateServer
from roborun.remote import run_lockstep

S = TaskState
SUPERVISES = {"recovery": {"agent"}}  # recovery may take from agent; not the reverse


@pytest.fixture(params=["in_process", "gate_process"])
def run(request):
    """Run ``main`` on a fresh fake backend, with the Gate in this process or behind a socket."""

    def in_process(main, until=30.0):
        async def configured():
            runtime().supervises.update(SUPERVISES)
            return await main()

        return asyncio.run(run_sim(configured, backend=FakeBackend(), until=until))

    def gate_process(main, until=30.0):
        async def go():
            path = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")
            server = GateServer(FakeBackend(), dt=0.01, supervises=SUPERVISES)
            await server.start(path)
            stepping = asyncio.ensure_future(server.run_lockstep())
            try:
                task = await run_lockstep(main, path, until=until)
                await stepping
            finally:
                await server.close()
            return task

        return asyncio.run(go())

    return in_process if request.param == "in_process" else gate_process


async def idle(seconds):
    await sleep(seconds)


async def goes_and_ignores_cancellation(target):
    """Generated code often looks like this: a command, then a wait, and no cleanup."""
    await send("goal", "arm", target=target)
    await sleep(100)


async def slow_to_stop(log, name, seconds=10.0, cleanup=0.5):
    """Holds what it was given; when cancelled, its cleanup takes a while."""
    log.append(name)
    try:
        await sleep(seconds)
    except asyncio.CancelledError:
        await sleep(cleanup)
        raise


def as_session(session, body, **kwargs):
    return runtime().start(body, session=session, detached=True, **kwargs)


# -- cancelled means stopped (SPEC §1.1) -------------------------------------------------------


def test_a_cancelled_task_leaves_the_arm_at_rest_without_any_cleanup(run):
    async def main():
        t = start(goes_and_ignores_cancellation((1.0, 0.0)), resources={"arm"})
        await sleep(0.2)
        t.cancel()
        moving_at_request = world.get("arm.moving").value
        state = await t.stopped()
        at_end = (world.get("arm.moving").value, world.get("arm.ee").value)
        await sleep(2.0)
        return state, moving_at_request, at_end, world.get("arm.ee").value

    state, moving_at_request, (moving, ee), ee_later = run(main).result
    assert state is S.CANCELLED
    assert moving_at_request is True  # a cancel request is not a stop
    assert moving is False
    assert ee[0] < 0.2 and tuple(ee_later) == tuple(ee)  # stopped short, and stayed there


def test_a_preempted_task_hands_the_arm_over_only_at_rest(run):
    async def main():
        first = start(goes_and_ignores_cancellation((1.0, 0.0)), resources={"arm"})
        await sleep(0.2)

        async def second():
            return world.get("arm.moving").value  # as soon as it owns the arm

        t = start(second(), resources={"arm"}, on_busy="take")
        return await t, (await first.stopped(), first.reason)

    moving_at_handover, first = run(main).result
    assert moving_at_handover is False
    assert first == (S.CANCELLED, "preempted")


def test_an_actuator_that_never_comes_to_rest_fails_the_cancellation(sim, monkeypatch):
    # SPEC §1.2: cancelling -> failed when cleanup fails.
    import roborun.task as task_module

    monkeypatch.setattr(task_module, "REST_TIMEOUT", 0.5)

    async def main():
        runtime().backend.at_rest = lambda actuator: False  # a stop that never completes
        t = start(goes_and_ignores_cancellation((1.0, 0.0)), resources={"arm"})
        await sleep(0.2)
        t.cancel()
        state = await t.stopped()
        return state, t.error.code

    task, _ = sim(main, until=10.0)
    assert task.result == (S.FAILED, "fault")


# -- fail never waits (SPEC §3.3) --------------------------------------------------------------


def test_a_fail_request_does_not_wait_for_an_actuator_reserved_by_an_earlier_request(run):
    # The arm is free, but an earlier request for the arm and the gripper waits for it.
    async def main():
        start(idle(5.0), resources={"gripper"}, name="holds_gripper")
        await sleep(0.1)
        waiter = start(idle(0.1), resources={"arm", "gripper"}, name="waits_for_both")
        await sleep(0.1)
        f = start(idle(0.1), resources={"arm"}, on_busy="fail")
        await sleep(0.1)
        state = f.state
        try:
            await f
        except TaskFailed as e:
            return state, e.root, waiter.state

    state, error, waiter_state = run(main).result
    assert state is S.FAILED
    assert isinstance(error, ResourceBusy) and "waits_for_both" in str(error)
    assert waiter_state is S.PENDING  # the earlier request keeps its place


# -- take (SPEC §3.3) --------------------------------------------------------------------------


def test_a_take_cancels_a_waiting_take_and_gets_the_arm_directly(run):
    async def main():
        log = []
        owner = start(slow_to_stop(log, "owner"), resources={"arm"})
        await sleep(0.1)
        k1 = start(slow_to_stop(log, "k1", 0.2), resources={"arm"}, on_busy="take")
        await sleep(0.1)
        k2 = start(slow_to_stop(log, "k2", 0.2), resources={"arm"}, on_busy="take")
        await sleep(0.05)
        k1_now = (k1.state, k1.reason)  # at once: it never ran, so there is nothing to stop
        await k2
        return log, k1_now, (await owner.stopped(), owner.reason)

    log, k1, owner = run(main).result
    assert log == ["owner", "k2"]  # k1 never ran; nothing came between the owner and k2
    assert k1 == (S.CANCELLED, "preempted")
    assert owner == (S.CANCELLED, "preempted")


def test_a_supervised_session_cannot_take_ahead_of_its_supervisor(run):
    # The agent has authority over the owner, its own task, but not over recovery's take.
    async def main():
        log = []
        as_session("agent", slow_to_stop(log, "agent.owner"), resources={"arm"})
        await sleep(0.1)
        recovery = as_session("recovery", slow_to_stop(log, "recovery", 0.2),
                              resources={"arm"}, on_busy="take")
        await sleep(0.1)
        again = as_session("agent", slow_to_stop(log, "agent.again", 0.2),
                           resources={"arm"}, on_busy="take")
        state = await again.stopped()
        waiting = recovery.state  # refusing the agent cancelled nothing
        await recovery
        return log, state, again.error, waiting

    log, state, error, recovery_meanwhile = run(main).result
    assert log == ["agent.owner", "recovery"]
    assert state is S.FAILED and isinstance(error, ResourceBusy)
    assert "authority" in str(error)
    assert recovery_meanwhile is S.PENDING


def test_a_supervisor_takes_ahead_of_a_supervised_take(run):
    async def main():
        log = []
        as_session("agent", slow_to_stop(log, "agent.owner"), resources={"arm"})
        await sleep(0.1)
        agent = as_session("agent", slow_to_stop(log, "agent.take", 0.2),
                           resources={"arm"}, on_busy="take")
        await sleep(0.1)
        recovery = as_session("recovery", slow_to_stop(log, "recovery", 0.2),
                              resources={"arm"}, on_busy="take")
        await recovery
        return log, (await agent.stopped(), agent.reason)

    log, agent = run(main).result
    assert log == ["agent.owner", "recovery"]
    assert agent == (S.CANCELLED, "preempted")


def test_a_task_waiting_in_hold_is_cancelled_through_cancelling(run):
    # A take that waits in `hold` belongs to a running task, which may hold other actuators:
    # it is cancelled like an owner, not ended at once.
    async def main():
        log = []
        start(slow_to_stop(log, "owner"), resources={"arm"})
        await sleep(0.1)

        async def program():
            try:
                async with hold("arm", on_busy="take"):
                    log.append("program got the arm")
            except asyncio.CancelledError:
                await sleep(0.3)  # its own cleanup
                log.append("program cleaned up")
                raise

        p = start(program(), resources={"gripper"})
        await sleep(0.1)
        k = start(slow_to_stop(log, "k", 0.2), resources={"arm"}, on_busy="take")
        await sleep(0.05)
        meanwhile = p.state
        await k
        return log, meanwhile, (await p.stopped(), p.reason)

    log, meanwhile, program = run(main).result
    assert meanwhile is S.CANCELLING
    assert program == (S.CANCELLED, "preempted")
    assert "program got the arm" not in log and "program cleaned up" in log


def test_a_take_does_not_cancel_a_take_of_its_own_ancestor(run):
    # Cancelling the ancestor would cancel the taker too.
    async def main():
        log = []
        start(slow_to_stop(log, "owner"), resources={"arm"})
        await sleep(0.1)
        result = {}

        async def program():
            async def child():
                await sleep(0.1)  # by now the program waits in hold
                c = start(idle(0.1), resources={"arm"}, on_busy="take")
                result["child"] = (await c.stopped(), c.error)

            start(child())
            async with hold("arm", on_busy="take"):
                log.append("program got the arm")
                await sleep(0.2)

        p = start(program())
        state = await p.stopped()
        return log, state, result["child"]

    log, state, (child_state, child_error) = run(main).result
    assert state is S.SUCCEEDED and "program got the arm" in log
    assert child_state is S.FAILED and isinstance(child_error, ResourceBusy)


# -- a child that uses its parent's token (SPEC §1.6, §3.3) ------------------------------------


def test_a_cancelled_child_cannot_move_the_arm_through_its_parents_token(run):
    # The child has no token of its own to revoke: the parent's, in `hold`, stays valid.
    async def main():
        seen = {}

        async def child():
            try:
                await send("goal", "arm", target=(1.0, 0.0))
                await sleep(100)
            except asyncio.CancelledError:
                try:
                    await send("goal", "arm", target=(0.0, 1.0))  # it keeps commanding
                    seen["after_cancel"] = "accepted"
                except asyncio.CancelledError:
                    seen["after_cancel"] = "rejected"
                raise

        async with hold("arm"):
            c = start(child(), resources={"arm"})
            await sleep(0.2)
            c.cancel()
            state = await c.stopped()
            seen["at_cancelled"] = (world.get("arm.moving").value, world.get("arm.ee").value)
            await sleep(1.0)
            seen["later"] = world.get("arm.ee").value
            # the parent still owns the arm and can use it
            await send("goal", "arm", target=(0.0, 0.0))
            await sleep(0.1)
            seen["parent_moves"] = world.get("arm.moving").value
            await send("stop", "arm")
        return state, seen

    state, seen = run(main).result
    moving, ee = seen["at_cancelled"]
    assert state is S.CANCELLED
    assert seen["after_cancel"] == "rejected"
    assert moving is False and ee[0] < 0.2 and tuple(seen["later"]) == tuple(ee)
    assert seen["parent_moves"] is True


def test_a_cancelled_child_ends_at_rest_also_when_the_arm_was_being_stopped(run):
    # CancelledAtRest, found by comparing the Gate's core with Ownership.tla (spec/compare).
    # A `stop` made the Gate forget whose command the arm was carrying out, so a child
    # cancelled while the arm was braking ended `cancelled` with the arm still moving.
    async def main():
        async def child():
            await send("goal", "arm", target=(1.0, 0.0))
            await sleep(100)

        async with hold("arm"):
            c = start(child(), resources={"arm"})
            await sleep(0.2)
            await send("stop", "arm")  # the parent stops the arm, which takes 0.1 s
            c.cancel()
            state = await c.stopped()
            moving = world.get("arm.moving").value
        return state, moving

    state, moving = run(main).result
    assert state is S.CANCELLED
    assert moving is False


def test_a_cancelled_child_does_not_wait_for_a_motion_its_parent_started_later(run):
    # Found by comparing the Gate's core with Ownership.tla (spec/compare). The Gate stops
    # what a cancelled child commanded and lets it end at rest. If its parent then sent a
    # new goal while the child was still in its cleanup, the child waited for that motion
    # too, and ended `failed` ("still moving") when it lasted long.
    async def main():
        async def child():
            try:
                await send("goal", "arm", target=(1.0, 0.0))
                await sleep(100)
            except asyncio.CancelledError:
                await sleep(0.3)  # a slow cleanup; the arm is at rest after 0.1 s
                raise

        async with hold("arm"):
            c = start(child(), resources={"arm"})
            await sleep(0.2)
            c.cancel()
            await sleep(0.2)
            await send("goal", "arm", target=(50.0, 0.0))  # the parent's own, long motion
            t0 = runtime().clock.now()
            state = await c.stopped()
            waited = runtime().clock.now() - t0
            moving = world.get("arm.moving").value
            await send("stop", "arm")
        return state, waited, moving

    state, waited, moving = run(main).result
    assert state is S.CANCELLED
    assert waited < 0.5 and moving is True  # it ended while its parent's motion went on

