"""Control authority and arbitration through the in-process Gate (SPEC §3.3)."""

from roborun import TaskCancelled, TaskFailed, TaskState, hold, runtime, send, sleep, start
from roborun.backends.fake import grasp, move_to, place
from roborun.errors import ResourceBusy

S = TaskState


def test_a_task_without_resources_cannot_command_an_actuator(sim):
    async def rogue():
        await send("goal", "arm", target=(1.0, 0.0))

    async def main():
        t = start(rogue())
        await t.stopped()
        return t.state, t.reason

    task, backend = sim(main)
    assert task.result == (S.CANCELLED, "rejected")
    assert not any(cmd.kind == "goal" for _, cmd in backend.log)


def test_stop_is_always_accepted(sim):
    async def anyone():
        await send("stop", "arm")

    task, _ = sim(lambda: start(anyone()).stopped())
    assert task.state is S.SUCCEEDED


def test_take_preempts_and_the_owner_stops_first(sim):
    async def main():
        first = start(move_to((1.0, 0.0)))
        await sleep(0.2)
        second = start(move_to((0.0, 0.3)), on_busy="take")
        try:
            await first
        except TaskCancelled as e:
            first_reason = e.reason
        await second
        return first, second, first_reason

    task, backend = sim(main)
    first, second, first_reason = task.result
    assert first_reason == "preempted"
    assert second.state is S.SUCCEEDED
    first_goal = min(
        t for t, c in backend.log if c.kind == "goal" and c.payload["target"] == (0.0, 0.3)
    )
    assert first_goal >= first.ended_at - 1e-9  # ownership moved only after the owner stopped


def test_waiters_are_served_first_come_first_served(sim):
    order = []

    async def use(tag, target):
        await move_to(target)
        order.append(tag)

    async def main():
        a = start(use("a", (0.1, 0.0)), resources={"arm"})
        await sleep(0.01)
        b = start(use("b", (0.0, 0.1)), resources={"arm"})
        await sleep(0.01)
        c = start(use("c", (0.1, 0.1)), resources={"arm"})
        for t in (a, b, c):
            await t

    sim(main)
    assert order == ["a", "b", "c"]


def test_a_task_waiting_for_actuators_is_pending_and_cancels_at_once(sim):
    async def main():
        a = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        b = start(move_to((0.0, 1.0)))
        await sleep(0.1)
        waiting = b.state
        b.cancel()
        after = b.state  # no cancelling phase: it never had any effect
        a.cancel()
        await a.stopped()
        return waiting, after

    task, _ = sim(main)
    assert task.result == (S.PENDING, S.CANCELLED)


def test_wait_timeout_fails_with_resource_busy(sim):
    async def main():
        a = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        b = start(move_to((0.0, 1.0)), wait_timeout=0.2)
        try:
            await b
        except TaskFailed as e:
            cause, state = type(e.cause), b.state
        a.cancel()
        await a.stopped()
        return cause, state, b.ended_at

    task, _ = sim(main)
    cause, state, ended_at = task.result
    assert cause is ResourceBusy and state is S.FAILED
    assert abs(ended_at - 0.3) < 0.02


def test_take_across_sessions_needs_a_supervisor(sim):
    async def main():
        rt = runtime()
        a = rt.start(move_to((1.0, 0.0)), session="agent")
        await sleep(0.1)
        denied = rt.start(move_to((0.0, 1.0)), on_busy="take", session="other")
        await denied.stopped()
        rt.supervises["recovery"] = {"agent"}
        allowed = rt.start(move_to((0.0, 0.2)), on_busy="take", session="recovery")
        await allowed.stopped()
        return denied.state, type(denied.error), a.reason, allowed.state

    task, _ = sim(main)
    assert task.result == (S.FAILED, ResourceBusy, "preempted", S.SUCCEEDED)


def test_hold_keeps_the_arm_between_grasp_and_place(sim):
    async def program():
        async with hold("arm", "gripper"):
            await grasp((0.2, 0.0))
            await sleep(0.3)  # a gap between two actions
            await place((0.0, 0.2))

    async def intruder():
        await sleep(0.5)  # arrives during the gap
        await move_to((0.5, 0.5))

    async def main():
        p = start(program())
        i = start(intruder())
        await p
        await i
        return p, i

    task, backend = sim(main, objects={"cup": (0.2, 0.0)})
    p, i = task.result
    assert p.state is S.SUCCEEDED and i.state is S.SUCCEEDED
    intruder_goals = [t for t, c in backend.log if c.kind == "goal" and c.payload["target"] == (0.5, 0.5)]
    assert min(intruder_goals) >= p.ended_at - 1e-9
    assert backend.objects["cup"] == (0.0, 0.2)


def test_on_busy_fail_raises_resource_busy(sim):
    async def main():
        a = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        b = start(move_to((0.0, 1.0)), on_busy="fail")
        try:
            await b
        except TaskFailed as e:
            a.cancel()
            return type(e.cause)

    task, _ = sim(main)
    assert task.result is ResourceBusy
