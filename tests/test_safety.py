"""Monitors, the watchdog, the fallback, the locked state, and reset (SPEC §3.4, §3.5)."""

import asyncio
import tempfile
from pathlib import Path

from roborun import (
    ResourceBusy,
    TaskCancelled,
    TaskFailed,
    TaskState,
    now,
    run_sim,
    runtime,
    send,
    sleep,
    start,
)
from roborun.backends.fake import FakeBackend, move_to, track
from roborun.checks import min_distance
from roborun.gateproc import GateServer, spawn_gate
from roborun.remote import run_lockstep

S = TaskState
FAR = (2.0, 2.0)


def run(main, **backend_kwargs):
    backend = FakeBackend(**backend_kwargs)
    task = asyncio.run(run_sim(main, backend=backend))
    return task, backend


def events(task):
    return [(e.kind, e.cause) for e in task.runtime.gate.events]


def person_monitor():
    return min_distance("arm.ee", "sim.person", 0.3, actuators={"arm"})


def test_a_monitor_stops_the_arm_locks_it_and_unlocks_when_the_person_leaves():
    async def main():
        runtime().gate.add_monitor(person_monitor())
        first = start(move_to((1.0, 0.0)))
        await first.stopped()
        stopped_at = now()
        second = start(move_to((0.0, 0.0)))  # waits while the arm is locked
        await sleep(0.1)
        waiting = second.state
        await second
        return first, stopped_at, waiting, second

    task, _backend = run(main, objects={"person": FAR},
                        nudges=[("person", (0.4, 0.1), 0.3), ("person", FAR, 1.0)])
    first, stopped_at, waiting, second = task.result
    assert (first.state, first.reason) == (S.CANCELLED, "rejected")
    assert first.detail["code"] == "monitor:min_distance"
    assert stopped_at < 1.0 and waiting is S.PENDING
    assert second.state is S.SUCCEEDED and second.started_at >= 1.0
    kinds = events(task)
    assert ("lock", "monitor:min_distance") in kinds
    assert ("fallback", "monitor:min_distance") in kinds
    assert ("unlock", "monitor:min_distance") in kinds


def test_a_monitor_whose_input_is_missing_fails_closed():
    async def main():
        runtime().gate.add_monitor(min_distance("arm.ee", "scene.humans", 0.3))
        await sleep(0.1)
        m = start(move_to((1.0, 0.0)), wait_timeout=0.2)
        try:
            await m
        except TaskFailed as e:
            return type(e.cause), runtime().gate.locks

    task, _ = run(main)
    cause, locks = task.result
    assert cause is ResourceBusy
    assert locks["arm"] == {"monitor:min_distance"}


def test_the_watchdog_stops_a_setpoint_stream_that_goes_silent():
    async def stream():
        for i in range(20):
            await send("setpoint", "arm", target=(0.01 * i, 0.0))
            await sleep(0.01)
        await sleep(10)  # a bug: keeps the arm but stops streaming

    async def main():
        t = start(stream(), resources={"arm"})
        await t.stopped()
        return t, now()

    task, backend = run(main)
    t, _ended = task.result
    assert (t.state, t.reason, t.detail["code"]) == (S.CANCELLED, "watchdog", "watchdog")
    last_setpoint = max(at for at, c in backend.log if c.kind == "setpoint")
    stop_at = min(at for at, c in backend.log if c.kind == "stop")
    assert 0.1 - 1e-6 <= stop_at - last_setpoint <= 0.12
    assert ("watchdog", "watchdog") in events(task)


def test_operator_stop_cancels_owners_only_and_needs_reset():
    async def main():
        tracker = start(track("person"), detached=True)
        m = start(move_to((1.0, 0.0)))
        await sleep(0.2)
        runtime().gate.operator_stop(["arm"])
        await m.stopped()
        tracker_after = tracker.state
        refused = start(move_to((0.0, 0.0)))
        await refused.stopped()
        runtime().gate.reset(["arm"])
        again = await move_to((0.0, 0.0))
        return m, tracker_after, refused, again

    task, _ = run(main, objects={"person": FAR})
    m, tracker_after, refused, again = task.result
    assert (m.state, m.reason, m.detail["code"]) == (S.CANCELLED, "operator", "operator")
    assert tracker_after is S.RUNNING  # holds no actuators: keeps the World fresh
    assert refused.state is S.FAILED and "locked until reset" in str(refused.error)
    assert again == (0.0, 0.0)
    assert ("unlock", "fault,operator") in events(task)


def test_a_rejected_command_runs_the_fallback_but_an_unauthorized_one_does_not():
    async def rogue():
        await send("goal", "arm", target=(0.0, 1.0))

    async def main():
        m = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        r = start(rogue())
        await r.stopped()
        await m
        return r

    task, backend = run(main)
    assert task.result.detail["code"] == "unauthorized"
    assert not any(c.kind == "stop" for _, c in backend.log)  # the owner kept moving
    assert backend.ee == (1.0, 0.0)


def test_operator_stop_and_reset_across_the_socket():
    async def main():
        gate = runtime().gate
        m = start(move_to((1.0, 0.0)))
        await sleep(0.2)
        await gate.operator_stop(["arm"])
        await m.stopped()
        await gate.reset(["arm"])
        await move_to((0.0, 0.0))
        return m

    async def go():
        path = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")
        server = GateServer(FakeBackend())
        await server.start(path)
        stepping = asyncio.ensure_future(server.run_lockstep())
        try:
            task = await run_lockstep(main, path)
            await stepping
        finally:
            await server.close()
        return task

    task = asyncio.run(go())
    m = task.result
    assert (m.reason, m.detail["code"]) == ("operator", "operator")
    assert task.state is S.SUCCEEDED


def test_a_library_monitor_configured_in_a_gate_process():
    async def main():
        first = start(move_to((1.0, 0.0)))
        await first.stopped()
        return first.reason, first.detail["code"]

    scene = {"objects": {"person": list(FAR)}, "nudges": [["person", [0.4, 0.1], 0.3]]}
    monitor = {"min_distance": {"a": "arm.ee", "b": "sim.person", "limit": 0.3,
                                "actuators": ["arm"]}}
    proc, path = spawn_gate("roborun.backends.fake:FakeBackend", scene, checks=monitor)
    try:
        task = asyncio.run(run_lockstep(main, path, until=5.0))
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert task.result == ("rejected", "monitor:min_distance")


def test_a_monitor_with_allow_lets_only_a_retreat_through_its_lock():
    from roborun.checks import Accept, Reject

    def away(cmd, view):  # the straight path starts by moving away from the person
        ee, person, t = view.value("arm.ee"), view.value("sim.person"), cmd.payload["target"]
        dot = sum((t[i] - ee[i]) * (ee[i] - person[i]) for i in range(2))
        return Accept() if dot > 0 else Reject("only away from the person")

    async def main():
        m = person_monitor()
        m.allow = away
        runtime().gate.add_monitor(m)
        first = start(move_to((1.0, 0.0)))
        await first.stopped()
        closer = start(move_to((1.0, 0.0)), wait_timeout=0.5)  # toward the person: refused
        try:
            await closer
        except (TaskFailed, TaskCancelled):
            pass
        back = start(move_to((0.0, 0.0)), wait_timeout=0.5)  # away: granted and admitted
        await back
        return first, closer, back

    task, _ = run(main, objects={"person": FAR}, nudges=[("person", (0.4, 0.1), 0.3)])
    assert task.error is None, task.error
    first, closer, back = task.result
    assert first.detail["code"] == "monitor:min_distance"
    assert (closer.state, closer.reason) == (S.CANCELLED, "rejected")
    assert closer.detail["code"] == "locked"
    assert "stopped by monitor min_distance" in closer.detail["message"]
    assert back.state is S.SUCCEEDED
    assert ("unlock", "monitor:min_distance") in events(task)
