"""The Gate across a process boundary (SPEC §3.1, §3.8, §2.1).

Most tests run the Gate server and the runtime in one event loop, talking over a real Unix
socket; the last ones run the Gate as a separate OS process with the MuJoCo backend.
"""

import asyncio
import tempfile
import types
from pathlib import Path

import pytest

from roborun import (
    ResourceBusy,
    TaskCancelled,
    TaskFailed,
    TaskState,
    changes,
    run_sim,
    sleep,
    start,
    world,
)
from roborun.backends.fake import (
    FakeBackend,
    GraspMissed,
    dist,
    grasp,
    grasp_blocking,
    move_to,
    place,
    track,
)
from roborun.gate import Command
from roborun.gateproc import GateServer
from roborun.remote import RemoteGate, run_lockstep

S = TaskState
CUP, NUDGED, TRAY = (0.5, 0.0), (0.5, 0.15), (0.0, 0.3)


def socket_path() -> str:
    return str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")


def over_socket(main, backend, *, until=30.0, **server_kwargs):
    async def go():
        path = socket_path()
        server = GateServer(backend, dt=0.01, **server_kwargs)
        await server.start(path)
        stepping = asyncio.ensure_future(server.run_lockstep())
        try:
            task = await run_lockstep(main, path, until=until)
            await stepping
        finally:
            await server.close()
        return task

    return asyncio.run(go())


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


def nudged_fake():
    b = FakeBackend(objects={"cup": CUP})
    b.nudge("cup", NUDGED, at=0.5)
    return b


def test_the_same_program_behaves_identically_in_process_and_over_the_socket():
    local_backend = nudged_fake()
    local = asyncio.run(run_sim(task_based, backend=local_backend))
    remote_backend = nudged_fake()
    remote = over_socket(task_based, remote_backend)
    assert remote.state is S.SUCCEEDED
    assert remote.ended_at == local.ended_at
    assert remote_backend.truth("cup") == local_backend.truth("cup")
    assert [(t, c.kind, c.payload) for t, c in remote_backend.log] == [
        (t, c.kind, {k: tuple(v) if isinstance(v, list) else v for k, v in c.payload.items()})
        for t, c in local_backend.log
    ]


def test_blocking_program_fails_over_the_socket():
    task = over_socket(blocking, nudged_fake())
    assert task.state is S.FAILED and isinstance(task.error.cause, GraspMissed)


def test_a_command_without_a_token_is_rejected_by_the_gate_process():
    async def rogue():
        await send("goal", "arm", target=(1.0, 0.0))

    from roborun import send

    async def main():
        t = start(rogue())
        await t.stopped()
        return t.state, t.reason, t.detail["code"]

    backend = FakeBackend()
    task = over_socket(main, backend)
    assert task.result == (S.CANCELLED, "rejected", "unauthorized")
    assert not any(c.kind == "goal" for _, c in backend.log)


def test_take_revokes_first_and_moves_ownership_after_the_owner_stops():
    async def main():
        first = start(move_to((1.0, 0.0)))
        await sleep(0.2)
        second = start(move_to((0.0, 0.3)), on_busy="take")
        try:
            await first
        except TaskCancelled as e:
            reason = e.reason
        await second
        return first, second, reason

    backend = FakeBackend()
    task = over_socket(main, backend)
    first, second, reason = task.result
    assert reason == "preempted" and second.state is S.SUCCEEDED
    first_goal = min(t for t, c in backend.log if c.kind == "goal" and c.payload["target"] == (0.0, 0.3))
    assert first_goal >= first.ended_at - 1e-9


def test_wait_timeout_runs_on_the_gate_clock():
    async def main():
        a = start(move_to((1.0, 0.0)))
        await sleep(0.1)
        b = start(move_to((0.0, 1.0)), wait_timeout=0.2)
        try:
            await b
        except TaskFailed as e:
            cause = type(e.cause)
        a.cancel()
        await a.stopped()
        return cause, b.ended_at

    cause, ended_at = over_socket(main, FakeBackend()).result
    assert cause is ResourceBusy and abs(ended_at - 0.3) < 0.02


def test_a_disconnected_client_loses_its_tokens_and_the_arm_stops():
    async def go():
        path = socket_path()
        backend = FakeBackend()
        server = GateServer(backend)
        await server.start(path)
        try:
            client = await RemoteGate.connect(path)
            await client.hello(session="agent", mode="client")
            me = types.SimpleNamespace(id="t1", name="agent-task", session="agent", parent=None)
            grant = await client.acquire(me, {"arm"})
            assert await client.send(me, Command("goal", "arm", {"target": (1.0, 0.0)}))
            assert server.gate.owner("arm") is not None
            client._writer.transport.abort()  # the runtime process dies
            for _ in range(50):
                await asyncio.sleep(0.01)
                if server.gate.owner("arm") is None:
                    break
            return backend, server, grant
        finally:
            await server.close()

    backend, server, _grant = asyncio.run(go())
    assert server.gate.owner("arm") is None and not server.tokens
    assert backend.log[-1][1].kind == "stop"


# -- a real Gate process with the MuJoCo backend ---------------------------------------

mj = pytest.importorskip("roborun.backends.mujoco")
needs_panda = pytest.mark.skipif(
    not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists(),
    reason="MuJoCo Menagerie Panda model not found",
)

MJ_CUP, MJ_TRAY = (0.5, 0.0, 0.04), (0.3, 0.35, 0.045)
MJ_SCENE = {"objects": {"cup": [0.5, 0.0]}, "nudges": [["cup", [0.5, 0.15], 1.0]]}


async def mj_task_based():
    start(mj.track("cup"), detached=True)
    cup = await world.next("cup")
    g = start(mj.grasp(cup.value))
    async for cup in changes("cup", while_running=g):
        if mj.dist(cup.value, g.target) > 0.02:
            g.update(target=cup.value)
    await g
    await mj.place(MJ_TRAY)


async def mj_blocking():
    await mj.grasp_blocking(MJ_CUP)
    await mj.place(MJ_TRAY)


async def mj_unreachable():
    try:
        await mj.move_to((1.5, 0.0, 0.3))
    except TaskFailed as e:
        return e.cause


def in_gate_process(main, scene=MJ_SCENE):
    from roborun.gateproc import spawn_gate

    proc, path = spawn_gate("roborun.backends.mujoco:MujocoBackend", scene)
    try:
        task = asyncio.run(run_lockstep(main, path, until=20.0))
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    return task


@needs_panda
def test_cup_nudge_in_a_separate_gate_process_matches_in_process():
    local_backend = mj.MujocoBackend(objects={"cup": (0.5, 0.0)}, nudges=[("cup", (0.5, 0.15), 1.0)])
    local = asyncio.run(run_sim(mj_task_based, backend=local_backend, until=20.0))
    remote = in_gate_process(mj_task_based)
    assert remote.state is S.SUCCEEDED, remote.error
    assert remote.ended_at == local.ended_at
    cup = remote.runtime.world.get("sim.cup").value
    assert mj.dist(cup[:2], MJ_TRAY[:2]) <= 0.015


@needs_panda
def test_blocking_fails_and_backend_errors_cross_the_process_boundary():
    blocking_task = in_gate_process(mj_blocking)
    assert blocking_task.state is S.FAILED
    assert isinstance(blocking_task.error.cause, mj.GraspMissed)
    err = in_gate_process(mj_unreachable).result
    assert err.code == "unreachable" and err.source == "mujoco.panda"
