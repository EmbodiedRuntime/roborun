"""Gate checks (SPEC §3.4): the Gate calls them under a budget and fails closed."""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from roborun import TaskCancelled, TaskState, run_sim, runtime, start, world
from roborun.backends.fake import FakeBackend, move_to
from roborun.checks import Accept, Reject, max_jump, workspace_box
from roborun.gateproc import GateServer
from roborun.remote import run_lockstep

S = TaskState
BOX = workspace_box(lo=(-1.0, -1.0), hi=(1.0, 1.0))


def run(main, *checks, **backend_kwargs):
    backend = FakeBackend(**backend_kwargs)

    async def with_checks():
        for c in checks:
            runtime().gate.add_check(c)
        return await main()

    task = asyncio.run(run_sim(with_checks, backend=backend))
    return task, backend


async def try_move(target):
    m = start(move_to(target))
    try:
        await m
    except TaskCancelled:
        pass
    return m, world.get("arm.moving").value


def goals(backend, target):
    return [t for t, c in backend.log if c.kind == "goal" and tuple(c.payload["target"]) == target]


def test_a_target_inside_the_workspace_is_accepted():
    task, _backend = run(lambda: try_move((0.3, 0.0)), BOX)
    m, _ = task.result
    assert m.state is S.SUCCEEDED
    assert [r.verdict for r in runtime_log(task)] == ["accept"]


def runtime_log(task):
    return task.runtime.gate.checks.log


def test_a_target_outside_the_workspace_is_rejected_and_the_arm_ends_at_rest():
    task, backend = run(lambda: try_move((1.5, 0.0)), BOX)
    m, moving = task.result
    assert (m.state, m.reason) == (S.CANCELLED, "rejected")
    assert m.detail["code"] == "check:workspace"
    assert "outside the workspace" in m.detail["message"]
    assert not goals(backend, (1.5, 0.0))  # the backend never saw it
    assert moving is False
    rec = runtime_log(task)[-1]
    assert (rec.check, rec.verdict, rec.kind) == ("workspace", "reject", "goal")


def test_stop_is_never_checked():
    def reject_all(cmd, view):
        return Reject("nothing moves")

    task, backend = run(lambda: try_move((0.3, 0.0)), reject_all)
    m, _ = task.result
    assert m.detail["code"] == "check:reject_all"
    # The Gate's fallback stops the arm, and the skill's own cleanup stop still gets through.
    assert [c.kind for _, c in backend.log] == ["stop", "stop"]


def test_a_missing_or_old_input_rejects_without_calling_the_check():
    called = []

    def needs_people(cmd, view):
        called.append(True)
        return Accept()

    async def main():
        runtime().gate.add_check(needs_people, inputs={"scene.humans": 0.2})
        return await try_move((0.3, 0.0))

    task, _ = run(main)
    m, _ = task.result
    assert m.detail["code"] == "stale:scene.humans"
    assert not called


def test_a_slow_check_is_rejected_at_its_budget_without_stalling_the_gate():
    def slow(cmd, view):
        time.sleep(0.5)
        return Accept()

    async def main():
        runtime().gate.add_check(slow, budget=0.02)
        t0 = time.perf_counter()
        result = await try_move((0.3, 0.0))
        return result, time.perf_counter() - t0

    task, backend = run(main)
    (m, moving), wall = task.result
    assert m.detail["code"] == "check_timeout:slow"
    assert wall < 0.4  # the program did not wait for the slow check
    assert backend.log[-1][1].kind == "stop" and moving is False


def test_a_check_that_raises_or_returns_garbage_fails_closed():
    def broken(cmd, view):
        raise RuntimeError("bug in the check")

    def sloppy(cmd, view):
        return True

    for check, code in ((broken, "check_error:broken"), (sloppy, "check_error:sloppy")):
        task, backend = run(lambda: try_move((0.3, 0.0)), check)
        assert task.result[0].detail["code"] == code
        assert not goals(backend, (0.3, 0.0))


def test_checks_see_stamped_inputs_from_the_gate():
    seen = []

    def records(cmd, view):
        seen.append((view.value("arm.ee"), view["arm.ee"].source, view.now))
        return Accept()

    async def main():
        runtime().gate.add_check(records, inputs={"arm.ee": 0.05})
        return await try_move((0.3, 0.0))

    run(main)
    assert seen and seen[0][0] == (0.0, 0.0) and seen[0][1] == "fake.arm"


def test_max_jump_rejects_a_large_setpoint_step():
    async def jump():
        from roborun import send

        await send("setpoint", "arm", target=(0.5, 0.0))

    async def main():
        runtime().gate.add_check(max_jump(0.05))
        t = start(jump(), resources={"arm"})
        await t.stopped()
        return t

    task, _ = run(main)
    assert task.result.detail["code"] == "check:max_jump"


def test_registering_a_check_for_stop_or_twice_is_an_error():
    def ok(cmd, view):
        return Accept()

    async def main():
        gate = runtime().gate
        with pytest.raises(ValueError):
            gate.add_check(ok, kinds={"stop"})
        gate.add_check(ok)
        with pytest.raises(ValueError):
            gate.add_check(ok)
        return True

    assert run(main)[0].result is True


def test_checks_run_in_the_gate_process_and_rejections_cross_the_socket():
    async def go():
        path = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")
        backend = FakeBackend()
        server = GateServer(backend)
        server.checks.add_check(workspace_box(lo=(-1.0, -1.0), hi=(1.0, 1.0)))
        await server.start(path)
        stepping = asyncio.ensure_future(server.run_lockstep())
        try:
            task = await run_lockstep(lambda: try_move((1.5, 0.0)), path)
            await stepping
        finally:
            await server.close()
        return task, backend, server

    task, backend, server = asyncio.run(go())
    m, moving = task.result
    assert m.detail["code"] == "check:workspace" and moving is False
    assert not goals(backend, (1.5, 0.0))
    assert server.checks.log[-1].verdict == "reject"


# -- on MuJoCo: a command that would hit the table is rejected -------------

mj = pytest.importorskip("roborun.backends.mujoco")
needs_panda = pytest.mark.skipif(
    not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists(),
    reason="MuJoCo Menagerie Panda model not found",
)
# The tool may not go below 2 cm above the table surface (z = 0).
TABLE_BOX = {"workspace_box": {"lo": [0.15, -0.6, 0.02], "hi": [0.8, 0.6, 0.8]}}


async def into_the_table():
    await mj.move_to((0.5, 0.0, 0.3))
    m = start(mj.move_to((0.5, 0.0, -0.05)))  # would drive the gripper into the table
    try:
        await m
    except TaskCancelled:
        pass
    return m.state, m.reason, m.detail, world.get("arm.moving").value, world.get("arm.ee").value


@needs_panda
def test_m2_a_command_that_would_hit_the_table_is_rejected_in_the_gate_process():
    from roborun.gateproc import spawn_gate

    proc, path = spawn_gate("roborun.backends.mujoco:MujocoBackend", {}, checks=TABLE_BOX)
    try:
        task = asyncio.run(run_lockstep(into_the_table, path, until=10.0))
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    state, reason, detail, moving, ee = task.result
    assert (state, reason, detail["code"]) == (S.CANCELLED, "rejected", "check:workspace")
    assert moving is False  # the robot ends at rest
    assert mj.dist(ee, (0.5, 0.0, 0.3)) <= mj.REACH_TOL  # where it was, not in the table
