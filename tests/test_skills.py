"""The backend-independent skills (roborun.skills): phases, and their checks as expectations."""

import asyncio

from roborun import (
    ExpectationFailed,
    TaskState,
    during,
    phase,
    run_sim,
    sleep,
    start,
    timeout,
    world,
)
from roborun.api import action, runtime
from roborun.backends.fake import ROBOT, FakeBackend, track
from roborun.expect import at_end, at_start
from roborun.skills import Dropped, GraspMissed, NothingHeld, library
from roborun.trace import load

S = TaskState
CUP, TRAY = (0.3, 0.0), (0.0, 0.3)
lib = library(ROBOT)


def names(task):
    return [p["name"] for p in task.phases]


def goals(backend, after=0.0):
    return [c for t, c in backend.log if c.kind == "goal" and t >= after]


# -- the sequences -----------------------------------------------------------------------------


def test_a_grasp_and_a_place_go_through_their_phases(sim):
    async def main():
        g = start(lib.grasp(CUP))
        await g
        p = start(lib.place(TRAY))
        await p
        return names(g), names(p), g.verified, p.verified, g.phases

    task, backend = sim(main, objects={"cup": CUP})
    grasp, place, g_verified, p_verified, phases = task.result
    assert grasp == ["open", "approach", "descend", "close", "lift"]
    assert place == ["approach", "lower", "release", "retreat"]
    assert g_verified is True and p_verified is True
    assert all(p["end"] >= p["start"] for p in phases)
    assert phases[-1]["start"] == phases[-2]["end"]  # one follows the other
    assert backend.truth("cup") == TRAY


def test_pick_and_place_is_one_task_that_keeps_the_arm(sim):
    async def main():
        start(track("cup"), detached=True)
        await sleep(0.1)
        t = start(lib.pick_and_place("cup", TRAY))
        await sleep(0.3)
        other = start(lib.move_to((0.5, 0.5)), on_busy="fail")  # between the steps too
        return await t.stopped(), names(t), await other.stopped(), [c.name for c in t.children]

    task, backend = sim(main, objects={"cup": CUP})
    state, phases, other, children = task.result
    assert (state, phases) == (S.SUCCEEDED, ["pick", "place"])
    assert other is S.FAILED and children == ["grasp", "place"]
    assert backend.truth("cup") == TRAY


# -- the skills' own checks are expectations ------------------------------------------------------


def test_closing_on_nothing_is_a_failed_expectation_named_grasp_missed(sim):
    async def main():
        t = start(lib.grasp(CUP))
        state = await t.stopped()
        return state, t.error

    task, backend = sim(main)  # no cup
    state, error = task.result
    assert state is S.FAILED and isinstance(error, GraspMissed)
    assert isinstance(error, ExpectationFailed)
    assert (error.name, error.when, error.phase) == ("grasped", "end", "close")
    assert str(error).startswith("closed on nothing at")
    assert error.evidence["gripper.holding"]["value"] is None
    assert backend.log[-1][1].kind == "effector"  # it did not lift: nothing was sent after


def test_a_slip_on_the_way_stops_the_arm_and_fails_as_dropped(sim):
    async def main():
        await lib.grasp(CUP)
        backend = runtime().backend
        backend.at(runtime().clock.now() + 0.2, backend.drop)
        t = start(lib.place((0.9, 0.0)))
        state = await t.stopped()
        return state, t.error, world.get("arm.moving").value, world.get("arm.ee").value

    state, error, moving, ee = sim(main, objects={"cup": CUP})[0].result
    assert state is S.FAILED and isinstance(error, Dropped)
    assert (error.name, error.when, error.phase) == ("holding", "during", "approach")
    assert "slipped out of the gripper" in str(error)
    assert moving is False and ee[0] < 0.6  # stopped well short of the target


def test_placing_with_an_empty_gripper_fails_before_anything_is_sent(sim):
    async def main():
        t = start(lib.place(TRAY))
        return await t.stopped(), t.error

    task, backend = sim(main)
    state, error = task.result
    assert state is S.FAILED and isinstance(error, NothingHeld)
    assert (error.when, error.phase) == ("start", None)
    assert backend.log == []


# -- did the command take effect ------------------------------------------------------------------


def test_a_gripper_that_does_not_respond_fails_the_grasp(sim):
    async def main():
        runtime().backend.gripper_jammed = True
        t = start(lib.grasp(CUP))
        state = await t.stopped()
        return state, t.error

    task, backend = sim(main, objects={"cup": CUP})
    state, error = task.result
    assert state is S.FAILED and type(error) is ExpectationFailed  # not "closed on nothing"
    assert (error.name, error.phase) == ("gripper responds", "close")
    assert "told to close and is still 0.080 m open" in str(error)
    assert backend.holding is None


def test_a_stop_that_does_not_end_fails_the_cancellation(sim, monkeypatch):
    import roborun.task as task_module

    monkeypatch.setattr(task_module, "REST_TIMEOUT", 0.5)

    async def main():
        runtime().backend.brakes = False
        t = start(lib.move_to((3.0, 0.0)))  # six seconds away
        await sleep(0.2)
        t.cancel()
        state = await t.stopped()
        return state, t.error, runtime().clock.now()

    task, _ = sim(main)
    state, error, ended = task.result
    assert state is S.FAILED and isinstance(error, ExpectationFailed)
    assert error.name == "arm stops" and "still moving" in str(error)
    assert 2.0 < ended < 3.5  # its cleanup did not wait for the arm for ever


def test_a_stop_that_ends_is_an_ordinary_cancellation(sim):
    async def main():
        t = start(lib.move_to((3.0, 0.0)))
        await sleep(0.2)
        t.cancel()
        return await t.stopped()

    assert sim(main)[0].result is S.CANCELLED


# -- phases ---------------------------------------------------------------------------------------


@action(resources={"arm"})
async def two_steps():
    await phase("first")
    await sleep(0.3)
    await phase("second")
    await sleep(0.3)


def flag(name, **kwargs):
    return during(lambda v: v.value(name), inputs={name: 1.0}, name=name, **kwargs)


def test_an_expectation_limited_to_a_phase_is_not_watched_outside_it(sim):
    async def main():
        world.write("ok", False)  # false all along
        t = start(two_steps().expect(flag("ok", phase="second")))
        state = await t.stopped()
        return state, t.error.phase, t.ended_at - t.started_at

    state, where, took = sim(main)[0].result
    assert state is S.FAILED and where == "second"
    assert 0.3 <= took < 0.35  # the first phase ran to its end


def test_expectations_at_the_start_and_the_end_of_a_phase(sim):
    ready = at_start(lambda v: v.value("ready"), inputs={"ready": 1.0}, phase="second", name="ready")
    done = at_end(lambda v: v.value("done"), inputs={"done": 1.0}, phase="first", name="done")

    async def run(ready_value, done_value, *exps):
        world.write("ready", ready_value)
        world.write("done", done_value)
        t = start(two_steps().expect(*exps))
        state = await t.stopped()
        return state, getattr(t.error, "name", None), names(t), t.verified

    async def main():
        return (await run(True, True, ready, done), await run(False, True, ready),
                await run(True, False, done))

    fine, not_ready, not_done = sim(main)[0].result
    assert fine == (S.SUCCEEDED, None, ["first", "second"], True)
    assert not_ready == (S.FAILED, "ready", ["first", "second"], None)
    assert not_done == (S.FAILED, "done", ["first"], None)  # it never entered the second


def test_phases_and_the_failing_phase_are_in_the_trace(tmp_path):
    async def main():
        await start(lib.grasp(CUP)).stopped()

    path = tmp_path / "t.jsonl"
    asyncio.run(run_sim(main, backend=FakeBackend(), trace=str(path)))
    _, spans = load(path)
    span = next(s for s in spans if s.get("name") == "grasp")
    assert [p["name"] for p in span["phases"]] == ["open", "approach", "descend", "close"]
    assert span["error"]["type"] == "GraspMissed" and span["error"]["phase"] == "close"
    assert span["expectations"][-1]["phase"] == "close"


# -- what can be said before the task runs --------------------------------------------------------


def test_expectations_that_cannot_work_are_reported_when_the_task_starts(sim):
    seen = during(lambda v: True, inputs={"cup": 0.02}, name="cup seen")  # tracked at 20 Hz
    missing = during(lambda v: True, inputs={"lid": 0.5}, name="lid seen")
    slow = during(lambda v: True, inputs={"arm.ee": 0.1}, grace=5.0, name="slow")
    fine = during(lambda v: True, inputs={"cup": 0.2}, name="fine")

    async def main():
        start(track("cup"), detached=True)
        await sleep(0.2)
        async with timeout(2.0):
            t = start(lib.move_to((0.1, 0.0)).expect(seen, missing, slow, fine))
            await t
        return t.warnings, t.state

    warnings, state = sim(main, objects={"cup": CUP})[0].result
    assert state is S.SUCCEEDED  # warnings do not fail it
    assert [w["name"] for w in warnings] == ["cup seen", "lid seen", "slow"]
    assert "shorter than the 0.050s period of track(cup)" in warnings[0]["warning"]
    assert "nothing has written 'lid'" in warnings[1]["warning"]
    assert "grace=5.0s does not fit in the task's timeout" in warnings[2]["warning"]
