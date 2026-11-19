"""Expectations and monitoring (roborun.expect): what a task must find true of the World."""

import json

from roborun import ExpectationFailed, TaskState, action, at_end, at_start, sleep, start, world
from roborun.backends.fake import grasp, move_to
from roborun.expect import near
from roborun.harness import status
from roborun.trace import load, show_failed

S = TaskState
CUP = (0.3, 0.0)

# The fake's move_to without the expectation it comes with, to test one expectation at a time.
bare_move = action(resources={"arm"})(move_to.__wrapped_body__)


def holding(name):
    return at_end(lambda v: v.value("gripper.holding") == name,
                  inputs={"gripper.holding": 0.05}, name=f"holding {name}")


arrived = at_end(lambda v: near(v.value("arm.ee"), v.params["target"], 0.01),
                 inputs={"arm.ee": 0.05}, name="arrived")


def test_an_expectation_that_holds_marks_the_success_verified(sim):
    async def main():
        t = start(bare_move((0.2, 0.0)).expect(arrived))
        await t
        return t.state, t.verified, t.checks, status(t)

    state, verified, checks, st = sim(main)[0].result
    assert state is S.SUCCEEDED and verified is True and st["verified"] is True
    assert [(c["name"], c["verdict"]) for c in checks] == [("arrived", "holds")]
    assert "arm.ee" in checks[0]["evidence"]  # the values the verdict rests on


def test_a_task_without_end_expectations_is_neither_verified_nor_unverified(sim):
    async def main():
        t = start(bare_move((0.2, 0.0)))
        await t
        return t.verified, "verified" in status(t)

    assert sim(main)[0].result == (None, False)


def test_a_false_success_becomes_a_failure_with_the_evidence(sim):
    # The move itself goes well; what the caller expected of the world is not so.
    async def main():
        t = start(bare_move((0.2, 0.0)).expect(holding("cup")))
        return await t.stopped(), t.error, t.verified

    task, _ = sim(main, objects={"cup": CUP})
    state, error, verified = task.result
    assert state is S.FAILED and isinstance(error, ExpectationFailed) and verified is None
    assert (error.name, error.when, error.verdict) == ("holding cup", "end", "violated")
    assert error.evidence["gripper.holding"]["value"] is None
    assert error.evidence["gripper.holding"]["source"] == "fake.gripper"


def test_stale_inputs_are_unknown_and_success_is_unverified(sim):
    # Nothing writes "cup": old data never counts as true.
    seen = at_end(lambda v: True, inputs={"cup": 0.1}, name="cup seen")

    async def main():
        t = start(bare_move((0.2, 0.0)).expect(seen))
        await t
        world.write("cup", CUP, source="test")
        await sleep(0.5)  # now it exists, but is too old
        t2 = start(bare_move((0.1, 0.0)).expect(seen))
        await t2
        return (t.state, t.verified, t.checks[0]["reason"]), (t2.verified, t2.checks[0]["reason"])

    first, second = sim(main)[0].result
    assert first == (S.SUCCEEDED, False, "no value for 'cup'")
    assert second[0] is False and "old (limit 0.1s)" in second[1]


def test_an_expectation_at_the_start_fails_the_task_before_it_sends_anything(sim):
    holds_something = at_start(lambda v: v.value("gripper.holding") is not None,
                               inputs={"gripper.holding": 0.05}, name="holds something")

    async def main():
        t = start(bare_move((0.5, 0.0)).expect(holds_something))
        state = await t.stopped()
        return state, t.error

    task, backend = sim(main)
    state, error = task.result
    assert state is S.FAILED and (error.when, error.verdict) == ("start", "violated")
    assert not backend.log  # no command reached the robot


def test_unknown_at_the_start_fails_too(sim):
    needs_cup = at_start(lambda v: True, inputs={"cup": 0.1}, name="cup seen")

    async def main():
        t = start(bare_move((0.5, 0.0)).expect(needs_cup))
        await t.stopped()
        return t.state, t.error.verdict

    task, backend = sim(main)
    assert task.result == (S.FAILED, "unknown") and not backend.log


def test_stable_requires_the_condition_to_keep_holding(sim):
    # The arm is at the target when move_to returns, then something else moves it away.
    stays = at_end(lambda v: near(v.value("arm.ee"), v.params["target"], 0.01),
                   inputs={"arm.ee": 0.05}, name="stays", stable=0.5)

    async def main():
        async def disturb():
            await sleep(0.55)  # move_to takes 0.4 s; this lands inside the window
            import roborun
            roborun.runtime().backend.ee = (0.5, 0.5)

        start(disturb())
        t = start(bare_move((0.2, 0.0)).expect(stays))
        state = await t.stopped()
        return state, [c["verdict"] for c in t.checks], t.ended_at

    state, verdicts, ended = sim(main)[0].result
    assert state is S.FAILED and verdicts == ["holds", "violated"]
    assert 0.55 <= ended < 0.7


def test_a_condition_that_raises_proves_nothing(sim):
    broken = at_end(lambda v: v.value("no such key"), inputs={"arm.ee": 0.05}, name="broken")

    async def main():
        t = start(bare_move((0.2, 0.0)).expect(broken))
        await t
        return t.verified, t.checks[0]["verdict"], t.checks[0]["reason"]

    verified, verdict, reason = sim(main)[0].result
    assert verified is False and verdict == "unknown" and "raised" in reason


def test_a_grasp_confirmed_by_what_the_gripper_holds(sim):
    async def main():
        t = start(grasp(CUP).expect(holding("cup")))
        await t
        return t.verified

    task, _ = sim(main, objects={"cup": CUP})
    assert task.result is True


def test_expectations_are_in_the_trace(sim, tmp_path):
    path = tmp_path / "run.jsonl"

    async def main():
        t = start(bare_move((0.2, 0.0)).expect(holding("cup")))
        await t.stopped()

    import asyncio

    from roborun import run_sim
    from roborun.backends.fake import FakeBackend

    asyncio.run(run_sim(main, backend=FakeBackend(objects={"cup": CUP}), trace=str(path)))
    _, spans = load(path)
    span = next(s for s in spans if s.get("name") == "move_to")
    assert span["status"] == "failed" and "verified" not in span
    assert span["expectations"][0]["verdict"] == "violated"
    assert span["error"]["expectation"] == "holding cup"
    assert span["error"]["evidence"]["gripper.holding"]["value"] is None
    json.dumps(span)  # plain data
    assert "holding cup" in show_failed(path)


# -- while the task runs -----------------------------------------------------------------------

from roborun import TaskFailed, during, hold, runtime, send
from roborun.backends.fake import track
from roborun.expect import after, consistent, disagree, no, not_stalled, relation


def test_a_stalled_goal_fails_the_task_instead_of_waiting_for_ever(sim):
    # The arm jams on the way. move_to would wait for an arrival that never comes.
    async def main():
        backend = runtime().backend
        backend.at(0.3, lambda: setattr(backend, "jammed", True))
        t = start(bare_move((1.0, 0.0)).expect(not_stalled(within=0.5)))
        state = await t.stopped()
        return state, t.error, t.reason, t.ended_at, world.get("arm.moving").value

    task, backend = sim(main, until=20.0)
    state, error, reason, ended, moving = task.result
    assert state is S.FAILED and reason is None and isinstance(error, ExpectationFailed)
    assert (error.name, error.when, error.verdict) == ("arm not stalled", "during", "violated")
    assert "at rest 0.8" in error.reason and "arm.ee" in error.evidence
    assert 0.8 <= ended < 1.0  # 0.3 s moving, then 0.5 s of grace
    assert moving is False and backend.log[-1][1].kind == "stop"  # the Gate stopped it


def test_without_the_expectation_the_same_move_never_ends(sim):
    async def main():
        backend = runtime().backend
        backend.at(0.3, lambda: setattr(backend, "jammed", True))
        t = start(bare_move((1.0, 0.0)))
        await sleep(5.0)
        state = t.state
        t.cancel()
        return state

    assert sim(main, until=20.0)[0].result is S.RUNNING


def test_a_brief_failure_within_the_grace_does_not_fail_the_task(sim):
    async def main():
        backend = runtime().backend
        backend.at(0.2, lambda: setattr(backend, "jammed", True))
        backend.at(0.4, lambda: setattr(backend, "jammed", False))
        t = start(bare_move((0.5, 0.0)).expect(not_stalled(within=0.5)))
        await t
        return t.verified, [c["verdict"] for c in t.checks]

    verified, verdicts = sim(main)[0].result
    assert verified is True and verdicts == ["holds", "violated", "holds"]


def test_a_parent_sees_the_failure_as_a_failed_child_with_the_cause(sim):
    async def main():
        backend = runtime().backend
        backend.at(0.1, lambda: setattr(backend, "jammed", True))
        try:
            await bare_move((1.0, 0.0)).expect(not_stalled(within=0.2))
        except TaskFailed as e:
            return type(e.root).__name__, e.path

    assert sim(main)[0].result == ("ExpectationFailed", ["move_to"])


def test_an_unknown_during_the_task_leaves_success_unverified(sim):
    watched = during(lambda v: True, inputs={"cup": 0.05}, name="cup seen")

    async def main():
        t = start(bare_move((0.2, 0.0)).expect(watched))  # nothing writes "cup"
        await t
        return t.state, t.verified

    assert sim(main)[0].result == (S.SUCCEEDED, False)


def test_a_command_that_does_not_take_effect(sim):
    # 0.3 s after a goal is accepted, the arm must have left where it was.
    def left_start(v, cmd):
        return v.value("arm.ee") != (0.0, 0.0) or no("the arm has not moved")

    responds = after("arm", "goal", left_start, within=0.3, inputs={"arm.ee": 0.05})

    async def main():
        runtime().backend.jammed = True
        t = start(bare_move((0.5, 0.0)).expect(responds))
        state = await t.stopped()
        return state, t.error.name, t.error.reason, round(t.ended_at, 2)

    assert sim(main)[0].result == (S.FAILED, "goal on arm takes effect", "the arm has not moved", 0.3)


def test_a_program_sees_the_commands_its_children_sent(sim):
    async def main():
        async def program():
            await move_to((0.2, 0.0))

        p = start(program())
        await p
        return p.commands["arm"]["kind"], p.commands["arm"]["payload"]["target"]

    assert sim(main)[0].result == ("goal", (0.2, 0.0))


# -- sources that disagree ---------------------------------------------------------------------


def held_is_at_the_gripper(v):
    """What the gripper says it holds must be where the gripper is."""
    name = v.value("gripper.holding")
    if name is None:
        return True
    d = sum((a - b) ** 2 for a, b in zip(v.value("cup"), v.value("arm.ee"))) ** 0.5
    return d <= 0.05 or disagree(f"gripper.holding says {name}, but cup is {d:.2f} m from arm.ee")


HELD = {"gripper.holding": 0.05, "cup": 0.2, "arm.ee": 0.05}


def test_a_relation_reports_that_sources_agree_then_that_they_do_not(sim):
    async def main():
        backend = runtime().backend
        start(track("cup"), detached=True)
        start(relation("held", held_is_at_the_gripper, inputs=HELD), detached=True)
        await grasp(CUP)
        await sleep(0.1)
        before = world.get("relation.held")
        backend.objects["cup"] = (0.9, 0.9)  # the cup is gone; the contact sensor is stuck
        backend.step = _stuck(backend)
        await sleep(0.2)
        return before, world.get("relation.held")

    before, after_ = sim(main, objects={"cup": CUP})[0].result
    assert before.value["status"] == "consistent" and before.source == "relation(held)"
    assert after_.value["status"] == "inconsistent"
    assert "gripper.holding says cup" in after_.value["reason"]
    assert set(after_.value["sources"]) == set(HELD)  # both sides, with stamps and sources
    assert after_.value["sources"]["cup"]["source"] == "track(cup)"


def _stuck(backend):
    """The fake's step, with a cup that no longer follows the gripper although it 'holds' it."""
    step = backend.step

    def stepped(dt):
        cup = backend.objects["cup"]
        step(dt)
        backend.objects["cup"] = cup
        backend._publish()

    return stepped


def test_a_task_that_depends_on_sources_that_disagree_fails_as_inconsistent(sim):
    async def main():
        backend = runtime().backend
        start(track("cup"), detached=True)
        start(relation("held", held_is_at_the_gripper, inputs=HELD), detached=True)
        async with hold("arm", "gripper"):
            await grasp(CUP)

            def lose_the_cup():
                backend.objects["cup"] = CUP
                backend.step = _stuck(backend)

            backend.at(runtime().clock.now() + 0.3, lose_the_cup)
            carry = start(bare_move((0.0, 1.0)).expect(consistent("held", grace=0.05)))
            state = await carry.stopped()
            await send("stop", "arm")
        return state, carry.error, world.get("arm.moving").value

    state, error, moving = sim(main, objects={"cup": CUP})[0].result
    assert state is S.FAILED and error.verdict == "inconsistent"
    assert "gripper.holding says cup" in error.reason
    sources = error.evidence["relation.held"]["value"]["sources"]
    assert sources["gripper.holding"]["value"] == "cup" and sources["cup"]["source"] == "track(cup)"
    assert moving is False


def test_a_relation_with_a_stale_source_is_unknown_not_consistent(sim):
    async def main():
        start(relation("held", held_is_at_the_gripper, inputs=HELD), detached=True)  # no tracker
        await sleep(0.2)
        t = start(bare_move((0.2, 0.0)).expect(consistent("held")))
        await t
        return world.get("relation.held").value["status"], t.verified

    assert sim(main, objects={"cup": CUP})[0].result == ("unknown", False)


def test_the_library_move_expects_not_to_stall(sim):
    # Skills that send goals come with this expectation; nobody has to remember it.
    async def main():
        backend = runtime().backend
        backend.at(0.3, lambda: setattr(backend, "jammed", True))
        t = start(move_to((1.0, 0.0)))
        state = await t.stopped()
        backend.jammed = False
        ok = start(move_to((0.3, 0.0)))
        await ok
        return state, t.error.name, round(t.ended_at, 1), ok.verified

    assert sim(main, until=20.0)[0].result == (S.FAILED, "arm not stalled", 1.3, True)
