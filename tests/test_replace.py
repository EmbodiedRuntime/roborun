"""Replacing a running program at its next safe point (SPEC §6)."""

import asyncio

from roborun import TaskState, run_sim, runtime, sleep, start, world
from roborun.backends.fake import FakeBackend, move_to

S = TaskState


def test_the_step_in_progress_finishes_then_the_new_version_takes_over():
    steps = []

    async def v1():
        for target in ((0.3, 0.0), (0.3, 0.3), (0.0, 0.3)):
            await move_to(target)  # the second start is the safe point
            steps.append(("v1", target))

    async def v2():
        steps.append(("v2", world.get("arm.ee").value))  # reads where v1 left the arm
        await move_to((0.0, 0.0))
        return "v2 done"

    async def main():
        old = start(v1(), name="program")
        await sleep(0.3)  # in the middle of the first move
        new = runtime().replace(old, v2)
        await new
        return old, new

    task = asyncio.run(run_sim(main, backend=FakeBackend()))
    old, new = task.result
    assert (old.state, old.reason) == (S.CANCELLED, "replaced")
    first_move = old.children[0]
    assert first_move.state is S.SUCCEEDED  # never cut off
    assert steps == [("v1", (0.3, 0.0)), ("v2", (0.3, 0.0))]
    assert new.state is S.SUCCEEDED and new.result == "v2 done"
    assert new.started_at <= old.ended_at  # it existed (and could be watched) from the start


def test_a_program_that_already_ended_is_not_replaced():
    async def quick():
        return 1

    async def main():
        old = start(quick(), name="program")
        await old
        new = runtime().replace(old, lambda: asyncio.sleep(0, result=2))
        return old.reason, await new

    task = asyncio.run(run_sim(main, backend=FakeBackend()))
    assert task.result == (None, 2)  # the old one was not marked; the new one just runs
