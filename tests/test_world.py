"""Stamped values, World, and changes (SPEC §2)."""

import pytest

from roborun import QUEUE, Overflow, StaleData, TaskState, changes, sleep, start, world
from roborun.backends.fake import track

S = TaskState


def test_get_raises_stale_data_with_the_source(sim):
    async def main():
        world.declare("cup", max_age=0.1)
        world.write("cup", (1.0, 2.0), source="camera")
        assert world.get("cup").value == (1.0, 2.0)
        await sleep(0.3)
        assert world.get("cup", max_age=1.0).value == (1.0, 2.0)  # explicit limit overrides
        with pytest.raises(StaleData) as e:
            world.get("cup")
        return str(e.value)

    task, _ = sim(main)
    assert "'cup'" in task.result and "camera" in task.result


def test_missing_value_says_to_check_the_producer(sim):
    async def main():
        with pytest.raises(StaleData, match="producer"):
            world.get("nothing")
        return True

    task, _ = sim(main)
    assert task.result is True


def test_changes_raise_stale_data_when_the_tracker_stops(sim):
    async def watcher():
        async for _ in changes("cup"):
            pass

    async def main():
        world.declare("cup", max_age=0.2)
        tracker = start(track("cup"), detached=True)
        await world.next("cup")
        w = start(watcher())
        await sleep(0.5)
        tracker.cancel()
        await w.stopped()
        return w, w.ended_at

    task, _ = sim(main, objects={"cup": (0.5, 0.0)})
    w, ended_at = task.result
    assert w.state is S.FAILED and isinstance(w.error, StaleData)
    assert 0.6 < ended_at < 0.85  # about max_age after the last value


def test_state_keys_skip_to_the_newest_value(sim):
    async def main():
        it = changes("x")
        for i in range(3):
            world.write("x", i)
        return (await anext(it)).value

    task, _ = sim(main)
    assert task.result == 2


def test_queue_keys_deliver_every_message_to_every_reader(sim):
    async def main():
        world.declare("instr", QUEUE, maxlen=8)
        world.write("instr", "red")
        world.write("instr", "blue")
        a, b = changes("instr"), changes("instr")  # new readers start at the oldest message
        world.write("instr", "green")
        got_a = [(await anext(a)).value for _ in range(3)]
        got_b = [(await anext(b)).value for _ in range(3)]
        return got_a, got_b

    task, _ = sim(main)
    assert task.result == (["red", "blue", "green"], ["red", "blue", "green"])


def test_queue_reader_that_falls_behind_raises_overflow(sim):
    async def main():
        world.declare("instr", QUEUE, maxlen=3)
        it = changes("instr")
        for i in range(5):
            world.write("instr", i)
        with pytest.raises(Overflow) as e:
            await anext(it)
        lost = e.value.lost
        return lost, (await anext(it)).value

    task, _ = sim(main)
    assert task.result == (2, 2)


def test_queue_skips_expired_messages_and_silence_is_not_an_error(sim):
    async def main():
        world.declare("instr", QUEUE, maxlen=8, max_age=0.5)
        world.write("instr", "old")
        await sleep(1.0)
        it = changes("instr")
        world.write("instr", "new")
        return (await anext(it)).value

    task, _ = sim(main)
    assert task.result == "new"
