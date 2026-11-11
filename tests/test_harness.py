"""The harness interface (SPEC §5.1): sessions, non-blocking calls, wake_on, programs."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from roborun.backends.fake import FakeBackend
from roborun.harness import Client, HarnessError, Host

CUP, NUDGED = (0.5, 0.0), (0.5, 0.15)

# Clients here connect with heartbeat=None. The simulation runs on simulated time, but the
# session watchdog is wall time (0.5 s); on a busy machine a late heartbeat would end a session
# mid-test. The watchdog has its own test (test_bench_crash.py), with a real, frozen process.


def sock() -> str:
    return str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "host.sock")


def with_host(scenario, backend=None):
    async def go():
        path = sock()
        host = await Host.local(backend or FakeBackend(), tools=["roborun.backends.fake"])
        await host.start(path)
        try:
            return await scenario(path, host)
        finally:
            await host.close()

    return asyncio.run(go())


def nudged():
    return FakeBackend(objects={"cup": CUP}, nudges=[("cup", NUDGED, 0.5)])


def test_start_returns_at_once_and_wait_returns_the_result():
    async def scenario(path, host):
        c = await Client.connect(path, session="agent", heartbeat=None)
        assert {t["name"] for t in c.tools} >= {"move_to", "grasp", "track"}
        s = await c.start("move_to", target=[0.2, 0.0])
        assert s["state"] in ("pending", "running") and s["session"] == c.session
        done = await c.wait(s["id"], timeout=10)
        await c.close()
        return done

    done = with_host(scenario)
    assert done["woke"] == "done" and done["state"] == "succeeded"
    assert tuple(done["result"]) == (0.2, 0.0)


def test_one_step_at_a_time_a_model_notices_the_cup_move_and_redirects_the_grasp():
    """SPEC Appendix B, the harness form: no code, only calls."""

    async def scenario(path, host):
        c = await Client.connect(path, session="agent", heartbeat=None)
        await c.start("track", name="cup")
        await c.wait((await c.start("move_to", target=[0.0, 0.0]))["id"], timeout=1)
        cup = (await c.world("cup"))["cup"]["value"]
        g = await c.start("grasp", target=cup)
        woke = await c.wait(g["id"], timeout=5, wake_on=["cup"])
        await c.update(g["id"], target=woke["value"]["value"])
        done = await c.wait(g["id"], timeout=5)
        await c.close()
        return woke, done

    woke, done = with_host(scenario, nudged())
    assert woke["woke"] == "cup" and woke["state"] == "running"
    assert tuple(woke["value"]["value"]) == NUDGED and woke["value"]["age"] < 0.1
    assert done["state"] == "succeeded" and tuple(done["result"]) == NUDGED


def test_wait_times_out_and_cancel_is_a_request():
    async def scenario(path, host):
        c = await Client.connect(path, heartbeat=None)
        s = await c.start("move_to", target=[1.0, 0.0])
        timed_out = await c.wait(s["id"], timeout=0.2)
        cancelling = await c.cancel(s["id"])
        done = await c.wait(s["id"], timeout=5)
        return timed_out, cancelling, done

    timed_out, cancelling, done = with_host(scenario)
    assert timed_out["woke"] == "timeout" and timed_out["state"] == "running"
    assert cancelling["state"] == "cancelling"
    assert (done["state"], done["reason"]) == ("cancelled", "requested")


def test_closing_a_session_stops_what_it_started():
    async def scenario(path, host):
        a = await Client.connect(path, session="a", heartbeat=None)
        s = await a.start("move_to", target=[1.0, 0.0])
        await a.wait(s["id"], timeout=0.2)
        await a.close()
        b = await Client.connect(path, session="b", heartbeat=None)
        done = await b.wait(s["id"], timeout=5)
        await asyncio.sleep(0.05)
        moving = (await b.world("arm.moving"))["arm.moving"]["value"]
        await b.close()
        return done, moving

    done, moving = with_host(scenario)
    assert (done["state"], done["reason"]) == ("cancelled", "parent")
    assert moving is False


def test_sessions_share_the_world_but_not_control():
    async def scenario(path, host):
        a = await Client.connect(path, session="a", heartbeat=None)
        b = await Client.connect(path, session="b", heartbeat=None)
        tracker = await a.call("start", tool="track", args={"name": "cup"}, detached=True)
        await a.wait((await a.start("move_to", target=[0.0, 0.0]))["id"], timeout=1)
        seen_by_b = (await b.world("cup"))["cup"]
        m = await a.start("move_to", target=[1.0, 0.0])
        with pytest.raises(HarnessError, match="belongs to session"):
            await b.cancel(m["id"])
        await a.close()
        await b.wait(m["id"], timeout=5)
        still = await b.status(tracker["id"])  # detached: outlives session a
        await b.close()
        return seen_by_b, still

    seen_by_b, still = with_host(scenario, FakeBackend(objects={"cup": CUP}))
    assert tuple(seen_by_b["value"]) == CUP and seen_by_b["source"] == "track(cup)"
    assert still["state"] == "running"


GENERATED = '''
async def main():
    await move_to((0.2, 0.0))
    await grasp_blocking((0.5, 0.0))
    await place((0.0, 0.3))
'''


def test_a_generated_program_runs_as_a_task_and_its_failure_is_explained():
    async def scenario(path, host):
        c = await Client.connect(path, session="agent", heartbeat=None)
        p = await c.program(GENERATED, generated_by="model-call-7")
        done = await c.wait(p["id"], timeout=20)
        explained = await c.trace(p["id"])
        task = host.rt.task_by_id(p["id"])
        await c.close()
        return done, explained, task.provenance

    done, explained, provenance = with_host(scenario, nudged())
    assert done["state"] == "failed"
    assert list(done["error"]["path"]) == ["grasp_blocking"]
    assert done["error"]["root"]["type"] == "GraspMissed"
    assert "GraspMissed: closed on nothing" in explained["text"]
    assert [s["name"] for s in explained["failures"][0]] == ["program", "grasp_blocking"]
    assert provenance["generated_by"] == "model-call-7" and "grasp_blocking" in provenance["source"]


def test_bad_calls_say_what_to_do():
    async def scenario(path, host):
        c = await Client.connect(path, heartbeat=None)
        errors = []
        for call in (c.start("fly", target=[0, 0]), c.program("def main(): pass"),
                     c.program("async def main(:"), c.status("t999")):
            with pytest.raises(HarnessError) as e:
                await call
            errors.append(str(e.value))
        missing = (await c.world("scene.humans"))["scene.humans"]
        await c.close()
        return errors, missing

    errors, missing = with_host(scenario)
    assert "no tool 'fly'; tools:" in errors[0] and "move_to" in errors[0]
    assert "async def main" in errors[1]
    assert "does not compile" in errors[2]
    assert "no task" in errors[3]
    assert "producer running" in missing["error"]


def test_operator_stop_and_reset_through_the_harness():
    async def scenario(path, host):
        c = await Client.connect(path, heartbeat=None)
        s = await c.start("move_to", target=[1.0, 0.0])
        await c.wait(s["id"], timeout=0.2)
        await c.operator_stop(["arm"])
        stopped = await c.wait(s["id"], timeout=5)
        refused = await c.wait((await c.start("move_to", target=[0.0, 0.0]))["id"], timeout=5)
        await c.reset(["arm"])
        again = await c.wait((await c.start("move_to", target=[0.0, 0.0]))["id"], timeout=5)
        await c.close()
        return stopped, refused, again

    stopped, refused, again = with_host(scenario)
    assert (stopped["state"], stopped["reason"]) == ("cancelled", "operator")
    assert refused["state"] == "failed" and "locked until reset" in refused["error"]["message"]
    assert again["state"] == "succeeded"
