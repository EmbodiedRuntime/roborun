"""Behaviour that the TLA+ model of the Gate's process boundary (spec/Protocol.tla) requires.

Each test is a counterexample TLC found against an earlier implementation.
"""

import asyncio
import tempfile
import threading
import time
from pathlib import Path

from roborun import TaskState, send, sleep, start
from roborun.backends.fake import FakeBackend
from roborun.checks import Accept
from roborun.gateproc import GateServer
from roborun.remote import run_lockstep

S = TaskState


def run(main, configure=None, until=30.0):
    """Run ``main`` against a Gate behind a socket; returns (task, backend, server)."""
    backend = FakeBackend()

    async def go():
        path = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")
        server = GateServer(backend, dt=0.01)
        if configure is not None:
            configure(server)
        await server.start(path)
        stepping = asyncio.ensure_future(server.run_lockstep())
        try:
            task = await run_lockstep(main, path, until=until)
            await stepping
        finally:
            await server.close()
        return task, backend, server

    return asyncio.run(go())


async def idle(seconds):
    await sleep(seconds)


def test_a_request_cancelled_while_its_grant_is_on_the_way_does_not_keep_the_actuator():
    # NoOrphanGrant: the Gate granted the request; the runtime, not having heard yet,
    # withdrew it and dropped the answer. Nobody was left to release the grant.
    async def main():
        owner = start(idle(0.3), resources={"gripper"})
        await sleep(0.05)
        waiter = start(idle(1.0), resources={"gripper"})
        await owner.stopped()  # the Gate grants the waiter now; the event is not here yet
        waiter.cancel()
        await sleep(0.2)
        again = start(idle(0.1), resources={"gripper"}, on_busy="fail")
        return waiter.state, await again.stopped(), again.error

    task, _, server = run(main)
    waiter, again, error = task.result
    assert waiter is S.CANCELLED
    assert again is S.SUCCEEDED, error
    assert server.tokens == {} and server.gate.owner("gripper") is None


def test_a_stop_does_not_wait_for_the_checks_of_an_earlier_command():
    # NextRequestNeverWaits (SPEC §3.4 rule 5), with the Gate in its own process.
    checking = threading.Event()

    def slow(cmd, view):
        checking.set()
        time.sleep(0.4)
        return Accept()

    async def main():
        order = []

        async def mover():
            try:
                await send("goal", "arm", target=(1.0, 0.0))
            finally:
                order.append("goal answered")
            await sleep(0.1)

        m = start(mover(), resources={"arm"})
        while not checking.is_set():  # until the goal is with the Gate's checks
            await asyncio.sleep(0.001)
        t0 = time.perf_counter()
        await send("stop", "arm")
        order.append("stop answered")
        waited = time.perf_counter() - t0
        await m.stopped()
        return order, waited

    task, _, _ = run(main, lambda server: server.gate.add_check(slow, kinds={"goal"}, budget=2.0))
    order, waited = task.result
    assert order == ["stop answered", "goal answered"]
    assert waited < 0.2
