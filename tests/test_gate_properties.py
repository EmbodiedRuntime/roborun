"""The invariants of the TLA+ models (spec/), checked on the real Gate under random use.

TLC checks the models exhaustively, but a model is not the code. Here random programs
(tasks three deep, `hold`, wait / fail / take from two sessions, cancellations, commands
sent with and without authority, operator stops and resets) run against the real runtime
on the fake backend, and after every step the invariants are evaluated on the Gate's own
state, under the names they have in the models.
"""

import asyncio
import random
import tempfile
from pathlib import Path

import pytest

from roborun import TaskState, hold, run_sim, runtime, send, sleep, start
from roborun.api import current
from roborun.backends.fake import FakeBackend
from roborun.gate import MANUAL, _is_self_or_ancestor, cancelled_below
from roborun.gateproc import GateServer
from roborun.remote import run_lockstep

S = TaskState
ACTUATORS = ("arm", "gripper")
SUPERVISES = {"recovery": {"agent"}}
ON_BUSY = ("wait", "wait", "fail", "take")
STEPS = 40


def some(r, items):
    items = sorted(items)
    return frozenset(x for x in items if r.random() < 0.5) or frozenset({r.choice(items)})


async def command(r):
    if r.random() < 0.7:
        await send("goal", "arm", target=(round(r.uniform(-1, 1), 2), round(r.uniform(-1, 1), 2)))
    else:
        await send("effector", "gripper", close=r.random() < 0.5)


async def worker(seed, depth=0):
    """A random program. It may command actuators it does not hold, start children that use
    its token or ask for their own, wait in `hold`, and go on commanding when cancelled."""
    r = random.Random(seed)
    try:
        for i in range(r.randint(1, 6)):
            k = r.random()
            if k < 0.4:
                await command(r)
            elif k < 0.6:
                await sleep(r.choice((0.01, 0.05, 0.2)))
            elif k < 0.8 and depth < 2:
                kwargs = {}
                if r.random() < 0.5:
                    kwargs = {"resources": some(r, ACTUATORS), "on_busy": r.choice(ON_BUSY)}
                child = start(worker(seed * 31 + i, depth + 1), **kwargs)
                if r.random() < 0.5:
                    await child.stopped()
            else:
                async with hold(*some(r, ACTUATORS), on_busy=r.choice(ON_BUSY)):
                    await command(r)
                    await sleep(r.choice((0.0, 0.05)))
                    if depth < 2 and r.random() < 0.5:
                        start(worker(seed * 37 + i, depth + 1))  # it uses this hold's token
                        await sleep(r.choice((0.0, 0.05)))
    except asyncio.CancelledError:
        if r.random() < 0.5:
            try:
                await command(r)  # generated code often keeps going
            except asyncio.CancelledError:
                pass
        if r.random() < 0.5:
            await sleep(r.choice((0.01, 0.1)))  # a slow cleanup
        raise


async def drive(seed, check, operator=True):
    """Start, cancel and stop at random; call ``check`` after every step."""
    r = random.Random(seed)
    rt, me = runtime(), current()
    for i in range(STEPS):
        k = r.random()
        live = [t for t in rt.all_tasks() if not t.state.terminal and t is not me]
        if k < 0.5:
            rt.start(worker(seed * 1000 + i), session=r.choice(("agent", "agent", "recovery")),
                     detached=True, resources=some(r, ACTUATORS), on_busy=r.choice(ON_BUSY),
                     wait_timeout=r.choice((None, None, 0.1)))
        elif k < 0.7 and live:
            r.choice(live).cancel()
        elif k < 0.8:
            await send("stop", r.choice(ACTUATORS))
        elif k < 0.9 and operator:
            stop = rt.gate.operator_stop(some(r, ACTUATORS))
            if asyncio.iscoroutine(stop):
                await stop
        elif operator:
            reset = rt.gate.reset()
            if asyncio.iscoroutine(reset):
                await reset
        await sleep(r.choice((0.01, 0.01, 0.05)))
        for _ in range(50):
            await asyncio.sleep(0)  # every task has reacted to this instant (the models' steps are atomic)
        check(i)
    # Everything ends: no cancellation hangs, and nothing is left owned.
    reset = rt.gate.reset()
    if asyncio.iscoroutine(reset):
        await reset
    tasks = [t for t in rt.all_tasks() if t is not me]
    for t in tasks:
        t.cancel()
    for t in tasks:
        await t.stopped()
    await sleep(0.5)  # an arm that the last task left moving has been stopped by now
    check("end")
    return tasks


def tree_invariants(rt, found, at):
    for t in rt.all_tasks():
        if t.state.terminal and any(not c.state.terminal for c in t.children):
            found.append((at, "ParentOutlivesChildren", t.name))
        if t.on_busy == "fail" and t.state is S.PENDING and t.created_at < rt.clock.now():
            found.append((at, "FailNeverWaits", t.name))  # still pending a step later


def gate_invariants(gate, backend, found, at):
    """What the Gate's own state shows, wherever the Gate runs."""
    waiting = [e for e in gate._queue if not e.fut.done()]
    for a in ACTUATORS:
        g = gate._owner.get(a)
        if g is None:
            if a == "arm" and backend.moving:
                found.append((at, "MotionHasOwner", a))
        else:
            if a not in g.resources:
                found.append((at, "OwnershipConsistent", a))
            if gate.locks.get(a) and not g.revoked:
                found.append((at, "LockedHasNoOwner", a, g.owner.name))
        if gate.locks.get(a, set()) & MANUAL and any(a in e.resources for e in waiting):
            found.append((at, "NoWaiterUntilReset", a))
        if sum(1 for e in waiting if e.take and a in e.resources) > 1:
            found.append((at, "OneTaker", a))


def watch_cancellations(rt, backend, found):
    """CancelledAtRest: when a task ends cancelled, the arm it owned is at rest. Looked at
    as the task gives up its grants, which is the moment it becomes cancelled."""
    gate = rt.gate
    release_all = gate.release_all

    def releasing(task):
        grants = gate._owner.values() if hasattr(gate, "_owner") else gate.grants.values()
        # An arm it gave up earlier (it left a `hold` with a goal on its way) is not its
        # own any more: the Gate is stopping that one, and keeps it until it is at rest.
        if (task.state is S.CANCELLED and backend.moving
                and any(g.owner is task and "arm" in g.resources
                        and not getattr(g, "released", False) for g in grants)):
            found.append((rt.clock.now(), "CancelledAtRest", task.name))
        release_all(task)

    gate.release_all = releasing


@pytest.mark.parametrize("seed", range(200))
def test_the_models_invariants_hold_on_the_gate_in_process(seed):
    found = []

    async def main():
        rt = runtime()
        rt.supervises.update(SUPERVISES)
        gate, backend = rt.gate, rt.backend
        watch_cancellations(rt, backend, found)
        sender = []

        send_, execute_ = gate.send, gate._execute

        async def sending(task, cmd, verdicts=None):
            sender.append(task)
            try:
                return await send_(task, cmd, verdicts)
            finally:
                sender.pop()

        def executing(cmd):
            # ExecutedForTheOwner, NoCommandWhenCancelled: what reaches the backend comes
            # from a running task with authority, for an actuator that is not locked.
            if cmd.kind != "stop":
                task, g = (sender[-1] if sender else None), gate._owner.get(cmd.actuator)
                ok = (task is not None and g is not None and not g.revoked
                      and _is_self_or_ancestor(g.owner, task) and task.state is S.RUNNING
                      and not cancelled_below(task, g.owner) and not gate.locks.get(cmd.actuator))
                if not ok:
                    found.append((rt.clock.now(), "ExecutedForTheOwner", getattr(task, "name", None)))
            execute_(cmd)

        gate.send, gate._execute = sending, executing

        def check(at):
            tree_invariants(rt, found, at)
            gate_invariants(gate, backend, found, at)
            for g in set(gate._owner.values()):
                # A grant given up while the arm moved is kept, with no task behind it,
                # until the arm is at rest.
                if g.owner.state not in (S.RUNNING, S.CANCELLING) and g not in gate._settling:
                    found.append((at, "OwnershipConsistent", g.owner.name, g.owner.state))
                if g.revoked and not g.released and g.owner.state is S.RUNNING:
                    found.append((at, "RevokedIsCancelling", g.owner.name))

        tasks = await drive(seed, check)
        return [t for t in tasks if not t.state.terminal], dict(gate._owner)

    task = asyncio.run(run_sim(main, backend=FakeBackend(), until=60.0))
    assert task.state is S.SUCCEEDED, task.error
    assert found == []
    assert task.result == ([], {})


@pytest.mark.parametrize("seed", range(40))
def test_the_models_invariants_hold_with_the_gate_in_its_own_process(seed):
    found = []
    backend = FakeBackend()

    async def main():
        rt = runtime()
        watch_cancellations(rt, backend, found)

        def check(at):
            tree_invariants(rt, found, at)
            gate_invariants(server.gate, backend, found, at)
            # MirrorIsSound: the runtime believes in no token the Gate has not granted,
            # and one it still takes for valid is valid at the Gate, or the revocation
            # is on its way (it arrives with the next reply).
            for token, g in rt.gate.grants.items():
                if token not in server.tokens:
                    found.append((at, "MirrorIsSound", g.owner.name))
            # OwnershipConsistent across the boundary: a token the Gate takes for valid
            # belongs to a task that has not ended, or its release is on its way.
            tasks = {t.id: t for t in rt.all_tasks()}
            for token, g in server.tokens.items():
                t = tasks.get(g.owner.task_id)
                if (not g.revoked and t is not None and t.state.terminal
                        and not rt.gate.outstanding):
                    found.append((at, "OwnershipConsistent", t.name, t.state))

        tasks = await drive(seed, check)
        return [t for t in tasks if not t.state.terminal]

    server = GateServer(backend, dt=0.01, supervises=SUPERVISES)

    async def go():
        path = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")
        log = []
        execute_ = backend.execute

        def executing(cmd):
            # ExecutedForTheOwner at the Gate: a valid token, and no lock.
            g = server.gate._owner.get(cmd.actuator)
            if backend.moving and server.gate._owner.get("arm") is None:
                log.append((server.clock.now(), "MotionHasOwner", "arm"))
            if cmd.kind != "stop" and (g is None or g.revoked or server.gate.locks.get(cmd.actuator)):
                log.append((server.clock.now(), "ExecutedForTheOwner", cmd.actuator))
            execute_(cmd)

        backend.execute = executing
        await server.start(path)
        stepping = asyncio.ensure_future(server.run_lockstep())
        try:
            task = await run_lockstep(main, path, until=60.0)
            await stepping
        finally:
            await server.close()
        # NoOrphanGrant: when the runtime has released everything, so has the Gate.
        return task, log, dict(server.tokens), dict(server.gate._owner)

    task, log, tokens, owners = asyncio.run(go())
    assert task.state is S.SUCCEEDED, task.error
    assert found == [] and log == []
    assert task.result == [] and tokens == {} and owners == {}
