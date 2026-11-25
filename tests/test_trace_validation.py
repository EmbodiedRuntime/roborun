"""Runs of the real Gate, put to TLC: is each one a behaviour of spec/Gate.tla?

The random tests in test_gate_properties.py evaluate the models' invariants on the Gate's
state. Here the whole run is checked: the Gate's core records every event (roborun.journal),
the events are turned into steps of the model (spec/validate), and TLC must find a behaviour
of Gate.tla that takes exactly these steps and passes through the states the Gate was in.

The programs stay within what Gate.tla describes: one actuator, and every command sent with
the sender's own token. Inside that they use everything the model has: requests that wait
or take, commands that checks reject, a backend fault, setpoint streams that stop, a monitor
with and without an `allow` check, cancellations, operator stops and resets.
"""

import asyncio
import functools
import importlib.util
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

from roborun import hold, run_sim, runtime, send, sleep
from roborun.api import current
from roborun.backends.fake import FakeBackend
from roborun.checks import Accept, Reject
from roborun.errors import BackendError
from roborun.gateproc import GateServer
from roborun.remote import run_lockstep

SPEC = Path(__file__).resolve().parents[1] / "spec"
pytestmark = pytest.mark.skipif(
    shutil.which("java") is None or not (SPEC / ".tools" / "tla2tools-1.7.4.jar").exists(),
    reason="needs java and TLC (spec/check.sh downloads it)")

SUPERVISES = {"recovery": {"agent"}}
FAULT = (0.33, 0.33)


@functools.lru_cache
def validator():
    sys.path[:0] = [str(SPEC / "validate"), str(SPEC / "compare")]
    spec = importlib.util.spec_from_file_location("validate_main", SPEC / "validate" / "__main__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Faulty(FakeBackend):
    """A backend whose controller reports a fault for one target."""

    def execute(self, cmd):
        if cmd.kind in ("goal", "setpoint") and tuple(cmd.payload["target"]) == FAULT:
            raise BackendError("fault", "joint 4 over torque", source="fake.arm", stamp=self.now())
        super().execute(cmd)


def limit(cmd, view):
    time.sleep(0.002)  # checks take time, and things happen meanwhile
    return Reject("out of reach") if abs(cmd.payload["target"][0]) > 0.8 else Accept()


def target(r):
    if r.random() < 0.1:
        return FAULT
    reach = r.choice((0.1, 0.3, 1.0))  # near targets are reached, far ones rarely
    return (round(r.uniform(-reach, reach), 2), round(r.uniform(-reach, reach), 2))


async def act(r, flying):
    """One thing a program does with the arm. True if it sent a command without waiting
    for the answer: the caller may then leave its `hold`, or end, while the checks run."""
    if flying:
        await asyncio.wait(flying)  # one command at a time, as the model has it
        flying.clear()
    k = r.random()
    if k < 0.4:
        await send("goal", "arm", target=target(r))
    elif k < 0.62:
        for _ in range(r.randint(1, 4)):  # a stream, which may stop without a word
            await send("setpoint", "arm", target=target(r))
            await sleep(0.02)
    elif k < 0.72:
        await send("stop", "arm")
    elif k < 0.87:
        f = asyncio.ensure_future(send("goal", "arm", target=target(r)))
        f.add_done_callback(lambda f: f.cancelled() or f.exception())
        flying.append(f)
        await asyncio.sleep(0.001)  # real time: the command is with the checks now
        return r.random() < 0.7
    await sleep(r.choice((0.0, 0.05, 0.3, 1.0)))
    return False


async def worker(seed, owns):
    """A random program for one arm. It commands with its own token: the one it started
    with, or one it asks for with `hold`. Without either its commands are refused."""
    r = random.Random(seed)
    flying = []
    try:
        for _ in range(r.randint(1, 5)):
            if owns or r.random() < 0.3:
                if await act(r, flying):
                    return
            else:
                async with hold("arm", on_busy=r.choice(("wait", "wait", "fail", "take")),
                                wait_timeout=r.choice((None, 0.1))):
                    for _ in range(r.randint(1, 3)):
                        if await act(r, flying):
                            break
    except asyncio.CancelledError:
        if r.random() < 0.5:
            try:
                await act(r, flying)  # generated code often keeps going
            except asyncio.CancelledError:
                pass
        raise


async def drive(seed, person, steps=30):
    r = random.Random(seed)
    rt, me = runtime(), current()

    async def maybe(x):
        if asyncio.iscoroutine(x):
            await x

    for i in range(steps):
        k = r.random()
        live = [t for t in rt.all_tasks() if not t.state.terminal and t is not me]
        if k < 0.5:
            owns = r.random() < 0.6
            kwargs = {"resources": {"arm"}, "on_busy": r.choice(("wait", "wait", "fail", "take")),
                      "wait_timeout": r.choice((None, None, 0.1))} if owns else {}
            rt.start(worker(seed * 1000 + i, owns), detached=True,
                     session=r.choice(("agent", "agent", "recovery")), **kwargs)
        elif k < 0.62 and live:
            r.choice(live).cancel()
        elif k < 0.67:
            await send("stop", "arm")
        elif k < 0.77:
            person["near"] = not person["near"]
        elif k < 0.82:
            await maybe(rt.gate.operator_stop(["arm"]))
        else:
            person["near"] = False
            await maybe(rt.gate.reset())
        await sleep(r.choice((0.01, 0.05, 0.2, 0.5)))
    person["near"] = False
    await maybe(rt.gate.reset())
    tasks = [t for t in rt.all_tasks() if t is not me]
    for t in tasks:
        t.cancel()
    for t in tasks:
        await t.stopped()
    await sleep(0.5)


def configure(gate, seed, person):
    gate.add_check(limit, kinds={"goal", "setpoint"}, budget=1.0)
    allow = (lambda cmd, view: Accept() if cmd.payload["target"][0] < 0 else Reject("towards")) \
        if seed % 2 else None
    gate.add_monitor(lambda view: Reject("a person is near") if person["near"] else Accept(),
                     name="person", rate=50.0, budget=1.0, actuators=["arm"], allow=allow)
    return gate.record()


def in_process(seed):
    person = {"near": False}
    box = {}

    async def main():
        rt = runtime()
        rt.supervises.update(SUPERVISES)
        box["journal"] = configure(rt.gate, seed, person)
        await drive(seed, person)

    task = asyncio.run(run_sim(main, backend=Faulty(), until=120.0))
    assert task.error is None, task.error
    return box["journal"]


def gate_process(seed):
    person = {"near": False}
    server = GateServer(Faulty(), dt=0.01, supervises=SUPERVISES)

    async def go():
        path = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "gate.sock")
        journal = configure(server.gate, seed, person)
        await server.start(path)
        stepping = asyncio.ensure_future(server.run_lockstep())
        try:
            task = await run_lockstep(lambda: drive(seed, person), path, until=120.0)
            await stepping
        finally:
            await server.close()
        assert task.error is None, task.error
        return journal

    return asyncio.run(go())


def check(journal, tmp_path):
    v = validator()
    r = v.validate(journal.meta, journal.entries, tmp_path)
    if not r.ok:
        journal.save(tmp_path / "run.journal.jsonl")
    assert r.ok, v.explain(r, journal.entries) + f"\n  kept in {tmp_path}"
    return r


@pytest.mark.parametrize("seed", range(20))
def test_a_run_of_the_gate_in_process_is_a_behaviour_of_the_model(seed, tmp_path):
    r = check(in_process(seed), tmp_path)
    assert r.total > 20


@pytest.mark.parametrize("seed", range(10))
def test_a_run_of_the_gate_in_its_own_process_is_a_behaviour_of_the_model(seed, tmp_path):
    r = check(gate_process(seed), tmp_path)
    assert r.total > 20


# -- the control: a Gate with one mechanism switched off is not a behaviour of the model -------

MUTATIONS = ("norecheck", "norecheckrelease", "faultkeepswaiters", "unauthfallback",
             "nostoponrelease")


@pytest.mark.parametrize("layout", [in_process, gate_process], ids=["in_process", "gate_process"])
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_a_gate_with_one_mechanism_off_is_rejected(mutation, layout):
    """The cores of spec/compare/mutations.py, each the Gate before one of its fixes, run
    the same programs. TLC must refuse one of the first runs: otherwise accepting a run
    would say nothing."""
    v = validator()
    from mutations import mutated

    for seed in range(40):
        with mutated(mutation):
            try:
                journal = layout(seed)
            except AssertionError:
                continue  # the runtime itself noticed
        try:
            if not v.validate(journal.meta, journal.entries).ok:
                return
        except v.gate_trace.Outside:
            continue
    pytest.fail(f"40 runs of the Gate with {mutation} were all accepted")
