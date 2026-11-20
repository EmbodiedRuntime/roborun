"""`roborun trace check`: the invariants that can be read from a trace, replayed over one."""

import asyncio
import json

import pytest
from test_gate_properties import SUPERVISES, drive

from roborun import run_sim, runtime
from roborun.backends.fake import FakeBackend
from roborun.cli import main as cli
from roborun.trace import check, load


def record(path, seed):
    async def main():
        runtime().supervises.update(SUPERVISES)
        await drive(seed, lambda at: None)

    asyncio.run(run_sim(main, backend=FakeBackend(), until=60.0, trace=str(path)))
    return load(path)[1]


@pytest.mark.parametrize("seed", range(25))
def test_traces_of_random_runs_satisfy_the_invariants(tmp_path, seed):
    spans = record(tmp_path / "run.jsonl", seed)
    assert any(s.get("owned") for s in spans)  # tasks did own actuators
    assert check(spans) == []


def test_a_trace_that_breaks_an_invariant_is_reported_by_name(tmp_path, capsys):
    path = tmp_path / "run.jsonl"
    spans = record(path, 1)
    tasks = {s["id"]: s for s in spans if s.get("kind") == "task"}
    command = next(s for s in spans if s.get("kind") == "gate" and s["status"] == "accepted"
                   and not s["name"].startswith("stop") and s.get("parent") in tasks)
    child = next(s for s in tasks.values() if s.get("parent") in tasks)
    a, b = [s for s in tasks.values() if "arm" in (s.get("owned") or ())][:2]

    late = tasks[command["parent"]]["end"] + 1.0
    command["start"] = command["end"] = late  # a command after its task ended
    child["end"] = tasks[child["parent"]]["end"] + 1.0  # a child that outlives its parent
    b["start"] = a["start"]  # two owners of the arm at once
    path.write_text("\n".join(json.dumps(s) for s in spans))

    names = {line.split(":")[0] for line in check(load(path)[1])}
    assert {"NoCommandWhenCancelled", "ParentOutlivesChildren", "OwnershipConsistent"} <= names
    assert cli(["trace", "check", str(path)]) == 1
    assert "NoCommandWhenCancelled: " in capsys.readouterr().out


def test_a_command_accepted_while_locked_until_reset_is_reported():
    spans = [
        {"id": "s1", "kind": "gate", "name": "lock arm", "start": 1.0, "end": 1.0,
         "status": "lock", "outputs": {"cause": "operator"}},
        {"id": "g1", "kind": "gate", "name": "goal arm", "start": 1.5, "end": 1.5,
         "status": "accepted"},
        {"id": "s2", "kind": "gate", "name": "unlock arm", "start": 2.0, "end": 2.0,
         "status": "unlock", "outputs": {"cause": "fault,operator"}},
        {"id": "g2", "kind": "gate", "name": "goal arm", "start": 2.5, "end": 2.5,
         "status": "accepted"},
    ]
    assert check(spans) == ["LockedBlocks: goal arm accepted at 1.500 while locked (operator)"]
