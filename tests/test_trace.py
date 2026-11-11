"""The trace (SPEC §4) and ``roborun trace show`` (SPEC §1.8)."""

import asyncio
import json
import subprocess
import sys

import pytest

from roborun import TaskCancelled, TaskFailed, run_sim, runtime, start, world
from roborun.backends.fake import FakeBackend, grasp_blocking, move_to, place, track
from roborun.checks import workspace_box
from roborun.cli import main as cli
from roborun.trace import failures, load, show_failed

TRAY = (0.0, 0.3)


def run(main, tmp_path, **backend_kwargs):
    path = tmp_path / "run.jsonl"
    backend = FakeBackend(**backend_kwargs)
    task = asyncio.run(run_sim(main, backend=backend, trace=path))
    return task, path


async def program():
    start(track("cup"), detached=True)
    await world.next("cup")
    await grasp_blocking((0.5, 0.0))
    await place(TRAY)


def test_spans_follow_the_task_tree_and_the_file_is_json_lines(tmp_path):
    _task, path = run(program, tmp_path, objects={"cup": (0.5, 0.0)},
                     nudges=[("cup", (0.5, 0.15), 0.5)])
    lines = [json.loads(s) for s in path.read_text().splitlines()]
    assert lines[0]["kind"] == "run" and lines[0]["version"] == 1
    _meta, spans = load(path)
    by_name = {s["name"]: s for s in spans if s["kind"] == "task"}
    main, grasp = by_name["program"], by_name["grasp_blocking"]
    assert grasp["parent"] == main["id"] and grasp["status"] == "failed"
    assert grasp["inputs"] == {"target": [0.5, 0.0]}
    assert grasp["error"]["type"] == "GraspMissed"
    assert main["error"]["path"] == ["grasp_blocking"]
    assert main["error"]["root"]["type"] == "GraspMissed"
    assert "arm.ee" in grasp["state_at_end"]  # the state when it failed
    goals = [s for s in spans if s["kind"] == "gate" and s["parent"] == grasp["id"]]
    assert goals and goals[0]["name"] == "goal arm" and goals[0]["status"] == "accepted"
    writes = [s for s in spans if s["kind"] == "world"]
    assert writes and all(s["name"] == "write cup" for s in writes)  # by the tracker task
    tracker = by_name["track"]
    assert all(s["parent"] == tracker["id"] for s in writes)


def test_repeated_identical_commands_are_counted_not_recorded(tmp_path):
    from roborun.backends.fake import grasp

    async def main():
        await grasp((0.3, 0.0))  # resends the same goal every cycle until it arrives

    _task, path = run(main, tmp_path, objects={"cup": (0.3, 0.0)})
    _, spans = load(path)
    g = next(s for s in spans if s["name"] == "grasp")
    goals = [s for s in spans if s["kind"] == "gate" and s["parent"] == g["id"]
             and s["name"] == "goal arm"]
    assert len(goals) == 1 and g["repeats"] > 10


def test_repeats_are_counted_even_when_checks_run(tmp_path):
    from roborun.backends.fake import grasp

    async def main():
        runtime().gate.add_check(workspace_box(lo=(-1, -1), hi=(1, 1)))
        await grasp((0.3, 0.0))

    _, path = run(main, tmp_path, objects={"cup": (0.3, 0.0)})
    _, spans = load(path)
    g = next(s for s in spans if s["name"] == "grasp")
    goals = [s for s in spans if s["kind"] == "gate" and s["parent"] == g["id"]
             and s["name"] == "goal arm"]
    assert len(goals) == 1 and goals[0]["verdicts"][0]["verdict"] == "accept"
    assert g["repeats"] > 10


def test_a_rejected_command_is_traced_with_its_verdict_and_links_to_its_task(tmp_path):
    async def main():
        runtime().gate.add_check(workspace_box(lo=(-1, -1), hi=(1, 1)))
        m = start(move_to((1.5, 0.0)), trace_parent="harness:turn-3")
        try:
            await m
        except TaskCancelled:
            pass

    _task, path = run(main, tmp_path)
    _, spans = load(path)
    m = next(s for s in spans if s["name"] == "move_to")
    assert m["external_parent"] == "harness:turn-3"
    rejected = [s for s in spans if s["kind"] == "gate" and s["status"] == "rejected"]
    assert len(rejected) == 1 and rejected[0]["parent"] == m["id"]
    assert rejected[0]["outputs"]["code"] == "check:workspace"
    assert rejected[0]["verdicts"][0]["verdict"] == "reject"
    assert any(s["kind"] == "gate" and s["status"] == "fallback" for s in spans)

    text = show_failed(path)
    assert "1 failure(s)" in text
    assert "move_to" in text and "check:workspace" in text and "outside the workspace" in text


def test_failure_chain_root_and_path():
    async def inner():
        raise ValueError("bad")

    async def middle():
        await start(inner(), name="inner")

    async def outer():
        try:
            await start(middle(), name="middle")
        except TaskFailed as e:
            return e.path, type(e.root)

    task = asyncio.run(run_sim(outer, backend=FakeBackend()))
    assert task.result == (["middle", "inner"], ValueError)


def test_show_failed_finds_the_innermost_failure(tmp_path, capsys):
    _task, path = run(program, tmp_path, objects={"cup": (0.5, 0.0)},
                     nudges=[("cup", (0.5, 0.15), 0.5)])
    chains = failures(load(path)[1])
    assert len(chains) == 1
    assert [s["name"] for s in chains[0]] == ["program", "grasp_blocking"]
    assert cli(["trace", "show", "--failed", str(path)]) == 0
    out = capsys.readouterr().out
    assert "GraspMissed: closed on nothing" in out and "state at end" in out
    assert cli(["trace", "show", str(path)]) == 0
    tree = capsys.readouterr().out
    assert "program" in tree and "track" in tree and "writes" in tree


def test_the_command_line_entry_point(tmp_path):
    _, path = run(program, tmp_path, objects={"cup": (0.5, 0.0)},
                  nudges=[("cup", (0.5, 0.15), 0.5)])
    out = subprocess.run([sys.executable, "-m", "roborun", "trace", "show", "--failed", "--json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    chains = json.loads(out)
    assert chains[0][-1]["error"]["type"] == "GraspMissed"


# -- a rejected command, fully: rejected, traced, and at rest --------------------------------

mj = pytest.importorskip("roborun.backends.mujoco")
needs_panda = pytest.mark.skipif(
    not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists(),
    reason="MuJoCo Menagerie Panda model not found",
)


async def into_the_table():
    await mj.move_to((0.5, 0.0, 0.3))
    await mj.move_to((0.5, 0.0, -0.05))


@needs_panda
def test_m2_done_a_command_into_the_table_is_rejected_traced_and_the_robot_at_rest(tmp_path):
    from roborun.gateproc import spawn_gate
    from roborun.remote import run_lockstep

    path = tmp_path / "table.jsonl"
    box = {"workspace_box": {"lo": [0.15, -0.6, 0.02], "hi": [0.8, 0.6, 0.8]}}
    proc, sock = spawn_gate("roborun.backends.mujoco:MujocoBackend", {}, checks=box)
    try:
        task = asyncio.run(run_lockstep(into_the_table, sock, until=10.0, trace=str(path)))
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert task.runtime.world.get("arm.moving").value is False  # at rest
    _, spans = load(path)
    rejected = [s for s in spans if s["kind"] == "gate" and s["status"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["outputs"]["code"] == "check:workspace"
    assert rejected[0]["inputs"]["target"] == [0.5, 0.0, -0.05]
    fallback = [s for s in spans if s["kind"] == "gate" and s["status"] == "fallback"]
    assert fallback  # the safety event from the Gate process reached the trace
    text = show_failed(path)
    assert "into_the_table" in text and "move_to" in text and "check:workspace" in text
