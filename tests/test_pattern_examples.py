"""The programming patterns from the blog, as runnable examples (examples/patterns, examples/agents)."""

import asyncio
import math
import sys
from pathlib import Path

import pytest

mj = pytest.importorskip("roborun.backends.mujoco")
if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
    pytest.skip("MuJoCo Menagerie Panda model not found", allow_module_level=True)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "examples" / "patterns"))
sys.path.insert(0, str(ROOT / "examples" / "agents"))

import event_replan
import graph_policy
import live_patch
import parallel_monitor
import two_rate
from robot import robot


def on_tray(p):
    return abs(p[0] - 0.3) < 0.08 and abs(p[1] - 0.38) < 0.055 and abs(p[2] - 0.05) < 0.008


def test_a_program_level_monitor_catches_the_slip_and_the_handler_follows_the_push():
    task, cup = parallel_monitor.run()
    assert task.state.value == "succeeded" and task.result == 2  # one slip, one retry
    assert on_tray(cup)


def test_two_rates_a_fast_loop_keeps_the_constraint_and_the_model_is_asked_when_it_cannot():
    task, backend = two_rate.run(two_rate.scripted_planner())
    log = task.result
    kinds = [e[0] for e in log]
    assert kinds == ["constraint", "lost", "constraint", "kept"]
    assert "cup_red" in log[1][2]  # lost: the cup was taken away
    assert backend.lag < 0.01  # within 1 cm of 12 cm above a cup moving at 3 cm/s
    assert sum(c.kind == "setpoint" for _, c in backend.log) > 100  # all through the Gate


def test_a_new_instruction_cancels_the_plan_and_the_model_replans_from_the_real_state():
    task, backend = event_replan.run(event_replan.scripted())
    kinds = [e[0] for e in task.result]
    assert kinds == ["heard", "plan", "heard", "cancelled", "plan", "done"]
    red, blue = backend.truth("cup_red"), backend.truth("cup_blue")
    assert on_tray(red) and on_tray(blue) and math.dist(red[:2], blue[:2]) > 0.06


def test_a_graph_written_by_a_model_recovers_from_a_slip_without_the_model():
    task, backend = graph_policy.run()
    _graph, path = task.result
    assert [n for n, _ in path] == ["grasp", "place", "grasp", "place", "check", "done"]
    assert path[1] == ("place", "Dropped")
    assert on_tray(backend.truth("cup_red"))


def test_a_fix_applied_while_the_program_runs_takes_effect_at_a_safe_point():
    async def go():
        async with robot(pace=None) as client:
            return await live_patch.live_patch(client, live_patch.scripted)

    r = asyncio.run(go())
    assert (r["v1"]["state"], r["v1"]["reason"]) == ("cancelled", "replaced")
    assert r["v2"]["state"] == "succeeded"
    assert r["swap"]["new"]["id"] != r["v1"]["id"]
    red, blue = r["cups"]["cup_red"], r["cups"]["cup_blue"]
    assert on_tray(red) and on_tray(blue) and math.dist(red[:2], blue[:2]) > 0.06
