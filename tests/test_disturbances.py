"""The disturbance benchmark and the skill behaviors it relies on."""

import asyncio

import pytest

mj = pytest.importorskip("roborun.backends.mujoco")
if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
    pytest.skip("MuJoCo Menagerie Panda model not found", allow_module_level=True)

from benchmarks import disturbances as d
from roborun import run_sim, sleep, world
from roborun.demos.cup_nudge import SCENE


def test_a_slip_while_carrying_is_noticed_and_the_cup_is_picked_up_again():
    task = d.run_one("slip", 1, "task_based")
    blind = d.run_one("slip", 1, "blocking")
    assert task.success and task.closes_on_nothing == 0
    assert not blind.success and blind.outcome == "false success"  # it said done; it was not


def test_the_grasp_holds_still_while_the_cup_is_occluded():
    backend = mj.MujocoBackend(objects=SCENE)
    backend.removed_at = None
    d.occluded_push(backend, 1)  # occluded from 1.0 s to 3.0 s, pushed at 1.5 s
    task = asyncio.run(run_sim(d.task_based, backend=backend, until=30))
    assert task.state.value == "succeeded"
    blind_goals = [t for t, c in backend.log
                   if c.kind == "goal" and 1.0 + mj.MAX_AGE + 0.1 < t < 3.0]
    assert not blind_goals  # no motion on stale data
    assert any(c.kind == "stop" and 1.0 < t < 1.5 for t, c in backend.log)  # it stopped


def test_when_the_cup_is_removed_the_task_gives_up_with_the_reason():
    r = d.run_one("removed", 1, "task_based")
    blind = d.run_one("removed", 1, "blocking_retry")
    assert r.outcome == "StaleData" and r.closes_on_nothing == 0
    assert r.completion - r.params["removed_at"] < mj.STALE_WAIT + 0.5
    assert blind.outcome == "GraspMissed" and blind.closes_on_nothing == 3


def test_simulation_actions_and_their_refusals():
    backend = mj.MujocoBackend(objects=SCENE)

    async def main():
        start = world.get("sim.cup_red").value
        backend.sim("move", object="cup_red", velocity=(0.0, 0.05), duration=1.0)
        await sleep(0.5)
        midway = world.get("sim.cup_red").value
        backend.sim("occlude", object="cup_red", duration=0.5)
        await sleep(0.3)
        hidden_age = world.get("sim.cup_red").age(backend.now())
        with pytest.raises(ValueError, match="not in the gripper"):
            backend.sim("drop", object="cup_red")
        with pytest.raises(ValueError, match="unknown simulation action"):
            backend.sim("teleport", object="cup_red")
        backend.sim("remove", object="cup_red")
        await sleep(0.5)
        return start, midway, hidden_age, world.get("sim.cup_red").age(backend.now())

    start, midway, hidden_age, removed_age = asyncio.run(
        run_sim(main, backend=backend, until=5)).result
    assert abs((midway[1] - start[1]) - 0.025) < 0.004  # steady 5 cm/s, not eased
    assert hidden_age > 0.25 and removed_age > 0.45  # perception stopped updating
    assert backend.truth("cup_red")[2] < -0.5  # on the floor, out of reach


def test_the_summary_table():
    results = [{"scenario": s, "variant": 0, "program": p, "success": p == "task_based",
                "completion": 10.0, "outcome": "placed" if p == "task_based" else "false success",
                "closes_on_nothing": 0, "params": {"removed_at": 1.0}}
               for s in d.SCENARIOS for p in d.PROGRAMS]
    table = d.summarize(results)
    assert "| slip | 0/1, — [1 false] |" in table and "1/1, 10.0 s |" in table
