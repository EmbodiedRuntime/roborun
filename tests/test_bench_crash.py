import pytest

pytest.importorskip("mujoco")

from benchmarks.crash import run_one


def test_a_killed_agent_stops_the_arm_on_roborun_only():
    base = run_one("kill", 0, "baseline")
    rr = run_one("kill", 0, "roborun")
    assert base.reached_goal  # the driver kept executing the dead agent's goal
    assert not rr.reached_goal and rr.moved_after < 0.1 and rr.rest_after is not None


def test_a_frozen_agent_is_caught_by_the_heartbeat_watchdog():
    rr = run_one("freeze", 0, "roborun")
    assert not rr.reached_goal and rr.rest_after is not None and rr.rest_after < 2.0
