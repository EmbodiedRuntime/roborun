import pytest

pytest.importorskip("mujoco")

from benchmarks.stopping import run_one


def test_cancel_returns_with_the_arm_at_rest_on_roborun():
    assert run_one("cancel", 0, "baseline").metrics["speed_when_cancel_returned"] > 0.1
    assert run_one("cancel", 0, "roborun").metrics["speed_when_cancel_returned"] < 0.02


def test_cancelled_code_cannot_move_the_arm_on_roborun():
    assert run_one("zombie", 0, "baseline").metrics["commands_reaching_arm_after_cancel"] > 10
    assert run_one("zombie", 0, "roborun").metrics["commands_reaching_arm_after_cancel"] == 0


def test_second_agent_waits_its_turn_on_roborun():
    assert run_one("conflict", 0, "baseline").metrics["target_switches"] > 10
    assert run_one("conflict", 0, "roborun").metrics["target_switches"] <= 1


def test_operator_stop_holds_against_retries_on_roborun():
    assert run_one("operator", 0, "baseline").metrics["reached_goal_anyway"]
    assert not run_one("operator", 0, "roborun").metrics["reached_goal_anyway"]
