import pytest

pytest.importorskip("mujoco")

from benchmarks import ablate, crash, safety, stopping
from roborun.gate import LocalGate
from roborun.stamped import Stamped
from roborun.task import Task


def test_ablation_switches_are_undone():
    before = (LocalGate.admit, Task._request_cancel, Stamped.age)
    for part in ablate.PARTS:
        with ablate.without(part):
            pass
    assert (LocalGate.admit, Task._request_cancel, Stamped.age) == before
    assert Stamped(1, 0.0).age(2.0) == 2.0


def test_parse():
    assert ablate.parse("all") == ablate.PARTS
    assert ablate.parse("gate, world") == ("gate", "world")
    with pytest.raises(SystemExit):
        ablate.parse("planner")


def test_without_the_gate_unsafe_commands_reach_the_arm():
    assert safety.run_one("keep_out_goal", 0, "roborun-no-gate").harm
    assert stopping.run_one("conflict", 0, "roborun-no-gate").metrics["target_switches"] > 10


def test_without_tasks_cancel_does_not_wait_and_a_dead_agent_keeps_the_arm_going():
    assert stopping.run_one("cancel", 0, "roborun-no-task").metrics["speed_when_cancel_returned"] > 0.1
    assert crash.run_one("kill", 1, "roborun-no-task").reached_goal
