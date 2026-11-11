import pytest

pytest.importorskip("mujoco")

from benchmarks.safety import run_one


@pytest.mark.parametrize("program", ["table", "through_zone", "sweep_cup"])
def test_unsafe_program_harms_on_baseline_and_is_stopped_on_roborun(program):
    base = run_one(program, 0, "baseline")
    rr = run_one(program, 0, "roborun")
    assert base.harm and base.state == "succeeded"  # it did the damage and reported success
    assert not rr.harm and rr.stopped_by is not None


def test_safe_program_is_not_stopped():
    r = run_one("pick_place", 0, "roborun")
    assert r.completed and not r.harm and r.stopped_by is None
