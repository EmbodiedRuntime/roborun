import pytest

pytest.importorskip("mujoco")

from benchmarks.concurrency import run_one


def test_planning_overlaps_motion_on_roborun():
    base, rr = run_one("baseline", 3.0), run_one("roborun", 3.0)
    assert base.placed == rr.placed == 2
    assert rr.total < 0.8 * base.total and rr.idle < base.idle / 4
