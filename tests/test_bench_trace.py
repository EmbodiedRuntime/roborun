import pytest

pytest.importorskip("mujoco")

from benchmarks.diagnosis import produce


def test_a_rejection_is_explained_by_the_trace_not_the_error_line():
    case = produce("rejected", 0)
    assert "rejected" in case.error_line and "workspace" not in case.error_line
    assert "check:workspace" in case.trace and "z = 0.000" in case.trace


def test_the_trace_shows_where_a_moved_cup_ended_up():
    case = produce("moved", 0)
    assert "GraspMissed" in case.error_line
    assert "cup_red=" in case.trace  # the state at the end: the cup is not where the grasp went
