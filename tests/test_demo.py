"""The cup-nudge demo and its benchmark measurements."""

import pytest

cn = pytest.importorskip("roborun.demos.cup_nudge")
mj = pytest.importorskip("roborun.backends.mujoco")
if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
    pytest.skip("MuJoCo Menagerie Panda model not found", allow_module_level=True)


def test_one_push_separates_the_programs_and_latency_is_measured():
    blocking = cn.run_one("blocking", 1.0, "+y")
    task = cn.run_one("task_based", 1.0, "+y")
    assert not blocking.success and blocking.outcome == "GraspMissed" and blocking.latency is None
    assert task.success and task.outcome == "placed"
    assert 0 < task.latency < 0.5


def test_retry_recovers_but_reacts_late():
    r = cn.run_one("blocking_retry", 1.0, "+y")
    assert r.success and r.latency > 1.5


def test_the_summary_table():
    results = [
        {"program": p, "push_time": t, "direction": "+y", "success": ok, "completion": 9.0,
         "latency": lat, "outcome": "placed" if ok else "GraspMissed"}
        for p, ok, lat in (("blocking", False, None), ("blocking_retry", True, 2.9),
                           ("task_based", True, 0.3))
        for t in (None, 1.0)
    ]
    table = cn.summarize(results)
    assert "| `blocking` | 0/1 | 0/1 | never | never |" in table
    assert "0.30 s (1/1 reacted)" in table
