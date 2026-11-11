import pytest

pytest.importorskip("mujoco")

from benchmarks.stale import run_one


def test_occluded_cup_no_empty_grasp_with_timestamps():
    assert run_one("occluded", 1, "roborun-no-world").metrics["closes_on_nothing"] >= 1
    assert run_one("occluded", 1, "roborun").metrics["closes_on_nothing"] == 0


def test_dead_tracker_is_noticed_and_restarted():
    without = run_one("tracker_died", 1, "roborun-no-world").metrics
    assert not without["placed"] and without["closes_on_nothing"] >= 1
    m = run_one("tracker_died", 1, "roborun").metrics
    assert m["placed"] and m["noticed_stale"] and m["closes_on_nothing"] == 0


def test_silent_sensor_stops_the_arm_before_the_hand_arrives():
    assert run_one("sensor_dropout", 0, "roborun-no-world").metrics["harm"]
    m = run_one("sensor_dropout", 0, "roborun").metrics
    assert not m["harm"] and m["stopped_for"] == ["stale data"]


def test_growing_margin_keeps_working_far_from_the_hand_and_still_stops_near_it():
    far = (2, "sensor_glitch")  # hand 0.8 m away, 0.3 s silences
    assert run_one(far[1], far[0], "roborun").metrics["moves"] <= 1
    assert run_one(far[1], far[0], "roborun-growing").metrics["moves"] >= 5
    assert not run_one("sensor_dropout", 0, "roborun-growing").metrics["harm"]
