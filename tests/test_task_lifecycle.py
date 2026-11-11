import itertools

from roborun import TaskState, can_transition

S = TaskState

ALLOWED = {
    (S.PENDING, S.RUNNING),
    (S.PENDING, S.CANCELLED),
    (S.PENDING, S.FAILED),  # could not start, e.g. actuators stayed busy
    (S.RUNNING, S.SUCCEEDED),
    (S.RUNNING, S.FAILED),
    (S.RUNNING, S.CANCELLING),
    (S.CANCELLING, S.CANCELLED),
    (S.CANCELLING, S.SUCCEEDED),
    (S.CANCELLING, S.FAILED),
}


def test_transitions_match_spec_exhaustively():
    for src, dst in itertools.product(S, S):
        assert can_transition(src, dst) == ((src, dst) in ALLOWED), (src, dst)


def test_terminal_states_never_change():
    for src in S:
        if src.terminal:
            assert not any(can_transition(src, dst) for dst in S)


def test_cancel_request_is_not_a_stop():
    assert not S.CANCELLING.terminal
    assert not can_transition(S.RUNNING, S.CANCELLED)
