"""The core with one mechanism switched off, as the models' mutation configs have it.

They are the control of the comparison: a core changed this way must differ from the model
as specified, as the model with the mechanism off must differ from the core.
"""

from __future__ import annotations

import contextlib

from roborun.checks import Rejection
from roborun.core import GateCore


def _norecheck(core_cls):
    """Gate.tla Recheck = FALSE: after the checks, a revoked token or a new lock is not
    looked at again."""
    def checked(self, p, rejection):
        a = p.cmd.actuator
        if rejection is not None:
            self.dropped(p)
            self.fallback([a], rejection.code)
            return rejection
        if self.owner.get(a) is not p.grant:
            self.dropped(p)
            return Rejection("unauthorized", f"token for {a!r} was released")
        return None
    return {"checked": checked}


def _norecheckrelease(core_cls):
    """Gate.tla RecheckRelease = FALSE: after the checks, a released token still counts."""
    def checked(self, p, rejection):
        a = p.cmd.actuator
        if rejection is not None:
            self.dropped(p)
            self.fallback([a], rejection.code)
            return rejection
        if p.grant.revoked or set(self.locks.get(a, ())) - p.passed:
            self.dropped(p)
            return Rejection("unauthorized", f"token for {a!r} was revoked")
        return None
    return {"checked": checked}


def _faultkeepswaiters(core_cls):
    """Gate.tla FaultRejects = FALSE: a fault leaves the requests for the actuator waiting."""
    def fault(self, p, message):
        self.dropped(p)
        a = p.cmd.actuator
        self.lock([a], "fault", message)
        self._cancel_owner(p.grant.owner, "rejected", {"code": "fault", "message": message})
    return {"fault": fault}


def _unauthfallback(core_cls):
    """Gate.tla UnauthFallback = TRUE: a command without a valid token runs the fallback."""
    command = core_cls.command

    def mutated(self, grant, cmd, task=None):
        r = command(self, grant, cmd, task)
        if isinstance(r, Rejection) and r.code == "unauthorized":
            self.fallback([cmd.actuator], "unauthorized")
        return r
    return {"command": mutated}


def _nostoponrelease(core_cls):
    """Gate.tla StopOnRelease = FALSE: a grant given up while the actuator moves is
    released at once."""
    release = core_cls.release

    def mutated(self, grant, moving=()):
        release(self, grant, ())
    return {"release": mutated}


MUTATIONS = {
    "norecheck": _norecheck,
    "norecheckrelease": _norecheckrelease,
    "faultkeepswaiters": _faultkeepswaiters,
    "unauthfallback": _unauthfallback,
    "nostoponrelease": _nostoponrelease,
}

@contextlib.contextmanager
def mutated(name: str | None):
    if name is None:
        yield
        return
    patch = MUTATIONS[name](GateCore)
    saved = {k: GateCore.__dict__[k] for k in patch}
    try:
        for k, v in patch.items():
            setattr(GateCore, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(GateCore, k, v)
