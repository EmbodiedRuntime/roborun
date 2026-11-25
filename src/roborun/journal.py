"""A journal of the Gate's core: every event, what it returned, its effects, and the state
after it, in order.

The core (``roborun.core``) is a state machine, so a run of the Gate is a sequence of its
events. Recorded, that sequence can be put to the TLA+ models in ``spec/``: ``spec/validate``
asks TLC whether the run is a behaviour of the model. The journal is for that and for
debugging; the Trace (``roborun.trace``) stays the record of a run for people and models.

    journal = gate.record()          # LocalGate, in-process or inside the Gate process
    ...
    journal.save("run.journal.jsonl")
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roborun import core as C
from roborun.checks import Command, Rejection


class Journal:
    """``probe()`` says which actuators are at rest; it is asked before every event, since
    the core is told about motion only through its events."""

    def __init__(self, probe: Callable[[], dict[str, bool]] | None = None) -> None:
        self.probe = probe or (dict)
        self.entries: list[dict] = []
        self.meta: dict = {}
        self._ids: dict[int, str] = {}
        self._keep: list[Any] = []  # the objects named, so that their ids stay theirs
        self._count: dict[str, int] = {}

    # -- names for the things the core handles ------------------------------------

    def _name(self, prefix: str, obj: Any) -> str:
        key = id(obj)
        if key not in self._ids:
            self._count[prefix] = n = self._count.get(prefix, 0) + 1
            self._ids[key] = f"{prefix}{n}"
            self._keep.append(obj)
        return self._ids[key]

    def task(self, t: Any) -> dict | None:
        if t is None:
            return None
        parent = getattr(t, "parent", None)
        return {"id": self._name("t", t), "name": getattr(t, "name", ""),
                "state": getattr(getattr(t, "state", None), "value", None),
                "parent": None if parent is None else self._name("t", parent)}

    def request(self, x: Any) -> str | None:
        """A request and the grant that answers it have one name."""
        return None if x is None else self._name("r", x)

    def _plain(self, x: Any) -> Any:
        if isinstance(x, C.Waiter):
            return {"request": self.request(x), "task": self.task(x.task),
                    "resources": sorted(x.resources), "take": x.take}
        if isinstance(x, C.Grant):
            return {"request": self.request(x), "task": self.task(x.owner),
                    "resources": sorted(x.resources), "revoked": x.revoked,
                    "released": x.released}
        if isinstance(x, C.Pending):
            return {"pending": self._name("p", x), "request": self.request(x.grant),
                    "kind": x.cmd.kind, "actuator": x.cmd.actuator,
                    "task": self.task(x.task), "passed": sorted(x.passed)}
        if isinstance(x, C.Watch):
            return {"watch": self.request(x.grant)}
        if isinstance(x, Command):
            return {"kind": x.kind, "actuator": x.actuator}
        if isinstance(x, Rejection):
            return {"rejected": x.code, "message": x.message}
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): self._plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set, frozenset)):
            items = [self._plain(v) for v in x]
            return sorted(items) if isinstance(x, (set, frozenset)) else items
        if hasattr(x, "state") and hasattr(x, "parent"):
            return self.task(x)
        return repr(x)

    def _effect(self, e: Any) -> dict:
        out = {"effect": type(e).__name__}
        for k, v in vars(e).items():
            out[k] = self._plain(v)
        return out

    def state(self, core: C.GateCore) -> dict:
        """The core's state, by name."""
        grants = list(dict.fromkeys(core.owner.values()))
        return {
            "owner": {a: self.request(g) for a, g in sorted(core.owner.items())},
            "revoked": sorted(self.request(g) for g in grants if g.revoked),
            "released": sorted(self.request(g) for g in grants if g.released),
            "queue": [self.request(w) for w in core.queue],
            "locks": {a: sorted(c) for a, c in sorted(core.locks.items())},
            "failing": sorted(core.failing),
            "pending": sorted(self._name("p", p) for p in core.pending),
            "watch": {a: self.request(w.grant) for a, w in sorted(core.watch.items())},
            "settling": sorted(self.request(g) for g in core.settling),
            "stopped": {self._name("t", t): sorted(a) for t, a in core.stopped.items()},
            "last": {a: self._name("t", t) for a, t in sorted(core.last.items())},
        }

    # -- called by the core ---------------------------------------------------------

    def before(self, core: C.GateCore) -> dict:
        if not self.meta:
            self.meta = {"actuators": sorted(core.actuators), "soft": sorted(core.soft)}
        return {"rest": dict(self.probe())}

    def record(self, core: C.GateCore, event: str, args: tuple, kwargs: dict, result: Any,
               error: BaseException | None, effects: list, before: dict) -> None:
        for e in effects:  # a grant has the name of the request it answers
            if isinstance(e, C.Granted):
                self._ids[id(e.grant)] = self.request(e.waiter)
                self._keep.append(e.grant)
        entry = {"n": len(self.entries) + 1, "event": event, "rest": before["rest"],
                 "args": [self._plain(a) for a in args]}
        if kwargs:
            entry["kwargs"] = self._plain(kwargs)
        if error is not None:
            entry["raised"] = type(error).__name__
        elif result is not None:
            entry["result"] = self._plain(result)
        entry["effects"] = [self._effect(e) for e in effects]
        entry["state"] = self.state(core)
        self.entries.append(entry)

    # -- files ------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"journal": 1, **self.meta, "soft": self.meta.get("soft", [])}) + "\n")
            f.writelines(json.dumps(e) + "\n" for e in self.entries)


def load(path: str | Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines[0], lines[1:]
