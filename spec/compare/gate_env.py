"""The core in the setting of Gate.tla: one actuator, tasks that each ask for it once.

Besides the core, the environment holds what the model's ``ts`` and ``motion`` describe:
where each task is, and whether the arm moves. It does to the core what the runtime and
``LocalGate`` do: a cancelled task has its tokens revoked, a task that ends gives up what
it owns, a rejected command cancels its sender, the grant kept after a release is given up
at rest.
"""

from __future__ import annotations

from types import SimpleNamespace

from tla import fn

from roborun import core as C
from roborun.checks import Command, Rejection

ARM = "arm"
MONITOR = "m"
# Where a task is in Gate.tla -> what the core reads of a task's state.
STATE = {"idle": "pending", "waiting": "pending", "owning": "running",
         "cancelling": "cancelling", "done": "succeeded"}


class Task:
    """What the core reads of a task (hashable by identity, as the runtime's tasks are)."""

    def __init__(self, name: str, parent=None, session: str = "s", supervises=None) -> None:
        self.name, self.parent, self.session = name, parent, session
        self.state = SimpleNamespace(value="pending")
        self.rt = SimpleNamespace(supervises=supervises or {})


task = Task


def grant_key(g):
    return None if g is None else (g.owner.name, tuple(sorted(g.resources)), g.revoked, g.released)


def core_key(core: C.GateCore) -> tuple:
    """Everything in the core that its future depends on, as a value."""
    return (
        tuple(sorted((a, grant_key(g)) for a, g in core.owner.items())),
        tuple((w.task.name, tuple(sorted(w.resources)), w.take) for w in core.queue),
        tuple(sorted((a, tuple(sorted(c))) for a, c in core.locks.items())),
        tuple(sorted(core.failing)),
        tuple(sorted((grant_key(p.grant), p.cmd.kind, p.cmd.actuator,
                      getattr(p.task, "name", None), tuple(sorted(p.passed)))
                     for p in core.pending)),
        tuple(sorted((a, grant_key(w.grant)) for a, w in core.watch.items())),
        tuple(sorted((t.name, tuple(sorted(s))) for t, s in core.stopped.items())),
        tuple(sorted((a, t.name) for a, t in core.last.items())),
        tuple(sorted(grant_key(g) for g in core.settling)),
    )


class GateEnv:
    inner = frozenset({"Dispatch"})

    def __init__(self, names=("a", "b", "c"), *, allow: bool = False) -> None:
        self.core = C.GateCore([ARM])
        if allow:
            self.core.soft.add(f"monitor:{MONITOR}")
        self.tasks = {n: task(n) for n in names}
        self.ts = dict.fromkeys(names, "idle")
        self.grant = dict.fromkeys(names)
        self.pend = dict.fromkeys(names)
        self.motion = "rest"
        self.step = ("init", None)

    # -- what the model sees -----------------------------------------------------

    def granted(self, n: str) -> str:
        g = self.grant[n]
        if g is None:
            return "none"
        if self.core.owner.get(ARM) is not g:
            return "released"
        return "revoked" if g.revoked else "valid"

    def locks(self) -> frozenset:
        return frozenset(c.split(":")[0] for c in self.core.locks.get(ARM, ()))

    def view(self) -> tuple:
        return fn({
            "ts": fn(self.ts),
            "grant": fn((n, self.granted(n)) for n in self.tasks),
            "pending": fn((n, None if p is None else fn({
                "kind": p.cmd.kind, "passed": frozenset(c.split(":")[0] for c in p.passed)}))
                for n, p in self.pend.items()),
            "locks": self.locks(),
            "failing": bool(self.core.failing),
            "motion": self.motion,
            "watch": ARM in self.core.watch,
            "step": fn({"a": self.step[0], "t": self.step[1]}),
        })

    def key(self) -> tuple:
        return (core_key(self.core), tuple(self.ts.items()), self.motion, self.step,
                tuple((n, self.granted(n)) for n in self.tasks))

    # -- what can happen -----------------------------------------------------------

    def events(self) -> list[tuple]:
        out: list[tuple] = []
        for n in self.tasks:
            ts, g, p = self.ts[n], self.granted(n), self.pend[n]
            if ts == "idle":
                out.append(("Acquire", n))
            if g != "valid":
                out.append(("Unauthorized", n))
            elif p is None:
                for kind in ("goal", "setpoint"):
                    out.append(("Send", n, kind, True))
                    if self.core.locks.get(ARM):
                        out.append(("Send", n, kind, False))  # the monitor's `allow` refuses
            if p is not None:
                out += [("Accepted", n, "taken"), ("Accepted", n, "fault"), ("Rejected", n)]
            if ts in ("owning", "waiting"):
                out.append(("Cancel", n))
            if ts == "owning":
                out.append(("Release", n))
            if ts == "done" and g == "revoked" and self.motion == "rest":
                out.append(("Settle", n))
            if ts == "cancelling" and self.motion == "rest":
                out.append(("End", n))
        out.append(("Stop",))
        if self.motion == "stopping":
            out.append(("Rest",))
        out.append(("MonitorAccepts",) if self.core.failing else ("MonitorFires",))
        out.append(("OperatorStop",))
        if self.core.locks.get(ARM, set()) & C.MANUAL:
            out.append(("Reset",))
        if ARM in self.core.watch:
            out.append(("Watchdog",))
        return out

    def apply(self, event: tuple) -> None:
        name, *args = event
        getattr(self, "_" + name)(*args)
        self._react()

    # -- the runtime's side ----------------------------------------------------------

    def _set(self, n: str, ts: str) -> None:
        self.ts[n] = ts
        self.tasks[n].state.value = STATE[ts]

    def _moving(self) -> list[str]:
        return [] if self.motion == "rest" else [ARM]

    def _brake(self) -> None:
        if self.motion == "moving":
            self.motion = "stopping"

    def _cancelled(self, n: str) -> None:
        """What `Task.cancel` does: a running task is revoked and goes on to its cleanup;
        one that waits for the arm ends at once."""
        if self.ts[n] == "owning":
            self._set(n, "cancelling")
            self.core.revoke(self.tasks[n])
        elif self.ts[n] == "waiting":
            self._set(n, "done")
            self.core.release_all(self.tasks[n], self._moving())

    def _react(self) -> None:
        effects = self.core.effects
        while effects:
            e = effects.pop(0)
            if isinstance(e, C.Stop):
                self._brake()
            elif isinstance(e, C.Cancel):
                self._cancelled(e.task.name)
            elif isinstance(e, C.Granted):
                n = e.waiter.task.name
                self._set(n, "owning")
                self.grant[n] = e.grant
                self.step = ("grant", n)
            elif isinstance(e, (C.Refused, C.Withdrawn)):
                self._set(e.waiter.task.name, "done")

    # -- the events ---------------------------------------------------------------------

    def _Acquire(self, n: str) -> None:
        try:
            self.core.request(self.tasks[n], [ARM], "wait")
        except C.ResourceBusy:
            self._set(n, "done")
            self.step = ("busy", n)
            return
        self._set(n, "waiting")
        self.step = ("wait", n)  # a grant in the same step shows in `_react`

    def _command(self, n: str, kind: str, allowed: bool) -> None:
        t = self.tasks[n]
        r = self.core.command(self.core.authority(t, ARM), Command(kind, ARM, {}), t)
        if isinstance(r, Rejection):
            self.step = (r.code, n)
        elif r.passed and not allowed:
            self.core.dropped(r)
            self.step = ("locked", n)
        else:
            self.pend[n] = r
            self.step = ("checking", n)

    def _Unauthorized(self, n: str) -> None:
        self._command(n, "goal", True)

    def _Send(self, n: str, kind: str, allowed: bool) -> None:
        self._command(n, kind, allowed)

    def _Accepted(self, n: str, backend: str) -> None:
        p, self.pend[n] = self.pend[n], None
        r = self.core.checked(p, None)
        if r is not None:
            self.step = (r.code, n)
        elif backend == "taken":
            self.motion = "moving"
            self.core.executed(p)
            self.step = ("execute", n)
        else:
            self.core.fault(p, "fault")
            self.step = ("fault", n)

    def _Rejected(self, n: str) -> None:
        p, self.pend[n] = self.pend[n], None
        self.core.checked(p, Rejection("rejected", "a check said no"))
        self._react()
        self._cancelled(n)  # `send` cancels the task whose command was rejected
        self.step = ("reject", n)

    def _Stop(self) -> None:
        self.core.command(None, Command("stop", ARM, {}))
        self._brake()
        self.step = ("stop", None)

    def _Rest(self) -> None:
        self.motion = "rest"
        self.step = ("rest", None)

    def _Cancel(self, n: str) -> None:
        self._cancelled(n)
        self.step = ("cancel", n)

    def _Release(self, n: str) -> None:
        moving = self._moving()
        self._set(n, "done")
        self.core.release_all(self.tasks[n], moving)
        self.step = ("release-stop" if moving else "release", n)

    def _Settle(self, n: str) -> None:
        self.core.release(self.grant[n], self._moving())
        self.step = ("settle", n)

    def _End(self, n: str) -> None:
        self.core.rested(self.tasks[n])
        self._set(n, "done")
        self.core.release_all(self.tasks[n], self._moving())
        self.step = ("end", n)

    def _MonitorFires(self) -> None:
        self.core.monitor(MONITOR, None, Rejection("rejected", "a person is near"))
        self.step = ("monitor", None)

    def _MonitorAccepts(self) -> None:
        self.core.monitor(MONITOR, None, None)
        self.step = ("unlock", None)

    def _OperatorStop(self) -> None:
        self.core.operator_stop()
        self.step = ("operator", None)

    def _Reset(self) -> None:
        self.core.reset()
        self.step = ("reset", None)

    def _Watchdog(self) -> None:
        self.core.watch_expired(ARM, self.core.watch[ARM])
        self.step = ("watchdog", None)
