"""The core in the setting of Ownership.tla: a tree of tasks, `hold` scopes, two actuators,
requests that wait, fail or take.

The environment does what the runtime does around the Gate: a task asks only for what
neither it nor an ancestor holds; cancelling a task cancels its subtree and revokes every
token in it; a cancelled task ends only when the actuators the Gate stopped for it are at
rest; a task that ends gives up everything it owns. Commands pass no checks here (they
are Gate.tla's), so one is executed or refused at once.

A task may end, or leave a `hold`, with an actuator still moving: the core then stops it
and keeps the grant until rest (the model's ``Kept``), and ``Pass`` is the shell finding
it at rest and giving it up.

The instances are those of ``MCOwnership.tla``; a config picks the tasks and scopes.
"""

from __future__ import annotations

import re

from gate_env import Task, core_key
from tla import SPEC, fn

from roborun import core as C
from roborun.checks import Command, Rejection

ACTUATORS = ("arm", "grip")
NEEDS = {"w1": {"arm", "grip"}, "w2": {"arm"}, "g": {"grip"}, "f": {"arm"}, "k1": {"arm"},
         "k2": {"arm"}, "k3": {"arm"}, "kb": {"arm", "grip"}, "c": {"arm"}, "gc": {"arm"},
         "cg": {"arm", "grip"}, "ck": {"grip"}, "hg": {"grip"}, "ha": {"arm"}, "hk": {"grip"}}
MODE = dict.fromkeys(NEEDS, "wait") | {"f": "fail", "k1": "take", "k2": "take", "k3": "take",
                                        "kb": "take", "ck": "take", "hk": "take"}
PARENT = {"c": "w2", "cg": "w2", "gc": "c", "ck": "c"}
OF = {"hg": "w2", "ha": "g", "hk": "w2"}
SUPERVISES = {"s2": {"s1"}}
# Ownership.tla's states of a task -> what the core reads of a task's state.
STATE = {"idle": "pending", "waiting": "pending", "running": "running",
         "cancelling": "cancelling", "terminal": "cancelled"}


def instance(cfg: str) -> dict:
    """The tasks, scopes and special requests a config of MCOwnership picks."""
    text = (SPEC / "cfg" / f"{cfg}.cfg").read_text()

    def members(name: str) -> tuple[str, ...]:
        m = re.search(rf"\b{name} = \{{([^}}]*)\}}", text)
        return tuple(x.strip() for x in m[1].split(",") if x.strip())

    return {"tasks": members("Tasks"), "scopes": members("Scopes"),
            "impatient": members("Impatient"), "repeating": members("Repeating")}


class OwnershipEnv:
    inner = frozenset()

    def __init__(self, tasks, scopes=(), impatient=(), repeating=()) -> None:
        self.core = C.GateCore(ACTUATORS)
        self.names, self.scopes = tuple(tasks), tuple(scopes)
        self.reqs = self.names + self.scopes
        self.impatient, self.repeating = frozenset(impatient), frozenset(repeating)
        self.tasks: dict[str, Task] = {}
        for n in self.names:  # parents first
            self.tasks[n] = Task(n, None, "s2" if n == "k2" else "s1", SUPERVISES)
        for n in self.names:
            if PARENT.get(n) in self.tasks:
                self.tasks[n].parent = self.tasks[PARENT[n]]
        self.st = dict.fromkeys(self.reqs, "idle")
        self.out = dict.fromkeys(self.names, "none")
        self.asks = {x: frozenset() for x in self.reqs}
        self.grant = dict.fromkeys(self.reqs)
        self.waiter = dict.fromkeys(self.reqs)
        self.motion = dict.fromkeys(ACTUATORS, "rest")
        # The model's `last`: the task whose command an actuator is carrying out with an
        # ancestor's token, until it is at rest. It is the world's history, kept here; what
        # the core must get right from its own records is when such a task may end.
        self.last = dict.fromkeys(ACTUATORS)

    # -- the tree ------------------------------------------------------------------

    def tk(self, x: str) -> str:
        return OF.get(x, x)

    def scopes_of(self, t: str) -> list[str]:
        return [h for h in self.scopes if OF[h] == t]

    def kids(self, t: str) -> list[str]:
        return [c for c in self.names if PARENT.get(c) == t]

    def blocked(self, t: str) -> bool:
        return any(self.st[h] == "waiting" for h in self.scopes_of(t))

    def uses(self, t: str) -> set[str]:
        return set(NEEDS[t]).union(*(NEEDS[h] for h in self.scopes_of(t)))

    # -- what the model sees -----------------------------------------------------------

    def _request_of(self, obj, table) -> str | None:
        return next((x for x, o in table.items() if o is obj), None)

    def view(self) -> tuple:
        owner = {a: self._request_of(self.core.owner.get(a), self.grant)
                 if a in self.core.owner else None for a in ACTUATORS}
        return fn({
            "st": fn(self.st),
            "out": fn(self.out),
            "asks": fn((x, a if self.st[x] not in ("idle", "terminal") or self._kept(x)
                        else frozenset()) for x, a in self.asks.items()),
            "owner": fn(owner),
            "revoked": frozenset(x for x, g in self.grant.items()
                                 if g is not None and g.revoked and self.core._owns(g)),
            "queue": ("seq", tuple(self._request_of(w, self.waiter) for w in self.core.queue)),
            "motion": fn(self.motion),
            "last": fn(self.last),
        })

    def key(self) -> tuple:
        return (core_key(self.core), tuple(self.st.items()), tuple(self.out.items()),
                tuple((x, tuple(sorted(a))) for x, a in self.asks.items()),
                tuple(self.motion.items()), tuple(self.last.items()),
                tuple((x, g is not None and self.core._owns(g)) for x, g in self.grant.items()))

    # -- what can happen -----------------------------------------------------------------

    def events(self) -> list[tuple]:
        out: list[tuple] = []
        for x in self.reqs:
            by = OF.get(x, PARENT.get(x))
            if self.st[x] == "idle" and (
                    by is None or (self.st.get(by) == "running" and not self.blocked(by))):
                out.append(("Request", x))
            if x in self.impatient and self.st[x] == "waiting":
                out.append(("Timeout", x))
        for t in self.names:
            st = self.st[t]
            if st in ("waiting", "running"):
                out.append(("Cancel", t))
            if st in ("running", "cancelling") and not self.blocked(t):
                for a in sorted(self.uses(t)):
                    out.append(("Command", t, a))
                    if self.motion[a] == "moving":
                        out.append(("TaskStop", t, a))
                if (all(self.st[c] in ("idle", "terminal") for c in self.kids(t))
                        and all(self.st[h] in ("idle", "terminal", "cancelling")
                                for h in self.scopes_of(t))
                        and (st == "running" or self._settled(t))):
                    for o in (("succeeded", "failed") if st == "running"
                              else ("cancelled", "succeeded", "failed")):
                        out.append(("End", t, o))
            if (t in self.repeating and st == "terminal"
                    and not any(self._kept(x) for x in (t, *self.scopes_of(t)))):
                out.append(("Again", t))
        for x in self.reqs:
            if self._kept(x) and self._resting(x):
                out.append(("Pass", x))
        for h in self.scopes:
            if self.st[h] == "running" and self.st[OF[h]] == "running":
                out.append(("Leave", h))
            if self.st[h] == "cancelling":
                out.append(("Drop", h))
        for a in ACTUATORS:
            if self.motion[a] == "moving":
                out.append(("Arrive", a))
            if self.motion[a] == "braking":
                out.append(("Brake", a))
        return out

    def apply(self, event: tuple) -> None:
        name, *args = event
        getattr(self, "_" + name)(*args)
        self._react()

    # -- the runtime's side ------------------------------------------------------------------

    def _set(self, x: str, st: str) -> None:
        self.st[x] = st
        if st == "idle":
            self.asks[x] = frozenset()
        if x in self.tasks:
            self.tasks[x].state.value = STATE[st]

    def _moving(self) -> list[str]:
        return [a for a in ACTUATORS if self.motion[a] != "rest"]

    def _resting(self, *reqs: str) -> bool:
        """Whether everything these requests own is at rest."""
        return all(self.motion[a] == "rest" for x in reqs if self.grant[x] is not None
                   for a in self.grant[x].resources if self.core.owner.get(a) is self.grant[x])

    def _kept(self, x: str) -> bool:
        """Whether the Gate keeps the grant of a request whose task has given it up."""
        g = self.grant[x]
        return self.st[x] == "terminal" and g is not None and self.core._owns(g)

    def _settled(self, t: str) -> bool:
        """`Task._settled`: the actuators the Gate stopped for the task are at rest."""
        return all(self.motion[a] == "rest" for a in self.core.stopped.get(self.tasks[t], ()))

    def _cancel(self, t: str) -> None:
        """`Task.cancel`: a task that waits to start ends at once; a running one is
        revoked, the `hold` it waits in is interrupted, and its children are cancelled."""
        if self.st[t] == "waiting":
            self._set(t, "terminal")
            self.out[t] = "cancelled"
            self.core.release_all(self.tasks[t], self._moving())
        elif self.st[t] == "running":
            self._set(t, "cancelling")
            self.core.revoke(self.tasks[t])
            for h in self.scopes_of(t):
                if self.st[h] == "running":
                    self._set(h, "cancelling")
                elif self.st[h] == "waiting":
                    self._set(h, "terminal")
                    self.core.withdraw(self.waiter[h])
            for c in self.kids(t):
                self._cancel(c)

    def _refused(self, x: str) -> None:
        self._set(x, "terminal")
        if x in self.tasks:
            self.out[x] = "failed"

    def _react(self) -> None:
        effects = self.core.effects
        while effects:
            e = effects.pop(0)
            if isinstance(e, C.Stop):
                if self.motion[e.actuator] == "moving":
                    self.motion[e.actuator] = "braking"
            elif isinstance(e, C.Cancel):
                self._cancel(e.task.name)
            elif isinstance(e, C.Granted):
                x = self._request_of(e.waiter, self.waiter)
                self.grant[x] = e.grant
                self._set(x, "running")
            elif isinstance(e, C.Refused):
                self._refused(self._request_of(e.waiter, self.waiter))
            # Withdrawn: the Cancel that follows ends the task or interrupts its `hold`.

    # -- the events ---------------------------------------------------------------------------

    def _Request(self, x: str) -> None:
        t = self.tasks[self.tk(x)]
        needed = frozenset(NEEDS[x]) - self.core.held_by_ancestors(t, NEEDS[x])
        if not needed:
            self._set(x, "running")  # it uses what is held above it
            return
        try:
            self.waiter[x] = self.core.request(t, needed, MODE[x])
        except C.ResourceBusy:
            self._refused(x)
            return
        self._set(x, "waiting")
        self.asks[x] = needed

    def _Cancel(self, t: str) -> None:
        self._cancel(t)

    def _Timeout(self, x: str) -> None:
        self.core.expire(self.waiter[x])

    def _Command(self, t: str, a: str) -> None:
        task = self.tasks[t]
        p = self.core.command(self.core.authority(task, a), Command("goal", a, {}), task)
        if isinstance(p, Rejection):
            return
        if self.core.checked(p, None) is None:
            self.motion[a] = "moving"
            self.last[a] = None if p.grant.owner is task else t
            self.core.executed(p)

    def _TaskStop(self, t: str, a: str) -> None:
        self.core.command(None, Command("stop", a, {}))
        self.motion[a] = "braking"

    def _Arrive(self, a: str) -> None:
        self.motion[a] = "rest"
        self.last[a] = None

    _Brake = _Arrive

    def _End(self, t: str, o: str) -> None:
        task = self.tasks[t]
        self.core.rested(task)
        for h in self.scopes_of(t):
            if self.st[h] != "idle":
                self._set(h, "terminal")
        self._set(t, "terminal")
        self.out[t] = o
        self.core.release_all(task, self._moving())

    def _Leave(self, h: str) -> None:
        self._set(h, "terminal")
        self.core.release(self.grant[h], self._moving())

    def _Pass(self, x: str) -> None:
        """`LocalGate._release_at_rest`: what was kept is at rest."""
        self.core.release(self.grant[x], self._moving())

    def _Drop(self, h: str) -> None:
        """The cancelled task unwinds out of its `hold`. The Gate keeps a revoked grant
        until its actuators are at rest, and gives it up then."""
        g = self.grant[h]
        self.core.release(g, self._moving())
        if not self.core._owns(g):
            self._set(h, "terminal")

    def _Again(self, t: str) -> None:
        for x in (t, *self.scopes_of(t)):
            self._set(x, "idle")
            self.grant[x] = self.waiter[x] = None
        self.out[t] = "none"
