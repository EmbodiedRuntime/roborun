"""The Gate's rules as a state machine (SPEC §3).

``GateCore`` decides who owns an actuator, which request is served next, which command
may reach the backend, and what a safety event does. It has no clock, no backend and no
``await``: every method is one event, it changes the state at once, and what has to happen
outside (stop an actuator, cancel a task, answer a request) is appended to ``effects`` for
the caller to carry out, in order. ``roborun.gate.LocalGate`` is that caller.

What the core reads and does not own: the task tree (``parent``, ``state``, ``session`` and
``name`` of the tasks it is given), and which actuators are moving, which the caller tells
it with the event.

Because of this the core can be run through every order of events, and its reachable
states compared with those of the TLA+ models in ``spec/`` (``spec/compare``). Each event
names the model action it corresponds to.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from roborun.checks import Command, Rejection
from roborun.errors import ResourceBusy

KINDS = frozenset({"goal", "setpoint", "effector", "stop"})
ON_BUSY = frozenset({"wait", "fail", "take"})
# Lock causes that only an explicit reset clears (SPEC §3.5).
MANUAL = frozenset({"operator", "fault"})


# -- the journal ---------------------------------------------------------------


def event(method):
    """An event of the core. With a journal attached, the event, what it returned and the
    effects it added are recorded in order (``roborun.journal``). An event called from
    inside another is part of that one."""

    @functools.wraps(method)
    def recorded(self, *args, **kwargs):
        journal = self.journal
        if journal is None or self._depth:
            return method(self, *args, **kwargs)
        before = journal.before(self)
        n = len(self.effects)
        self._depth += 1
        result = error = None
        try:
            result = method(self, *args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            self._depth -= 1
            journal.record(self, method.__name__, args, kwargs, result, error,
                           self.effects[n:], before)

    return recorded


# -- state ---------------------------------------------------------------------


@dataclass(eq=False)
class Grant:
    """Ownership of some actuators; doubles as the capability token."""

    owner: Any  # Task
    resources: frozenset[str]
    revoked: bool = False
    released: bool = False  # given up while moving: kept, with nobody behind it, until at rest


@dataclass(eq=False)
class Waiter:
    """A request for actuators that has not been answered."""

    task: Any
    resources: frozenset[str]
    take: bool = False


@dataclass(eq=False)
class Pending:
    """A command whose checks are running."""

    grant: Grant
    cmd: Command
    task: Any  # the task that sent it, when the Gate knows (in-process)
    passed: frozenset[str]  # the locks it was let through: monitors with an `allow` check


@dataclass(eq=False)
class Watch:
    """The watchdog of a setpoint stream."""

    grant: Grant


# -- effects: what the caller carries out, in order -----------------------------


@dataclass(frozen=True)
class Stop:
    """Send ``stop`` to the backend for this actuator."""

    actuator: str


@dataclass(frozen=True)
class Cancel:
    """Cancel this task. Its tokens are revoked already (SPEC §3.1)."""

    task: Any
    reason: str
    detail: dict | None = None


@dataclass(frozen=True)
class Granted:
    waiter: Waiter
    grant: Grant


@dataclass(frozen=True)
class Refused:
    waiter: Waiter
    message: str


@dataclass(frozen=True)
class Withdrawn:
    """A later take ended this request; its task is cancelled."""

    waiter: Waiter


@dataclass(frozen=True)
class Safety:
    """A safety event, for the trace."""

    kind: str
    actuators: tuple[str, ...]
    cause: str
    detail: str = ""


@dataclass(frozen=True)
class Arm:
    """Call ``watch_expired(actuator, watch)`` when the setpoint timeout has passed."""

    actuator: str
    watch: Watch


@dataclass(frozen=True)
class Settle:
    """Call ``release(grant, moving)`` again when its actuators are at rest."""

    grant: Grant


# -- what the core reads of the task tree ----------------------------------------


def may_take(requester: Any, owner: Any) -> bool:
    """Authority to take actuators from ``owner``, or ahead of its waiting take (SPEC §3.3).

    Within one session it is always allowed; across sessions only for a supervisor.
    """
    if requester.session == owner.session:
        return True
    return owner.session in requester.rt.supervises.get(requester.session, ())


def _is_self_or_ancestor(candidate: Any, task: Any) -> bool:
    node = task
    while node is not None:
        if node is candidate:
            return True
        node = node.parent
    return False


_ENDING = frozenset({"cancelling", "succeeded", "failed", "cancelled"})


def ending(task: Any) -> bool:
    """Whether ``task`` is being cancelled or has ended."""
    return getattr(getattr(task, "state", None), "value", None) in _ENDING


def cancelled_below(task: Any, owner: Any) -> bool:
    """Whether ``task``, or a task between it and its ancestor ``owner``, is cancelled or
    has ended. Such a task uses ``owner``'s token, which is still valid; it must not be
    able to move the robot all the same (SPEC §3.3)."""
    node = task
    while node is not None and node is not owner:
        if ending(node):
            return True
        node = node.parent
    return False


def _lock_message(locks: dict[str, set[str]]) -> str:
    return "; ".join(f"{a} ({', '.join(sorted(c))})" for a, c in sorted(locks.items()))


def _busy_message(resources: frozenset[str], owners: Iterable[Any]) -> str:
    names = ", ".join(sorted({o.name for o in owners}))
    return f"{sorted(resources)} are owned by {names}"


class GateCore:
    def __init__(self, actuators: Iterable[str] = (), *, setpoint_timeout: float = 0.1) -> None:
        self.actuators = frozenset(actuators)
        self.setpoint_timeout = setpoint_timeout  # only for the watchdog's message
        self.owner: dict[str, Grant] = {}
        self.queue: list[Waiter] = []
        self.locks: dict[str, set[str]] = {}  # actuator -> causes
        self.soft: set[str] = set()  # lock causes of monitors that have an `allow` check
        self.failing: set[str] = set()  # monitors that reject
        self.pending: list[Pending] = []
        self.watch: dict[str, Watch] = {}
        self.stopped: dict[Any, set[str]] = {}  # task -> actuators stopped when revoked
        # actuator -> the task that sent its last command. A `stop` does not clear it: the
        # actuator is then still coming to rest from that task's command, and a cancelled
        # task ends only when what it commanded is at rest (SPEC §1.1).
        self.last: dict[str, Any] = {}
        self.settling: list[Grant] = []  # grants given up while moving, kept until rest
        self.effects: list[Any] = []
        self.journal: Any = None  # records every event and its effects, when set
        self._depth = 0

    # -- queries ---------------------------------------------------------------

    def grants_of(self, task: Any) -> list[Grant]:
        """The grants ``task`` owns, in the order of their first actuator."""
        return [g for g in dict.fromkeys(self.owner[a] for a in sorted(self.owner))
                if g.owner is task]

    def owners_of(self, resources: Iterable[str]) -> list[Any]:
        """The tasks that own ``resources``. A grant given up while moving has none: the
        task behind it is done with the actuator, and a take must not cancel it."""
        seen: list[Any] = []
        for r in sorted(resources):
            g = self.owner.get(r)
            if g is not None and not g.released and g.owner not in seen:
                seen.append(g.owner)
        return seen

    def held_by_ancestors(self, task: Any, resources: Iterable[str]) -> set[str]:
        """Resources that ``task`` may already use through itself or an ancestor."""
        held = set()
        for r in resources:
            g = self.owner.get(r)
            if g is not None and not g.revoked and _is_self_or_ancestor(g.owner, task):
                held.add(r)
        return held

    def authority(self, task: Any, actuator: str) -> Grant | None:
        """The grant ``task`` may command ``actuator`` with: its own or an ancestor's, unless
        it or a task between them is cancelled (SPEC §3.3)."""
        g = self.owner.get(actuator)
        if (g is not None and task is not None and _is_self_or_ancestor(g.owner, task)
                and not cancelled_below(task, g.owner)):
            return g
        return None

    def hard_locks(self, actuator: str) -> set[str]:
        """Lock causes that no command passes: all but monitors with an ``allow`` check."""
        return {c for c in self.locks.get(actuator, ()) if c not in self.soft}

    def _owns(self, grant: Grant) -> bool:
        return any(self.owner.get(r) is grant for r in grant.resources)

    # -- requests (SPEC §3.3; Ownership.tla Request, Gate.tla Acquire) -----------

    @event
    def request(self, task: Any, resources: Iterable[str], on_busy: str = "wait") -> Waiter:
        """A request for ``resources``. Raises ``ResourceBusy`` if it fails at once;
        otherwise it is queued, and a ``Granted`` effect answers it now or later.

        Waiters are served first come, first served. ``on_busy="take"`` cancels the current
        owners and the takes already waiting for these actuators (reason ``preempted``), and
        is served first once the owners are terminal; it requires authority over all of
        them, and cancels none without it. So at most one take waits for an actuator, and
        ownership goes directly to it. ``on_busy="fail"`` fails at once when the resources
        are owned or waited for.
        Actuators locked by an operator stop or a fault cannot be acquired until ``reset``;
        actuators locked by a monitor are waited for, like owned ones, unless the monitor has
        an ``allow`` check: then they are granted, and each command is put to that check.
        """
        resources = frozenset(resources)
        if on_busy not in ON_BUSY:
            raise ValueError(f"on_busy must be one of {sorted(ON_BUSY)}, got {on_busy!r}")
        manual = {r: self.locks[r] & MANUAL for r in resources if self.locks.get(r, set()) & MANUAL}
        if manual:
            raise ResourceBusy(f"locked until reset: {_lock_message(manual)}")
        owners = self.owners_of(resources)
        stopping = sorted(r for r in resources if getattr(self.owner.get(r), "released", False))
        locked = [r for r in resources if self.hard_locks(r)]
        # A request still in the queue reserves its actuators, so a free actuator that an
        # earlier request waits for is busy too: `fail` never waits (SPEC §3.3).
        ahead = [e.task for e in self.queue if e.resources & resources]
        if (owners or stopping or locked or ahead) and on_busy == "fail":
            msg = _busy_message(resources, owners) if owners else ""
            if stopping:
                msg += ("; " if msg else "") + (
                    f"{stopping} still stopping after "
                    + ", ".join(sorted({self.owner[r].owner.name for r in stopping})))
            if locked:
                held = {a: set(self.locks[a]) for a in locked}
                msg += ("; " if msg else "") + f"locked: {_lock_message(held)}"
            if ahead:
                msg += ("; " if msg else f"{sorted(resources)}: ") + (
                    "waited for by " + ", ".join(sorted({t.name for t in ahead})))
            raise ResourceBusy(msg)
        if on_busy == "take":
            takers = [e for e in self.queue if e.take and e.resources & resources]
            # A take would cancel a take its own ancestor waits on, and with it itself.
            mine = [e.task for e in takers if _is_self_or_ancestor(e.task, task)]
            if mine:
                raise ResourceBusy(
                    f"{sorted(resources)} are being taken by {mine[0].name}, which started "
                    f"this task; wait for it instead of taking")
            denied = [o for o in owners if not may_take(task, o)]
            if denied:
                raise ResourceBusy(
                    _busy_message(resources, denied)
                    + "; take requires authority over the owner (same session or a supervisor)"
                )
            denied = [e.task for e in takers if not may_take(task, e.task)]
            if denied:
                raise ResourceBusy(
                    f"{sorted(resources)} are being taken by "
                    f"{', '.join(sorted({t.name for t in denied}))}; take requires authority "
                    f"over a waiting take (same session or a supervisor)")
            for o in owners:
                self._cancel_owner(o, "preempted")
            for e in takers:  # their requests are withdrawn; the tasks are cancelled
                self.queue.remove(e)
                self.effects.append(Withdrawn(e))
                self._cancel_owner(e.task, "preempted")
        entry = Waiter(task, resources, take=on_busy == "take")
        if on_busy == "take":
            self.queue.insert(0, entry)
        else:
            self.queue.append(entry)
        self._dispatch()
        return entry

    def _dispatch(self) -> None:
        """Serve waiters in order; a waiter never jumps ahead of an earlier one it overlaps.
        (Ownership.tla Grant, Gate.tla Dispatch.)"""
        reserved: set[str] = set()
        for entry in list(self.queue):
            free = all(r not in self.owner and not self.hard_locks(r) for r in entry.resources)
            if free and not (entry.resources & reserved):
                grant = Grant(entry.task, entry.resources)
                for r in entry.resources:
                    self.owner[r] = grant
                self.queue.remove(entry)
                self.effects.append(Granted(entry, grant))
            else:
                reserved |= entry.resources

    @event
    def expire(self, waiter: Waiter) -> None:
        """The request waited as long as it was willing to (Ownership.tla Timeout)."""
        if waiter not in self.queue:
            return
        self.queue.remove(waiter)
        owners = self.owners_of(waiter.resources)
        self.effects.append(Refused(
            waiter,
            f"waited too long for {sorted(waiter.resources)}"
            + (f"; owned by {', '.join(o.name for o in owners)}" if owners else "")))
        self._dispatch()

    @event
    def withdraw(self, waiter: Waiter) -> None:
        """The task behind a waiting request was cancelled."""
        if waiter in self.queue:
            self.queue.remove(waiter)
        self._dispatch()

    # -- giving up and losing a grant ----------------------------------------------

    @event
    def release(self, grant: Grant, moving: Iterable[str] = ()) -> None:
        """Give up ``grant``; ``moving`` are the actuators that are not at rest now.

        Actuators pass on only at rest, so whatever moves has an owner and the next owner
        never receives a moving actuator: a revoked grant waits for them to stop, and one
        given up while they move (a task that ended, or left its `hold`, without waiting for
        a goal) has them stopped by the Gate first (SPEC §3.3).
        (Gate.tla Release, Settle, End; Ownership.tla End, Leave, Drop.)"""
        moving = tuple(sorted(a for a in moving if a in grant.resources))
        if self._owns(grant) and moving:
            if not grant.revoked:
                grant.revoked = grant.released = True
                # `last` is left as it is: a child that commanded this motion with the
                # token still waits for it to come to rest when it is cancelled.
                for a in moving:
                    self.watch.pop(a, None)
                    self.effects.append(Stop(a))
                self.effects.append(
                    Safety("fallback", moving, "released", f"left moving by {grant.owner.name}"))
            if grant not in self.settling:
                self.settling.append(grant)
                self.effects.append(Settle(grant))
            return
        if grant in self.settling:
            self.settling.remove(grant)
        stopped = self.stopped.get(grant.owner)
        if stopped is not None:
            stopped -= grant.resources
            if not stopped:
                del self.stopped[grant.owner]
        for r in grant.resources:
            if self.owner.get(r) is grant:
                del self.owner[r]
            w = self.watch.get(r)
            if w is not None and w.grant is grant:
                del self.watch[r]
        self._dispatch()

    @event
    def release_all(self, task: Any, moving: Iterable[str] = ()) -> None:
        """``task`` has ended: everything it owns is given up."""
        moving = frozenset(moving)
        self.stopped.pop(task, None)
        self._forget_requests(task)
        for g in self.grants_of(task):
            self.release(g, moving)
        self._dispatch()

    def _forget_requests(self, task: Any) -> None:
        """A task that is cancelled or has ended waits for nothing: a request of its own
        must not be granted in the moment before it is withdrawn."""
        self.queue[:] = [w for w in self.queue if w.task is not task]

    @event
    def rested(self, task: Any) -> None:
        """The actuators stopped when ``task`` was cancelled are at rest."""
        self.stopped.pop(task, None)

    @event
    def revoke(self, task: Any) -> None:
        """``task`` is cancelled: revoke its tokens. A task that used an ancestor's token
        has none to revoke; the Gate stops the motion it commanded, and ``authority``
        refuses what it sends from now on. (Ownership.tla Cancel, Gate.tla Cancel.)"""
        for g in self.grants_of(task):
            self.revoke_grant(g)
        for a in [a for a, t in self.last.items() if t is task]:
            del self.last[a]
            g = self.owner.get(a)
            if g is not None and g.owner is not task:
                self.stopped.setdefault(task, set()).add(a)
                self.watch.pop(a, None)
                self.effects.append(Stop(a))
        if any(w.task is task for w in self.queue):
            self._forget_requests(task)
            self._dispatch()

    @event
    def revoke_grant(self, grant: Grant) -> None:
        """Revoke a token and stop its actuators. The owner can no longer command them, so
        the Gate stops them itself instead of relying on the owner's cleanup (SPEC §1.1)."""
        if grant.revoked:
            return
        grant.revoked = True
        self.stopped.setdefault(grant.owner, set()).update(grant.resources)
        for a in sorted(grant.resources):
            self.watch.pop(a, None)
            self.effects.append(Stop(a))

    def _cancel_owner(self, owner: Any, reason: str, detail: dict | None = None) -> None:
        """A cancellation that starts in the Gate: revoke first, then cancel (SPEC §3.1)."""
        self.revoke(owner)
        self.effects.append(Cancel(owner, reason, detail))

    # -- commands (SPEC §3.4; Gate.tla Send, Unauthorized, Accepted, Rejected, Stop) -----

    @event
    def command(self, grant: Grant | None, cmd: Command, task: Any = None) -> Pending | Rejection | None:
        """A command arrives with ``grant`` (see ``authority``).

        ``stop`` is always accepted: None, and the caller executes it. Otherwise a valid
        grant and no lock are needed; then the checks start, and the caller answers the
        returned ``Pending`` with ``checked``. An unauthorized command changes nothing, so a
        task cannot stop someone else's motion by sending commands it may not send.
        """
        if cmd.kind not in KINDS:
            raise ValueError(f"unknown command kind {cmd.kind!r}; expected one of {sorted(KINDS)}")
        a = cmd.actuator
        if cmd.kind == "stop":
            self.watch.pop(a, None)
            return None
        if grant is None or grant.revoked or a not in grant.resources:
            return Rejection("unauthorized", f"no valid token for {a!r}")
        if self.hard_locks(a):
            return Rejection("locked", f"{a!r} is locked: {', '.join(sorted(self.locks[a]))}")
        p = Pending(grant, cmd, task, frozenset(self.locks.get(a, ())))
        self.pending.append(p)
        return p

    @event
    def dropped(self, p: Pending) -> None:
        """The command is over without reaching the backend: a monitor's ``allow`` check
        refused it, or its sender went away."""
        if p in self.pending:
            self.pending.remove(p)

    @event
    def checked(self, p: Pending, rejection: Rejection | None) -> Rejection | None:
        """The checks have answered. A rejection runs the fallback for the actuator
        (SPEC §3.5). None means: execute the command now, then call ``executed`` or
        ``fault``."""
        a = p.cmd.actuator
        if rejection is not None:
            self.dropped(p)
            self.fallback([a], rejection.code)
            return rejection
        # Checks take time. Meanwhile the token may have been revoked, the actuator locked,
        # or the grant released (its task left a `hold` with this command still in flight)
        # and the actuator given to someone else.
        if p.grant.revoked or set(self.locks.get(a, ())) - p.passed:
            self.dropped(p)
            return Rejection("unauthorized", f"token for {a!r} was revoked")
        if self.owner.get(a) is not p.grant:
            self.dropped(p)
            return Rejection("unauthorized", f"token for {a!r} was released")
        return None

    @event
    def executed(self, p: Pending) -> None:
        """The backend took the command."""
        self.dropped(p)
        a = p.cmd.actuator
        if p.cmd.kind == "setpoint":
            w = self.watch[a] = Watch(p.grant)
            self.effects.append(Arm(a, w))
        elif p.cmd.kind == "goal":
            self.watch.pop(a, None)
        if p.task is not None:
            self.last[a] = p.task
        # The actuator now carries out this command. A cancelled task that waits for what
        # it commanded here to come to rest waits no longer: that motion is over.
        for t in [t for t, acts in self.stopped.items() if a in acts]:
            self.stopped[t].discard(a)
            if not self.stopped[t]:
                del self.stopped[t]

    @event
    def fault(self, p: Pending, message: str) -> None:
        """The backend reported a fault for the command (SPEC §3.5)."""
        self.dropped(p)
        a = p.cmd.actuator
        self.lock([a], "fault", message)
        self._cancel_owner(p.grant.owner, "rejected", {"code": "fault", "message": message})
        self._reject_waiters({a}, "fault")

    # -- fallback, locks, reset (SPEC §3.5) -------------------------------------------

    @event
    def fallback(self, actuators: Iterable[str], cause: str) -> None:
        """The pre-verified behavior: stop. ``stop`` is never checked, so this cannot fail
        because of a check."""
        actuators = tuple(sorted(actuators))
        for a in actuators:
            self.watch.pop(a, None)
            self.effects.append(Stop(a))
        self.effects.append(Safety("fallback", actuators, cause))

    @event
    def lock(self, actuators: Iterable[str], cause: str, detail: str = "") -> None:
        actuators = tuple(sorted(actuators))
        for a in actuators:
            self.locks.setdefault(a, set()).add(cause)
        self.effects.append(Safety("lock", actuators, cause, detail))

    def _unlock(self, actuators: Iterable[str], causes: Iterable[str], kind: str) -> None:
        causes = set(causes)
        freed = []
        for a in sorted(actuators):
            held = self.locks.get(a)
            if not held or not (held & causes):
                continue
            held -= causes
            if not held:
                del self.locks[a]
                freed.append(a)
        if freed:
            self.effects.append(Safety(kind, tuple(freed), ",".join(sorted(causes))))
            self._dispatch()

    @event
    def safety_stop(
        self, actuators: Iterable[str], cause: str, reason: str, detail: dict[str, str]
    ) -> None:
        """Lock ``actuators``, cancel their owners (revoking first), and run the fallback."""
        actuators = frozenset(actuators)
        self.lock(actuators, cause, detail.get("message", ""))
        for owner in self.owners_of(actuators):
            self._cancel_owner(owner, reason, dict(detail))
        if cause in MANUAL:
            self._reject_waiters(actuators, cause)
        self.fallback(actuators, cause)

    def _reject_waiters(self, actuators: Iterable[str], cause: str) -> None:
        """Nothing waits for actuators that are locked until ``reset`` (SPEC §3.5)."""
        actuators = frozenset(actuators)
        for entry in [e for e in self.queue if e.resources & actuators]:
            self.queue.remove(entry)
            self.effects.append(Refused(entry, f"locked until reset: {cause}"))

    @event
    def operator_stop(self, actuators: Iterable[str] | None = None) -> None:
        """An operator stop (SPEC §3.5; Gate.tla OperatorStop): only the named actuators;
        unlocks only on ``reset``. Tasks that hold no actuators keep running."""
        actuators = frozenset(actuators) if actuators is not None else self.actuators
        self.safety_stop(actuators, "operator", "operator",
                         {"code": "operator", "message": "operator stop"})

    @event
    def reset(self, actuators: Iterable[str] | None = None) -> None:
        """Clear operator and fault locks (Gate.tla Reset)."""
        actuators = frozenset(actuators) if actuators is not None else frozenset(self.locks)
        self.effects.append(Safety("reset", tuple(sorted(actuators)), "reset"))
        self._unlock(actuators, MANUAL, "unlock")

    @event
    def monitor(self, name: str, actuators: Iterable[str] | None, rejection: Rejection | None) -> None:
        """A monitor has answered (Gate.tla MonitorFires, MonitorAccepts). Its lock is
        there exactly while it rejects, and goes away by itself (SPEC §3.5)."""
        actuators = frozenset(actuators) if actuators is not None else self.actuators
        cause = f"monitor:{name}"
        if rejection is not None:
            if name not in self.failing:
                self.failing.add(name)
                self.safety_stop(actuators, cause, "rejected",
                                 {"code": rejection.code, "message": rejection.message})
            return
        if name in self.failing:
            self.failing.discard(name)
            self._unlock(actuators, {cause}, "unlock")

    @event
    def watch_expired(self, actuator: str, w: Watch) -> None:
        """No setpoint came within the timeout (Gate.tla Watchdog)."""
        if self.watch.get(actuator) is not w:
            return  # a newer setpoint, a goal, or a stop came in time
        del self.watch[actuator]
        if w.grant.revoked or self.owner.get(actuator) is not w.grant:
            return
        message = f"no setpoint for {actuator!r} within {self.setpoint_timeout}s"
        self.effects.append(Safety("watchdog", (actuator,), "watchdog", message))
        self._cancel_owner(w.grant.owner, "watchdog", {"code": "watchdog", "message": message})
        self.fallback([actuator], "watchdog")
