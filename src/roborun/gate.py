"""The Gate (SPEC §3): control authority and safety over actuators.

``LocalGate`` holds the Gate's logic: ownership, arbitration (first come, first served;
``take`` with authority), checks and monitors (``roborun.checks``), the setpoint watchdog,
the fallback, and the locked state. It runs in-process for tests and ``run_sim``, and inside
the Gate process (``roborun.gateproc``), which adds capability tokens and a socket.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roborun.checks import Checks, Command, Monitor, Rejection
from roborun.clock import Clock
from roborun.errors import BackendError, ResourceBusy

if TYPE_CHECKING:
    from roborun.task import Task

__all__ = ["KINDS", "ON_BUSY", "Command", "Grant", "LocalGate", "may_take"]

KINDS = frozenset({"goal", "setpoint", "effector", "stop"})
ON_BUSY = frozenset({"wait", "fail", "take"})
# Lock causes that only an explicit reset clears (SPEC §3.5).
MANUAL = frozenset({"operator", "fault"})
REST_POLL = 0.01  # how often the Gate asks the backend whether a stopped actuator is at rest


@dataclass(eq=False)
class Grant:
    """Ownership of some actuators; doubles as the capability token."""

    owner: Any  # Task
    resources: frozenset[str]
    revoked: bool = False
    released: bool = False  # given up while moving: kept, with nobody behind it, until at rest


@dataclass(eq=False)
class _Waiter:
    task: Any
    resources: frozenset[str]
    fut: asyncio.Future
    take: bool = False


@dataclass(frozen=True)
class SafetyEvent:
    """A safety event, for the trace: a lock, an unlock, a watchdog, a fallback, a reset."""

    t: float
    kind: str
    actuators: tuple[str, ...]
    cause: str
    detail: str = ""


@dataclass
class _Watch:
    grant: Grant
    deadline: float
    gen: int


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


class LocalGate:
    def __init__(
        self,
        backend: Any,
        clock: Clock,
        observe: Callable[[str], Any] | None = None,
        *,
        setpoint_timeout: float = 0.1,
    ) -> None:
        self._backend = backend
        self._clock = clock
        self._owner: dict[str, Grant] = {}
        self._queue: list[_Waiter] = []
        self.checks = Checks(observe or (lambda key: None), clock.now)
        self.actuators = frozenset(getattr(backend, "actuators", ()) or ())
        self.setpoint_timeout = setpoint_timeout
        self.locks: dict[str, set[str]] = {}  # actuator -> causes
        self.events: list[SafetyEvent] = []
        self._watch: dict[str, _Watch] = {}
        self._gen = 0
        self._stopped: dict[Any, set[str]] = {}  # task -> actuators stopped when revoked
        self._last: dict[str, Any] = {}  # actuator -> the task whose command it carries out
        self._settling: dict[Grant, asyncio.Future] = {}  # releases waiting for rest
        self._monitor_tasks: list[asyncio.Task] = []
        self.listeners: list[Callable[[SafetyEvent], None]] = []  # e.g. the trace
        # Checks and monitors in flight; the simulation drivers wait for them before stepping.
        self.activity = 0
        self._busy = 0
        self._quiet = asyncio.Event()
        self._quiet.set()

    # -- configuration -----------------------------------------------------

    def add_check(self, *args: Any, **kwargs: Any) -> Any:
        """Register a command check (SPEC §3.4); see ``Checks.add_check``."""
        return self.checks.add_check(*args, **kwargs)

    def add_monitor(self, *args: Any, **kwargs: Any) -> Monitor:
        """Register a monitor (SPEC §3.4); it runs at its rate on the Gate's clock."""
        monitor = self.checks.add_monitor(*args, **kwargs)
        self._monitor_tasks.append(asyncio.ensure_future(self._monitor_loop(monitor)))
        return monitor

    def close(self) -> None:
        for t in [*self._monitor_tasks, *self._settling.values()]:
            t.cancel()
        self.checks.close()

    # -- quiescence (for lockstep drivers) -----------------------------------

    @property
    def outstanding(self) -> int:
        return self._busy

    async def quiet(self) -> None:
        await self._quiet.wait()

    @contextlib.contextmanager
    def _working(self):
        self.activity += 1
        self._busy += 1
        self._quiet.clear()
        try:
            yield
        finally:
            self._busy -= 1
            if not self._busy:
                self._quiet.set()

    # -- ownership ---------------------------------------------------------

    def owner(self, actuator: str) -> Any:
        g = self._owner.get(actuator)
        return None if g is None else g.owner

    def held_by_ancestors(self, task: Task, resources: Iterable[str]) -> set[str]:
        """Resources that ``task`` may already use through itself or an ancestor."""
        held = set()
        for r in resources:
            g = self._owner.get(r)
            if g is not None and not g.revoked and _is_self_or_ancestor(g.owner, task):
                held.add(r)
        return held

    async def acquire(
        self,
        task: Task,
        resources: Iterable[str],
        *,
        on_busy: str = "wait",
        wait_timeout: float | None = None,
    ) -> Grant:
        """Grant ``resources`` to ``task`` (SPEC §3.3).

        Waiters are served first come, first served. ``on_busy="take"`` cancels the current
        owners and the takes already waiting for these actuators (reason ``preempted``), and
        is served first once the owners are terminal; it requires authority over all of
        them, and cancels none without it. So at most one take waits for an actuator, and
        ownership goes directly to it. ``on_busy="fail"`` fails at once when the resources
        are owned or waited for.
        Actuators locked by an operator stop or a fault cannot be acquired until ``reset``;
        actuators locked by a monitor are waited for, like owned ones, unless the monitor has
        an ``allow`` check: then they are granted, and ``admit`` asks it about each command.
        """
        resources = frozenset(resources)
        if on_busy not in ON_BUSY:
            raise ValueError(f"on_busy must be one of {sorted(ON_BUSY)}, got {on_busy!r}")
        manual = {r: self.locks[r] & MANUAL for r in resources if self.locks.get(r, set()) & MANUAL}
        if manual:
            raise ResourceBusy(f"locked until reset: {self._lock_message(manual)}")
        owners = self._owners_of(resources)
        stopping = sorted(r for r in resources if getattr(self._owner.get(r), "released", False))
        locked = [r for r in resources if self._hard_locks(r)]
        # A request still in the queue reserves its actuators, so a free actuator that an
        # earlier request waits for is busy too: `fail` never waits (SPEC §3.3).
        ahead = [e.task for e in self._queue if not e.fut.done() and e.resources & resources]
        if (owners or stopping or locked or ahead) and on_busy == "fail":
            msg = self._busy_message(resources, owners) if owners else ""
            if stopping:
                msg += ("; " if msg else "") + (
                    f"{stopping} still stopping after "
                    + ", ".join(sorted({self._owner[r].owner.name for r in stopping})))
            if locked:
                msg += ("; " if msg else "") + f"locked: {self._lock_message(self._locks_of(locked))}"
            if ahead:
                msg += ("; " if msg else f"{sorted(resources)}: ") + (
                    "waited for by " + ", ".join(sorted({t.name for t in ahead})))
            raise ResourceBusy(msg)
        if on_busy == "take":
            takers = [e for e in self._queue
                      if e.take and not e.fut.done() and e.resources & resources]
            # A take would cancel a take its own ancestor waits on, and with it itself.
            mine = [e.task for e in takers if _is_self_or_ancestor(e.task, task)]
            if mine:
                raise ResourceBusy(
                    f"{sorted(resources)} are being taken by {mine[0].name}, which started "
                    f"this task; wait for it instead of taking")
            denied = [o for o in owners if not may_take(task, o)]
            if denied:
                raise ResourceBusy(
                    self._busy_message(resources, denied)
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
                self._queue.remove(e)
                e.fut.cancel()
                self._cancel_owner(e.task, "preempted")
        entry = _Waiter(task, resources, asyncio.get_running_loop().create_future(),
                        take=on_busy == "take")
        if on_busy == "take":
            self._queue.insert(0, entry)
        else:
            self._queue.append(entry)
        self._dispatch()
        timer = None
        if wait_timeout is not None and not entry.fut.done():
            timer = self._clock.timer(self._clock.now() + wait_timeout)
            timer.add_done_callback(lambda f: None if f.cancelled() else self._expire(entry))
        try:
            return await entry.fut
        except asyncio.CancelledError:
            self._withdraw(entry)
            raise
        finally:
            if timer is not None and not timer.done():
                timer.cancel()

    def _owners_of(self, resources: frozenset[str]) -> list[Any]:
        """The tasks that own ``resources``. A grant given up while moving has none: the
        task behind it is done with the actuator, and a take must not cancel it."""
        seen: list[Any] = []
        for r in sorted(resources):
            g = self._owner.get(r)
            if g is not None and not g.released and g.owner not in seen:
                seen.append(g.owner)
        return seen

    def _monitor_of(self, cause: str) -> Monitor | None:
        if not cause.startswith("monitor:"):
            return None
        name = cause.removeprefix("monitor:")
        return next((m for m in self.checks.monitors if m.name == name), None)

    def _hard_locks(self, actuator: str) -> set[str]:
        """Lock causes that no command passes: all but monitors with an ``allow`` check."""
        return {c for c in self.locks.get(actuator, ())
                if (m := self._monitor_of(c)) is None or m.allow is None}

    def _locks_of(self, actuators: Iterable[str]) -> dict[str, set[str]]:
        return {a: set(self.locks[a]) for a in actuators if self.locks.get(a)}

    @staticmethod
    def _lock_message(locks: dict[str, set[str]]) -> str:
        return "; ".join(f"{a} ({', '.join(sorted(c))})" for a, c in sorted(locks.items()))

    @staticmethod
    def _busy_message(resources: frozenset[str], owners: Iterable[Any]) -> str:
        names = ", ".join(sorted({o.name for o in owners}))
        return f"{sorted(resources)} are owned by {names}"

    def _dispatch(self) -> None:
        """Serve waiters in order; a waiter never jumps ahead of an earlier one it overlaps."""
        reserved: set[str] = set()
        for entry in list(self._queue):
            if entry.fut.done():
                self._queue.remove(entry)
                continue
            free = all(r not in self._owner and not self._hard_locks(r) for r in entry.resources)
            if free and not (entry.resources & reserved):
                grant = Grant(entry.task, entry.resources)
                for r in entry.resources:
                    self._owner[r] = grant
                self._queue.remove(entry)
                entry.fut.set_result(grant)
            else:
                reserved |= entry.resources

    def _expire(self, entry: _Waiter) -> None:
        if entry.fut.done():
            return
        self._queue.remove(entry)
        owners = self._owners_of(entry.resources)
        entry.fut.set_exception(
            ResourceBusy(
                f"waited too long for {sorted(entry.resources)}"
                + (f"; owned by {', '.join(o.name for o in owners)}" if owners else "")
            )
        )
        self._dispatch()

    def _withdraw(self, entry: _Waiter) -> None:
        if entry in self._queue:
            self._queue.remove(entry)
        elif entry.fut.cancelled():
            pass  # a later take withdrew this request
        elif entry.fut.done() and entry.fut.exception() is None:
            self.release(entry.fut.result())  # granted just as the waiter was cancelled
        self._dispatch()

    def at_rest(self, actuators: Iterable[str]) -> bool:
        """Whether the backend reports every one of ``actuators`` at rest. A backend
        without ``at_rest`` cannot say, and counts as at rest."""
        probe = getattr(self._backend, "at_rest", None)
        return probe is None or all(probe(a) for a in actuators)

    async def settled(self, task: Task, timeout: float) -> bool:
        """Wait until the actuators stopped when ``task``'s token was revoked are at rest
        (SPEC §1.1: ``cancelled`` means the robot is stopped). False after ``timeout``."""
        actuators = self._stopped.get(task, ())
        deadline = self._clock.now() + timeout
        while not self.at_rest(actuators):
            if self._clock.now() >= deadline:
                return False
            await self._clock.sleep(REST_POLL)
        self._stopped.pop(task, None)
        return True

    def release(self, grant: Grant) -> None:
        """Give up ``grant``. Its actuators pass on only at rest, so whatever moves has an
        owner and the next owner never receives a moving actuator: a revoked grant waits
        for them to stop, and one given up while they move (a task that ended, or left its
        `hold`, without waiting for a goal) has them stopped by the Gate first (SPEC §3.3)."""
        if self._owner_of_any(grant) and not self.at_rest(grant.resources):
            if not grant.revoked:
                grant.revoked = grant.released = True
                moving = tuple(sorted(a for a in grant.resources if not self.at_rest([a])))
                for a in moving:
                    self._unwatch(a)
                    self._last.pop(a, None)
                    with contextlib.suppress(Exception):
                        self._execute(Command("stop", a, {}))
                self._event("fallback", moving, "released", f"left moving by {grant.owner.name}")
            if grant not in self._settling:
                self._settling[grant] = asyncio.ensure_future(self._release_at_rest(grant))
            return
        self._settling.pop(grant, None)
        stopped = self._stopped.get(grant.owner)
        if stopped is not None:
            stopped -= grant.resources
            if not stopped:
                del self._stopped[grant.owner]
        for r in grant.resources:
            if self._owner.get(r) is grant:
                del self._owner[r]
            w = self._watch.get(r)
            if w is not None and w.grant is grant:
                del self._watch[r]
        self._dispatch()

    def _owner_of_any(self, grant: Grant) -> bool:
        return any(self._owner.get(r) is grant for r in grant.resources)

    async def _release_at_rest(self, grant: Grant) -> None:
        while not self.at_rest(grant.resources):
            await self._clock.sleep(REST_POLL)
        self.release(grant)

    def release_all(self, task: Task) -> None:
        self._stopped.pop(task, None)
        for g in {g for g in self._owner.values() if g.owner is task}:
            self.release(g)

    def revoke(self, task: Task) -> None:
        """``task`` is cancelled: revoke its tokens. A task that used an ancestor's token
        has none to revoke; the Gate stops the motion it commanded, and ``send`` refuses
        what it sends from now on."""
        for g in {g for g in self._owner.values() if g.owner is task}:
            self.revoke_grant(g)
        for a in [a for a, t in self._last.items() if t is task]:
            del self._last[a]
            g = self._owner.get(a)
            if g is not None and g.owner is not task:
                self._stopped.setdefault(task, set()).add(a)
                self._unwatch(a)
                with contextlib.suppress(Exception):
                    self._execute(Command("stop", a, {}))

    def revoke_grant(self, grant: Grant) -> None:
        """Revoke a token and stop its actuators. The owner can no longer command them, so
        the Gate stops them itself instead of relying on the owner's cleanup (SPEC §1.1)."""
        if grant.revoked:
            return
        grant.revoked = True
        self._stopped.setdefault(grant.owner, set()).update(grant.resources)
        for a in sorted(grant.resources):
            self._unwatch(a)
            with contextlib.suppress(Exception):
                self._execute(Command("stop", a, {}))

    def _cancel_owner(self, owner: Any, reason: str, detail: dict | None = None) -> None:
        """A cancellation that starts in the Gate: revoke first, then cancel (SPEC §3.1)."""
        self.revoke(owner)
        if detail is not None:
            owner.detail = detail
        owner.cancel(reason)

    # -- commands ----------------------------------------------------------

    async def send(self, task: Task | None, cmd: Command, verdicts: list | None = None) -> bool:
        """Forward ``cmd`` if ``task`` may issue it and it passes the Gate.

        Returns False if rejected; the reason is left in ``task.detail`` (SPEC §3.8).
        """
        if cmd.kind not in KINDS:
            raise ValueError(f"unknown command kind {cmd.kind!r}; expected one of {sorted(KINDS)}")
        grant = None
        if cmd.kind != "stop":
            g = self._owner.get(cmd.actuator)
            if (g is not None and task is not None and _is_self_or_ancestor(g.owner, task)
                    and not cancelled_below(task, g.owner)):
                grant = g
        rejection = await self.admit(grant, cmd, verdicts)
        if rejection is None:
            if cmd.kind == "stop":
                self._last.pop(cmd.actuator, None)
            else:
                self._last[cmd.actuator] = task
            return True
        if task is not None:
            task.detail = {"code": rejection.code, "message": rejection.message}
        return False

    async def admit(
        self, grant: Grant | None, cmd: Command, verdicts: list | None = None
    ) -> Rejection | None:
        """Everything between a command and the backend, for both Gate layouts.

        ``stop`` is always accepted. Otherwise: a valid grant, the actuator not locked, every
        check accepts, then the backend. A command that a check rejects runs the fallback
        for its actuator (SPEC §3.5); an unauthorized one does not, so a task cannot stop
        someone else's motion by sending commands it may not send.
        """
        a = cmd.actuator
        if cmd.kind == "stop":
            self._unwatch(a)
            self._execute(cmd)
            return None
        if grant is None or grant.revoked or a not in grant.resources:
            return Rejection("unauthorized", f"no valid token for {a!r}")
        if self._hard_locks(a):
            return Rejection("locked", f"{a!r} is locked: {', '.join(sorted(self.locks[a]))}")
        passed = set(self.locks.get(a, ()))  # monitor locks with an allow check
        for cause in sorted(passed):
            monitor = self._monitor_of(cause)
            assert monitor is not None and monitor.allow is not None
            with self._working():
                rejection = await self.checks.run_allow(monitor, cmd, verdicts)
            if rejection is not None:
                return Rejection("locked", f"{a!r} is stopped by monitor {monitor.name} "
                                           f"({monitor.reason}); {rejection.message}")
        with self._working():
            rejection = await self.checks.run(cmd, verdicts)
        if rejection is not None:
            self._fallback([a], rejection.code)
            return rejection
        # Checks take time. Meanwhile the token may have been revoked, the actuator locked,
        # or the grant released (its task left a `hold` with this command still in flight)
        # and the actuator given to someone else.
        if grant.revoked or set(self.locks.get(a, ())) - passed:
            return Rejection("unauthorized", f"token for {a!r} was revoked")
        if self._owner.get(a) is not grant:
            return Rejection("unauthorized", f"token for {a!r} was released")
        try:
            self._execute(cmd)
        except BackendError as e:
            if e.code == "fault":
                self.lock([a], "fault", e.message)
                self._cancel_owner(grant.owner, "rejected", {"code": "fault", "message": e.message})
                self._reject_waiters({a}, "fault")
            raise
        if cmd.kind == "setpoint":
            self._arm_watchdog(a, grant)
        elif cmd.kind == "goal":
            self._unwatch(a)
        return None

    def _execute(self, cmd: Command) -> None:
        self._backend.execute(cmd)

    # -- fallback, locks, reset (SPEC §3.5) -----------------------------------

    def _fallback(self, actuators: Iterable[str], cause: str) -> None:
        """The pre-verified behavior: stop. ``stop`` is never checked, so this cannot fail
        because of a check."""
        actuators = tuple(sorted(actuators))
        for a in actuators:
            self._unwatch(a)
            with contextlib.suppress(Exception):
                self._execute(Command("stop", a, {}))
        self._event("fallback", actuators, cause)

    def lock(self, actuators: Iterable[str], cause: str, detail: str = "") -> None:
        actuators = tuple(sorted(actuators))
        for a in actuators:
            self.locks.setdefault(a, set()).add(cause)
        self._event("lock", actuators, cause, detail)

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
            self._event(kind, tuple(freed), ",".join(sorted(causes)))
            self._dispatch()

    def safety_stop(
        self, actuators: Iterable[str], cause: str, reason: str, detail: dict[str, str]
    ) -> None:
        """Lock ``actuators``, cancel their owners (revoking first), and run the fallback."""
        actuators = frozenset(actuators)
        self.lock(actuators, cause, detail.get("message", ""))
        for owner in self._owners_of(actuators):
            self._cancel_owner(owner, reason, dict(detail))
        if cause in MANUAL:
            self._reject_waiters(actuators, cause)
        self._fallback(actuators, cause)

    def _reject_waiters(self, actuators: Iterable[str], cause: str) -> None:
        """Nothing waits for actuators that are locked until ``reset`` (SPEC §3.5)."""
        actuators = frozenset(actuators)
        for entry in [e for e in self._queue if e.resources & actuators]:
            if not entry.fut.done():
                entry.fut.set_exception(ResourceBusy(f"locked until reset: {cause}"))

    def operator_stop(self, actuators: Iterable[str] | None = None) -> None:
        """An operator stop (SPEC §3.5): only the named actuators; unlocks only on ``reset``.

        Tasks that hold no actuators keep running.
        """
        actuators = frozenset(actuators) if actuators is not None else self.actuators
        self.safety_stop(actuators, "operator", "operator",
                         {"code": "operator", "message": "operator stop"})

    def reset(self, actuators: Iterable[str] | None = None) -> None:
        """Clear operator and fault locks; ask the backend to recover from a fault."""
        actuators = frozenset(actuators) if actuators is not None else frozenset(self.locks)
        faulted = [a for a in actuators if "fault" in self.locks.get(a, ())]
        if faulted and hasattr(self._backend, "recover"):
            self._backend.recover(faulted)
        self._event("reset", tuple(sorted(actuators)), "reset")
        self._unlock(actuators, MANUAL, "unlock")

    def sim(self, action: str, **kwargs: Any) -> dict:
        """Change the simulated world (not the robot), e.g. push an object. Not a command: it
        needs no token and passes no check, and only a simulator backend supports it."""
        if not hasattr(self._backend, "sim"):
            raise ValueError("this backend is not a simulator")
        return self._backend.sim(action, **kwargs)

    def _event(self, kind: str, actuators: tuple[str, ...], cause: str, detail: str = "") -> None:
        ev = SafetyEvent(self._clock.now(), kind, actuators, cause, detail)
        self.events.append(ev)
        for listener in self.listeners:
            listener(ev)

    # -- monitors ------------------------------------------------------------

    async def _monitor_loop(self, monitor: Monitor) -> None:
        period = 1.0 / monitor.rate
        while True:
            await self._clock.sleep(period)
            with self._working():
                rejection = await self.checks.run_monitor(monitor)
            self._monitor_result(monitor, rejection)

    def _monitor_result(self, monitor: Monitor, rejection: Rejection | None) -> None:
        actuators = monitor.actuators if monitor.actuators is not None else self.actuators
        cause = f"monitor:{monitor.name}"
        if rejection is not None:
            monitor.reason = rejection.message
            if not monitor.failing:
                monitor.failing = True
                self.safety_stop(actuators, cause, "rejected",
                                 {"code": rejection.code, "message": rejection.message})
            return
        if monitor.failing:
            monitor.failing, monitor.reason = False, ""
            self._unlock(actuators, {cause}, "unlock")  # automatic (SPEC §3.5)

    # -- watchdog ------------------------------------------------------------

    def _arm_watchdog(self, actuator: str, grant: Grant) -> None:
        self._gen += 1
        w = _Watch(grant, self._clock.now() + self.setpoint_timeout, self._gen)
        self._watch[actuator] = w
        timer = self._clock.timer(w.deadline)
        timer.add_done_callback(lambda f: None if f.cancelled() else self._expire_watch(actuator, w))

    def _unwatch(self, actuator: str) -> None:
        self._watch.pop(actuator, None)

    def _expire_watch(self, actuator: str, w: _Watch) -> None:
        if self._watch.get(actuator) is not w:
            return  # a newer setpoint, a goal, or a stop came in time
        del self._watch[actuator]
        if w.grant.revoked or self._owner.get(actuator) is not w.grant:
            return
        message = f"no setpoint for {actuator!r} within {self.setpoint_timeout}s"
        self._event("watchdog", (actuator,), "watchdog", message)
        self._cancel_owner(w.grant.owner, "watchdog", {"code": "watchdog", "message": message})
        self._fallback([actuator], "watchdog")
