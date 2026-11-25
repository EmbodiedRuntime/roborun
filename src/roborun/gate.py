"""The Gate (SPEC §3): control authority and safety over actuators.

The rules are in ``roborun.core.GateCore``, a state machine with no clock and no I/O:
ownership, arbitration (first come, first served; ``take`` with authority), what happens to
a command before and after its checks, the fallback, and the locked state. ``LocalGate`` is
the shell around it. It turns calls, timers and the answers of checks and monitors
(``roborun.checks``) into the core's events, and carries out what the core decides: commands
to the backend, answers to waiting requests, cancellations, safety events. It runs
in-process for tests and ``run_sim``, and inside the Gate process (``roborun.gateproc``),
which adds capability tokens and a socket.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from roborun import core as _core
from roborun.checks import Checks, Command, Monitor, Rejection
from roborun.clock import Clock
from roborun.core import (  # noqa: F401 - part of this module's interface
    KINDS,
    MANUAL,
    ON_BUSY,
    GateCore,
    Grant,
    _is_self_or_ancestor,
    cancelled_below,
    ending,
    may_take,
)
from roborun.errors import BackendError, ResourceBusy
from roborun.journal import Journal

if TYPE_CHECKING:
    from roborun.task import Task

__all__ = ["KINDS", "ON_BUSY", "Command", "Grant", "LocalGate", "may_take"]

REST_POLL = 0.01  # how often the Gate asks the backend whether a stopped actuator is at rest


@dataclass(frozen=True)
class SafetyEvent:
    """A safety event, for the trace: a lock, an unlock, a watchdog, a fallback, a reset."""

    t: float
    kind: str
    actuators: tuple[str, ...]
    cause: str
    detail: str = ""


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
        self.checks = Checks(observe or (lambda key: None), clock.now)
        self.actuators = frozenset(getattr(backend, "actuators", ()) or ())
        self.setpoint_timeout = setpoint_timeout
        self.core = GateCore(self.actuators, setpoint_timeout=setpoint_timeout)
        self.events: list[SafetyEvent] = []
        self._futs: dict[_core.Waiter, asyncio.Future] = {}  # requests waiting for an answer
        self._settling: dict[Grant, asyncio.Future] = {}  # releases waiting for rest
        self._monitor_tasks: list[asyncio.Task] = []
        self.listeners: list[Callable[[SafetyEvent], None]] = []  # e.g. the trace
        # Checks and monitors in flight; the simulation drivers wait for them before stepping.
        self.activity = 0
        self._busy = 0
        self._quiet = asyncio.Event()
        self._quiet.set()

    # -- the core's state, as the Gate shows it ----------------------------------

    @property
    def locks(self) -> dict[str, set[str]]:
        return self.core.locks

    @property
    def _owner(self) -> dict[str, Grant]:
        return self.core.owner

    @property
    def _queue(self) -> list[_core.Waiter]:
        return self.core.queue

    @property
    def _watch(self) -> dict[str, _core.Watch]:
        return self.core.watch

    def record(self) -> Journal:
        """Record every event of the core from now on (``roborun.journal``)."""
        self.core.journal = journal = Journal(
            lambda: {a: self.at_rest([a]) for a in sorted(self.actuators)})
        return journal

    # -- carrying out what the core decided ----------------------------------------

    def _perform(self) -> None:
        """Carry out the core's effects, in order. Cancelling a task comes back as events
        (its children's tokens are revoked); their effects join the same queue."""
        effects = self.core.effects
        while effects:
            e = effects.pop(0)
            if isinstance(e, _core.Stop):
                with contextlib.suppress(Exception):
                    self._execute(Command("stop", e.actuator, {}))
            elif isinstance(e, _core.Cancel):
                if e.detail is not None:
                    e.task.detail = e.detail
                e.task.cancel(e.reason)
            elif isinstance(e, _core.Granted):
                fut = self._futs[e.waiter]
                if fut.done():  # its caller went away a moment ago: nobody would release it
                    self.core.release(e.grant, self._moving(e.grant.resources))
                else:
                    fut.set_result(e.grant)
            elif isinstance(e, _core.Refused):
                fut = self._futs[e.waiter]
                if not fut.done():
                    fut.set_exception(ResourceBusy(e.message))
            elif isinstance(e, _core.Withdrawn):
                self._futs[e.waiter].cancel()
            elif isinstance(e, _core.Safety):
                ev = SafetyEvent(self._clock.now(), e.kind, e.actuators, e.cause, e.detail)
                self.events.append(ev)
                for listener in self.listeners:
                    listener(ev)
            elif isinstance(e, _core.Arm):
                timer = self._clock.timer(self._clock.now() + self.setpoint_timeout)
                timer.add_done_callback(
                    lambda f, e=e: None if f.cancelled() else self._expire_watch(e.actuator, e.watch))
            elif isinstance(e, _core.Settle):
                self._settling[e.grant] = asyncio.ensure_future(self._release_at_rest(e.grant))

    def _moving(self, actuators: Iterable[str]) -> list[str]:
        return [a for a in actuators if not self.at_rest([a])]

    # -- configuration -----------------------------------------------------

    def add_check(self, *args: Any, **kwargs: Any) -> Any:
        """Register a command check (SPEC §3.4); see ``Checks.add_check``."""
        return self.checks.add_check(*args, **kwargs)

    def add_monitor(self, *args: Any, **kwargs: Any) -> Monitor:
        """Register a monitor (SPEC §3.4); it runs at its rate on the Gate's clock."""
        monitor = self.checks.add_monitor(*args, **kwargs)
        if monitor.allow is not None:
            self.core.soft.add(f"monitor:{monitor.name}")
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
        g = self.core.owner.get(actuator)
        return None if g is None else g.owner

    def held_by_ancestors(self, task: Task, resources: Iterable[str]) -> set[str]:
        """Resources that ``task`` may already use through itself or an ancestor."""
        return self.core.held_by_ancestors(task, resources)

    async def acquire(
        self,
        task: Task,
        resources: Iterable[str],
        *,
        on_busy: str = "wait",
        wait_timeout: float | None = None,
    ) -> Grant:
        """Grant ``resources`` to ``task`` (SPEC §3.3); the rules are in
        ``GateCore.request``. Raises ``ResourceBusy`` when the request fails at once, when
        ``wait_timeout`` passes, or when the actuators are locked until ``reset``."""
        waiter = self.core.request(task, resources, on_busy)
        fut = self._futs[waiter] = asyncio.get_running_loop().create_future()
        self._perform()
        timer = None
        if wait_timeout is not None and not fut.done():
            timer = self._clock.timer(self._clock.now() + wait_timeout)
            timer.add_done_callback(lambda f: None if f.cancelled() else self._expire(waiter))
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self.release(fut.result())  # granted just as the waiter was cancelled
            else:
                self.core.withdraw(waiter)
                self._perform()
            raise
        finally:
            self._futs.pop(waiter, None)
            if timer is not None and not timer.done():
                timer.cancel()

    def _expire(self, waiter: _core.Waiter) -> None:
        self.core.expire(waiter)
        self._perform()

    def at_rest(self, actuators: Iterable[str]) -> bool:
        """Whether the backend reports every one of ``actuators`` at rest. A backend
        without ``at_rest`` cannot say, and counts as at rest."""
        probe = getattr(self._backend, "at_rest", None)
        return probe is None or all(probe(a) for a in actuators)

    async def settled(self, task: Task, timeout: float) -> bool:
        """Wait until the actuators stopped when ``task``'s token was revoked are at rest
        (SPEC §1.1: ``cancelled`` means the robot is stopped). False after ``timeout``."""
        deadline = self._clock.now() + timeout
        while not self.at_rest(self.core.stopped.get(task, ())):
            if self._clock.now() >= deadline:
                return False
            await self._clock.sleep(REST_POLL)
        self.core.rested(task)
        return True

    def release(self, grant: Grant) -> None:
        """Give up ``grant``. Its actuators pass on only at rest (SPEC §3.3): see
        ``GateCore.release``."""
        self.core.release(grant, self._moving(grant.resources))
        if grant not in self.core.settling:
            self._settling.pop(grant, None)
        self._perform()

    async def _release_at_rest(self, grant: Grant) -> None:
        while not self.at_rest(grant.resources):
            await self._clock.sleep(REST_POLL)
        self.release(grant)

    def release_all(self, task: Task) -> None:
        grants = self.core.grants_of(task)
        self.core.release_all(task, self._moving(a for g in grants for a in g.resources))
        for g in grants:
            if g not in self.core.settling:
                self._settling.pop(g, None)
        self._perform()

    def revoke(self, task: Task) -> None:
        """``task`` is cancelled: revoke its tokens and stop what it commanded."""
        self.core.revoke(task)
        self._perform()

    def revoke_grant(self, grant: Grant) -> None:
        """Revoke a token and stop its actuators (SPEC §1.1)."""
        self.core.revoke_grant(grant)
        self._perform()

    # -- commands ----------------------------------------------------------

    async def send(self, task: Task | None, cmd: Command, verdicts: list | None = None) -> bool:
        """Forward ``cmd`` if ``task`` may issue it and it passes the Gate.

        Returns False if rejected; the reason is left in ``task.detail`` (SPEC §3.8).
        """
        if cmd.kind not in KINDS:
            raise ValueError(f"unknown command kind {cmd.kind!r}; expected one of {sorted(KINDS)}")
        grant = None if cmd.kind == "stop" else self.core.authority(task, cmd.actuator)
        rejection = await self.admit(grant, cmd, verdicts, task=task)
        if rejection is None:
            return True
        if task is not None:
            task.detail = {"code": rejection.code, "message": rejection.message}
        return False

    async def admit(
        self, grant: Grant | None, cmd: Command, verdicts: list | None = None, *, task: Any = None
    ) -> Rejection | None:
        """Everything between a command and the backend, for both Gate layouts.

        ``stop`` is always accepted. Otherwise: a valid grant, the actuator not locked, every
        check accepts, then the backend. A command that a check rejects runs the fallback
        for its actuator (SPEC §3.5); an unauthorized one does not, so a task cannot stop
        someone else's motion by sending commands it may not send. The decisions are the
        core's (``command``, ``checked``, ``executed``, ``fault``); the checks run here.
        """
        a = cmd.actuator
        p = self.core.command(grant, cmd, task)
        if p is None:
            self._execute(cmd)
            return None
        if isinstance(p, Rejection):
            return p
        try:
            for cause in sorted(p.passed):  # monitor locks with an allow check
                monitor = self._monitor_of(cause)
                assert monitor is not None and monitor.allow is not None
                with self._working():
                    rejection = await self.checks.run_allow(monitor, cmd, verdicts)
                if rejection is not None:
                    return Rejection("locked", f"{a!r} is stopped by monitor {monitor.name} "
                                               f"({monitor.reason}); {rejection.message}")
            with self._working():
                rejection = await self.checks.run(cmd, verdicts)
            verdict = self.core.checked(p, rejection)
            self._perform()
            if verdict is not None:
                return verdict
            try:
                self._execute(cmd)
            except BackendError as e:
                if e.code == "fault":
                    self.core.fault(p, e.message)
                    self._perform()
                raise
            self.core.executed(p)
            self._perform()
            return None
        finally:
            self.core.dropped(p)

    def _execute(self, cmd: Command) -> None:
        self._backend.execute(cmd)

    def _monitor_of(self, cause: str) -> Monitor | None:
        if not cause.startswith("monitor:"):
            return None
        name = cause.removeprefix("monitor:")
        return next((m for m in self.checks.monitors if m.name == name), None)

    # -- locks, reset (SPEC §3.5) ---------------------------------------------

    def lock(self, actuators: Iterable[str], cause: str, detail: str = "") -> None:
        self.core.lock(actuators, cause, detail)
        self._perform()

    def _fallback(self, actuators: Iterable[str], cause: str) -> None:
        """The pre-verified behavior: stop (SPEC §3.5)."""
        self.core.fallback(actuators, cause)
        self._perform()

    def safety_stop(
        self, actuators: Iterable[str], cause: str, reason: str, detail: dict[str, str]
    ) -> None:
        """Lock ``actuators``, cancel their owners (revoking first), and run the fallback."""
        self.core.safety_stop(actuators, cause, reason, detail)
        self._perform()

    def operator_stop(self, actuators: Iterable[str] | None = None) -> None:
        """An operator stop (SPEC §3.5): only the named actuators; unlocks only on ``reset``.

        Tasks that hold no actuators keep running.
        """
        self.core.operator_stop(actuators)
        self._perform()

    def reset(self, actuators: Iterable[str] | None = None) -> None:
        """Clear operator and fault locks; ask the backend to recover from a fault."""
        chosen = frozenset(actuators) if actuators is not None else frozenset(self.core.locks)
        faulted = [a for a in chosen if "fault" in self.core.locks.get(a, ())]
        if faulted and hasattr(self._backend, "recover"):
            self._backend.recover(faulted)
        self.core.reset(chosen)
        self._perform()

    def sim(self, action: str, **kwargs: Any) -> dict:
        """Change the simulated world (not the robot), e.g. push an object. Not a command: it
        needs no token and passes no check, and only a simulator backend supports it."""
        if not hasattr(self._backend, "sim"):
            raise ValueError("this backend is not a simulator")
        return self._backend.sim(action, **kwargs)

    # -- monitors ------------------------------------------------------------

    async def _monitor_loop(self, monitor: Monitor) -> None:
        period = 1.0 / monitor.rate
        while True:
            await self._clock.sleep(period)
            with self._working():
                rejection = await self.checks.run_monitor(monitor)
            self._monitor_result(monitor, rejection)

    def _monitor_result(self, monitor: Monitor, rejection: Rejection | None) -> None:
        if rejection is not None:
            monitor.reason = rejection.message
        self.core.monitor(monitor.name, monitor.actuators, rejection)
        monitor.failing = monitor.name in self.core.failing
        if not monitor.failing:
            monitor.reason = ""
        self._perform()

    # -- watchdog ------------------------------------------------------------

    def _expire_watch(self, actuator: str, w: _core.Watch) -> None:
        self.core.watch_expired(actuator, w)
        self._perform()
