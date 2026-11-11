"""Gate checks (SPEC §3.4): the Gate calls them; the integrator decides what they check.

A check is a function ``fn(cmd, view) -> Accept() | Reject(reason)``, registered with the
command kinds it applies to, the inputs it reads (each with a ``max_age``), and a budget.
The Gate guarantees, whatever the function does:

1. every applicable check runs before a command reaches the backend;
2. fail closed: a check that exceeds its budget, raises, or returns anything else rejects;
3. freshness first: if a declared input is missing or too old, the check is not called;
4. checks run in worker threads, so a slow or hung check cannot stall the Gate's loop;
5. checks only judge: they get a read-only view and return a verdict;
6. every verdict is recorded.

``stop`` is never checked. The bottom of this module has library checks: examples of policy,
not part of the Gate.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from roborun.stamped import Stamped


@dataclass(frozen=True)
class Command:
    """A structured command from the fixed vocabulary (SPEC §3.2)."""

    kind: str
    actuator: str
    payload: dict[str, Any] = field(default_factory=dict)


_EPS = 1e-9


@dataclass(frozen=True)
class Accept:
    pass


@dataclass(frozen=True)
class Reject:
    reason: str


Verdict = Accept | Reject


class View(Mapping[str, Stamped]):
    """A read-only snapshot of the stamped inputs a check declared."""

    def __init__(self, items: dict[str, Stamped], now: float) -> None:
        self._items, self.now = dict(items), now

    def __getitem__(self, key: str) -> Stamped:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def value(self, key: str) -> Any:
        return self._items[key].value


@dataclass
class Check:
    fn: Callable[[Command, View], Verdict]
    name: str
    kinds: frozenset[str] = frozenset({"goal", "setpoint", "effector"})
    inputs: dict[str, float] = field(default_factory=dict)  # key -> max_age
    # Seconds of wall time. Python checks share the interpreter with whatever else runs in the
    # Gate's process (a simulator, other threads), so budgets of a few ms time out under load.
    budget: float = 0.05


@dataclass
class Monitor:
    """A check over state, run at ``rate`` Hz whatever is being commanded (SPEC §3.4).

    ``actuators`` are the ones it protects; None means every actuator of the backend.

    While it rejects, its actuators stay locked. ``allow``, if given, is a check over the same
    inputs that may let single commands through the lock, typically ones that move away from
    what the monitor guards (a retreat); every other check still applies to them. Without
    ``allow`` nothing moves until the monitor accepts again.
    """

    fn: Callable[[View], Verdict]
    name: str
    inputs: dict[str, float] = field(default_factory=dict)
    rate: float = 50.0
    budget: float = 0.02
    actuators: frozenset[str] | None = None
    allow: Callable[[Command, View], Verdict] | None = None
    failing: bool = False
    reason: str = ""  # the last rejection, while failing


@dataclass(frozen=True)
class Rejection:
    """Why the Gate rejected a command; ``code`` follows SPEC §3.8."""

    code: str
    message: str


@dataclass(frozen=True)
class Record:
    t: float  # time base of the runtime
    check: str
    kind: str
    actuator: str
    verdict: str  # accept, reject, stale, timeout, error
    reason: str
    took: float  # wall seconds


class Checks:
    """The check registry and runner shared by the in-process and process Gates."""

    def __init__(
        self,
        observe: Callable[[str], Stamped | None],
        now: Callable[[], float],
        *,
        workers: int = 4,
    ) -> None:
        self._observe, self._now = observe, now
        self.checks: list[Check] = []
        self.monitors: list[Monitor] = []
        self.log: list[Record] = []
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="roborun-check")

    def add_check(
        self,
        check: Check | Callable[[Command, View], Verdict],
        *,
        name: str | None = None,
        kinds: Iterable[str] | None = None,
        inputs: dict[str, float] | None = None,
        budget: float | None = None,
    ) -> Check:
        if not isinstance(check, Check):
            check = Check(check, name or getattr(check, "__name__", "check"))
        if name is not None:
            check.name = name
        if kinds is not None:
            check.kinds = frozenset(kinds)
        if inputs is not None:
            check.inputs = dict(inputs)
        if budget is not None:
            check.budget = budget
        if "stop" in check.kinds:
            raise ValueError("stop is always accepted (SPEC §3.2); a check cannot apply to it")
        if any(c.name == check.name for c in self.checks):
            raise ValueError(f"a check named {check.name!r} is already registered")
        self.checks.append(check)
        return check

    def add_monitor(
        self,
        monitor: Monitor | Callable[[View], Verdict],
        *,
        name: str | None = None,
        inputs: dict[str, float] | None = None,
        rate: float | None = None,
        budget: float | None = None,
        actuators: Iterable[str] | None = None,
        allow: Callable[[Command, View], Verdict] | None = None,
    ) -> Monitor:
        if not isinstance(monitor, Monitor):
            monitor = Monitor(monitor, name or getattr(monitor, "__name__", "monitor"))
        if name is not None:
            monitor.name = name
        if inputs is not None:
            monitor.inputs = dict(inputs)
        if rate is not None:
            monitor.rate = rate
        if budget is not None:
            monitor.budget = budget
        if actuators is not None:
            monitor.actuators = frozenset(actuators)
        if allow is not None:
            monitor.allow = allow
        if monitor.rate <= 0:
            raise ValueError("a monitor needs a positive rate")
        if any(m.name == monitor.name for m in self.monitors):
            raise ValueError(f"a monitor named {monitor.name!r} is already registered")
        self.monitors.append(monitor)
        return monitor

    async def run_monitor(self, monitor: Monitor) -> Rejection | None:
        """Run one monitor once; its failures follow the same fail-closed rules as checks."""
        probe = Command("monitor", "*", {})
        return await self._run_one(
            Check(lambda _cmd, view: monitor.fn(view), monitor.name, frozenset(),
                  monitor.inputs, monitor.budget),
            probe,
            prefix="monitor",
        )

    async def run_allow(
        self, monitor: Monitor, cmd: Command, verdicts: list | None = None
    ) -> Rejection | None:
        """Ask a failing monitor's ``allow`` whether ``cmd`` may pass its lock (fail closed)."""
        assert monitor.allow is not None
        rejection = await self._run_one(
            Check(monitor.allow, f"{monitor.name}.allow", frozenset({cmd.kind}), monitor.inputs,
                  monitor.budget),
            cmd,
            prefix="allow",
        )
        if verdicts is not None:
            r = self.log[-1]
            verdicts.append({"check": r.check, "verdict": r.verdict, "reason": r.reason,
                             "took": round(r.took, 6)})
        return rejection

    async def run(self, cmd: Command, verdicts: list | None = None) -> Rejection | None:
        """Run every applicable check in order; return the first rejection, or None.

        Each verdict is also appended to ``verdicts``, if given, for the trace.
        """
        if cmd.kind == "stop":
            return None
        for check in self.checks:
            if cmd.kind not in check.kinds:
                continue
            rejection = await self._run_one(check, cmd)
            if verdicts is not None:
                r = self.log[-1]
                verdicts.append({"check": r.check, "verdict": r.verdict, "reason": r.reason,
                                 "took": round(r.took, 6)})
            if rejection is not None:
                return rejection
        return None

    async def _run_one(
        self, check: Check, cmd: Command, *, prefix: str = "check"
    ) -> Rejection | None:
        now = self._now()
        items: dict[str, Stamped] = {}
        for key, max_age in check.inputs.items():
            s = self._observe(key)
            if s is None or s.age(now) > max_age + _EPS:
                age = "missing" if s is None else f"{s.age(now):.3f}s old (limit {max_age}s)"
                self._record(check, cmd, "stale", f"{key} {age}", 0.0)
                return Rejection(f"stale:{key}", f"{check.name}: input {key!r} is {age}")
            items[key] = s
        view = View(items, now)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        try:
            verdict = await asyncio.wait_for(
                loop.run_in_executor(self._pool, check.fn, cmd, view), check.budget
            )
        except asyncio.TimeoutError:
            took = time.perf_counter() - t0
            self._record(check, cmd, "timeout", f"over budget {check.budget}s", took)
            return Rejection(
                f"{prefix}_timeout:{check.name}", f"{check.name} took longer than {check.budget}s"
            )
        except Exception as e:  # noqa: BLE001 - fail closed on any error in a check
            took = time.perf_counter() - t0
            self._record(check, cmd, "error", repr(e), took)
            return Rejection(f"{prefix}_error:{check.name}", f"{check.name} raised {e!r}")
        took = time.perf_counter() - t0
        if isinstance(verdict, Accept):
            self._record(check, cmd, "accept", "", took)
            return None
        if isinstance(verdict, Reject):
            self._record(check, cmd, "reject", verdict.reason, took)
            return Rejection(f"{prefix}:{check.name}", verdict.reason)
        self._record(check, cmd, "error", f"returned {verdict!r}", took)
        return Rejection(
            f"{prefix}_error:{check.name}",
            f"{check.name} returned {verdict!r}, not Accept or Reject",
        )

    def _record(self, check: Check, cmd: Command, verdict: str, reason: str, took: float) -> None:
        self.log.append(
            Record(self._now(), check.name, cmd.kind, cmd.actuator, verdict, reason, took)
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# -- library checks (SPEC Appendix A): examples of policy, not part of the Gate ------------


def workspace_box(lo: Iterable[float], hi: Iterable[float], *, name: str = "workspace") -> Check:
    """Reject ``goal`` and ``setpoint`` targets outside an axis-aligned box.

    Use it to keep the tool above a table: a box whose lower z is the table height plus a
    margin rejects any target that would drive the tool into it.
    """
    lo, hi = tuple(float(v) for v in lo), tuple(float(v) for v in hi)

    def check(cmd: Command, view: View) -> Verdict:
        target = tuple(cmd.payload["target"])
        if len(target) != len(lo):
            return Reject(f"target {target} has {len(target)} coordinates; the box has {len(lo)}")
        for axis, (v, a, b) in enumerate(zip(target, lo, hi)):
            if not a <= v <= b:
                return Reject(
                    f"target {tuple(round(x, 3) for x in target)} is outside the workspace: "
                    f"axis {'xyz'[axis] if len(lo) <= 3 else axis} = {v:.3f}, allowed [{a}, {b}]"
                )
        return Accept()

    return Check(check, name, kinds=frozenset({"goal", "setpoint"}), budget=0.05)


def max_jump(limit: float, *, key: str = "arm.ee", max_age: float = 0.05,
             name: str = "max_jump") -> Check:
    """Reject a ``setpoint`` farther than ``limit`` from the measured position in ``key``.

    A setpoint stream should move in small steps; a large jump usually means a bug.
    """

    def check(cmd: Command, view: View) -> Verdict:
        here = view.value(key)
        target = tuple(cmd.payload["target"])
        d = sum((a - b) ** 2 for a, b in zip(here, target)) ** 0.5
        if d > limit:
            return Reject(f"setpoint is {d:.3f} m from the current position (limit {limit} m)")
        return Accept()

    return Check(check, name, kinds=frozenset({"setpoint"}), inputs={key: max_age}, budget=0.02)


def min_distance(
    a: str,
    b: str,
    limit: float,
    *,
    max_age: float = 0.1,
    actuators: Iterable[str] | None = None,
    rate: float = 50.0,
    name: str = "min_distance",
) -> Monitor:
    """A monitor: the positions in keys ``a`` and ``b`` must stay at least ``limit`` apart.

    E.g. ``min_distance("arm.ee", "scene.person", 0.3)`` stops the arm when a person comes
    closer than 30 cm, and lets it move again once they have left.
    """

    def monitor(view: View) -> Verdict:
        pa, pb = view.value(a), view.value(b)
        d = sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
        if d < limit:
            return Reject(f"{a} is {d:.3f} m from {b} (minimum {limit} m)")
        return Accept()

    return Monitor(
        monitor, name, inputs={a: max_age, b: max_age}, rate=rate,
        actuators=None if actuators is None else frozenset(actuators),
    )


LIBRARY: dict[str, Callable[..., Check | Monitor]] = {
    "workspace_box": workspace_box,
    "max_jump": max_jump,
    "min_distance": min_distance,
}


def from_config(config: Mapping[str, Mapping[str, Any]]) -> list[Check | Monitor]:
    """Build library checks and monitors from ``{"workspace_box": {"lo": [...], ...}, ...}``."""
    out = []
    for kind, kwargs in config.items():
        if kind not in LIBRARY:
            raise ValueError(f"unknown library check {kind!r}; known: {sorted(LIBRARY)}")
        out.append(LIBRARY[kind](**kwargs))
    return out
