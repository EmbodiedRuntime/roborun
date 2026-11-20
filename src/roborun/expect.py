"""Expectations (SPEC Appendix A): what a task must find true of the World, and monitoring.

An expectation is a condition over stamped World values, checked for one task: before its
body runs (``at_start``), while it runs (``during``), or after its body returned
(``at_end``). Checking them while the task runs is *monitoring*. It is not verification:
it judges one run, against measurements. It can say that a run went wrong and which
condition broke, with the values that show it; it cannot say that runs do not go wrong.

    in_tray = at_end(lambda v: near(v.value("cup_red"), v.value("tray"), 0.05),
                     inputs={"cup_red": 0.3, "tray": 0.3}, name="cup_in_tray")
    await place("tray").expect(in_tray, not_stalled())

A verdict has four values, because a measurement can be missing and two can disagree:

- ``holds``;
- ``violated``: the inputs are fresh and the condition is false;
- ``inconsistent``: two sources contradict each other, so at least one is wrong (see
  ``relation``); it has the same consequences as ``violated``, under its own name;
- ``unknown``: an input is missing or older than its ``max_age``, or the condition raised.
  Old data never counts as true.

What follows from a verdict:

| when     | violated or inconsistent                          | unknown                         |
|----------|---------------------------------------------------|---------------------------------|
| at_start | the task fails before its body runs               | the same                        |
| during   | after ``grace`` seconds: the task is cancelled, the Gate stops its actuators, and it ends ``failed`` | recorded; success is unverified |
| at_end   | the task fails                                    | the task succeeds, unverified   |

A failure is ``ExpectationFailed``, with the stamped values the verdict rests on. A task
whose expectations all held throughout has ``verified=True``; one that succeeded with an
unknown has ``False``; one with none during or at the end has ``None``.

A condition returns a truth value, or says why not: ``no("...")``, ``disagree("...")``,
``unknown("...")``. It reads its declared inputs (``v.value(key)``), the task's parameters
(``v.params``) and result, and the last command the task or its children had accepted for
an actuator (``v.command("arm")``), which is how it can tell whether a command took effect.

A skill names its phases as it goes (``await phase("lift")``), and an expectation can be
limited to some of them: ``during(holds, phase=("approach", "lower"))`` is watched only
there, ``at_end(holds, phase="close")`` is checked when that phase ends, and a failure says
in which phase it happened. Phases follow one another; they do not nest.

When a task starts, its expectations are also checked against what is known without
running it (``lint``): an input nothing has written, a ``max_age`` shorter than the period
its producer writes at, a ``grace`` longer than the task's timeout. These are recorded as
warnings; they do not fail the task.

The Gate's checks and monitors are a different thing: they guard actuators against unsafe
commands, for every task, and cannot be bypassed. Expectations run in the runtime's
process and say whether a task achieves what it claims.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from roborun.checks import View
from roborun.errors import RoborunError

POLL = 0.01  # how often a `stable` expectation is sampled

_EPS = 1e-9


class Snapshot(View):
    """The fresh stamped inputs an expectation declared, with the task's parameters and,
    at the end, its result."""

    def __init__(self, items: dict, now: float, params: Mapping[str, Any], result: Any,
                 commands: Mapping[str, dict] | None = None, phase: str | None = None) -> None:
        super().__init__(items, now)
        self.params, self.result, self.phase = dict(params), result, phase
        self._commands = dict(commands or {})

    def command(self, actuator: str) -> dict | None:
        """The last command the task, or a task it started, had accepted for ``actuator``:
        ``{"kind", "payload", "t"}``, or None."""
        return self._commands.get(actuator)


class _Why(str):
    """A reason that is false as a truth value: what a condition returns to say why not."""

    status = "violated"

    def __bool__(self) -> bool:
        return False


class _Disagree(_Why):
    status = "inconsistent"


class _Unknown(_Why):
    status = "unknown"


def no(reason: str) -> _Why:
    """Return this from a condition: it does not hold, and why."""
    return _Why(reason)


def disagree(reason: str) -> _Why:
    """Return this from a condition: its sources contradict each other."""
    return _Disagree(reason)


def unknown(reason: str) -> _Why:
    """Return this from a condition: it cannot tell."""
    return _Unknown(reason)


@dataclass(frozen=True)
class Expectation:
    fn: Callable[[Snapshot], Any]
    name: str
    when: str  # "start", "during" or "end"
    inputs: Mapping[str, float] = field(default_factory=dict)  # key -> max_age
    stable: float = 0.0  # at the end: it must hold for this long
    grace: float = 0.0  # during: how long it may fail before the task does
    phases: tuple = ()  # the phases of the task it applies to; empty: the whole task
    error: type | None = None  # the ExpectationFailed subclass a failure is reported as


@dataclass(frozen=True)
class Verdict:
    status: str  # holds, violated, inconsistent, unknown

    @property
    def failed(self) -> bool:
        return self.status in ("violated", "inconsistent")
    reason: str
    evidence: dict  # key -> {value, stamp, source, age}

    def record(self, e: Expectation, t: float) -> dict:
        return {"name": e.name, "when": e.when, "verdict": self.status, "reason": self.reason,
                "evidence": self.evidence, "t": t}

    def error(self, e: Expectation, phase: str | None) -> ExpectationFailed:
        return (e.error or ExpectationFailed)(e.name, e.when, self, phase)


class ExpectationFailed(RoborunError):
    """A task did not meet one of its expectations. ``evidence`` holds the stamped values
    that the verdict rests on; ``verdict`` is ``violated`` or ``inconsistent``, or
    ``unknown`` at the start; ``phase`` is the phase the task was in, if it names them.

    A skill can give a failure its own name by subclassing (``error=GraspMissed``); such a
    subclass sets ``plain`` and its message is the condition's reason."""

    plain = False

    def __init__(self, name: str, when: str, verdict: Verdict, phase: str | None = None) -> None:
        self.name, self.when, self.phase = name, when, phase
        self.verdict, self.reason, self.evidence = verdict.status, verdict.reason, verdict.evidence
        what = {"violated": "does not hold", "inconsistent": "its sources disagree"}.get(
            verdict.status, "cannot be checked")
        of = f" of {phase!r}" if phase and when != "during" else ""
        where = (f"during {phase!r}" if phase else "during the task") if when == "during" \
            else f"at the {when}{of}"
        text = f"expected {name!r} {where}: {what}" + (f" ({verdict.reason})" if verdict.reason else "")
        super().__init__(verdict.reason if self.plain and verdict.reason else text)


def _named(fn: Callable, name: str | None) -> str:
    return name or getattr(fn, "__name__", "expectation").strip("<>")


def _phases(phase: Any) -> tuple:
    return () if phase is None else (phase,) if isinstance(phase, str) else tuple(phase)


def at_start(fn: Callable[[Snapshot], Any], *, inputs: Mapping[str, float],
             name: str | None = None, phase: Any = None, error: type | None = None) -> Expectation:
    """``fn`` must hold before the task's body runs; otherwise the task fails at once,
    having sent no command. With ``phase``: when the task enters that phase."""
    return Expectation(fn, _named(fn, name), "start", dict(inputs), phases=_phases(phase),
                       error=error)


def at_end(fn: Callable[[Snapshot], Any], *, inputs: Mapping[str, float],
           name: str | None = None, stable: float = 0.0, phase: Any = None,
           error: type | None = None) -> Expectation:
    """``fn`` must hold when the task's body has returned, and for ``stable`` seconds after
    (an object that was just released may still be moving). With ``phase``: when the task
    leaves that phase, before it goes on."""
    return Expectation(fn, _named(fn, name), "end", dict(inputs), stable, phases=_phases(phase),
                       error=error)


def during(fn: Callable[[Snapshot], Any], *, inputs: Mapping[str, float],
           name: str | None = None, grace: float = 0.0, phase: Any = None,
           error: type | None = None) -> Expectation:
    """``fn`` must hold while the task's body runs. Once it has failed for ``grace`` seconds
    without a break (a contact that flickers is not a slip), the task is cancelled, the Gate
    stops its actuators, and it ends ``failed``. With ``phase`` (one name or several): only
    while the task is in one of those phases."""
    return Expectation(fn, _named(fn, name), "during", dict(inputs), grace=grace,
                       phases=_phases(phase), error=error)


def evaluate(e: Expectation, world: Any, now: float, params: Mapping[str, Any],
             result: Any = None, commands: Mapping[str, dict] | None = None,
             phase: str | None = None) -> Verdict:
    """One verdict for ``e`` on the World as it is now."""
    items, evidence = {}, {}
    for key, max_age in e.inputs.items():
        sv = world.latest(key)
        if sv is None:
            return Verdict("unknown", f"no value for {key!r}", evidence)
        age = sv.age(now)
        evidence[key] = {"value": sv.value, "stamp": sv.stamp, "source": sv.source,
                         "age": round(age, 4)}
        if age > max_age + _EPS:
            return Verdict("unknown", f"{key!r} is {age:.3f}s old (limit {max_age}s)", evidence)
        items[key] = sv
    try:
        ok = e.fn(Snapshot(items, now, params, result, commands, phase))
    except Exception as err:  # noqa: BLE001 - a condition that cannot be evaluated proves nothing
        return Verdict("unknown", f"the condition raised {err!r}", evidence)
    if isinstance(ok, _Why):
        return Verdict(ok.status, str(ok), evidence)
    return Verdict("holds", "", evidence) if ok else Verdict("violated", "", evidence)


def _record(task: Any, e: Expectation, v: Verdict, now: float) -> None:
    rec = v.record(e, now)
    if task.phase is not None:
        rec["phase"] = task.phase
    task.checks.append(rec)


async def check(task: Any, when: str, phase: str | None = None) -> None:
    """Check ``task``'s expectations for ``when`` (start or end), of the whole task or of
    ``phase``; record every verdict on the task.

    Raises ``ExpectationFailed`` when one fails, and for an unknown one at the start. At
    the end of the task it also settles ``task.verified``.
    """
    rt = task.rt
    mine = [e for e in task.expectations
            if e.when == when and (phase in e.phases if phase is not None else not e.phases)]
    unsure, first, last = False, True, {}
    began = rt.clock.now()
    until = began + max((e.stable for e in mine), default=0.0)
    while mine:
        now = rt.clock.now()
        for i, e in enumerate(mine):
            if not first and now > began + e.stable + _EPS:
                continue  # its window is over
            v = evaluate(e, rt.world, now, task.params, task.result, task.commands, task.phase)
            if last.get(i) != v.status:  # one record per change, not one per sample
                last[i] = v.status
                _record(task, e, v, now)
            if v.failed or (v.status == "unknown" and when == "start"):
                raise v.error(e, task.phase)
            unsure = unsure or v.status == "unknown"
        first = False
        if now >= until - _EPS:
            break
        await rt.clock.sleep(POLL)
    if phase is not None:
        task.unwatched = task.unwatched or unsure  # an unknown in a phase: unverified
    elif when == "end" and any(e.when in ("during", "end") for e in task.expectations):
        task.verified = not (unsure or task.unwatched)


async def phase(name: str) -> None:
    """Say that the task whose body is running now enters phase ``name``.

    The phase it was in ends first: its ``at_end`` expectations are checked, and a failed one
    raises here, before anything else is sent. Then the new phase's ``at_start`` expectations
    are checked. Phases are recorded on the task and in its trace span.
    """
    from roborun.api import current

    task = current()
    if task.phase == name:
        return
    await leave(task)
    task.phase = name
    task.phases.append({"name": name, "start": task.rt.clock.now(), "end": None})
    await check(task, "start", name)


async def leave(task: Any) -> None:
    """End the phase ``task`` is in, checking that phase's ``at_end`` expectations."""
    if task.phase is None:
        return
    try:
        await check(task, "end", task.phase)
    finally:
        task.phases[-1]["end"] = task.rt.clock.now()
        task.phase = None


def watch(task: Any) -> asyncio.Future | None:
    """Start monitoring ``task``'s ``during`` expectations; cancel the result to stop."""
    if not any(e.when == "during" for e in task.expectations):
        return None
    return asyncio.ensure_future(_watch(task))


async def _watch(task: Any) -> None:
    rt = task.rt
    mine = [e for e in task.expectations if e.when == "during"]
    since: dict[int, float] = {}  # when each one began to fail
    last: dict[int, str] = {}
    while True:
        now = rt.clock.now()
        for i, e in enumerate(mine):
            if e.phases and task.phase not in e.phases:
                since.pop(i, None)
                last.pop(i, None)
                continue
            v = evaluate(e, rt.world, now, task.params, None, task.commands, task.phase)
            if last.get(i) != v.status:
                last[i] = v.status
                _record(task, e, v, now)
            if v.status == "unknown":
                task.unwatched = True  # it could not be watched all the way
            if not v.failed:
                since.pop(i, None)
            elif now - since.setdefault(i, now) >= e.grace - _EPS:
                task._fail_expectation(v.error(e, task.phase))
                return
        await rt.clock.sleep(POLL)


# -- ready-made expectations ---------------------------------------------------------------


def after(actuator: str, kind: str, fn: Callable[[Snapshot, dict], Any], *, within: float,
          inputs: Mapping[str, float], name: str | None = None, grace: float = 0.0,
          phase: Any = None, error: type | None = None) -> Expectation:
    """A command must take effect: ``within`` seconds after the task's last ``kind`` command
    for ``actuator`` was accepted, ``fn(view, command)`` must hold, and go on holding."""

    def took_effect(v: Snapshot) -> Any:
        c = v.command(actuator)
        if c is None or c["kind"] != kind or v.now - c["t"] < within - _EPS:
            return True
        return fn(v, c)

    return during(took_effect, inputs=inputs, grace=grace, phase=phase, error=error,
                  name=name or f"{kind} on {actuator} takes effect")


def not_stalled(actuator: str = "arm", *, position: str = "arm.ee", moving: str = "arm.moving",
                tol: float = 0.01, within: float = 1.0, max_age: float = 0.1) -> Expectation:
    """A goal is being worked on or has been reached: the actuator must not sit still, away
    from its goal, for ``within`` seconds. (Blocked, jammed, or a controller that gave up
    without saying so.)"""

    def working(v: Snapshot) -> Any:
        c = v.command(actuator)
        if c is None or c["kind"] != "goal" or v.value(moving):
            return True
        if v[moving].stamp <= c["t"]:
            return True  # measured before the command: it says nothing about it yet
        here, goal = v.value(position), c["payload"]["target"]
        d = sum((x - y) ** 2 for x, y in zip(here, goal)) ** 0.5
        return d <= tol or no(f"{actuator} is at rest {d:.3f} m from its goal")

    return during(working, inputs={position: max_age, moving: max_age}, grace=within,
                  name=f"{actuator} not stalled")


def stops(actuator: str = "arm", *, moving: str = "arm.moving", within: float = 2.0,
          max_age: float = 0.1) -> Expectation:
    """A stop ends: ``within`` seconds after the task told ``actuator`` to stop, it must be
    at rest. Without it, a skill that waits for the arm to stop waits for as long as a
    controller that ignores the stop keeps moving."""
    return after(actuator, "stop",
                 lambda v, c: not v.value(moving) or no(f"{actuator} is still moving "
                                                        f"{v.now - c['t']:.1f}s after its stop"),
                 within=within, inputs={moving: max_age}, name=f"{actuator} stops")


def gripper_responds(actuator: str = "gripper", *, width: str = "gripper.width",
                     holding: str = "gripper.holding", open_width: float = 0.07,
                     within: float = 3.0, max_age: float = 0.1) -> Expectation:
    """The gripper does what it was told: ``within`` seconds after a close, the fingers have
    closed (on something, or all the way); after an open, they are ``open_width`` apart."""

    def responded(v: Snapshot, c: dict) -> Any:
        w = v.value(width)
        if c["payload"].get("close"):
            return (v.value(holding) is not None or w < open_width
                    or no(f"{actuator} was told to close and is still {w:.3f} m open"))
        return w >= open_width or no(f"{actuator} was told to open and is {w:.3f} m apart")

    return after(actuator, "effector", responded, within=within,
                 inputs={width: max_age, holding: max_age}, name=f"{actuator} responds")


# -- what can be said before the task runs -------------------------------------------------


def lint(task: Any) -> list[dict]:
    """Check ``task``'s expectations against what is known when it starts, without running
    it, and return a warning for each one that cannot work as written:

    - an input that nothing has written (no provider for it was started);
    - a ``max_age`` shorter than the period at which the key is written, so the value is
      stale between two samples and the verdict is mostly ``unknown``;
    - a ``grace`` or ``stable`` time that does not fit in the task's timeout, so the
      expectation can never decide.

    The task keeps them in ``task.warnings`` and its trace span.
    """
    world, now, out = task.rt.world, task.rt.clock.now(), []

    def warn(e: Expectation, reason: str) -> None:
        out.append({"name": e.name, "when": e.when, "warning": reason, "t": now})

    left = None if task.deadline is None else task.deadline - now
    for e in task.expectations:
        for key, max_age in e.inputs.items():
            sv = world.latest(key)
            if sv is None:
                if e.when != "start":  # at the start it fails as unknown anyway
                    warn(e, f"nothing has written {key!r}: is its provider started?")
                continue
            period = world.period(key)
            if period is not None and max_age < period - _EPS:
                warn(e, f"max_age {max_age}s for {key!r} is shorter than the {period:.3f}s "
                        f"period of {sv.source or 'its producer'}: it is stale between samples")
        bound = e.grace if e.when == "during" else e.stable
        if left is not None and bound > left + _EPS:
            what = "grace" if e.when == "during" else "stable"
            warn(e, f"{what}={bound}s does not fit in the task's timeout ({left:.3f}s left)")
    return out


# -- relations between sources -------------------------------------------------------------


def relation(name: str, fn: Callable[[Snapshot], Any], *, inputs: Mapping[str, float],
             rate: float = 20.0) -> Any:
    """A provider that keeps checking that several sources agree, and writes the verdict
    to the World key ``relation.<name>``.

    ``fn`` returns a truth value or ``disagree("...")``. The value written is
    ``{"status": "consistent" | "inconsistent" | "unknown", "reason", "sources"}``, stamped
    with its oldest input (SPEC §2.2), so a reader's ``max_age`` covers the inputs too. It
    judges the sensors, not any task: start it once, detached, and let tasks that depend on
    those sources expect ``consistent(name)``.
    """
    from roborun.api import now, provider, sleep, world

    probe = Expectation(fn, name, "during", dict(inputs))

    async def relate() -> None:
        while True:
            v = evaluate(probe, world, now(), {})
            status = {"holds": "consistent", "unknown": "unknown"}.get(v.status, "inconsistent")
            stamps = [s["stamp"] for s in v.evidence.values()]
            world.write(f"relation.{name}",
                        {"status": status, "reason": v.reason, "sources": v.evidence},
                        stamp=min(stamps) if stamps else now(), source=f"relation({name})")
            await sleep(1.0 / rate)

    relate.__qualname__ = f"relation({name})"
    return provider(relate)()


def consistent(name: str, *, max_age: float = 0.2, grace: float = 0.0,
               when: Callable[[Snapshot], Any] | None = None, phase: Any = None) -> Expectation:
    """While the task runs, the sources of ``relation(name)`` must agree. If they do not,
    the task fails with verdict ``inconsistent`` and both sources as evidence: what it was
    acting on cannot all be true. ``phase`` limits it to the phases of the task that depend
    on those sources, e.g. a place until it releases; ``when`` does the same with a
    condition, for tasks that do not name their phases."""
    key = f"relation.{name}"

    def agree(v: Snapshot) -> Any:
        if when is not None and not when(v):
            return True
        r = v.value(key)
        if r["status"] == "inconsistent":
            return disagree(r["reason"])
        return True if r["status"] == "consistent" else unknown(r["reason"])

    return during(agree, inputs={key: max_age}, grace=grace, phase=phase,
                  name=f"{name} consistent")


def near(a: Any, b: Any, tol: float) -> bool:
    """Whether two points are within ``tol`` of each other."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 <= tol
