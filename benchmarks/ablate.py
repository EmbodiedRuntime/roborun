"""Ablations: roborun with one part switched off, everything else unchanged.

A suite compares a baseline (no runtime) with the whole of roborun; that shows the runtime
helps, not that every part is needed. An ablated arm, ``roborun-no-<part>``, runs the same
program on roborun with only that part replaced by what a program would otherwise have:

- ``gate``: commands go straight to the driver. No checks or monitors, no tokens (so no
  arbitration and no revocation), no locks; an operator stop sends ``stop`` and nothing more.
- ``task``: tasks behave like plain asyncio tasks. Cancel returns at once instead of when the
  robot is at rest, it does not revoke the task's commands, a parent does not wait for its
  children, and a harness session does not own what it starts (closing it stops nothing).
- ``world``: values carry no usable time. Every value reads as current, so a stale position,
  a dead tracker or a silent sensor looks the same as a fresh one, to skills and checks alike.
- ``trace``: no trace; a failure is known only by its error line. This changes nothing while
  a program runs, so only the trace suite measures it (its error-line condition).

The switches are applied inside the process that runs one job and undone afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any

PARTS = ("gate", "task", "world", "trace")
PREFIX = "roborun-no-"


def arms(ablate: tuple[str, ...] | list[str] = ()) -> list[str]:
    """The arms a suite runs: baseline, roborun, and one per ablated part."""
    return ["baseline", "roborun", *(PREFIX + p for p in ablate)]


def is_baseline(arm: str) -> bool:
    return arm == "baseline"


def part_of(arm: str) -> str | None:
    return arm.removeprefix(PREFIX) if arm.startswith(PREFIX) else None


def parse(spec: str | None) -> tuple[str, ...]:
    """``--ablate gate,task`` or ``--ablate all``."""
    if not spec:
        return ()
    parts = PARTS if spec == "all" else tuple(p.strip() for p in spec.split(",") if p.strip())
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise SystemExit(f"unknown part(s) to ablate: {', '.join(unknown)}; choose from {', '.join(PARTS)}")
    return parts


@contextlib.contextmanager
def without(part: str | None) -> Iterator[None]:
    """Switch off ``part`` (None: nothing) for the duration of the block."""
    if part in (None, "trace"):
        yield
        return
    patches = {"gate": _gate, "task": _task, "world": _world}[part]()
    saved = [(obj, name, obj.__dict__[name]) for obj, name, _ in patches]
    try:
        for obj, name, new in patches:
            setattr(obj, name, new)
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


def _gate() -> list[tuple[Any, str, Any]]:
    from roborun.gate import Grant, LocalGate

    async def admit(self: LocalGate, grant: Any, cmd: Any, verdicts: Any = None) -> None:
        self._execute(cmd)  # whoever sends it, whatever it is

    async def acquire(self: LocalGate, task: Any, resources: Any, **_: Any) -> Grant:
        return Grant(task, frozenset(resources))  # nobody owns an actuator

    async def monitor_loop(self: LocalGate, monitor: Any) -> None:
        return None

    def operator_stop(self: LocalGate, actuators: Any = None) -> None:
        self._fallback(self.actuators if actuators is None else actuators, "operator")

    return [(LocalGate, "admit", admit), (LocalGate, "acquire", acquire),
            (LocalGate, "_monitor_loop", monitor_loop), (LocalGate, "operator_stop", operator_stop)]


def _task() -> list[tuple[Any, str, Any]]:
    from roborun.harness import Host
    from roborun.task import Task, TaskState

    def request_cancel(self: Task, reason: str, *, interrupt: bool) -> None:
        if self.state.terminal or self.state is TaskState.CANCELLING:
            return
        self.reason = reason
        if self.state is TaskState.PENDING:
            if self._coro is not None:
                self._coro.close()
            if self._atask is not None:
                self._atask.cancel()
            self._finish(TaskState.CANCELLED)
            return
        self._set_state(TaskState.CANCELLING)
        for child in list(self.children):  # asyncio propagates a cancel down an await chain
            child.cancel("parent")
        if interrupt and self._atask is not None:
            self._atask.cancel()

    async def cancel_and_wait(self: Task, reason: str = "requested") -> TaskState:
        self.cancel(reason)
        await asyncio.sleep(0)  # like asyncio: cancel is a request, it returns at once
        return self.state

    async def reap_children(self: Task) -> None:
        for child in list(self.children):
            if not child.state.terminal:
                child.cancel("parent")

    orig_launch = Host._launch

    def launch(self: Host, session: Any, body: Any, **kw: Any) -> Any:
        kw["detached"] = True  # nothing ties a task to the client that started it
        return orig_launch(self, session, body, **kw)

    return [(Task, "_request_cancel", request_cancel), (Task, "cancel_and_wait", cancel_and_wait),
            (Task, "_reap_children", reap_children), (Host, "_launch", launch)]


def _world() -> list[tuple[Any, str, Any]]:
    from roborun.stamped import Stamped
    from roborun.world import World

    orig_declare = World.declare

    def age(self: Stamped, now: float) -> float:
        return 0.0  # no timestamps: every value looks current

    def declare(self: World, key: str, kind: str = "state", *, max_age: Any = None,
                maxlen: Any = None) -> None:
        orig_declare(self, key, kind, max_age=None, maxlen=maxlen)  # silence is not noticed

    return [(Stamped, "age", age), (World, "declare", declare)]
