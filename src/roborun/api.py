"""The user-facing API: module-level functions and decorators.

    from roborun import action, provider, start, changes, world, sleep, send
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any

from roborun.errors import TaskCancelled
from roborun.gate import Command
from roborun.runtime import runtime
from roborun.task import Body, Task, _current
from roborun.world import Changes

_scopes: ContextVar[tuple] = ContextVar("roborun_scopes", default=())


# -- tasks -------------------------------------------------------------------


def start(
    body: Body | Coroutine,
    *,
    resources: Any = None,
    detached: bool = False,
    on_busy: str = "wait",
    wait_timeout: float | None = None,
    name: str | None = None,
    trace_parent: str | None = None,
) -> Task:
    """Start ``body`` as a task and return its handle immediately (SPEC §1.3).

    ``on_busy`` decides what happens when the resources are owned: ``wait`` (first come,
    first served, optionally bounded by ``wait_timeout``), ``fail``, or ``take`` (SPEC §3.3).
    """
    return runtime().start(
        body,
        resources=resources,
        detached=detached,
        on_busy=on_busy,
        wait_timeout=wait_timeout,
        name=name,
        trace_parent=trace_parent,
    )


def current() -> Task:
    """The task whose body is running here."""
    t = _current.get()
    if t is None:
        raise RuntimeError("not inside a roborun task")
    return t


def now() -> float:
    return runtime().clock.now()


async def sleep(seconds: float) -> None:
    await runtime().clock.sleep(seconds)


# -- library roles (SPEC Appendix A) -----------------------------------------


def action(*, resources: Any) -> Callable:
    """Mark an async function as an Action: a task that holds actuators."""

    def deco(fn: Callable[..., Coroutine]) -> Callable[..., Body]:
        @functools.wraps(fn)
        def make(*args: Any, **kwargs: Any) -> Body:
            return Body(fn, args, kwargs, role="action", resources=frozenset(resources))

        make.__wrapped_body__ = fn
        return make

    return deco


def provider(fn: Callable[..., Coroutine]) -> Callable[..., Body]:
    """Mark an async function as a Provider: a task that produces stamped values and holds no actuators."""

    @functools.wraps(fn)
    def make(*args: Any, **kwargs: Any) -> Body:
        return Body(fn, args, kwargs, role="provider")

    make.__wrapped_body__ = fn
    return make


def on_change(key: str, *, while_running: Task | None = None) -> Callable:
    """Start a child task that runs the decorated function for each new value of ``key``."""

    def deco(fn: Callable[[Any], Coroutine]) -> Task:
        async def loop() -> None:
            async for value in changes(key, while_running=while_running):
                await fn(value)

        loop.__qualname__ = f"on_change({key!r}, {fn.__qualname__})"
        return start(Body(loop, role="handler"))

    return deco


# -- world -------------------------------------------------------------------


class _WorldProxy:
    """``world`` refers to the active runtime's World."""

    def __getattr__(self, item: str) -> Any:
        return getattr(runtime().world, item)


world = _WorldProxy()


def changes(key: str, while_running: Task | None = None) -> Changes:
    return runtime().world.changes(key, while_running=while_running)


# -- commands ----------------------------------------------------------------


async def send(kind: str, actuator: str, **payload: Any) -> None:
    """Send a command to the Gate. A rejected command cancels the issuing task (SPEC §3.3)."""
    rt = runtime()
    me = _current.get()
    cmd = Command(kind, actuator, payload)
    t = rt.clock.now()
    verdicts: list = []
    try:
        ok = rt.gate.send(me, cmd, verdicts)
        if inspect.isawaitable(ok):
            ok = await ok  # a Gate in another process answers asynchronously (SPEC §3.8)
    except Exception as e:
        rt.trace.command(me, cmd, t, status="error", verdicts=verdicts, error=e)
        raise
    if ok:
        rt.trace.command(me, cmd, t, status="accepted", verdicts=verdicts)
        return
    rt.trace.command(me, cmd, t, status="rejected", verdicts=verdicts,
                     detail=me.detail if me is not None else None)
    if me is not None:
        # Mark the task cancelled without interrupting a later await: its cleanup
        # (stopping the robot) must be able to run.
        me._request_cancel("rejected", interrupt=False)
    detail = (me.detail if me is not None else None) or {"code": "unauthorized"}
    raise asyncio.CancelledError(f"{kind} on {actuator!r} rejected: {detail['code']}")


# -- scopes ------------------------------------------------------------------


class timeout:
    """``async with timeout(s):`` cancels the tasks started inside it after ``s`` seconds.

    They pass through ``cancelling``; the scope exits only after they are terminal,
    then raises ``TimeoutError``. Expiry never abandons a moving robot (SPEC §1.4).

    Limitation: it bounds tasks started inside the scope, not other awaits there.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.expired = False
        self._tasks: list[Task] = []

    def add(self, task: Task) -> None:
        self._tasks.append(task)
        if self.expired:
            task.cancel("timeout")

    def _expire(self, fut: asyncio.Future) -> None:
        if fut.cancelled():
            return
        self.expired = True
        for t in self._tasks:
            t.cancel("timeout")

    async def __aenter__(self) -> timeout:
        clock = runtime().clock
        self._token = _scopes.set(_scopes.get() + (self,))
        self._timer = clock.timer(clock.now() + self.seconds)
        self._timer.add_done_callback(self._expire)
        return self

    async def __aexit__(self, et: object, e: object, tb: object) -> bool:
        _scopes.reset(self._token)
        if not self._timer.done():
            self._timer.cancel()
        if self.expired:
            for t in self._tasks:
                await t.stopped()
            if et is None or (isinstance(e, TaskCancelled) and e.reason == "timeout"):
                raise TimeoutError(f"timed out after {self.seconds}s; the robot has stopped")
        return False


class hold:
    """``async with hold("arm", "gripper"):`` owns those actuators for the whole scope (SPEC §1.6)."""

    def __init__(
        self, *resources: str, on_busy: str = "wait", wait_timeout: float | None = None
    ) -> None:
        self.resources = frozenset(resources)
        self.on_busy = on_busy
        self.wait_timeout = wait_timeout
        self._grant = None

    async def __aenter__(self) -> hold:
        me = current()
        gate = runtime().gate
        needed = self.resources - gate.held_by_ancestors(me, self.resources)
        if needed:
            self._grant = await gate.acquire(
                me, needed, on_busy=self.on_busy, wait_timeout=self.wait_timeout
            )
        return self

    async def __aexit__(self, *exc: object) -> bool:
        if self._grant is not None:
            runtime().gate.release(self._grant)
        return False
