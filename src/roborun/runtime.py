"""The runtime object and the simulation driver."""

from __future__ import annotations

import asyncio
import itertools
import os
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from roborun.clock import Clock, SteppedClock
from roborun.errors import Replaced
from roborun.gate import LocalGate
from roborun.task import Body, Task, _current, _Root
from roborun.world import World

_runtime: ContextVar[Runtime | None] = ContextVar("roborun_runtime", default=None)


def runtime() -> Runtime:
    rt = _runtime.get()
    if rt is None:
        raise RuntimeError("no roborun runtime is active; run inside run_sim(...) or Runtime.active()")
    return rt


class Runtime:
    def __init__(
        self,
        backend: Any,
        clock: Clock | None = None,
        *,
        gate: Any = None,
        supervises: dict[str, set[str]] | None = None,
        trace: Any = None,
    ) -> None:
        """With ``backend``, the Gate runs in-process (``LocalGate``). With ``gate`` (a
        ``RemoteGate``), the backend lives in the Gate process and ``backend`` is None."""
        self.clock = clock or SteppedClock()
        self.backend = backend
        self.world = World(self.clock)
        self.gate = gate if gate is not None else LocalGate(backend, self.clock, self.world.latest)
        # session -> sessions it may take actuators from (SPEC §3.3); configuration only.
        self.supervises: dict[str, set[str]] = dict(supervises or {})
        self.root = _Root()
        self._ids = itertools.count(1)
        self._by_id: dict[str, Task] = {}
        # SPEC §4: a Trace, a path to write one to, or None for in-memory spans only.
        from roborun.trace import Trace

        if trace is None or isinstance(trace, (str, os.PathLike)):
            trace = Trace(trace, meta={"backend": type(backend).__name__ if backend else "remote"})
        self.trace: Trace = trace
        self.world.on_write = self._traced_write
        if hasattr(self.gate, "listeners"):
            self.gate.listeners.append(self._traced_safety)
        if backend is not None:
            backend.attach(self)

    def _traced_write(self, key: str, sv: Any) -> None:
        task = _current.get()
        if task is not None:  # writes by tasks; backend observations are not traced one by one
            self.trace.world(task, key, sv)

    def _traced_safety(self, ev: Any) -> None:
        self.trace.safety({"t": ev.t, "kind": ev.kind, "actuators": list(ev.actuators),
                           "cause": ev.cause, "detail": ev.detail})

    @contextmanager
    def active(self) -> Iterator[Runtime]:
        token = _runtime.set(self)
        try:
            yield self
        finally:
            _runtime.reset(token)

    def start(
        self,
        body: Body | Coroutine,
        *,
        resources: Any = None,
        detached: bool = False,
        on_busy: str = "wait",
        wait_timeout: float | None = None,
        name: str | None = None,
        session: str | None = None,
        trace_parent: str | None = None,
        parent: Task | None = None,
    ) -> Task:
        """Start a task. ``parent`` is for hosts that start tasks on a session's behalf;
        otherwise the parent is the current task, or the root if detached."""
        from roborun.api import _scopes

        current = _current.get() if parent is None else parent
        if current is not None and getattr(current, "replace_requested", False):
            # A safe point: the program asks for its next step after a new version was
            # requested. The step it just finished ran under the old version.
            raise Replaced(f"{current.name} is replaced by a new version")
        parent = self.root if (detached or current is None) else current
        if isinstance(body, Body):
            res = body.resources if resources is None else frozenset(resources)
        else:
            res = frozenset(resources or ())
        # The session is the principal, inherited from the creator even for detached tasks.
        if session is None:
            session = current.session if current is not None else "local"
        task = Task(
            self,
            body,
            parent,
            resources=res,
            on_busy=on_busy,
            wait_timeout=wait_timeout,
            name=name,
            session=session,
        )
        task.id = f"t{next(self._ids)}"
        task.trace_parent = trace_parent
        self._by_id[task.id] = task
        parent.children.append(task)
        for scope in _scopes.get():
            scope.add(task)
        task._atask = asyncio.get_running_loop().create_task(task._run())
        return task

    def replace(self, old: Task, make_body: Callable[[], Coroutine], **start_kwargs: Any) -> Task:
        """Replace a running program with a new version at its next safe point (SPEC §6).

        The next time ``old`` starts a step, it ends with reason ``replaced``; the step it is
        running now finishes first. The new version is a task from now on (so it can be
        named and watched), waits for ``old`` to end, and then runs ``make_body()``. State
        passes through the World, which the new version reads.
        """
        old.replace_requested = True

        async def successor() -> Any:
            await old.stopped()
            return await make_body()

        parent = old.parent if old.parent is not self.root else None
        start_kwargs.setdefault("name", old.name)
        start_kwargs.setdefault("session", old.session)
        return self.start(successor(), parent=parent, detached=parent is None, **start_kwargs)

    def task_by_id(self, task_id: str) -> Task | None:
        return self._by_id.get(task_id)

    def all_tasks(self) -> list[Task]:
        out, stack = [], list(self.root.children)
        while stack:
            t = stack.pop()
            out.append(t)
            stack.extend(t.children)
        return out


async def _settle(n: int) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def settle_with_gate(gate: Any, n: int) -> None:
    """Let every task react, including to the Gate's answers, before time advances.

    Settled means: no command is waiting for the Gate, and a full round of the event loop
    sent no new one. Checks run in worker threads (SPEC §3.4), so this cannot be a fixed
    number of loop iterations.
    """
    while True:
        before = gate.activity
        await _settle(n)
        if gate.outstanding:
            await gate.quiet()
            continue
        if gate.activity == before:
            return


async def _drive(
    rt: Runtime,
    main: Body | Coroutine | Callable[[], Coroutine],
    *,
    settle: Callable[[], Awaitable[None]],
    step: Callable[[], Awaitable[None]],
    until: float,
) -> Task:
    """Run ``main`` as a task, alternating ``settle`` (every task reacts) and ``step``."""
    if callable(main) and not isinstance(main, Body):
        main = main()
    task = rt.start(main, name=getattr(main, "__qualname__", None))

    async def step_until(done: Callable[[], bool], limit: float) -> None:
        while not done() and rt.clock.now() < limit - 1e-9:
            await settle()
            if done():
                break
            await step()
        await settle()

    await step_until(lambda: task.state.terminal, until)
    if not task.state.terminal:
        task.cancel("timeout")
        await step_until(lambda: task.state.terminal, until + 5.0)
    # Shut down detached tasks (e.g. trackers) and let them stop.
    for t in list(rt.root.children):
        t.cancel("parent")
    await step_until(lambda: all(t.state.terminal for t in rt.root.children), rt.clock.now() + 5.0)
    task.runtime = rt  # for inspection after the run
    return task


async def run_sim(
    main: Body | Coroutine | Callable[[], Coroutine],
    *,
    backend: Any,
    dt: float = 0.01,
    until: float = 30.0,
    settle: int = 64,
    trace: Any = None,
    pace: float | None = None,
) -> Task:
    """Run ``main`` as a task on a stepped clock, stepping ``backend`` every ``dt`` seconds.

    Between steps the event loop runs ``settle`` iterations, so every task reacts to a
    step before the next one. Simulated time advances only here; results are deterministic.
    With ``pace`` (e.g. 1.0), simulated time also keeps up with wall time at that rate, so
    work outside the simulation, such as a model call, takes simulated time too.
    The Gate runs in-process; ``roborun.remote.run_lockstep`` runs against a Gate process.
    """
    rt = Runtime(backend, SteppedClock(), trace=trace)
    loop = asyncio.get_running_loop()
    wall0 = loop.time()

    async def step() -> None:
        nonlocal wall0
        backend.step(dt)
        rt.clock.advance(dt)
        if pace:  # simulated seconds per wall second, e.g. while a real model thinks
            delay = wall0 + rt.clock.now() / pace - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            elif delay < -0.5:
                wall0 -= delay  # fell behind: do not try to catch up

    with rt.active():
        try:
            return await _drive(
                rt, main, settle=lambda: settle_with_gate(rt.gate, settle), step=step, until=until
            )
        finally:
            rt.gate.close()
            rt.trace.close()
