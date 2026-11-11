"""Tasks (SPEC §1).

A task keeps running after the call that started it returns. Requesting cancellation
moves it to CANCELLING, not CANCELLED: the robot may still be moving.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Any

from roborun.errors import InvalidState, Replaced, TaskCancelled, TaskFailed

if TYPE_CHECKING:
    from roborun.runtime import Runtime


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset({TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED})

_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    # A pending task fails when it cannot start, e.g. its actuators stay busy.
    TaskState.PENDING: frozenset({TaskState.RUNNING, TaskState.CANCELLED, TaskState.FAILED}),
    TaskState.RUNNING: frozenset(
        {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLING}
    ),
    # The motion may finish before it can be stopped, and the stop itself may fail.
    TaskState.CANCELLING: frozenset(
        {TaskState.CANCELLED, TaskState.SUCCEEDED, TaskState.FAILED}
    ),
    TaskState.SUCCEEDED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


def can_transition(src: TaskState, dst: TaskState) -> bool:
    """Return True if SPEC §1.2 allows moving from ``src`` to ``dst``."""
    return dst in _TRANSITIONS[src]


# The task whose body is running in the current asyncio context.
_current: ContextVar[Task | None] = ContextVar("roborun_current_task", default=None)


class Body:
    """An unstarted task body: a function and its arguments, plus how to run it.

    Calling a function decorated with ``@action`` or ``@provider`` returns a Body.
    ``await body`` starts it as a child task and waits for it (SPEC §1.3).
    """

    def __init__(
        self,
        fn: Callable[..., Coroutine],
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        role: str = "task",
        resources: frozenset[str] = frozenset(),
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs or {}
        self.role = role
        self.resources = frozenset(resources)

    @property
    def name(self) -> str:
        return self.fn.__qualname__

    def params(self) -> dict[str, Any]:
        try:
            bound = inspect.signature(self.fn).bind(*self.args, **self.kwargs)
        except TypeError:
            return dict(self.kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def coroutine(self) -> Coroutine:
        return self.fn(*self.args, **self.kwargs)

    def __await__(self):
        from roborun.api import start

        return start(self).__await__()


class _Root:
    """The root of the task tree. Detached tasks are its children."""

    name = "root"
    parent = None
    session = "local"
    state = TaskState.RUNNING

    def __init__(self) -> None:
        self.children: list[Task] = []


class Task:
    def __init__(
        self,
        rt: Runtime,
        body: Body | Coroutine,
        parent: Task | _Root,
        *,
        resources: frozenset[str],
        on_busy: str,
        wait_timeout: float | None,
        name: str | None,
        session: str,
    ) -> None:
        if isinstance(body, Body):
            self._coro = None
            self._body = body
            self.name = name or body.name
            self.role = body.role
            self.params: dict[str, Any] = body.params()
            code = f"{body.fn.__module__}.{body.fn.__qualname__}"
        else:
            self._coro = body
            self._body = None
            self.name = name or getattr(body, "__qualname__", "task")
            self.role = "task"
            self.params = {}
            code = getattr(body, "__qualname__", "task")
        self.rt = rt
        self.parent = parent
        self.children: list[Task] = []
        self.resources = frozenset(resources)
        self.on_busy = on_busy
        self.wait_timeout = wait_timeout
        self.session = session
        self.state = TaskState.PENDING
        self.reason: str | None = None
        self.detail: dict[str, Any] | None = None  # structured rejection detail (SPEC §3.8)
        self.id = ""  # assigned by the runtime
        self.result: Any = None
        self.error: BaseException | None = None
        self.created_at: float = rt.clock.now()
        self.started_at: float | None = None
        self.trace_parent: str | None = None  # e.g. the harness turn that started it (SPEC §4)
        self.replace_requested = False  # a new version waits for the next safe point (SPEC §6)
        self._repeats = 0
        self.ended_at: float | None = None
        # SPEC §1.7: where the body came from. Generated code sets generated_by.
        self.provenance: dict[str, Any] = {"code": code, "generated_by": None}
        self._done: asyncio.Future = asyncio.get_running_loop().create_future()
        self._atask: asyncio.Task | None = None

    def __repr__(self) -> str:
        reason = f" {self.reason}" if self.reason else ""
        return f"<Task {self.name} {self.state.value}{reason}>"

    def __getattr__(self, item: str) -> Any:
        # Updatable parameters read as attributes: ``g.target``.
        params = self.__dict__.get("params", {})
        if item in params:
            return params[item]
        raise AttributeError(item)

    # -- state -------------------------------------------------------------

    def _set_state(self, new: TaskState) -> None:
        if not can_transition(self.state, new):
            raise InvalidState(f"{self.name}: {self.state.value} -> {new.value} is not allowed")
        self.state = new

    def _finish(self, state: TaskState) -> None:
        if self.state is TaskState.RUNNING and state is TaskState.CANCELLED:
            self._set_state(TaskState.CANCELLING)
        self._set_state(state)
        self.ended_at = self.rt.clock.now()
        self.rt.gate.release_all(self)
        if self.rt.trace is not None:
            self.rt.trace.task(self, self.rt.world)
        if not self._done.done():
            self._done.set_result(state)

    # -- operations (SPEC §1.3) --------------------------------------------

    def update(self, **params: Any) -> None:
        if self.state is not TaskState.RUNNING:
            raise InvalidState(
                f"cannot update {self.name}: it is {self.state.value}; "
                f"update is valid only while running"
            )
        self.params.update(params)

    def cancel(self, reason: str = "requested") -> None:
        """Request cancellation. Synchronous: the request is sent even without ``await``."""
        self._request_cancel(reason, interrupt=True)

    def _request_cancel(self, reason: str, *, interrupt: bool) -> None:
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
        self.rt.gate.revoke(self)
        for child in list(self.children):
            child.cancel("parent")
        if interrupt and self._atask is not None:
            self._atask.cancel()

    async def stopped(self) -> TaskState:
        """Wait for a terminal state. Never raises."""
        return await asyncio.shield(self._done)

    async def cancel_and_wait(self, reason: str = "requested") -> TaskState:
        self.cancel(reason)
        return await self.stopped()

    async def _wait(self) -> Any:
        state = await self.stopped()
        if state is TaskState.SUCCEEDED:
            return self.result
        if state is TaskState.FAILED:
            raise TaskFailed(self.name, self.error) from self.error
        reason = self.reason or "requested"
        raise TaskCancelled(reason, f"task {self.name!r} cancelled: {reason}")

    def __await__(self):
        return self._wait().__await__()

    # -- execution ---------------------------------------------------------

    async def _run(self) -> None:
        from roborun.api import _scopes

        _current.set(self)
        _scopes.set(())
        # A task waiting for actuators is still pending: it has no effect yet (SPEC §1.1).
        needed = self.resources - self.rt.gate.held_by_ancestors(self, self.resources)
        if needed:
            try:
                await self.rt.gate.acquire(
                    self, needed, on_busy=self.on_busy, wait_timeout=self.wait_timeout
                )
            except asyncio.CancelledError:
                if not self.state.terminal:  # cancelled from outside the task tree
                    self.reason = self.reason or "requested"
                    self._finish(TaskState.CANCELLED)
                return
            except Exception as e:  # noqa: BLE001 - ResourceBusy, or a bug: never stay pending
                self.error = e
                self._finish(TaskState.FAILED)  # could not start (SPEC §1.2)
                return
            if self.state.terminal:
                return
        self._set_state(TaskState.RUNNING)
        self.started_at = self.rt.clock.now()
        outcome = "error"
        try:
            if self._coro is None:
                self._coro = self._body.coroutine()
            self.result = await self._coro
            outcome = "ok"
        except asyncio.CancelledError:
            outcome = "cancelled"
        except Replaced:
            outcome = "cancelled"  # a safe point: the new version takes over (SPEC §6)
            self.reason = "replaced"
        except TaskCancelled as e:
            outcome = "cancelled" if self.state is TaskState.CANCELLING else "error"
            self.error = e
        except BaseException as e:  # noqa: BLE001 - the task records any failure
            self.error = e
        finally:
            if (
                self._coro is not None
                and inspect.getcoroutinestate(self._coro) == inspect.CORO_CREATED
            ):
                self._coro.close()
        await self._reap_children()
        if outcome == "ok":
            self._finish(TaskState.SUCCEEDED)
        elif outcome == "cancelled":
            if self.reason is None:
                self.reason = "requested"
            self._finish(TaskState.CANCELLED)
        else:
            self._finish(TaskState.FAILED)

    async def _reap_children(self) -> None:
        """No orphans: cancel children still running and wait for them (SPEC §1.5)."""
        for child in list(self.children):
            if not child.state.terminal:
                child.cancel("parent")
        while True:
            pending = [c._done for c in self.children if not c.state.terminal]
            if not pending:
                return
            try:
                await asyncio.wait(pending)
            except asyncio.CancelledError:
                continue  # a parent never becomes terminal before its children
