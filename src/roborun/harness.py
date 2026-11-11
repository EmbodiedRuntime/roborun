"""The upper boundary (SPEC §5.1): a long-running runtime that harnesses connect to.

``Host`` owns one runtime: it advances time (in-process or following a Gate process), and
serves harness sessions over a Unix socket. Each connection is a **session**: a session
task under the root, so closing the connection cancels everything the harness started
and the robot stops (SPEC §5.1). Every call returns at once; motion never blocks the
harness:

- ``start(tool, args)`` returns a task id immediately;
- ``status``, ``cancel``, ``update`` act on a task;
- ``wait(id, timeout, wake_on=[keys])`` returns when the task ends, the timeout expires, or
  a key in ``wake_on`` gets a new value, and says which;
- ``world(keys)`` reads stamped values; ``program(code)`` runs generated code as a task
  with provenance; ``trace(id)`` explains its failures;
- ``operator_stop`` and ``reset`` go straight to the Gate.

``Client`` is the Python side of the socket; the MCP adapter (``roborun mcp``) uses it.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import os
import time
from collections.abc import Callable, Iterable
from typing import Any

from roborun import wire
from roborun.clock import SteppedClock
from roborun.errors import InvalidState, RoborunError, StaleData
from roborun.runtime import Runtime, _runtime, settle_with_gate
from roborun.task import Body, Task
from roborun.tools import Tool, collect
from roborun.trace import describe_error, plain, show_failed_spans


class HarnessError(RoborunError):
    """A harness call failed; the message says why and what to do."""


def status(task: Task) -> dict:
    out = {
        "id": task.id, "name": task.name, "state": task.state.value, "session": task.session,
        "started": task.started_at, "ended": task.ended_at, "now": task.rt.clock.now(),
    }
    if task.reason:
        out["reason"] = task.reason
    if task.detail:
        out["detail"] = task.detail
    if task.state.value == "succeeded":
        out["result"] = plain(task.result)
    if task.state.value == "failed":
        out["error"] = describe_error(task.error)
    return out


def _stamped(sv: Any, now: float) -> dict:
    return {"value": plain(sv.value), "stamp": sv.stamp, "age": round(now - sv.stamp, 6),
            "source": sv.source}


# The names a generated program sees, besides the tools (SPEC Appendix A, program helpers).
def _program_namespace(tools: dict[str, Tool]) -> dict[str, Any]:
    import roborun

    ns: dict[str, Any] = {name: getattr(roborun, name) for name in roborun.__all__}
    ns.update({name: t.fn for name, t in tools.items()})
    return ns


class _Session:
    def __init__(self, name: str, task: Task, writer: asyncio.StreamWriter,
                 heartbeat: float | None = None) -> None:
        self.name, self.task, self.writer = name, task, writer
        self.heartbeat = heartbeat  # seconds between the client's pings; None: no watchdog
        self.last_seen = time.monotonic()  # wall time: an agent's liveness is not simulated

    def expired(self) -> bool:
        return self.heartbeat is not None and time.monotonic() - self.last_seen > max(
            0.5, WATCHDOG_MISSED * self.heartbeat)


WATCHDOG_MISSED = 4  # heartbeats a session may miss before the watchdog stops what it started


async def _forever() -> None:
    await asyncio.get_running_loop().create_future()


class Host:
    def __init__(
        self,
        rt: Runtime,
        *,
        step: Callable[[], Any],
        tools: dict[str, Tool] | None = None,
        pace: float | None = None,
        settle: int = 64,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        self.rt, self._step, self.tools, self.pace = rt, step, dict(tools or {}), pace
        self._settle_n, self._on_close = settle, on_close
        self._server: asyncio.AbstractServer | None = None
        self._loop_task: asyncio.Task | None = None
        self._running = False
        self._sessions: set[_Session] = set()
        self._names = itertools.count(1)
        self._programs = itertools.count(1)

    # -- construction --------------------------------------------------------

    @classmethod
    async def local(
        cls,
        backend: Any,
        *,
        dt: float = 0.01,
        tools: Iterable[Any] = (),
        pace: float | None = None,
        trace: Any = None,
    ) -> Host:
        """A host whose Gate runs in this process (tests, quick experiments)."""
        rt = Runtime(backend, SteppedClock(), trace=trace)

        async def step() -> None:
            backend.step(dt)
            rt.clock.advance(dt)

        def close() -> None:
            rt.gate.close()

        return cls(rt, step=step, tools=collect(tools), pace=pace, on_close=close)

    @classmethod
    async def remote(
        cls, gate_path: str, *, tools: Iterable[Any] = (), pace: float | None = 1.0,
        trace: Any = None,
    ) -> Host:
        """A host that follows a Gate process (SPEC §3.1): the robot's one runtime process."""
        from roborun.remote import connect_lockstep

        rt, gate, step = await connect_lockstep(gate_path, session="host", trace=trace)

        async def close() -> None:
            gate.ack(done=True)
            await gate.close()

        return cls(rt, step=step, tools=collect(tools), pace=pace, on_close=close)

    # -- lifecycle -------------------------------------------------------------

    async def start(self, path: str | os.PathLike) -> None:
        # Everything started from here (handlers, tasks) sees this runtime.
        _runtime.set(self.rt)
        self._running = True
        self._loop_task = asyncio.ensure_future(self._run())
        self._server = await asyncio.start_unix_server(self._handle, path=str(path))

    async def close(self) -> None:
        """Stop serving, cancel every task, let the robot stop, and release the Gate."""
        if self._server is not None:
            self._server.close()
            for s in list(self._sessions):
                s.writer.close()
        self._running = False
        if self._loop_task is not None:
            await self._loop_task
        for t in list(self.rt.root.children):
            t.cancel("parent")
        deadline = self.rt.clock.now() + 5.0
        while any(not t.state.terminal for t in self.rt.root.children):
            if self.rt.clock.now() > deadline:
                break
            await settle_with_gate(self.rt.gate, self._settle_n)
            await self._step()
        if self._on_close is not None:
            r = self._on_close()
            if asyncio.iscoroutine(r):
                await r
        self.rt.trace.close()
        if self._server is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._server.wait_closed(), 2.0)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        wall0, sim0 = loop.time(), self.rt.clock.now()
        while self._running:
            for session in [s for s in self._sessions if s.expired()]:
                self._expire(session)
            await settle_with_gate(self.rt.gate, self._settle_n)
            await self._step()
            if self.pace:
                delay = wall0 + (self.rt.clock.now() - sim0) / self.pace - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -0.5:  # fell behind (e.g. a slow step): do not try to catch up
                    wall0 -= delay
            else:
                await asyncio.sleep(0)

    # -- sessions ----------------------------------------------------------------

    def _expire(self, session: _Session) -> None:
        """A session that stopped sending heartbeats is treated as gone (SPEC §5.1): what it
        started is cancelled with reason ``watchdog`` and the connection is closed."""
        self._sessions.discard(session)
        session.task.cancel("watchdog")
        session.writer.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        session: _Session | None = None
        inflight: set[asyncio.Task] = set()
        try:
            while (msg := await wire.read(reader)) is not None:
                if session is not None:
                    session.last_seen = time.monotonic()
                if msg.get("op") == "hello":
                    name = f"{msg.get('session') or 'harness'}-{next(self._names)}"
                    task = self.rt.start(_forever(), detached=True, name=f"session:{name}",
                                         session=name, trace_parent=msg.get("trace_parent"))
                    session = _Session(name, task, writer, heartbeat=msg.get("heartbeat"))
                    self._sessions.add(session)
                    wire.write(writer, {"id": msg["id"], "ok": {"session": name,
                                                                 "tools": self._tool_list()}})
                    continue
                if session is None:
                    wire.write(writer, {"id": msg.get("id"),
                                        "error": {"type": "HarnessError",
                                                  "message": "send hello first"}})
                    continue
                t = asyncio.ensure_future(self._call(session, msg))
                inflight.add(t)
                t.add_done_callback(inflight.discard)
        finally:
            for t in inflight:
                t.cancel()
            if session is not None:
                self._sessions.discard(session)
                session.task.cancel()  # closing the connection stops what it started
            writer.close()

    async def _call(self, session: _Session, msg: dict) -> None:
        try:
            handler = getattr(self, f"_op_{msg.get('op')}", None)
            if handler is None:
                raise HarnessError(f"unknown call {msg.get('op')!r}")
            args = {k: v for k, v in msg.items() if k not in ("op", "id")}
            reply = {"id": msg["id"], "ok": plain(await handler(session, **args))}
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - every failure goes back to the harness
            reply = {"id": msg["id"], "error": {"type": type(e).__name__, "message": str(e)}}
        with contextlib.suppress(Exception):
            wire.write(session.writer, reply)

    # -- calls -------------------------------------------------------------------------

    def _tool_list(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "schema": t.schema}
                for t in self.tools.values()]

    def _task(self, task_id: str) -> Task:
        task = self.rt.task_by_id(task_id)
        if task is None:
            raise HarnessError(f"no task {task_id!r}")
        return task

    def _owned(self, session: _Session, task_id: str) -> Task:
        task = self._task(task_id)
        if task.session != session.name and task.session not in self.rt.supervises.get(
            session.name, ()
        ):
            raise HarnessError(f"task {task_id} belongs to session {task.session!r}")
        return task

    def _launch(self, session: _Session, body: Any, *, detached: bool = False,
                on_busy: str = "wait", wait_timeout: float | None = None,
                trace_parent: str | None = None, name: str | None = None) -> Task:
        return self.rt.start(body, parent=session.task, detached=detached, on_busy=on_busy,
                             wait_timeout=wait_timeout, session=session.name,
                             trace_parent=trace_parent, name=name)

    async def _op_ping(self, session: _Session) -> dict:
        return {}

    async def _op_tools(self, session: _Session) -> list[dict]:
        return self._tool_list()

    async def _op_start(self, session: _Session, tool: str, args: dict | None = None,
                        **opts: Any) -> dict:
        t = self.tools.get(tool)
        if t is None:
            raise HarnessError(f"no tool {tool!r}; tools: {', '.join(sorted(self.tools))}")
        body = t.fn(**(args or {}))
        if not isinstance(body, Body) and not asyncio.iscoroutine(body):
            raise HarnessError(f"tool {tool!r} did not return a task body")
        return status(self._launch(session, body, **opts))

    async def _op_status(self, session: _Session, task: str) -> dict:
        return status(self._task(task))

    async def _op_tasks(self, session: _Session) -> list[dict]:
        return [status(t) for t in self.rt.all_tasks()
                if t.session == session.name and not t.state.terminal and t is not session.task]

    async def _op_cancel(self, session: _Session, task: str) -> dict:
        t = self._owned(session, task)
        t.cancel()
        return status(t)

    async def _op_update(self, session: _Session, task: str, params: dict) -> dict:
        t = self._owned(session, task)
        try:
            t.update(**params)
        except InvalidState as e:
            raise HarnessError(str(e)) from e
        return status(t)

    async def _op_wait(self, session: _Session, task: str, timeout: float | None = None,
                       wake_on: list[str] | None = None) -> dict:
        """Return when the task ends, the timeout expires, or a ``wake_on`` key changes."""
        task = self._task(task)
        if task.state.terminal:
            return {"woke": "done", **status(task)}
        waiters: dict[asyncio.Future, str] = {asyncio.ensure_future(task.stopped()): "done"}
        if timeout is not None:
            waiters[self.rt.clock.timer(self.rt.clock.now() + timeout)] = "timeout"
        for key in wake_on or ():
            waiters[asyncio.ensure_future(self._changed(key))] = key
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for f in waiters:
                if not f.done():
                    f.cancel()
        first = min(done, key=lambda f: list(waiters).index(f))  # done wins a tie
        woke = waiters[first]
        out = {"woke": woke, **status(task)}
        if woke not in ("done", "timeout"):
            if first.exception() is not None:
                out["wake_error"] = str(first.exception())
            else:
                out["value"] = _stamped(first.result(), self.rt.clock.now())
        return out

    async def _changed(self, key: str) -> Any:
        """The next value of ``key`` that differs from its value now (a tracker that keeps
        publishing the same pose does not wake anyone)."""
        before = self.rt.world.latest(key)
        async for sv in self.rt.world.changes(key):
            if before is None or sv.value != before.value:
                return sv
        raise StaleData(key, None, None, None)

    async def _op_world(self, session: _Session, keys: list[str]) -> dict:
        now, out = self.rt.clock.now(), {}
        for key in keys:
            try:
                sv = self.rt.world.get(key)
            except StaleData as e:
                out[key] = {"error": str(e)}
                continue
            out[key] = None if sv is None else _stamped(sv, now)
        return out

    def _compile(self, code: str) -> Callable:
        ns = _program_namespace(self.tools)
        try:
            exec(compile(code, f"<program {next(self._programs)}>", "exec"), ns)  # noqa: S102
        except SyntaxError as e:
            raise HarnessError(f"the program does not compile: {e}") from e
        main = ns.get("main")
        if not asyncio.iscoroutinefunction(main):
            raise HarnessError("a program must define `async def main():`")
        return main

    async def _op_program(self, session: _Session, code: str,
                          generated_by: str | None = None, **opts: Any) -> dict:
        """Run generated code that defines ``async def main()`` as a task (SPEC §1.7)."""
        main = self._compile(code)
        task = self._launch(session, main(), name="program", **opts)
        digest = hashlib.sha256(code.encode()).hexdigest()[:12]
        task.provenance = {"code": f"program:{digest}", "source": code,
                           "generated_by": generated_by or f"harness:{session.name}"}
        return status(task)

    async def _op_replace(self, session: _Session, task: str, code: str,
                          generated_by: str | None = None) -> dict:
        """Replace a running program with new code at its next safe point (SPEC §6)."""
        old = self._owned(session, task)
        if old.state.terminal:
            raise HarnessError(f"task {task} has already ended ({old.state.value}); "
                               "start a new program")
        main = self._compile(code)
        new = self.rt.replace(old, main)
        version = int(old.provenance.get("version", 1)) + 1
        digest = hashlib.sha256(code.encode()).hexdigest()[:12]
        new.provenance = {"code": f"program:{digest}", "source": code, "version": version,
                          "replaces": old.id,
                          "generated_by": generated_by or f"harness:{session.name}"}
        return {"old": status(old), "new": status(new)}

    async def _op_trace(self, session: _Session, task: str) -> dict:
        """The failures in this task's subtree, as text and as span chains."""
        task = self._task(task)
        ids = {task.id}
        stack = list(task.children)
        while stack:
            t = stack.pop()
            ids.add(t.id)
            stack.extend(t.children)
        spans = [s for s in self.rt.trace.spans if s.get("id") in ids or s.get("parent") in ids]
        text, chains = show_failed_spans(spans, title=f"task {task.id} ({task.name})")
        return {"text": text, "failures": chains}

    async def _op_operator_stop(self, session: _Session,
                                actuators: list[str] | None = None) -> dict:
        r = self.rt.gate.operator_stop(actuators)
        if asyncio.iscoroutine(r):
            await r
        return {"stopped": actuators or "all"}

    async def _op_sim(self, session: _Session, action: str, args: dict | None = None) -> dict:
        """Change the simulated world, e.g. push an object (not a robot command)."""
        try:
            r = self.rt.gate.sim(action, **(args or {}))
            if asyncio.iscoroutine(r):
                r = await r
        except ValueError as e:
            raise HarnessError(str(e)) from e
        self.rt.trace.sim(session.task, action, args or {}, r, self.rt.clock.now())
        return r

    async def _op_reset(self, session: _Session, actuators: list[str] | None = None) -> dict:
        r = self.rt.gate.reset(actuators)
        if asyncio.iscoroutine(r):
            await r
        return {"reset": actuators or "all"}


class Client:
    """A harness's connection to a host. Every call returns without waiting for motion."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader, self._writer = reader, writer
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task = asyncio.ensure_future(self._read())
        self._ping_task: asyncio.Task | None = None
        self.session = ""
        self.tools: list[dict] = []

    @classmethod
    async def connect(cls, path: str, *, session: str = "harness",
                      trace_parent: str | None = None, heartbeat: float | None = 0.25) -> Client:
        """``heartbeat`` (seconds) keeps the session alive; if the client stops sending for
        several heartbeats (its process froze or hangs), the host stops what it started.
        ``None`` turns this off: then only a closed connection ends the session."""
        reader, writer = await asyncio.open_unix_connection(path)
        c = cls(reader, writer)
        hello = await c.call("hello", session=session, trace_parent=trace_parent,
                             heartbeat=heartbeat)
        c.session, c.tools = hello["session"], hello["tools"]
        if heartbeat:
            c._ping_task = asyncio.ensure_future(c._ping(heartbeat))
        return c

    async def _ping(self, every: float) -> None:
        while True:
            await asyncio.sleep(every)
            with contextlib.suppress(Exception):
                wire.write(self._writer, {"op": "ping", "id": next(self._ids)})  # reply ignored

    async def _read(self) -> None:
        while (msg := await wire.read(self._reader)) is not None:
            fut = self._pending.pop(msg.get("id"), None)
            if fut is None or fut.done():
                continue
            if "error" in msg:
                fut.set_exception(HarnessError(f"{msg['error']['type']}: {msg['error']['message']}"))
            else:
                fut.set_result(msg["ok"])
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("the host closed the connection"))

    async def call(self, op: str, **args: Any) -> Any:
        if "id" in args:
            raise ValueError("'id' is the request id; name the task argument 'task'")
        rid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        wire.write(self._writer, {"op": op, "id": rid, **args})
        return await fut

    async def start(self, tool: str, **args: Any) -> dict:
        return await self.call("start", tool=tool, args=args)

    async def status(self, task_id: str) -> dict:
        return await self.call("status", task=task_id)

    async def wait(self, task_id: str, timeout: float | None = None,
                   wake_on: Iterable[str] = ()) -> dict:
        return await self.call("wait", task=task_id, timeout=timeout, wake_on=list(wake_on))

    async def cancel(self, task_id: str) -> dict:
        return await self.call("cancel", task=task_id)

    async def update(self, task_id: str, **params: Any) -> dict:
        return await self.call("update", task=task_id, params=params)

    async def world(self, *keys: str) -> dict:
        return await self.call("world", keys=list(keys))

    async def program(self, code: str, **opts: Any) -> dict:
        return await self.call("program", code=code, **opts)

    async def trace(self, task_id: str) -> dict:
        return await self.call("trace", task=task_id)

    async def replace(self, task_id: str, code: str, **opts: Any) -> dict:
        """Replace a running program with new code at its next safe point."""
        return await self.call("replace", task=task_id, code=code, **opts)

    async def operator_stop(self, actuators: Iterable[str] | None = None) -> dict:
        return await self.call("operator_stop",
                               actuators=None if actuators is None else list(actuators))

    async def reset(self, actuators: Iterable[str] | None = None) -> dict:
        return await self.call("reset", actuators=None if actuators is None else list(actuators))

    async def push(self, obj: str, to: Iterable[float], duration: float = 0.6) -> dict:
        """Simulation only: slide ``obj`` to ``to`` (x, y), as a person pushing it."""
        return await self.call("sim", action="push",
                               args={"object": obj, "to": list(to), "duration": duration})

    async def close(self) -> None:
        if self._ping_task is not None:
            self._ping_task.cancel()
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task
