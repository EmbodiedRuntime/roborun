"""The runtime's side of the Gate process boundary (SPEC §3.1, §3.8, §2.1).

``RemoteGate`` offers the same interface as the in-process ``LocalGate``, so tasks, ``hold``,
and ``send`` do not know which one they use. The Gate is authoritative for ownership; this
side keeps a mirror of the grants its tasks hold, to find the token for a command.

``run_lockstep`` drives a program against a Gate process in lockstep: it follows the Gate's
ticks, lets every task react, and acknowledges.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from roborun import wire
from roborun.clock import SteppedClock
from roborun.errors import BackendError, ResourceBusy
from roborun.gate import Command


@dataclass(eq=False)
class RemoteGrant:
    owner: Any  # Task
    resources: frozenset[str]
    token: str
    revoked: bool = False


def _is_self_or_ancestor(candidate: Any, task: Any) -> bool:
    node = task
    while node is not None:
        if node is candidate:
            return True
        node = node.parent
    return False


class RemoteGate:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader, self._writer = reader, writer
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._waiters: dict[int, asyncio.Future] = {}
        self._quiet = asyncio.Event()
        self._quiet.set()
        self._ticks: asyncio.Queue = asyncio.Queue()
        self.grants: dict[str, RemoteGrant] = {}
        self.activity = 0  # requests sent so far; the lockstep settle loop watches it
        self.rt: Any = None
        self._reader_task = asyncio.ensure_future(self._read_loop())

    @classmethod
    async def connect(cls, path: str) -> RemoteGate:
        reader, writer = await asyncio.open_unix_connection(path)
        return cls(reader, writer)

    # -- transport -------------------------------------------------------------

    def _request(self, msg: dict) -> asyncio.Future:
        return self._request_with_id(msg)[1]

    def _request_with_id(self, msg: dict) -> tuple[int, asyncio.Future]:
        rid = next(self._ids)
        msg["id"] = rid
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        self._quiet.clear()
        self.activity += 1
        wire.write(self._writer, msg)
        return rid, fut

    async def _read_loop(self) -> None:
        while (msg := await wire.read(self._reader)) is not None:
            if msg.get("op") == "tick":
                await self._ticks.put(msg)
                continue
            self._apply(msg.get("events", ()))
            fut = self._pending.pop(msg["id"], None)
            if fut is not None and not fut.done():
                fut.set_result(msg)
            if not self._pending:
                self._quiet.set()
        # The Gate went away: wake everyone who waits on it.
        for fut in list(self._pending.values()) + list(self._waiters.values()):
            if not fut.done():
                fut.set_exception(ConnectionError("the Gate process closed the connection"))
        self._pending.clear()
        self._quiet.set()
        await self._ticks.put(None)

    def _apply(self, events: Iterable[dict]) -> None:
        for ev in events:
            kind = ev.get("ev")
            if kind == "cancel":
                for token in ev.get("revoked", ()):
                    if token in self.grants:
                        self.grants[token].revoked = True
                task = self.rt.task_by_id(ev["task"]) if self.rt is not None else None
                if task is not None:
                    if ev.get("detail"):
                        task.detail = dict(ev["detail"])
                    task._request_cancel(ev["reason"], interrupt=True)
            elif kind == "safety":
                if self.rt is not None and self.rt.trace is not None:
                    self.rt.trace.safety(ev)
            elif kind == "grant":
                fut = self._waiters.pop(ev["waiter"], None)
                if fut is not None and not fut.done():
                    fut.set_result(ev)

    @property
    def outstanding(self) -> int:
        return len(self._pending)

    async def quiet(self) -> None:
        await self._quiet.wait()

    async def next_tick(self) -> dict | None:
        return await self._ticks.get()

    async def hello(self, *, session: str = "local", mode: str = "lockstep") -> dict:
        return await self._request({"op": "hello", "session": session, "mode": mode})

    async def close(self) -> None:
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()
        self._reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader_task

    def ack(self, *, done: bool = False) -> None:
        wire.write(self._writer, {"op": "ack", "done": done})

    # -- the LocalGate interface -------------------------------------------------

    def held_by_ancestors(self, task: Any, resources: Iterable[str]) -> set[str]:
        held = set()
        for g in self.grants.values():
            if not g.revoked and _is_self_or_ancestor(g.owner, task):
                held |= g.resources & set(resources)
        return held

    async def acquire(
        self,
        task: Any,
        resources: Iterable[str],
        *,
        on_busy: str = "wait",
        wait_timeout: float | None = None,
    ) -> RemoteGrant:
        resources = frozenset(resources)
        rid, fut = self._request_with_id({
            "op": "acquire", "task": task.id, "name": task.name, "session": task.session,
            "resources": sorted(resources), "on_busy": on_busy, "wait_timeout": wait_timeout,
        })
        waiter = asyncio.get_running_loop().create_future()
        self._waiters[rid] = waiter  # registered before any grant event can arrive
        try:
            reply = await fut
            if reply.get("queued"):
                reply = await waiter
            else:
                self._waiters.pop(rid, None)
        except asyncio.CancelledError:
            self._waiters.pop(rid, None)
            self._request({"op": "withdraw", "waiter": rid})
            if waiter.done() and not waiter.cancelled() and "granted" in waiter.result():
                self._request({"op": "release", "token": waiter.result()["granted"]})
            raise
        if "busy" in reply:
            raise ResourceBusy(reply["busy"])
        if "granted" not in reply:
            raise ResourceBusy(reply.get("message", "acquire failed"))
        grant = RemoteGrant(task, resources, reply["granted"])
        self.grants[grant.token] = grant
        return grant

    def release(self, grant: RemoteGrant) -> None:
        if self.grants.pop(grant.token, None) is not None:
            self._request({"op": "release", "token": grant.token})

    def release_all(self, task: Any) -> None:
        for g in [g for g in self.grants.values() if g.owner is task]:
            self.release(g)

    def revoke(self, task: Any) -> None:
        for g in self.grants.values():
            if g.owner is task and not g.revoked:
                g.revoked = True
                self._request({"op": "revoke", "token": g.token})

    async def operator_stop(self, actuators: Iterable[str] | None = None) -> None:
        """An operator stop: locks the actuators until ``reset`` (SPEC §3.5)."""
        await self._request({"op": "operator_stop",
                             "actuators": None if actuators is None else sorted(actuators)})

    async def reset(self, actuators: Iterable[str] | None = None) -> None:
        await self._request({"op": "reset",
                             "actuators": None if actuators is None else sorted(actuators)})

    async def sim(self, action: str, **kwargs: Any) -> dict:
        """Change the simulated world in the Gate process (e.g. push an object)."""
        reply = await self._request({"op": "sim", "action": action, "args": kwargs})
        if "rejected" in reply:
            raise ValueError(reply.get("message", "rejected"))
        return reply["ok"]

    def owner(self, actuator: str) -> Any:
        for g in self.grants.values():
            if actuator in g.resources:
                return g.owner
        return None

    def _token_for(self, task: Any, actuator: str) -> str | None:
        node = task
        while node is not None:
            for g in self.grants.values():
                if g.owner is node and not g.revoked and actuator in g.resources:
                    return g.token
            node = node.parent
        return None

    async def send(self, task: Any, cmd: Command, verdicts: list | None = None) -> bool:
        token = self._token_for(task, cmd.actuator) if task is not None else None
        reply = await self._request({
            "op": "send", "token": token, "kind": cmd.kind, "actuator": cmd.actuator,
            "payload": cmd.payload,
        })
        if verdicts is not None:
            verdicts.extend(dict(v) for v in reply.get("verdicts", ()))
        if "error" in reply:
            e = reply["error"]
            raise BackendError(e["code"], e["message"], source=e.get("source", ""),
                               raw=e.get("raw"), stamp=e.get("stamp"))
        if "rejected" in reply:
            if task is not None:
                task.detail = {"code": reply["rejected"], "message": reply.get("message", "")}
            return False
        return True


async def connect_lockstep(
    path: str, *, session: str = "local", trace: Any = None
) -> tuple[Any, RemoteGate, Any]:
    """Connect a new runtime to the Gate process at ``path`` as its lockstep stepper.

    Returns the runtime, its ``RemoteGate``, and ``step()``: acknowledge the current tick
    and apply the next one.
    """
    from roborun.runtime import Runtime

    gate = await RemoteGate.connect(path)
    clock = SteppedClock()
    rt = Runtime(None, clock, gate=gate, trace=trace)
    gate.rt = rt
    await gate.hello(session=session, mode="lockstep")

    def apply(tick: dict) -> None:
        clock.advance(tick["t"] - clock.now())
        for key, value, stamp, source in tick["obs"]:
            rt.world.write(key, value, stamp=stamp, source=source)  # never re-stamped
        gate._apply(tick["events"])

    first = await gate.next_tick()
    if first is None:
        raise ConnectionError("the Gate process closed before the first tick")
    apply(first)

    async def step() -> None:
        gate.ack()
        tick = await gate.next_tick()
        if tick is None:
            raise ConnectionError("the Gate process closed the connection")
        apply(tick)

    return rt, gate, step


async def run_lockstep(
    main: Any,
    path: str,
    *,
    until: float = 30.0,
    session: str = "local",
    settle: int = 64,
    trace: str | None = None,
) -> Any:
    """Run ``main`` against the Gate process at ``path``, following its ticks (SPEC §2.1).

    Between ticks every task reacts, including to replies from the Gate, before the
    runtime acknowledges; results are as deterministic as with ``run_sim``.
    """
    from roborun.runtime import _drive, settle_with_gate

    rt, gate, step = await connect_lockstep(path, session=session, trace=trace)

    async def settle_all() -> None:
        await settle_with_gate(gate, settle)

    try:
        with rt.active():
            task = await _drive(rt, main, settle=settle_all, step=step, until=until)
        gate.ack(done=True)
    finally:
        await gate.close()
        if rt.trace is not None:
            rt.trace.close()
    return task
