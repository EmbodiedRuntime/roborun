"""The Gate process (SPEC §3.1, §3.8, §2.1): the only process connected to the backend.

It owns the backend, the ownership table, and the capability tokens. In lockstep mode it
also owns simulated time: it steps the backend and sends ``tick(t, observations)``, and steps
again only when the runtime answers ``ack``.

Every request from the runtime gets a reply. A reply (and every tick) carries the events the
Gate raised for that connection: cancellation orders, grants to waiting requests, and
revoked tokens. The Gate revokes a token *before* it tells the runtime to cancel the task.

Run it as ``python -m roborun.gateproc --socket PATH --backend module:Factory``, or start it
from Python with ``spawn_gate``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roborun import wire
from roborun.checks import Monitor, from_config
from roborun.clock import SteppedClock
from roborun.errors import BackendError, ResourceBusy
from roborun.gate import KINDS, Command, Grant, LocalGate
from roborun.runtime import settle_with_gate
from roborun.stamped import Stamped


class _Collector:
    """Stands in for the World inside the Gate process: collects stamped observations."""

    def __init__(self, clock: SteppedClock) -> None:
        self._clock = clock
        self._items: list[tuple] = []
        self._latest: dict[str, Stamped] = {}

    def write(self, key: str, value: Any, *, stamp: float | None = None, source: str = "") -> None:
        # The producer stamps (SPEC §2.1); the runtime keeps this stamp.
        stamp = self._clock.now() if stamp is None else stamp
        self._items.append((key, value, stamp, source))
        self._latest[key] = Stamped(value, stamp, source)

    def latest(self, key: str) -> Stamped | None:
        """What checks see: the latest observation the Gate itself has (SPEC §3.4)."""
        return self._latest.get(key)

    def drain(self) -> list[tuple]:
        out, self._items = self._items, []
        return out


class _Host:
    """What a backend's ``attach`` expects: a clock and somewhere to write observations."""

    def __init__(self, clock: SteppedClock) -> None:
        self.clock = clock
        self.world = _Collector(clock)


class _Config:
    def __init__(self, supervises: dict[str, set[str]]) -> None:
        self.supervises = supervises


class _Owner:
    """A remote task, as the ownership table sees it."""

    parent = None

    def __init__(self, conn: _Conn, task_id: str, name: str, session: str, cfg: _Config):
        self.conn, self.task_id, self.name, self.session, self.rt = conn, task_id, name, session, cfg
        self.detail: dict | None = None

    def cancel(self, reason: str) -> None:
        self.conn.order_cancel(self, reason)


class _Conn:
    def __init__(self, server: GateServer, writer: asyncio.StreamWriter) -> None:
        self.server, self.writer = server, writer
        self.session = "local"
        self.events: list[dict] = []
        self.owners: dict[str, _Owner] = {}
        self.waiters: dict[int, asyncio.Future] = {}

    def owner(self, msg: dict) -> _Owner:
        o = self.owners.get(msg["task"])
        if o is None:
            o = _Owner(self, msg["task"], msg.get("name", msg["task"]), msg["session"],
                       self.server.cfg)
            self.owners[msg["task"]] = o
        return o

    def order_cancel(self, owner: _Owner, reason: str) -> None:
        # Revoke first, so the robot is safe at once; then tell the runtime (SPEC §3.1).
        revoked = self.server.revoke_owner(owner)
        self.events.append({
            "ev": "cancel", "task": owner.task_id, "reason": reason, "revoked": revoked,
            "detail": owner.detail,
        })

    def drain(self) -> list[dict]:
        out, self.events = self.events, []
        return out


class GateServer:
    def __init__(
        self,
        backend: Any,
        *,
        dt: float = 0.01,
        supervises: dict[str, set[str]] | None = None,
        setpoint_timeout: float = 0.1,
    ) -> None:
        self.backend, self.dt = backend, dt
        self.clock = SteppedClock()
        self.host = _Host(self.clock)
        self.cfg = _Config({k: set(v) for k, v in (supervises or {}).items()})
        self.gate = LocalGate(
            backend, self.clock, self.host.world.latest, setpoint_timeout=setpoint_timeout
        )
        self.checks = self.gate.checks
        self.tokens: dict[str, Grant] = {}
        self._token_of: dict[Grant, str] = {}
        self.conns: set[_Conn] = set()
        self._stepper: _Conn | None = None
        self._stepper_ready: asyncio.Event | None = None
        self._ack: asyncio.Future | None = None
        self._server: asyncio.AbstractServer | None = None
        backend.attach(self.host)
        self.gate.listeners.append(self._broadcast)

    def _broadcast(self, ev: Any) -> None:
        """Safety events go to every connected runtime, for its trace (SPEC §4)."""
        msg = {"ev": "safety", "t": ev.t, "kind": ev.kind, "actuators": list(ev.actuators),
               "cause": ev.cause, "detail": ev.detail}
        for c in self.conns:
            c.events.append(dict(msg))

    # -- serving -------------------------------------------------------------

    async def start(self, path: str | os.PathLike) -> None:
        self._stepper_ready = asyncio.Event()
        self._server = await asyncio.start_unix_server(self._handle, path=str(path))

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            for c in list(self.conns):
                c.writer.close()
            self.gate.close()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._server.wait_closed(), 5.0)

    async def run_lockstep(self) -> None:
        """Step the backend each time the lockstep runtime acknowledges a tick (SPEC §2.1)."""
        await self._stepper_ready.wait()
        while self._stepper is not None:
            await self._quiesce()
            await settle_with_gate(self.gate, 8)  # monitors that woke on this step finish first
            conn = self._stepper
            self._ack = asyncio.get_running_loop().create_future()
            wire.write(conn.writer, {
                "op": "tick", "t": self.clock.now(),
                "obs": self.host.world.drain(), "events": conn.drain(),
            })
            ack = await self._ack
            if ack.get("done") or self._stepper is None:
                break
            self.backend.step(self.dt)
            self.clock.advance(self.dt)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = _Conn(self, writer)
        self.conns.add(conn)
        try:
            while (msg := await wire.read(reader)) is not None:
                if msg["op"] == "ack":
                    if self._ack is not None and not self._ack.done():
                        self._ack.set_result(msg)
                    continue
                reply = await self._dispatch(conn, msg)
                await self._quiesce()
                reply["id"] = msg["id"]
                reply["events"] = conn.drain()
                wire.write(writer, reply)
        finally:
            self._disconnect(conn)
            writer.close()

    async def _quiesce(self) -> None:
        # Let grants and cancellations raised inside this process settle before replying.
        for _ in range(8):
            await asyncio.sleep(0)

    # -- requests ---------------------------------------------------------------

    async def _dispatch(self, conn: _Conn, msg: dict) -> dict:
        op = msg["op"]
        if op == "hello":
            conn.session = msg.get("session", "local")
            if msg.get("mode") == "lockstep":
                self._stepper = conn
                self._stepper_ready.set()
            return {"t": self.clock.now()}
        if op == "acquire":
            return await self._acquire(conn, msg)
        if op == "withdraw":
            fut = conn.waiters.pop(msg["waiter"], None)
            if fut is not None:
                fut.cancel()
            return {}
        if op == "release":
            grant = self.tokens.pop(msg["token"], None)
            if grant is not None:
                self._token_of.pop(grant, None)
                self.gate.release(grant)
            return {}
        if op == "revoke":
            grant = self.tokens.get(msg["token"])
            if grant is not None:
                grant.revoked = True
            return {}
        if op == "send":
            return await self._send(msg)
        if op == "operator_stop":
            self.gate.operator_stop(msg.get("actuators"))
            return {}
        if op == "reset":
            self.gate.reset(msg.get("actuators"))
            return {}
        if op == "sim":
            try:
                return {"ok": self.gate.sim(msg["action"], **dict(msg.get("args") or {}))}
            except (ValueError, KeyError, TypeError) as e:
                return {"rejected": "invalid", "message": str(e)}
        return {"rejected": "invalid", "message": f"unknown op {op!r}"}

    async def _acquire(self, conn: _Conn, msg: dict) -> dict:
        owner = conn.owner(msg)
        fut = asyncio.ensure_future(
            self.gate.acquire(
                owner, msg["resources"], on_busy=msg["on_busy"], wait_timeout=msg["wait_timeout"]
            )
        )
        await self._quiesce()
        if fut.done():
            return self._grant_reply(fut)
        wid = msg["id"]
        conn.waiters[wid] = fut

        def granted(f: asyncio.Future) -> None:
            conn.waiters.pop(wid, None)
            if f.cancelled():
                return
            reply = self._grant_reply(f)
            reply.update(ev="grant", waiter=wid)
            conn.events.append(reply)

        fut.add_done_callback(granted)
        return {"queued": True}

    def _grant_reply(self, fut: asyncio.Future) -> dict:
        err = fut.exception()
        if isinstance(err, ResourceBusy):
            return {"busy": str(err)}
        if err is not None:
            raise err
        grant = fut.result()
        token = secrets.token_hex(16)  # unforgeable (SPEC §3.1)
        self.tokens[token] = grant
        self._token_of[grant] = token
        return {"granted": token, "resources": sorted(grant.resources)}

    async def _send(self, msg: dict) -> dict:
        kind, actuator = msg["kind"], msg["actuator"]
        if kind not in KINDS:
            return {"rejected": "invalid", "message": f"unknown command kind {kind!r}"}
        cmd = Command(kind, actuator, dict(msg.get("payload") or {}))
        grant = self.tokens.get(msg.get("token") or "")
        verdicts: list = []
        try:
            rejection = await self.gate.admit(grant, cmd, verdicts)
        except BackendError as e:
            return {"error": {"code": e.code, "message": e.message, "source": e.source,
                              "raw": _plain(e.raw), "stamp": e.stamp}}
        except (ValueError, KeyError, TypeError) as e:
            return {"rejected": "invalid", "message": str(e)}
        if rejection is not None:
            return {"rejected": rejection.code, "message": rejection.message,
                    "verdicts": verdicts}
        return {"ok": True, "verdicts": verdicts}

    def revoke_owner(self, owner: _Owner) -> list[str]:
        self.gate.revoke(owner)
        return [t for t, g in self.tokens.items() if g.owner is owner]

    def _disconnect(self, conn: _Conn) -> None:
        """A closed connection loses every token it held; the robot stops (SPEC §3.1)."""
        self.conns.discard(conn)
        for fut in list(conn.waiters.values()):
            fut.cancel()
        for token, grant in list(self.tokens.items()):
            if getattr(grant.owner, "conn", None) is conn:
                grant.revoked = True
                for actuator in sorted(grant.resources):
                    with contextlib.suppress(Exception):
                        self.backend.execute(Command("stop", actuator, {}))
                del self.tokens[token]
                self._token_of.pop(grant, None)
                self.gate.release(grant)
        if conn is self._stepper:
            self._stepper = None
            if self._ack is not None and not self._ack.done():
                self._ack.set_result({"done": True})


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


async def serve(
    backend: Any,
    path: str | os.PathLike,
    *,
    configure: Callable[[GateServer], None] | None = None,
    **kwargs: Any,
) -> None:
    """Serve one lockstep runtime until it says it is done or disconnects."""
    server = GateServer(backend, **kwargs)
    if configure is not None:
        configure(server)
    await server.start(path)
    try:
        await server.run_lockstep()
    finally:
        await server.close()
        if hasattr(backend, "close"):
            backend.close()  # e.g. the MuJoCo viewer, which crashes if left open at exit


def _factory(spec: str) -> Any:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def spawn_gate(
    backend: str,
    kwargs: dict | None = None,
    *,
    dt: float = 0.01,
    supervises: dict[str, list[str]] | None = None,
    checks: dict[str, dict] | None = None,
    config: str | None = None,
    path: str | None = None,
    timeout: float = 30.0,
) -> tuple[subprocess.Popen, str]:
    """Start a Gate process for ``backend`` (``"module:Factory"``); return it and its socket."""
    if path is None:
        path = str(Path(tempfile.mkdtemp(prefix="roborun-")) / "gate.sock")
    cmd = [sys.executable, "-m", "roborun.gateproc", "--socket", path, "--backend", backend,
           "--kwargs", json.dumps(kwargs or {}), "--dt", str(dt),
           "--supervises", json.dumps(supervises or {}), "--checks", json.dumps(checks or {})]
    if config:
        cmd += ["--config", config]
    proc = subprocess.Popen(cmd)
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if proc.poll() is not None:
            raise RuntimeError(f"Gate process exited with code {proc.returncode}")
        if time.monotonic() > deadline:
            proc.kill()
            raise TimeoutError(f"Gate process did not open {path} within {timeout}s")
        time.sleep(0.02)
    return proc, path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m roborun.gateproc")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--backend", required=True, help="module:Factory, e.g. roborun.backends.mujoco:MujocoBackend")
    ap.add_argument("--kwargs", default="{}", help="JSON keyword arguments for the factory")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--setpoint-timeout", type=float, default=0.1)
    ap.add_argument("--supervises", default="{}", help='JSON, e.g. {"recovery": ["agent"]}')
    ap.add_argument("--checks", default="{}",
                    help='library checks as JSON, e.g. {"workspace_box": {"lo": [...], "hi": [...]}}')
    ap.add_argument("--config", help="module:function called with the GateServer to add checks")
    args = ap.parse_args(argv)
    backend = _factory(args.backend)(**json.loads(args.kwargs))
    supervises = {k: set(v) for k, v in json.loads(args.supervises).items()}

    def configure(server: GateServer) -> None:
        # Checks are configuration, set when the Gate starts (SPEC §3.7).
        for item in from_config(json.loads(args.checks)):
            if isinstance(item, Monitor):
                server.gate.add_monitor(item)
            else:
                server.gate.add_check(item)
        if args.config:
            _factory(args.config)(server)

    asyncio.run(serve(backend, args.socket, dt=args.dt, supervises=supervises,
                      setpoint_timeout=args.setpoint_timeout, configure=configure))


if __name__ == "__main__":
    main()
