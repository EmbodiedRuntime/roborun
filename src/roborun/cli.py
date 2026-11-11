"""The ``roborun`` command line.

    roborun serve --backend roborun.backends.mujoco:MujocoBackend --tools roborun.backends.mujoco \\
                  --socket /tmp/roborun.sock [--kwargs JSON] [--checks JSON] [--pace 1.0]
        Start the robot's Gate process and its runtime; harnesses connect to --socket.
    roborun mcp --socket /tmp/roborun.sock
        An MCP server over stdio for a harness such as Claude Code.
    roborun demo cup-nudge [--benchmark] [--gif OUT.gif] [--view]
    roborun sim push --socket HOST.sock cup_red 0.45 0.1   push an object (simulation only)
    roborun stop --gate GATE.sock [ACTUATOR ...]     operator stop, straight to the Gate
    roborun reset --gate GATE.sock [ACTUATOR ...]
    roborun trace show [--failed] [--json] RUN.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import tempfile
from pathlib import Path

from roborun import trace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="roborun")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sv = sub.add_parser("serve", help="run the robot: a Gate process and a runtime host")
    sv.add_argument("--backend", required=True, help="module:Factory")
    sv.add_argument("--kwargs", default="{}", help="JSON keyword arguments for the backend")
    sv.add_argument("--tools", default="", help="comma-separated modules whose @tool skills to export")
    sv.add_argument("--socket", required=True, help="where harnesses connect")
    sv.add_argument("--gate-socket", help="the Gate's socket (default: next to --socket)")
    sv.add_argument("--checks", default="{}", help="library checks and monitors as JSON")
    sv.add_argument("--config", help="module:function that adds checks to the Gate")
    sv.add_argument("--pace", type=float, default=1.0,
                    help="simulated seconds per wall second; 0 runs as fast as possible")
    sv.add_argument("--dt", type=float, default=0.01)
    sv.add_argument("--trace", help="write the trace here (JSON Lines)")

    mc = sub.add_parser("mcp", help="MCP server over stdio, connected to a host")
    mc.add_argument("--socket", required=True)
    mc.add_argument("--session", default="mcp")

    for name in ("stop", "reset"):
        p = sub.add_parser(name, help=f"operator {name}, sent straight to the Gate")
        p.add_argument("--gate", required=True, help="the Gate's socket")
        p.add_argument("actuators", nargs="*")

    dm = sub.add_parser("demo", help="run a demo")
    dms = dm.add_subparsers(dest="demo", required=True)
    cn = dms.add_parser("cup-nudge", help="someone pushes the cup while the robot reaches for it")
    cn.add_argument("--benchmark", action="store_true", help="sweep push times and directions")
    cn.add_argument("--out", help="with --benchmark: write the results as JSON")
    cn.add_argument("--gif", help="render the blocking and task-based programs side by side")
    cn.add_argument("--view", action=argparse.BooleanOptionalAction, default=None,
                    help="open the MuJoCo viewer (default: if a display is available)")

    sm = sub.add_parser("sim", help="change the simulated world through the runtime")
    sms = sm.add_subparsers(dest="sim_cmd", required=True)
    push = sms.add_parser("push", help="slide an object to (x, y), as a person pushing it")
    push.add_argument("--socket", required=True, help="the host's socket (as for roborun mcp)")
    push.add_argument("object")
    push.add_argument("x", type=float)
    push.add_argument("y", type=float)
    push.add_argument("--duration", type=float, default=0.6)

    tr = sub.add_parser("trace", help="inspect a run's trace")
    trs = tr.add_subparsers(dest="trace_cmd", required=True)
    show = trs.add_parser("show", help="print a trace")
    show.add_argument("path")
    show.add_argument("--failed", action="store_true", help="only the causal chain of each failure")
    show.add_argument("--json", action="store_true", help="print failure chains as JSON")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        return asyncio.run(_serve(args))
    if args.cmd == "mcp":
        from roborun.mcp import serve_stdio

        asyncio.run(serve_stdio(args.socket, session=args.session))
        return 0
    if args.cmd == "demo":
        from roborun.demos import cup_nudge

        return cup_nudge.main(args)
    if args.cmd == "sim":
        return asyncio.run(_push(args))
    if args.cmd in ("stop", "reset"):
        return asyncio.run(_operator(args.cmd, args.gate, args.actuators or None))
    if args.cmd == "trace" and args.trace_cmd == "show":
        if args.json:
            _, spans = trace.load(args.path)
            print(json.dumps(trace.failures(spans), indent=2))
        elif args.failed:
            print(trace.show_failed(args.path))
        else:
            print(trace.show_tree(args.path))
        return 0
    return 2


async def _serve(args: argparse.Namespace) -> int:
    from roborun.gateproc import spawn_gate
    from roborun.harness import Host

    gate_socket = args.gate_socket or str(
        Path(tempfile.mkdtemp(prefix="roborun-")) / "gate.sock"
    )
    proc, gate_socket = spawn_gate(
        args.backend, json.loads(args.kwargs), dt=args.dt, checks=json.loads(args.checks),
        config=args.config, path=gate_socket,
    )
    tools = [m for m in args.tools.split(",") if m]
    host = await Host.remote(gate_socket, tools=tools, pace=args.pace or None, trace=args.trace)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(args.socket)
    await host.start(args.socket)
    print(f"roborun: harnesses connect to {args.socket}; operator stop: "
          f"roborun stop --gate {gate_socket}", file=sys.stderr, flush=True)
    done = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, done.set)
    await done.wait()
    await host.close()
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        proc.kill()
    return 0


async def _push(args: argparse.Namespace) -> int:
    from roborun.harness import Client

    c = await Client.connect(args.socket, session="operator")
    try:
        r = await c.push(args.object, (args.x, args.y), args.duration)
    finally:
        await c.close()
    print(f"pushed {r['object']} from {tuple(round(v, 3) for v in r['from'])} "
          f"to {tuple(round(v, 3) for v in r['to'])}")
    return 0


async def _operator(cmd: str, gate: str, actuators: list[str] | None) -> int:
    """The stop channel that bypasses models (SPEC §5.1): straight to the Gate process."""
    from roborun.remote import RemoteGate

    g = await RemoteGate.connect(gate)
    await g.hello(session="operator", mode="client")
    await (g.operator_stop(actuators) if cmd == "stop" else g.reset(actuators))
    await g.close()
    print(f"{cmd}: {', '.join(actuators) if actuators else 'all actuators'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
