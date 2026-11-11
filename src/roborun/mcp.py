"""An MCP server over stdio that connects a harness (Claude Code, Codex, ...) to a host.

``roborun mcp --socket HOST.sock`` is what the harness launches. It is a thin translation
(SPEC §5.1: model-facing tool formats belong to the harness):

- every skill the host exports becomes an MCP tool that **starts a task and returns at
  once** with its id;
- ``task_wait`` waits for a task with a timeout and ``wake_on`` keys, so a model working one
  step at a time notices the world change during a motion;
- ``task_status``, ``task_cancel``, ``task_update``, ``task_list``, ``world_read``,
  ``run_program``, ``task_trace``, ``robot_stop``, ``robot_reset`` act on the session.

The MCP connection is one session: when the harness exits, everything it started stops.
Messages are JSON-RPC 2.0, one per line (MCP's stdio transport). No MCP SDK is needed.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from roborun import __version__
from roborun.harness import Client, HarnessError

PROTOCOL = "2025-06-18"

INSTRUCTIONS = """\
roborun drives a robot through tasks. Tools named after skills start a task and return at
once with its id; the robot keeps moving while you think. Use task_wait to wait for a task,
with a timeout and wake_on keys (e.g. ["cup"]) to be woken when the world changes; then
task_update to redirect a running task, or task_cancel to stop it. world_read returns
stamped values with their age. run_program runs Python that defines `async def main():`
using the same skills, as one task; task_trace explains why a task failed. A rejected
command cancels its task; the reason says why.

world_read(["scene.objects"]) lists what is on the table. Skills accept object names
(grasp(target="cup_red"), place(target="tray")) once the object is tracked
(track(name="cup_red")); a grasp by name follows the object if it moves. In simulation,
sim_push moves an object, as a person would, to test how running tasks react."""


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


TASK = {"task": {"type": "string", "description": "the task id, e.g. 't7'"}}
ACTUATORS = {"actuators": {"type": "array", "items": {"type": "string"},
                           "description": "actuators to act on; all if omitted"}}

CONTROL = {
    "task_status": ("The state of a task, its reason or error, and the current time.",
                    _obj(TASK, ["task"])),
    "task_wait": (("Wait until the task ends, `timeout` seconds pass, or a key in `wake_on` "
                  "changes value; says which (`woke`) and, for a key, its new stamped value."),
                  _obj({**TASK, "timeout": {"type": "number", "default": 5.0},
                        "wake_on": {"type": "array", "items": {"type": "string"}}}, ["task"])),
    "task_cancel": (("Request cancellation. The task passes through `cancelling` while the robot "
                    "stops; wait for it to end."), _obj(TASK, ["task"])),
    "task_update": ("Change a running task's parameters, e.g. a grasp's target.",
                    _obj({**TASK, "params": {"type": "object"}}, ["task", "params"])),
    "task_list": ("This session's tasks that have not ended.", _obj({})),
    "world_read": ("Read World keys: value, stamp, age in seconds, and source.",
                   _obj({"keys": {"type": "array", "items": {"type": "string"}}}, ["keys"])),
    "run_program": (("Run Python code that defines `async def main():` as one task. The skills "
                    "are available by name, along with start, sleep, world, changes."),
                    _obj({"code": {"type": "string"}}, ["code"])),
    "replace_program": (("Replace a running program with new code. It takes effect at the "
                        "program's next step: the step running now finishes first. Returns "
                        "the old and the new task; the new version should read the World to "
                        "see where the old one left off."),
                        _obj({**TASK, "code": {"type": "string"}}, ["task", "code"])),
    "task_trace": (("Why a task (or anything it started) failed: the chain of tasks, the "
                   "rejection or root error, the last commands, and the state at the end."),
                   _obj(TASK, ["task"])),
    "robot_stop": ("Operator stop: stops the actuators and locks them until robot_reset.",
                   _obj(ACTUATORS)),
    "robot_reset": ("Clear an operator stop or a fault.", _obj(ACTUATORS)),
    "sim_push": (("Simulation only: slide an object on the table to (x, y), as a person pushing "
                 "it. It changes the world, not the robot; use it to see how running tasks "
                 "react. Fails if the object is in the gripper or fixed."),
                 _obj({"object": {"type": "string"},
                       "x": {"type": "number"}, "y": {"type": "number"},
                       "duration": {"type": "number", "default": 0.6}}, ["object", "x", "y"])),
}


class Server:
    def __init__(self, client: Client, out: Any = None) -> None:
        self.client = client
        self.out = out or sys.stdout
        self.skills = {t["name"]: t for t in client.tools}

    def tool_list(self) -> list[dict]:
        tools = [
            {"name": name,
             "description": (t["description"] + "\n\n" if t["description"] else "")
             + "Starts a task and returns at once with its id; use task_wait to follow it.",
             "inputSchema": t["schema"]}
            for name, t in self.skills.items()
        ]
        tools += [{"name": n, "description": d, "inputSchema": s} for n, (d, s) in CONTROL.items()]
        return tools

    async def call_tool(self, name: str, args: dict) -> Any:
        c = self.client
        if name in self.skills:
            return await c.start(name, **args)
        if name == "task_status":
            return await c.status(args["task"])
        if name == "task_wait":
            return await c.wait(args["task"], args.get("timeout", 5.0), args.get("wake_on", ()))
        if name == "task_cancel":
            return await c.cancel(args["task"])
        if name == "task_update":
            return await c.update(args["task"], **args["params"])
        if name == "task_list":
            return await c.call("tasks")
        if name == "world_read":
            return await c.world(*args["keys"])
        if name == "run_program":
            return await c.program(args["code"])
        if name == "replace_program":
            return await c.replace(args["task"], args["code"])
        if name == "task_trace":
            return (await c.trace(args["task"]))["text"]
        if name == "robot_stop":
            return await c.operator_stop(args.get("actuators"))
        if name == "robot_reset":
            return await c.reset(args.get("actuators"))
        if name == "sim_push":
            return await c.push(args["object"], (args["x"], args["y"]), args.get("duration", 0.6))
        raise HarnessError(f"no tool {name!r}")

    async def handle(self, msg: dict) -> dict | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # a notification, e.g. notifications/initialized
            return None
        params = msg.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", PROTOCOL),
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "roborun", "version": __version__},
                      "instructions": INSTRUCTIONS}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": self.tool_list()}
        elif method == "tools/call":
            try:
                value = await self.call_tool(params["name"], params.get("arguments") or {})
                text = value if isinstance(value, str) else json.dumps(value)
                result = {"content": [{"type": "text", "text": text}], "isError": False}
            except (HarnessError, KeyError, TypeError) as e:
                result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
        else:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method not found: {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def send(self, msg: dict) -> None:
        self.out.write(json.dumps(msg) + "\n")
        self.out.flush()


async def serve_stdio(socket: str, *, session: str = "mcp") -> None:
    client = await Client.connect(socket, session=session)
    server = Server(client)
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=2**24)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    pending: set[asyncio.Task] = set()

    async def one(msg: dict) -> None:
        reply = await server.handle(msg)
        if reply is not None:
            server.send(reply)

    while line := await reader.readline():
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            server.send({"jsonrpc": "2.0", "id": None,
                         "error": {"code": -32700, "message": f"parse error: {e}"}})
            continue
        t = asyncio.ensure_future(one(msg))  # a long task_wait must not block other calls
        pending.add(t)
        t.add_done_callback(pending.discard)
    for t in pending:
        t.cancel()
    await client.close()
