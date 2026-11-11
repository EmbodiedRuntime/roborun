"""End to end: a harness speaks MCP to `roborun mcp`, which drives `roborun serve`."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


class Proc:
    def __init__(self, *args):
        self.p = subprocess.Popen([sys.executable, "-m", "roborun", *args], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.next_id = 0

    def rpc(self, method, params=None):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        assert line, self.p.stderr.read()
        reply = json.loads(line)
        assert reply["id"] == self.next_id
        return reply

    def notify(self, method):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.p.stdin.flush()

    def tool(self, name, **arguments):
        r = self.rpc("tools/call", {"name": name, "arguments": arguments})["result"]
        text = r["content"][0]["text"]
        try:
            return r["isError"], json.loads(text)
        except json.JSONDecodeError:
            return r["isError"], text

    def close(self):
        if self.p.stdin:
            self.p.stdin.close()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()


def wait_for(path, proc, timeout=20):
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        assert proc.poll() is None, proc.stderr.read()
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)


@pytest.fixture
def served():
    d = Path(tempfile.mkdtemp(prefix="roborun-test-"))
    host, gate = str(d / "host.sock"), str(d / "gate.sock")
    server = subprocess.Popen(
        [sys.executable, "-m", "roborun", "serve", "--backend", "roborun.backends.fake:FakeBackend",
         "--kwargs", json.dumps({"objects": {"cup": [0.5, 0.0]}}), "--tools",
         "roborun.backends.fake", "--socket", host, "--gate-socket", gate, "--pace", "4",
         "--trace", str(d / "run.jsonl")],
        stderr=subprocess.PIPE, text=True)
    wait_for(host, server)
    yield host, gate, d
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()


def test_a_harness_drives_the_robot_over_mcp(served):
    host, _gate, _d = served
    mcp = Proc("mcp", "--socket", host)
    try:
        init = mcp.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                      "clientInfo": {"name": "test", "version": "0"}})
        assert init["result"]["serverInfo"]["name"] == "roborun"
        assert "task_wait" in init["result"]["instructions"]
        mcp.notify("notifications/initialized")
        tools = {t["name"]: t for t in mcp.rpc("tools/list")["result"]["tools"]}
        assert {"move_to", "grasp", "task_wait", "run_program", "replace_program",
                "robot_stop", "sim_push"} <= set(tools)
        assert tools["move_to"]["inputSchema"]["required"] == ["target"]

        err, started = mcp.tool("move_to", target=[0.2, 0.0])
        assert not err and started["state"] in ("pending", "running")
        _, done = mcp.tool("task_wait", task=started["id"], timeout=10)
        assert done["state"] == "succeeded" and done["woke"] == "done"

        err, world = mcp.tool("world_read", keys=["arm.ee"])
        assert world["arm.ee"]["value"] == [0.2, 0.0] and world["arm.ee"]["age"] < 0.1

        code = "async def main():\n    await grasp_blocking((0.9, 0.9))\n"
        err, prog = mcp.tool("run_program", code=code)
        err, done = mcp.tool("task_wait", task=prog["id"], timeout=20)
        assert done["state"] == "failed"
        err, why = mcp.tool("task_trace", task=prog["id"])
        assert "GraspMissed" in why and "grasp_blocking" in why

        err, pushed = mcp.tool("sim_push", object="cup", x=0.4, y=0.2)
        assert not err and pushed["to"] == [0.4, 0.2]
        for _ in range(50):  # the World sees it on the next tick
            err, world = mcp.tool("world_read", keys=["sim.cup"])
            if world["sim.cup"]["value"] == [0.4, 0.2]:
                break
            time.sleep(0.02)
        assert world["sim.cup"]["value"] == [0.4, 0.2]

        err, msg = mcp.tool("fly")
        assert err and "no tool" in msg
    finally:
        mcp.close()


def test_operator_stop_bypasses_the_harness_and_goes_to_the_gate(served):
    host, gate, _d = served
    mcp = Proc("mcp", "--socket", host)
    try:
        mcp.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "test", "version": "0"}})
        _, started = mcp.tool("move_to", target=[5.0, 0.0])  # long: 10 s at 0.5 m/s
        out = subprocess.run([sys.executable, "-m", "roborun", "stop", "--gate", gate, "arm"],
                             check=False, capture_output=True, text=True, timeout=20)
        assert out.returncode == 0, out.stderr
        _, done = mcp.tool("task_wait", task=started["id"], timeout=10)
        assert (done["state"], done["reason"]) == ("cancelled", "operator")
        _, refused = mcp.tool("move_to", target=[0.0, 0.0])
        _, refused = mcp.tool("task_wait", task=refused["id"], timeout=5)
        assert refused["state"] == "failed"
        subprocess.run([sys.executable, "-m", "roborun", "reset", "--gate", gate],
                       check=True, capture_output=True, timeout=20)
        _, again = mcp.tool("move_to", target=[0.0, 0.0])
        _err, again = mcp.tool("task_wait", task=again["id"], timeout=20)
        assert again["state"] == "succeeded"
    finally:
        mcp.close()
