"""A natural scene: a table, two cups, a tray; skills by object name; pushes at run time."""

import asyncio
import tempfile
from pathlib import Path

import pytest

mj = pytest.importorskip("roborun.backends.mujoco")
if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
    pytest.skip("MuJoCo Menagerie Panda model not found", allow_module_level=True)

from roborun import TaskState, run_sim, runtime, sleep, start, world

S = TaskState
SCENE = {
    "cup_red": {"kind": "cup", "pos": [0.5, -0.1], "color": "red"},
    "cup_blue": {"kind": "cup", "pos": [0.42, 0.12], "color": "blue"},
    "tray": {"kind": "tray", "pos": [0.3, 0.38]},
}


def in_tray(cup, tray=(0.3, 0.38)):
    return (abs(cup[0] - tray[0]) < 0.1 - mj.CUP_RADIUS
            and abs(cup[1] - tray[1]) < 0.075 - mj.CUP_RADIUS
            and abs(cup[2] - (0.01 + mj.CUP_HALF_HEIGHT)) < 0.006)


def test_objects_by_name_cup_into_the_tray():
    async def main():
        scene = world.get("scene.objects").value
        start(mj.track("cup_red"), detached=True)
        start(mj.track("tray"), detached=True)
        await mj.grasp("cup_red")
        await mj.place("tray")
        return scene

    backend = mj.MujocoBackend(objects=SCENE)
    task = asyncio.run(run_sim(main, backend=backend, until=20))
    assert task.state is S.SUCCEEDED, task.error
    assert task.result["tray"]["kind"] == "tray" and task.result["cup_red"]["height"] == 0.08
    assert in_tray(backend.truth("cup_red"))
    assert mj.dist(backend.truth("cup_blue")[:2], (0.42, 0.12)) < 0.005  # untouched


def test_a_grasp_by_name_follows_a_pushed_cup_without_update():
    async def main():
        start(mj.track("cup_red"), detached=True)
        g = start(mj.grasp("cup_red"))
        await sleep(1.0)  # still approaching
        runtime().gate.sim("push", object="cup_red", to=(0.45, 0.02))
        await g
        return world.get("gripper.holding").value

    backend = mj.MujocoBackend(objects=SCENE)
    task = asyncio.run(run_sim(main, backend=backend, until=20))
    assert task.state is S.SUCCEEDED, task.error
    assert task.result == "cup_red"


def test_a_push_slides_the_cup_and_cannot_move_a_held_or_fixed_object():
    backend = mj.MujocoBackend(objects=SCENE)

    async def main():
        backend.sim("push", object="cup_blue", to=(0.42, 0.25), duration=0.5)
        await sleep(0.25)
        halfway = world.get("sim.cup_blue").value
        await sleep(0.5)
        with pytest.raises(ValueError, match="fixed"):
            backend.sim("push", object="tray", to=(0, 0))
        with pytest.raises(ValueError, match="no object"):
            backend.sim("push", object="mug", to=(0, 0))
        return halfway, world.get("sim.cup_blue").value

    task = asyncio.run(run_sim(main, backend=backend, until=5))
    halfway, end = task.result
    assert 0.13 < halfway[1] < 0.24  # it slid; it did not jump
    assert mj.dist(end[:2], (0.42, 0.25)) < 0.002 and abs(end[2] - 0.04) < 0.002


def test_an_agent_pushes_the_cup_through_the_runtime_and_the_grasp_follows():
    from roborun.gateproc import spawn_gate
    from roborun.harness import Client, Host

    async def go():
        proc, gate = spawn_gate("roborun.backends.mujoco:MujocoBackend", {"objects": SCENE})
        sock = str(Path(tempfile.mkdtemp(prefix="roborun-test-")) / "host.sock")
        host = await Host.remote(gate, tools=["roborun.backends.mujoco"], pace=None)
        await host.start(sock)
        try:
            c = await Client.connect(sock, session="agent")
            await c.start("track", name="cup_red")
            await c.start("track", name="tray")
            g = await c.start("grasp", target="cup_red")
            await c.wait(g["id"], timeout=1.0)
            pushed = await c.push("cup_red", (0.45, 0.02))
            grasped = await c.wait(g["id"], timeout=15)
            p = await c.start("place", target="tray")
            placed = await c.wait(p["id"], timeout=15)
            cup = (await c.world("cup_red"))["cup_red"]["value"]
            spans = list(host.rt.trace.spans)
            await c.close()
            return pushed, grasped, placed, cup, spans
        finally:
            await host.close()
            proc.wait(timeout=10)

    pushed, grasped, placed, cup, spans = asyncio.run(go())
    assert pushed["object"] == "cup_red"
    assert grasped["state"] == "succeeded" and placed["state"] == "succeeded"
    assert in_tray(cup)
    assert any(s["kind"] == "world" and s["name"] == "sim push cup_red" for s in spans)
