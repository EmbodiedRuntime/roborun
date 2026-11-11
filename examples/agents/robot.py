"""Connect an agent to a robot: an existing ``roborun serve`` socket, or a simulated one here.

Either way the agent gets a harness ``Client``: every call returns at once, the robot keeps
moving while the model thinks, and closing the session stops what it started.
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from roborun.harness import Client, Host

SCENE = {
    "cup_red": {"pos": [0.5, -0.1], "color": "red"},
    "cup_blue": {"pos": [0.42, 0.12], "color": "blue"},
    "tray": {"kind": "tray", "pos": [0.3, 0.38]},
}
CHECKS = {"workspace_box": {"lo": [0.15, -0.6, 0.02], "hi": [0.8, 0.6, 0.8]}}


@contextlib.asynccontextmanager
async def robot(socket: str | None = None, *, pace: float | None = 1.0, view: bool = False,
                session: str = "agent") -> AsyncIterator[Client]:
    """Yield a Client. Without ``socket``, simulate the table scene in this process.

    ``pace=1.0`` runs simulated time at wall speed, so a model that thinks for five seconds
    watches the robot move for five seconds; ``pace=None`` runs as fast as possible (tests).
    """
    if socket is not None:
        client = await Client.connect(socket, session=session)
        try:
            yield client
        finally:
            await client.close()
        return

    from roborun.backends.mujoco import MujocoBackend
    from roborun.checks import from_config

    host = await Host.local(MujocoBackend(objects=SCENE, view=view),
                            tools=["roborun.backends.mujoco"], pace=pace)
    for check in from_config(CHECKS):
        host.rt.gate.add_check(check)
    path = str(Path(tempfile.mkdtemp(prefix="roborun-agent-")) / "host.sock")
    await host.start(path)
    client = await Client.connect(path, session=session)
    try:
        yield client
    finally:
        await client.close()
        await host.close()


def push_later(client: Client, obj: str, to: tuple[float, float], after: float) -> asyncio.Task:
    """Someone pushes ``obj`` to ``to`` after ``after`` seconds of wall time."""

    async def push() -> None:
        await asyncio.sleep(after)
        r = await client.push(obj, to)
        print(f"  [world] someone pushed {obj} to {tuple(r['to'])}")

    return asyncio.ensure_future(push())
