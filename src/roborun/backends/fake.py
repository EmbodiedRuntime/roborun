"""A fake kinematic backend for tests and examples: one planar arm, a gripper, and objects.

It has just enough physics to make the runtime's semantics observable:

- the arm moves toward its target at a fixed speed;
- stopping takes ``stop_time`` seconds, during which the arm keeps moving at half speed
  (so ``cancelling`` is not ``cancelled``);
- objects can be nudged at a given time;
- a closed gripper holds the object under it, and the object moves with the arm.

World keys written every step: ``arm.ee``, ``arm.moving``, ``gripper.holding``, and
``sim.<object>`` (ground truth, read by the ``track`` provider).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from typing import Any

from roborun.api import action, provider, send, sleep, world
from roborun.gate import Command
from roborun.tools import tool

Point = tuple[float, float]

CYCLE = 0.01  # control period of the skills below
REACH_TOL = 0.005


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FakeBackend:
    actuators = ("arm", "gripper")
    def __init__(
        self,
        *,
        ee: Point = (0.0, 0.0),
        objects: dict[str, Point] | None = None,
        speed: float = 0.5,
        stop_time: float = 0.1,
        grip_tol: float = 0.02,
        nudges: list[tuple[str, Point, float]] | None = None,
    ) -> None:
        self.ee = tuple(ee)
        self.objects = {k: tuple(v) for k, v in (objects or {}).items()}
        self.speed = speed
        self.stop_time = stop_time
        self.grip_tol = grip_tol
        self.holding: str | None = None
        self.log: list[tuple[float, Command]] = []
        self._target: Point | None = None
        self._stopping_until: float | None = None
        self._schedule: list[tuple[float, Callable[[], None]]] = []
        self.rt = None
        for name, to, t in nudges or ():
            self.nudge(name, tuple(to), at=t)

    # -- lifecycle -----------------------------------------------------------

    def attach(self, rt: Any) -> None:
        self.rt = rt
        self._publish()

    def now(self) -> float:
        return self.rt.clock.now()

    @property
    def moving(self) -> bool:
        return self._target is not None and dist(self.ee, self._target) > 1e-9

    # -- test controls ---------------------------------------------------------

    def at(self, t: float, fn: Callable[[], None]) -> None:
        self._schedule.append((t, fn))

    def nudge(self, name: str, to: Point, at: float) -> None:
        self.at(at, lambda: self.objects.__setitem__(name, to))

    def sim(self, action: str, **kwargs: Any) -> dict:
        """Change the simulated world: ``push`` moves an object to ``to`` at once."""
        if action != "push":
            raise ValueError(f"unknown simulation action {action!r}; known: push")
        name, to = kwargs["object"], tuple(float(v) for v in kwargs["to"])
        if name not in self.objects:
            raise ValueError(f"no object {name!r}; objects: {', '.join(sorted(self.objects))}")
        if self.holding == name:
            raise ValueError(f"{name!r} is in the gripper")
        before = self.objects[name]
        self.objects[name] = to
        return {"object": name, "from": list(before), "to": list(to)}

    def truth(self, name: str) -> Point:
        return self.objects[name]

    # -- called by the Gate ----------------------------------------------------

    def execute(self, cmd: Command) -> None:
        self.log.append((self.now(), cmd))
        if cmd.kind in ("goal", "setpoint"):
            self._target = tuple(cmd.payload["target"])
            self._stopping_until = None
        elif cmd.kind == "stop":
            if self._target is not None and self._stopping_until is None:
                self._stopping_until = self.now() + self.stop_time
        elif cmd.kind == "effector":
            if cmd.payload.get("close"):
                near = [n for n, p in self.objects.items() if dist(p, self.ee) <= self.grip_tol]
                self.holding = near[0] if near else None
            else:
                self.holding = None

    # -- physics ---------------------------------------------------------------

    def step(self, dt: float) -> None:
        now = self.now()
        due = [fn for t, fn in self._schedule if t <= now + 1e-9]
        self._schedule = [(t, fn) for t, fn in self._schedule if t > now + 1e-9]
        for fn in due:
            fn()
        speed = self.speed
        if self._stopping_until is not None:
            if now >= self._stopping_until - 1e-9:
                self._target = None
                self._stopping_until = None
            else:
                speed *= 0.5  # decelerating: still moving
        if self._target is not None:
            d = dist(self.ee, self._target)
            stride = speed * dt
            if d <= stride:
                self.ee = self._target
            else:
                f = stride / d
                self.ee = (
                    self.ee[0] + (self._target[0] - self.ee[0]) * f,
                    self.ee[1] + (self._target[1] - self.ee[1]) * f,
                )
        if self.holding is not None:
            self.objects[self.holding] = self.ee
        self._publish()

    def _publish(self) -> None:
        w = self.rt.world
        w.write("arm.ee", self.ee, source="fake.arm")
        w.write("arm.moving", self.moving, source="fake.arm")
        w.write("gripper.holding", self.holding, source="fake.gripper")
        for name, pos in self.objects.items():
            w.write(f"sim.{name}", pos, source="fake.sim")  # ground truth, like a mocap system


# -- example skills and providers on the fake backend ---------------------------


class GraspMissed(Exception):
    pass


async def _arrive(target: Point) -> None:
    """Wait until the arm is at ``target`` and no longer moving."""
    while world.get("arm.moving").value or dist(world.get("arm.ee").value, target) > REACH_TOL:
        await sleep(CYCLE)


async def wait_stopped() -> None:
    while world.get("arm.moving").value:
        await sleep(CYCLE)


@tool
@provider
async def track(name: str, rate: float = 20.0) -> None:
    """A perfect tracker: republishes the simulator's ground truth at ``rate`` Hz."""
    while True:
        truth = world.get(f"sim.{name}")
        world.write(name, truth.value, stamp=truth.stamp, source=f"track({name})")
        await sleep(1.0 / rate)


@tool
@action(resources={"arm"})
async def move_to(target: Point) -> Point:
    try:
        await send("goal", "arm", target=target)
        await _arrive(target)
        return target
    except asyncio.CancelledError:
        await send("stop", "arm")
        await wait_stopped()
        if dist(world.get("arm.ee").value, target) <= REACH_TOL:
            return target  # the motion finished before it could be stopped
        raise


def _me() -> Any:
    from roborun.api import current

    return current()


@tool
@action(resources={"arm", "gripper"})
async def grasp(target: Point) -> Point:
    """Reach for ``target`` and close the gripper. ``update(target=...)`` redirects it."""
    me = _me()
    try:
        while True:
            tgt = me.params["target"]
            await send("goal", "arm", target=tgt)
            if dist(world.get("arm.ee").value, tgt) <= REACH_TOL:
                break
            await sleep(CYCLE)
        await send("effector", "gripper", close=True)
        await sleep(CYCLE)
        if world.get("gripper.holding").value is None:
            raise GraspMissed(f"closed on nothing at {tgt}")
        return tgt
    except asyncio.CancelledError:
        await send("stop", "arm")
        await wait_stopped()
        raise


@tool
@action(resources={"arm", "gripper"})
async def grasp_blocking(target: Point) -> Point:
    """The same grasp as one blocking call: the target is fixed when it starts."""
    await send("goal", "arm", target=target)
    while dist(world.get("arm.ee").value, target) > REACH_TOL:
        await sleep(CYCLE)
    await send("effector", "gripper", close=True)
    await sleep(CYCLE)
    if world.get("gripper.holding").value is None:
        raise GraspMissed(f"closed on nothing at {target}")
    return target


@tool
@action(resources={"arm", "gripper"})
async def place(target: Point) -> Point:
    await send("goal", "arm", target=target)
    await _arrive(target)
    await send("effector", "gripper", close=False)
    await sleep(CYCLE)
    return target
