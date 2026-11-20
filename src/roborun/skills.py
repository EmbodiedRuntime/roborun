"""Skills that do not depend on a backend (SPEC Appendix A): move, grasp, place, pick and place.

A skill here is a fixed sequence of named phases, and what must be true in each is written
as expectations (``roborun.expect``), not as checks inside the motion code:

    grasp   open -> approach -> descend -> close -> lift
    place   approach -> lower -> release -> retreat

    | expectation        | where                    | fails as      |
    |--------------------|--------------------------|---------------|
    | holding something  | place, at the start      | NothingHeld   |
    | grasped            | grasp, end of `close`    | GraspMissed   |
    | holding            | grasp `lift`; place `approach`, `lower` | Dropped |
    | arm not stalled    | every skill that moves   | ExpectationFailed |
    | arm stops          | every skill that moves   | ExpectationFailed |
    | gripper responds   | every skill that grips   | ExpectationFailed |

So a failure says which phase it happened in and shows the measurements, the runtime stops
the arm whatever the skill's code does, and a caller can add expectations of its own to a
phase by name: ``place("tray").expect(consistent("held", phase=("approach", "lower")))``.

A backend says what the skills need to know about it with a ``Robot`` and gets its skills
from ``library(robot)``. The skills use the commands ``goal``, ``stop`` and ``effector``,
and the World keys named in the ``Robot``.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from roborun.api import action, current, hold, now, send, sleep, world
from roborun.errors import StaleData
from roborun.expect import (
    ExpectationFailed,
    Snapshot,
    at_end,
    at_start,
    during,
    gripper_responds,
    no,
    not_stalled,
    phase,
    stops,
)
from roborun.tools import tool


class GraspMissed(ExpectationFailed):
    """The gripper closed on nothing."""

    plain = True


class Dropped(ExpectationFailed):
    """The object slipped out of the gripper before it was put down."""

    plain = True


class NothingHeld(ExpectationFailed):
    """``place`` was asked to put something down, but the gripper holds nothing."""

    plain = True


@dataclass(frozen=True)
class Robot:
    """What the skills need to know about a backend: its actuators, World keys and sizes."""

    arm: str = "arm"
    gripper: str = "gripper"
    ee: str = "arm.ee"  # the position goals are given for
    moving: str = "arm.moving"
    holding: str = "gripper.holding"  # the name of the held object, or None
    width: str = "gripper.width"  # the distance between the fingers
    scene: str | None = None  # a key with {object: {"height": ...}}, used to place on surfaces
    cycle: float = 0.01  # control period of the skills
    reach_tol: float = 0.01  # at a goal
    align_tol: float = 0.01  # over an object, so it is safe to go down
    approach: float = 0.0  # approach and lift height along the last axis; 0 for a planar arm
    open_width: float = 0.075  # fingers at least this far apart are open
    empty_width: float = 0.01  # fingers closer than this have nothing between them
    object_height: float = 0.08  # of a held object the scene does not describe
    max_age: float = 0.3  # a tracked position older than this is not acted on
    stale_wait: float = 3.0  # how long a skill holds still waiting for fresh data
    slip_time: float = 0.05  # how long the gripper must be empty before it counts as a slip
    stall_time: float = 1.0  # how long the arm may sit still away from its goal
    grip_time: float = 2.0  # how long the fingers may take to do what they were told
    fresh: float = 0.1  # max_age of the robot's own state in expectations


def dist(a: Any, b: Any) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _rounded(p: Any) -> tuple:
    return tuple(round(float(v), 3) for v in p)


def library(r: Robot, *, point: Any = tuple, target: Any = Any) -> SimpleNamespace:
    """The skills for robot ``r``: ``move_to``, ``gripper``, ``grasp``, ``place`` and
    ``pick_and_place``, with the expectations they carry and the helpers they are built from.

    ``point`` and ``target`` are the types of a position and of a position or object name;
    they go into the skills' signatures, from which tool schemas are made."""

    # -- expectations ------------------------------------------------------------------

    state = {r.holding: r.fresh, r.width: r.fresh, r.ee: r.fresh}

    def has_something(v: Snapshot) -> Any:
        return v.value(r.holding) is not None or no("the gripper holds nothing to place")

    def closed_on_something(v: Snapshot) -> Any:
        return (v.value(r.holding) is not None
                or no(f"closed on nothing at {_rounded(v.value(r.ee))}"))

    def still_holding(v: Snapshot) -> Any:
        # Contact flickers when the object touches a surface, but the fingers stay apart
        # around it. After a slip they close on nothing.
        if v.value(r.holding) is None and v.value(r.width) < r.empty_width:
            return no(f"the object slipped out of the gripper at {_rounded(v.value(r.ee))}")
        return True

    def holding(*phases: str) -> Any:
        # Empty in every sample for slip_time: the first of them is not yet a duration.
        return during(still_holding, inputs=state, grace=r.slip_time - r.cycle, phase=phases,
                      error=Dropped, name="holding")

    # A goal that the arm neither reaches nor moves toward (blocked, or a controller that
    # gave up) fails the skill, which would otherwise wait for an arrival that never comes.
    moves = (not_stalled(r.arm, position=r.ee, moving=r.moving, tol=r.reach_tol,
                         within=r.stall_time, max_age=r.fresh),
             stops(r.arm, moving=r.moving, max_age=r.fresh))
    grips = (gripper_responds(r.gripper, width=r.width, holding=r.holding,
                              open_width=r.open_width - 0.005, within=r.grip_time,
                              max_age=r.fresh),)
    grasped = at_end(closed_on_something, inputs=state, phase="close", error=GraspMissed,
                     name="grasped")
    something = at_start(has_something, inputs=state, error=NothingHeld,
                         name="holding something")

    # -- helpers -----------------------------------------------------------------------

    def above(p: Any) -> tuple:
        return (*p[:-1], p[-1] + r.approach)

    def level(a: Any, b: Any) -> float:
        """How far apart two points are, ignoring height when the arm approaches from above."""
        return dist(a[:-1], b[:-1]) if r.approach else dist(a, b)

    async def position(target: Any, on_stale: Any = None) -> tuple:
        """A position, or the latest *fresh* tracked position of the object named ``target``.

        A name follows the object (SPEC Appendix A, key-following skills). If its value is
        older than ``max_age`` (the object is occluded, or its tracker stopped), the skill
        does not act on it: ``on_stale()`` is awaited once (e.g. to stop the arm) and it
        waits for fresh data, failing with ``StaleData`` after ``stale_wait`` seconds.
        """
        if not isinstance(target, str):
            return tuple(float(v) for v in target)
        deadline, warned = now() + r.stale_wait, False
        while True:
            sv = world.latest(target)
            if sv is not None and sv.age(now()) <= r.max_age:
                return tuple(sv.value)
            if now() >= deadline:
                raise StaleData(target, None if sv is None else sv.age(now()), r.max_age,
                                None if sv is None else sv.source)
            if not warned and on_stale is not None:
                warned = True
                await on_stale()
            await sleep(r.cycle)

    async def arrive(target: Any) -> None:
        """Wait until the arm is at ``target`` and at rest."""
        while world.get(r.moving).value or dist(world.get(r.ee).value, target) > r.reach_tol:
            await sleep(r.cycle)

    async def wait_stopped() -> None:
        while world.get(r.moving).value:
            await sleep(r.cycle)

    async def stop() -> None:
        await send("stop", r.arm)
        await wait_stopped()

    async def go(target: Any) -> None:
        await send("goal", r.arm, target=target)
        await arrive(target)

    async def set_gripper(close: bool) -> None:
        """Open or close, and return once the fingers have settled. When the object is off
        center, one finger touches it first and stops for a moment while the other is still
        closing, so a single still sample is not the end of the motion."""
        await send("effector", r.gripper, close=close)
        await sleep(5 * r.cycle)  # let the fingers start moving
        # Closing ends when the fingers hold something and the opening is steady (finger
        # speed jitters while squeezing), or when they have closed all the way on nothing.
        # In between a finger may be slowly pushing an off-center object toward the middle,
        # pausing on static friction, so a steady opening alone does not mean done.
        widths: list[float] = []
        for _ in range(250):  # at most 2.5 s
            width = world.get(r.width).value
            widths.append(width)
            steady = len(widths) > 10 and abs(widths[-1] - widths[-11]) < 0.0005
            if close:
                if world.get(r.holding).value is not None and steady:
                    return
                if width < r.empty_width:
                    return
            elif width > r.open_width or steady:
                return
            await sleep(r.cycle)

    async def place_point(target: Any) -> tuple:
        """Where the held object's center goes. On a named surface (e.g. "tray"), that is
        the surface height plus half the object's height."""
        if not isinstance(target, str):
            return tuple(float(v) for v in target)
        surface = await position(target)
        if not r.approach:
            return surface
        held = world.get(r.holding).value
        scene = world.get(r.scene).value if r.scene else {}
        height = scene.get(held, {}).get("height", r.object_height) if held else 0.0
        return (*surface[:-1], surface[-1] + height / 2 + 0.003)

    def skill(resources: set, expect: tuple, name: str | None = None) -> Any:
        def deco(fn: Any) -> Any:
            fn.__qualname__ = fn.__name__ = name or fn.__name__  # the task's name
            kinds = {"point": point, "target": target}
            fn.__annotations__ = {k: kinds.get(v, v) for k, v in fn.__annotations__.items()}
            return tool(action(resources=resources, expect=expect)(fn))

        return deco

    # -- skills ------------------------------------------------------------------------

    @skill({r.arm}, moves)
    async def move_to(target: point) -> point:
        """Move the gripper to ``target`` and end at rest."""
        try:
            await go(target)
            return target
        except asyncio.CancelledError:
            await stop()
            if dist(world.get(r.ee).value, target) <= r.reach_tol:
                return target  # the motion finished before it could be stopped
            raise

    @skill({r.gripper}, grips, name="gripper")
    async def gripper(close: bool) -> str | None:
        """Close (true) or open (false) the gripper and wait until the fingers settle;
        returns the name of the object it holds afterwards, or null."""
        await set_gripper(close)
        return world.get(r.holding).value

    @skill({r.arm, r.gripper}, (*moves, *grips, grasped, holding("lift")))
    async def grasp(target: target) -> point:
        """Pick up an object and lift it.

        ``target`` is an object name (which must be tracked) or a position. With a name the
        grasp follows the object if it moves; ``update(target=...)`` redirects it. Fails
        with GraspMissed if the gripper closes on nothing, and with Dropped if the object
        slips out while it is lifted. While the gripper is not over the object, the arm goes
        to the approach height first, so it never sweeps through it.
        """
        me = current()
        try:
            await phase("open")
            await set_gripper(False)
            while True:
                tgt = await position(me.params["target"], on_stale=stop)
                ee = world.get(r.ee).value
                if level(ee, tgt) <= r.align_tol:
                    await phase("descend")
                    goal = tgt
                    if dist(ee, tgt) <= r.reach_tol and not world.get(r.moving).value:
                        break
                else:
                    await phase("approach")
                    if r.approach and ee[-1] < tgt[-1] + r.approach - 0.03:
                        goal = (*ee[:-1], tgt[-1] + r.approach)  # rise before moving sideways
                    else:
                        goal = above(tgt)
                await send("goal", r.arm, target=goal)
                await sleep(r.cycle)
            await phase("close")
            await set_gripper(True)
            await phase("lift")  # GraspMissed here, before the arm moves, if it holds nothing
            await go(above(tgt))
            return tgt
        except asyncio.CancelledError:
            await stop()
            raise

    @skill({r.arm, r.gripper}, (*moves, *grips, something, holding("approach", "lower")))
    async def place(target: target) -> point:
        """Put down what the gripper holds, release it, and move away.

        ``target`` is a surface name (which must be tracked) or the position where the held
        object's center should end up. Fails at once with NothingHeld if the gripper is
        empty, and with Dropped, stopping the arm, if the object slips out on the way.
        """
        try:
            point = await place_point(target)
            await phase("approach")
            await go(above(point))
            await phase("lower")
            await go(point)
            await phase("release")
            await set_gripper(False)
            await phase("retreat")
            await go(above(point))
            return point
        except asyncio.CancelledError:
            await stop()
            raise

    @skill({r.arm, r.gripper}, ())
    async def pick_and_place(object: target, target: target) -> point:
        """Grasp ``object`` and place it at ``target``, keeping the arm and the gripper for
        the whole sequence. Its phases are ``pick`` and ``place``; a failure is the failure
        of the step it happened in."""
        async with hold(r.arm, r.gripper):
            await phase("pick")
            await grasp(object)
            await phase("place")
            return await place(target)

    return SimpleNamespace(
        robot=r, move_to=move_to, gripper=gripper, grasp=grasp, place=place,
        pick_and_place=pick_and_place,
        moves=moves, grips=grips, grasped=grasped, holding=holding, something=something,
        position=position, arrive=arrive, wait_stopped=wait_stopped, stop=stop, go=go,
        set_gripper=set_gripper, place_point=place_point, above=above, level=level,
    )
