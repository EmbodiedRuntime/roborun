"""A MuJoCo backend: a Franka Panda arm, its gripper, and free objects on the floor.

The robot model comes from MuJoCo Menagerie (Apache-2.0) and is referenced, not copied
(SPEC §5.2). Call ``fetch_menagerie()`` once, or point ``ROBORUN_MENAGERIE`` at a checkout.

The controller belongs to the backend, below the Gate (SPEC §3):

- a ``goal`` or ``setpoint`` is a TCP position; a Cartesian reference moves toward it with
  bounded speed and acceleration, so stopping takes time and the arm keeps moving while it
  brakes (``cancelling`` is not ``cancelled``);
- differential IK (damped least squares, orientation held pointing down, null-space pull
  toward the home posture) runs on a separate kinematic copy of the model and produces joint
  targets for the Panda's position actuators;
- ``effector`` opens or closes the gripper.

World keys written every step: ``arm.ee`` (TCP position), ``arm.q``, ``arm.moving``,
``gripper.width``, ``gripper.moving``, ``gripper.holding`` (the object touching both fingers,
or None), and ``sim.<object>`` (ground truth, read by the ``track`` provider).
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from roborun.api import action, current, now, provider, send, sleep, world
from roborun.errors import BackendError, StaleData
from roborun.gate import Command
from roborun.tools import tool

Point = tuple[float, float, float]

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034  # from the hand frame to the point between the fingertips
SHOULDER = np.array([0.0, 0.0, 0.333])
REACH = 0.80  # a conservative reach from the shoulder for a downward-pointing gripper
CUP_RADIUS, CUP_HALF_HEIGHT = 0.02, 0.04
GRIP_SCALE = 10.0  # gripper gain and stiffness relative to the Menagerie model
PAD_FRICTION = 1.0  # rubber finger pads on a ceramic cup


def menagerie_path() -> Path:
    return Path(
        os.environ.get("ROBORUN_MENAGERIE", Path.home() / ".cache/roborun/mujoco_menagerie")
    )


def fetch_menagerie(path: Path | None = None) -> Path:
    """Sparse-clone the Panda model from MuJoCo Menagerie (about 36 MB)."""
    path = Path(path or menagerie_path())
    if not (path / "franka_emika_panda/panda.xml").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             MENAGERIE_URL, str(path)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "sparse-checkout", "set", "franka_emika_panda"], check=True
        )
    return path


def _panda_xml() -> Path:
    xml = menagerie_path() / "franka_emika_panda/panda.xml"
    if not xml.exists():
        raise FileNotFoundError(
            f"Panda model not found at {xml}. Run roborun.backends.mujoco.fetch_menagerie(), "
            f"or set ROBORUN_MENAGERIE to a mujoco_menagerie checkout."
        )
    return xml


def dist(a: Any, b: Any) -> float:
    return math.dist(a, b)


def _rot_error(r: np.ndarray, r_des: np.ndarray) -> np.ndarray:
    """Small-angle orientation error that rotates ``r`` toward ``r_des``."""
    return 0.5 * sum(np.cross(r[:, i], r_des[:, i]) for i in range(3))


COLORS = {
    "red": [0.84, 0.22, 0.18, 1], "blue": [0.18, 0.4, 0.82, 1], "green": [0.25, 0.65, 0.3, 1],
    "yellow": [0.95, 0.78, 0.2, 1], "white": [0.92, 0.92, 0.9, 1], "grey": [0.6, 0.62, 0.66, 1],
}
TABLE_TOP = 0.0  # the robot is mounted on the table; coordinates are relative to its top
TRAY_SIZE, TRAY_BASE, TRAY_RIM = (0.2, 0.15), 0.01, 0.025
BLOCK = 0.025  # half the side of a block


def _object_spec(raw: Any) -> dict:
    """``(x, y)`` is a red cup; otherwise ``{"kind": "cup" | "block" | "tray", "pos": (x, y),
    "color": ...}``."""
    if isinstance(raw, dict):
        out = {"kind": "cup", "color": None, **raw}
    else:
        out = {"kind": "cup", "pos": raw, "color": None}
    out["pos"] = tuple(float(v) for v in out["pos"])[:2]
    if out["kind"] not in ("cup", "block", "tray"):
        raise ValueError(f"unknown object kind {out['kind']!r}; use cup, block, or tray")
    return out


def _add_room(spec: Any) -> None:
    """A table the robot is mounted on, a floor, a sky, and light."""
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.32, 0.34, 0.37],
                     rgb2=[0.28, 0.3, 0.33], width=300, height=300)
    floor_mat = spec.add_material(name="floor", texrepeat=[6, 6], texuniform=True, reflectance=0.1)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.62, 0.7, 0.78],
                     rgb2=[0.2, 0.22, 0.26], width=512, height=3072)
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                pos=[0, 0, TABLE_TOP - 0.76], material="floor")
    wood = [0.62, 0.47, 0.33, 1]
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, 0.6, 0.025],
                pos=[0.35, 0, TABLE_TOP - 0.025], rgba=wood, friction=[0.4, 0.005, 0.0001],
                priority=1)  # a cup slides on wood (0.4), whatever its own friction
    for i, (x, y) in enumerate([(-0.15, -0.54), (-0.15, 0.54), (0.85, -0.54), (0.85, 0.54)]):
        wb.add_geom(name=f"table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.03, 0.03, 0.355], pos=[x, y, TABLE_TOP - 0.405],
                    rgba=[0.45, 0.33, 0.22, 1])
    sun = wb.add_light(pos=[0.3, -0.3, 2.0], dir=[-0.15, 0.25, -1], diffuse=[0.6, 0.6, 0.58],
                       castshadow=True)
    sun.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL  # short, crisp shadows
    fill = wb.add_light(pos=[-0.6, 0.6, 1.2], dir=[0.5, -0.4, -1], diffuse=[0.3, 0.3, 0.33],
                        castshadow=False)
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL


def _add_object(spec: Any, name: str, o: dict) -> dict:
    """Add an object; return what ``scene.objects`` says about it."""
    x, y = o["pos"]
    kind = o["kind"]
    if kind == "tray":
        # Static, with its origin on the surface where things are put down.
        color = COLORS.get(o["color"] or "grey", COLORS["grey"])
        body = spec.worldbody.add_body(name=name, pos=[x, y, TABLE_TOP + TRAY_BASE])
        sx, sy = TRAY_SIZE[0] / 2, TRAY_SIZE[1] / 2
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, TRAY_BASE / 2],
                      pos=[0, 0, -TRAY_BASE / 2], rgba=color)
        t = 0.004
        for px, py, hx, hy in ((sx, 0, t, sy), (-sx, 0, t, sy), (0, sy, sx, t), (0, -sy, sx, t)):
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, TRAY_RIM / 2],
                          pos=[px, py, TRAY_RIM / 2], rgba=color)
        return {"kind": "tray", "color": o["color"] or "grey", "surface": TABLE_TOP + TRAY_BASE,
                "size": list(TRAY_SIZE)}
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint()
    if kind == "cup":
        color = COLORS.get(o["color"] or "red", COLORS["red"])
        body.pos = [x, y, TABLE_TOP + CUP_HALF_HEIGHT]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS, CUP_HALF_HEIGHT, 0],
                      mass=0.1, friction=[1.5, 0.01, 0.001], condim=4, rgba=color)
        # The opening and the lip, for looks only.
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS * 0.82, 0.0006, 0],
                      pos=[0, 0, CUP_HALF_HEIGHT], rgba=[0.12, 0.08, 0.07, 1],
                      contype=0, conaffinity=0, mass=0)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_RADIUS * 1.06, 0.003, 0],
                      pos=[0, 0, CUP_HALF_HEIGHT - 0.003], rgba=[c * 0.85 for c in color[:3]] + [1],
                      contype=0, conaffinity=0, mass=0)
        return {"kind": "cup", "color": o["color"] or "red", "height": 2 * CUP_HALF_HEIGHT,
                "radius": CUP_RADIUS}
    color = COLORS.get(o["color"] or "green", COLORS["green"])
    body.pos = [x, y, TABLE_TOP + BLOCK]
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BLOCK, BLOCK, BLOCK], mass=0.08,
                  friction=[1.5, 0.01, 0.001], rgba=color)
    return {"kind": "block", "color": o["color"] or "green", "height": 2 * BLOCK}


class MujocoBackend:
    actuators = ("arm", "gripper")
    def __init__(
        self,
        *,
        objects: dict[str, tuple[float, float]] | None = None,
        speed: float = 0.25,
        accel: float = 1.0,
        nudges: list[tuple[str, tuple[float, float], float]] | None = None,
        view: bool = False,
    ) -> None:
        """``view=True`` opens the MuJoCo viewer and paces stepping to wall time."""
        self.speed, self.accel = speed, accel
        self.log: list[tuple[float, Command]] = []
        self._schedule: list[tuple[float, Callable[[], None]]] = []
        self.rt = None
        self._build(objects or {})
        # Called after every step with the backend: scripted disturbances, recorders, tests.
        self.watchers: list[Callable[[MujocoBackend], None]] = []
        for name, to, t in nudges or ():
            self.nudge(name, tuple(to), at=t)
        self._viewer = None
        if view:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self._viewer.lock():
                cam = self._viewer.cam
                cam.lookat[:] = [0.35, 0.1, 0.12]
                cam.distance, cam.azimuth, cam.elevation = 1.55, 200.0, -28.0
            self._wall0: float | None = None

    # -- model -----------------------------------------------------------------

    def _build(self, objects: dict[str, Any]) -> None:
        spec = mujoco.MjSpec.from_file(str(_panda_xml()))
        for body in spec.bodies:
            if body.name not in ("world", "link0"):
                body.gravcomp = 1.0  # the Panda's own controller compensates gravity
        # The Menagerie gripper squeezes with about 2 N (stiffness 100 N/m on the finger
        # tendon); a held cup creeps out of it within seconds. Ten times the gain and stiffness
        # keeps the same open and closed positions and gives about 20 N, nearer the real
        # gripper's 70 N.
        grip = spec.actuator("actuator8")
        grip.gainprm[0] *= GRIP_SCALE
        grip.biasprm[1] *= GRIP_SCALE
        grip.biasprm[2] *= GRIP_SCALE
        # Rubber finger pads: their friction governs every contact they make (priority).
        for name in ("left_finger", "right_finger"):
            for geom in spec.body(name).geoms:
                geom.friction = [PAD_FRICTION, 0.005, 0.0001]
                geom.priority = 1
        tcp = spec.body("hand").add_site()
        tcp.name, tcp.pos = "tcp", [0, 0, TCP_OFFSET]
        _add_room(spec)
        self.scene: dict[str, dict] = {}
        for name, raw in objects.items():
            self.scene[name] = _add_object(spec, name, _object_spec(raw))
        self.model = m = spec.compile()
        # Soft contacts let a held object creep out of the fingers (12 mm in 5 s with the
        # default pyramidal cone). An elliptic cone with impratio 4 keeps it to 2 mm in 10 s;
        # higher values also make a cup too stiff to slide when a finger centers it.
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        m.opt.impratio = 4.0
        self.data = mujoco.MjData(m)
        self._kin = mujoco.MjData(m)  # kinematics only: IK never touches the physics state
        mujoco.mj_resetDataKeyframe(m, self.data, m.key("home").id)
        mujoco.mj_forward(m, self.data)

        self._qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self._dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self._qlo = np.array([m.jnt_range[m.joint(j).id][0] for j in ARM_JOINTS])
        self._qhi = np.array([m.jnt_range[m.joint(j).id][1] for j in ARM_JOINTS])
        self._fadr = [m.jnt_qposadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)]
        self._fdadr = [m.jnt_dofadr[m.joint(f"finger_joint{i}").id] for i in (1, 2)]
        self._tcp = m.site("tcp").id
        self._fingers = {m.body("left_finger").id, m.body("right_finger").id}
        self._objects = {name: m.body(name).id for name in objects}
        self._slides: dict[str, tuple] = {}
        self._hidden: dict[str, float] = {}  # object -> time until which perception misses it

        self.q_des = self.data.qpos[self._qadr].copy()
        self.r_des = self.data.site_xmat[self._tcp].reshape(3, 3).copy()
        self.x_ref = self.data.site_xpos[self._tcp].copy()
        self.v_ref = np.zeros(3)
        self.goal: np.ndarray | None = None
        self.stopping = False
        self.grip = 255.0  # actuator8: 255 open, 0 closed
        self._x_prev = self.x_ref.copy()
        self._ee_speed = 0.0

    # -- lifecycle ---------------------------------------------------------------

    def attach(self, rt: Any) -> None:
        self.rt = rt
        # What is on the table, for whoever plans: kinds, colors, sizes (positions are sim.*).
        rt.world.write("scene.objects", self.scene, source="mujoco.scene")
        self._publish()

    def now(self) -> float:
        return self.rt.clock.now()

    @property
    def ee(self) -> Point:
        return tuple(float(v) for v in self.data.site_xpos[self._tcp])

    @property
    def moving(self) -> bool:
        return float(np.linalg.norm(self.v_ref)) > 1e-6 or self._ee_speed > 0.005

    # -- test controls -------------------------------------------------------------

    def at(self, t: float, fn: Callable[[], None]) -> None:
        self._schedule.append((t, fn))

    def nudge(self, name: str, to: tuple[float, float], at: float) -> None:
        """Move an object to ``to`` (x, y) at time ``at``, as if someone pushed it."""

        def push() -> None:
            jnt = self.model.body_jntadr[self._objects[name]]
            qa, da = self.model.jnt_qposadr[jnt], self.model.jnt_dofadr[jnt]
            self.data.qpos[qa : qa + 3] = [to[0], to[1], CUP_HALF_HEIGHT]
            self.data.qvel[da : da + 6] = 0
            mujoco.mj_forward(self.model, self.data)

        self.at(at, push)

    SIM_ACTIONS = ("push", "move", "occlude", "remove", "drop")

    def sim(self, action: str, **kwargs: Any) -> dict:
        """Change the simulated world (not the robot). Actions:

        - ``push(object, to, duration=0.6)``: slide it to ``to`` (x, y), as a hand would;
        - ``move(object, velocity, duration)``: slide it steadily, like a conveyor;
        - ``occlude(object, duration)``: perception loses it (``sim.<object>`` stops updating);
        - ``remove(object)``: someone takes it away: off the table, and out of sight;
        - ``drop(object)``: it slips out of the gripper onto the table.
        """
        if action not in self.SIM_ACTIONS:
            raise ValueError(f"unknown simulation action {action!r}; "
                             f"known: {', '.join(self.SIM_ACTIONS)}")
        if action == "push":
            return self.slide(**kwargs)
        if action == "move":
            o, (vx, vy), dur = kwargs["object"], kwargs["velocity"], float(kwargs["duration"])
            x, y = self.truth(o)[:2]
            return self.slide(o, (x + vx * dur, y + vy * dur), dur, ease=False)
        if action == "occlude":
            o = self._check_object(kwargs["object"])
            self._hidden[o] = self.now() + float(kwargs["duration"])
            return {"object": o, "until": self._hidden[o]}
        if action == "remove":
            o = self._check_object(kwargs["object"], movable=True)
            self._slides.pop(o, None)
            self._place_object(o, (2.0, 2.0), z=TABLE_TOP - 0.76)  # on the floor, far away
            self._hidden[o] = math.inf
            return {"object": o, "removed": True}
        o = self._check_object(kwargs["object"])
        if self._holding() != o:
            raise ValueError(f"{o!r} is not in the gripper")
        ee = self.ee
        side = float(kwargs.get("side", 0.07))
        self._place_object(o, (ee[0], ee[1] + side))
        return {"object": o, "dropped_at": [ee[0], ee[1] + side]}

    def _check_object(self, name: str, *, movable: bool = False) -> str:
        if name not in self._objects:
            raise ValueError(f"no object {name!r}; objects: {', '.join(sorted(self._objects))}")
        if movable and self.scene[name]["kind"] == "tray":
            raise ValueError(f"{name!r} is fixed to the table")
        return name

    def _place_object(self, name: str, xy: Any, z: float | None = None) -> None:
        jnt = self.model.body_jntadr[self._objects[name]]
        qa, da = self.model.jnt_qposadr[jnt], self.model.jnt_dofadr[jnt]
        z = (TABLE_TOP if z is None else z) + self.scene[name]["height"] / 2
        self.data.qpos[qa : qa + 3] = [xy[0], xy[1], z]
        self.data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
        self.data.qvel[da : da + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def slide(self, object: str, to: Any, duration: float = 0.6,
              ease: bool = True) -> dict:
        """Slide ``object`` upright along the table to ``to`` (x, y) over ``duration`` seconds,
        as a person pushing it (``ease``) or a conveyor (steady). A slide stops when the
        gripper takes hold of the object."""
        if object not in self._objects:
            raise ValueError(f"no object {object!r}; objects: {', '.join(sorted(self._objects))}")
        if self.scene[object]["kind"] == "tray":
            raise ValueError(f"{object!r} is fixed to the table")
        if self._holding() == object:
            raise ValueError(f"{object!r} is in the gripper")
        start = np.array(self.truth(object)[:2])
        end = np.array([float(to[0]), float(to[1])])
        self._slides[object] = (start, end, self.now(), max(float(duration), 1e-3), ease)
        return {"object": object, "from": [float(v) for v in start],
                "to": [float(v) for v in end], "duration": duration}

    def _apply_slides(self, t: float) -> None:
        held = self._holding() if self._slides else None
        for name, (start, end, t0, dur, ease) in list(self._slides.items()):
            if name == held:  # the robot has it now; nobody pushes it out of the gripper
                del self._slides[name]
                continue
            a = min(1.0, max(0.0, (t - t0) / dur))
            if ease:
                a = a * a * (3 - 2 * a)  # ease in and out, like a hand
            jnt = self.model.body_jntadr[self._objects[name]]
            qa, da = self.model.jnt_qposadr[jnt], self.model.jnt_dofadr[jnt]
            xy = start + (end - start) * a
            self.data.qpos[qa : qa + 3] = [xy[0], xy[1], TABLE_TOP + self.scene[name]["height"] / 2]
            self.data.qpos[qa + 3 : qa + 7] = [1, 0, 0, 0]
            self.data.qvel[da : da + 6] = 0
            if a >= 1.0:
                del self._slides[name]

    def truth(self, name: str) -> Point:
        return tuple(float(v) for v in self.data.xpos[self._objects[name]])

    # -- called by the Gate ----------------------------------------------------------

    def execute(self, cmd: Command) -> None:
        self.log.append((self.now(), cmd))
        if cmd.kind in ("goal", "setpoint"):
            target = np.asarray(cmd.payload["target"], dtype=float)
            if target.shape != (3,):
                raise ValueError(f"target must be (x, y, z), got {cmd.payload['target']!r}")
            self._check_reachable(target)
            self.goal, self.stopping = target, False
        elif cmd.kind == "stop":
            self.goal, self.stopping = None, True
        elif cmd.kind == "effector":
            self.grip = 0.0 if cmd.payload.get("close") else 255.0

    def _check_reachable(self, target: np.ndarray) -> None:
        r = float(np.linalg.norm(target - SHOULDER))
        radial = float(np.hypot(target[0], target[1]))
        if r > REACH or target[2] < 0.0 or radial < 0.15:
            raise BackendError(
                "unreachable",
                f"target {tuple(round(float(v), 3) for v in target)} is outside the workspace "
                f"(distance from shoulder {r:.3f} m, limit {REACH} m)",
                source="mujoco.panda",
                raw={"distance": r, "radial": radial},
                stamp=self.now(),
            )

    # -- physics and control -------------------------------------------------------------

    def step(self, dt: float) -> None:
        now = self.now()
        due = [fn for t, fn in self._schedule if t <= now + 1e-9]
        self._schedule = [(t, fn) for t, fn in self._schedule if t > now + 1e-9]
        for fn in due:
            fn()
        h = self.model.opt.timestep
        for k in range(max(1, round(dt / h))):
            if self._slides:
                self._apply_slides(now + k * h)
            self._reference(h)
            self._ik()
            self.data.ctrl[:7] = self.q_des
            self.data.ctrl[7] = self.grip
            mujoco.mj_step(self.model, self.data)
        x = self.data.site_xpos[self._tcp]
        self._ee_speed = float(np.linalg.norm(x - self._x_prev)) / dt
        self._x_prev = x.copy()
        for w in list(self.watchers):
            w(self)
        self._publish()
        if self._viewer is not None:
            self._show(now + dt)

    def _show(self, t: float) -> None:
        import time

        self._viewer.sync()
        if self._wall0 is None:
            self._wall0 = time.monotonic() - t
        lag = self._wall0 + t - time.monotonic()
        if lag > 0:
            time.sleep(lag)

    def close(self) -> None:
        """Close the viewer and wait for its thread; exiting with it open can crash."""
        if self._viewer is None:
            return
        import time

        self._viewer.close()
        deadline = time.monotonic() + 2.0
        while self._viewer.is_running() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)
        self._viewer = None

    def _reference(self, h: float) -> None:
        """Move the Cartesian reference toward the goal with bounded speed and acceleration."""
        v_cmd = np.zeros(3)
        if self.goal is not None:
            e = self.goal - self.x_ref
            d = float(np.linalg.norm(e))
            if d <= max(float(np.linalg.norm(self.v_ref)) * h, 1e-9):
                self.x_ref, self.v_ref = self.goal.copy(), np.zeros(3)
                return
            v_cmd = e / d * min(self.speed, math.sqrt(2 * self.accel * d))
        dv = v_cmd - self.v_ref
        n = float(np.linalg.norm(dv))
        if n > self.accel * h:
            dv *= self.accel * h / n
        self.v_ref = self.v_ref + dv
        if self.stopping and float(np.linalg.norm(self.v_ref)) < 1e-6:
            self.v_ref, self.stopping = np.zeros(3), False
        self.x_ref = self.x_ref + self.v_ref * h

    def _ik(self) -> None:
        """One damped-least-squares step that moves q_des so FK(q_des) follows x_ref."""
        m, k = self.model, self._kin
        k.qpos[self._qadr] = self.q_des
        mujoco.mj_kinematics(m, k)
        mujoco.mj_comPos(m, k)
        x = k.site_xpos[self._tcp]
        r = k.site_xmat[self._tcp].reshape(3, 3)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, k, jp, jr, self._tcp)
        jac = np.vstack([jp[:, self._dadr], jr[:, self._dadr]])
        err = np.concatenate([self.x_ref - x, _rot_error(r, self.r_des)])
        pinv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-4 * np.eye(6))
        dq = pinv @ err + (np.eye(7) - pinv @ jac) @ (0.01 * (HOME_Q - self.q_des))
        self.q_des = np.clip(self.q_des + dq, self._qlo, self._qhi)

    # -- observations -------------------------------------------------------------------

    def _holding(self) -> str | None:
        touching: dict[int, set[int]] = {}
        for c in self.data.contact[: self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            for f, o in ((b1, b2), (b2, b1)):
                if f in self._fingers:
                    touching.setdefault(o, set()).add(f)
        for name, bid in self._objects.items():
            if touching.get(bid) == self._fingers:
                return name
        return None

    def _publish(self) -> None:
        w, d = self.rt.world, self.data
        w.write("arm.ee", self.ee, source="mujoco.panda")
        w.write("arm.q", tuple(float(v) for v in d.qpos[self._qadr]), source="mujoco.panda")
        w.write("arm.moving", self.moving, source="mujoco.panda")
        w.write("gripper.width", float(d.qpos[self._fadr].sum()), source="mujoco.panda")
        w.write(
            "gripper.moving", bool(np.abs(d.qvel[self._fdadr]).max() > 1e-3), source="mujoco.panda"
        )
        w.write("gripper.holding", self._holding(), source="mujoco.panda")
        now = self.now()
        for name in self._objects:
            if self._hidden.get(name, -1.0) > now:
                continue  # occluded or removed: perception has nothing new
            w.write(f"sim.{name}", self.truth(name), source="mujoco.sim")  # ground truth


# -- example skills and providers on the MuJoCo backend -------------------------------

CYCLE = 0.01
REACH_TOL = 0.01
ALIGN_TOL = 0.01
APPROACH = 0.12  # approach and lift height above a grasp or place point


class GraspMissed(Exception):
    pass


class Dropped(Exception):
    """The object slipped out of the gripper before it was put down."""


class NothingHeld(Exception):
    """``place`` was asked to put something down, but the gripper holds nothing."""


MAX_AGE = 0.3  # a tracked position older than this is not acted on
STALE_WAIT = 3.0  # how long a skill holds still waiting for fresh data before failing
SLIP_TIME = 0.05  # how long the gripper must be empty before it counts as a slip
EMPTY_WIDTH = 0.01  # fingers closer than this have nothing between them


Target = Point | str  # a position, or the name of a tracked object


async def position(target: Target, on_stale: Any = None) -> Point:
    """A position, or the latest *fresh* tracked position of the object named ``target``.

    A name follows the object (SPEC Appendix A, key-following skills). If its value is older
    than ``MAX_AGE`` (the object is occluded, or its tracker stopped), the skill does not act
    on it: ``on_stale()`` is awaited once (e.g. to stop the arm) and it waits for fresh data,
    failing with ``StaleData`` after ``STALE_WAIT`` seconds.
    """
    if not isinstance(target, str):
        return tuple(float(v) for v in target)
    deadline, warned = now() + STALE_WAIT, False
    while True:
        sv = world.latest(target)
        if sv is not None and sv.age(now()) <= MAX_AGE:
            return tuple(sv.value)
        if now() >= deadline:
            raise StaleData(target, None if sv is None else sv.age(now()), MAX_AGE,
                            None if sv is None else sv.source)
        if not warned and on_stale is not None:
            warned = True
            await on_stale()
        await sleep(CYCLE)


def _above(p: Any, h: float = APPROACH) -> Point:
    return (p[0], p[1], p[2] + h)


def _horizontal(a: Any, b: Any) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


async def _arrive(target: Point, holding: str | None = None) -> None:
    """Wait until the TCP is at ``target`` and the arm is at rest. With ``holding``,
    raise ``Dropped`` as soon as the gripper no longer holds that object."""
    missing = 0.0
    while world.get("arm.moving").value or dist(world.get("arm.ee").value, target) > REACH_TOL:
        if (holding is not None and world.get("gripper.holding").value != holding
                and world.get("gripper.width").value < EMPTY_WIDTH):
            # Contact flickers when the object touches a surface, but the fingers stay apart
            # around it. After a slip they close on nothing.
            missing += CYCLE
            if missing >= SLIP_TIME:
                await _stop()
                raise Dropped(f"{holding} slipped out of the gripper at "
                              f"{tuple(round(v, 3) for v in world.get('arm.ee').value)}")
        else:
            missing = 0.0
        await sleep(CYCLE)


async def wait_stopped() -> None:
    while world.get("arm.moving").value:
        await sleep(CYCLE)


async def _gripper(close: bool) -> None:
    """Open or close, and return once the fingers have settled. When the object is off
    center, one finger touches it first and stops for a moment while the other is still
    closing, so a single still sample is not the end of the motion."""
    await send("effector", "gripper", close=close)
    await sleep(5 * CYCLE)  # let the fingers start moving
    # Closing ends when the fingers hold something and the opening is steady (finger speed
    # jitters while squeezing), or when they have closed all the way on nothing. In between
    # a finger may be slowly pushing an off-center object toward the middle, pausing on
    # static friction, so a steady opening alone does not mean done.
    widths: list[float] = []
    for _ in range(250):  # at most 2.5 s
        width = world.get("gripper.width").value
        widths.append(width)
        steady = len(widths) > 10 and abs(widths[-1] - widths[-11]) < 0.0005
        if close:
            if world.get("gripper.holding").value is not None and steady:
                return
            if width < EMPTY_WIDTH:
                return
        elif width > 0.075 or steady:
            return
        await sleep(CYCLE)


async def _stop() -> None:
    await send("stop", "arm")
    await wait_stopped()


@tool
@provider
async def track(name: str, rate: float = 20.0) -> None:
    """Track an object: publish its position (x, y, z of its center, in metres) under its
    name, e.g. "cup_red", at ``rate`` Hz, until cancelled.

    Start it before referring to an object by name. The objects on the table are listed in
    the World key "scene.objects". (In simulation it republishes ground truth, ``sim.<name>``.)
    """
    while True:
        truth = world.get(f"sim.{name}")
        world.write(name, truth.value, stamp=truth.stamp, source=f"track({name})")
        await sleep(1.0 / rate)


@tool
@action(resources={"arm"})
async def move_to(target: Point) -> Point:
    """Move the gripper (the point between the fingertips) to ``target`` (x, y, z) in metres,
    pointing down, and end at rest. The robot base is at the origin; the table top is z = 0."""
    try:
        await send("goal", "arm", target=target)
        await _arrive(target)
        return target
    except asyncio.CancelledError:
        await _stop()
        if dist(world.get("arm.ee").value, target) <= REACH_TOL:
            return target  # the motion finished before it could be stopped
        raise


@tool
@action(resources={"gripper"})
async def gripper(close: bool) -> str | None:
    """Close (true) or open (false) the gripper and wait until the fingers settle; returns
    the name of the object it holds afterwards, or null."""
    await _gripper(close=close)
    return world.get("gripper.holding").value


@tool
@action(resources={"arm", "gripper"})
async def grasp(target: Target) -> Point:
    """Pick up an object from above and lift it.

    ``target`` is an object name (e.g. "cup_red", which must be tracked) or a position. With
    a name the grasp follows the object if it moves; ``update(target=...)`` redirects it.
    Fails with GraspMissed if the gripper closes on nothing. While the gripper is not over the
    object, the arm goes to the approach height first, so it never sweeps through it.
    """
    me = current()
    try:
        await _gripper(close=False)
        while True:
            tgt = await position(me.params["target"], on_stale=_stop)
            ee = world.get("arm.ee").value
            if _horizontal(ee, tgt) <= ALIGN_TOL:
                goal = tgt
                if dist(ee, tgt) <= REACH_TOL and not world.get("arm.moving").value:
                    break
            elif ee[2] < tgt[2] + APPROACH - 0.03:
                goal = (ee[0], ee[1], tgt[2] + APPROACH)  # rise before moving sideways
            else:
                goal = _above(tgt)
            await send("goal", "arm", target=goal)
            await sleep(CYCLE)
        await _gripper(close=True)
        held = world.get("gripper.holding").value
        if held is None:
            raise GraspMissed(f"closed on nothing at {tuple(round(v, 3) for v in tgt)}")
        await send("goal", "arm", target=_above(tgt))
        await _arrive(_above(tgt), holding=held)  # a slip while lifting is a failure too
        return tgt
    except asyncio.CancelledError:
        await _stop()
        raise


@tool
@action(resources={"arm", "gripper"})
async def grasp_blocking(target: Target) -> Point:
    """The same grasp as one blocking call: the target is fixed when it starts, so it does
    not follow an object that moves."""
    target = await position(target)
    await _gripper(close=False)
    ee = world.get("arm.ee").value
    if ee[2] < target[2] + APPROACH - 0.03:  # rise first, as grasp does: never sweep sideways
        await send("goal", "arm", target=(ee[0], ee[1], target[2] + APPROACH))
        await _arrive((ee[0], ee[1], target[2] + APPROACH))
    for goal in (_above(target), target):
        await send("goal", "arm", target=goal)
        await _arrive(goal)
    await _gripper(close=True)
    if world.get("gripper.holding").value is None:
        raise GraspMissed(f"closed on nothing at {tuple(round(v, 3) for v in target)}")
    await send("goal", "arm", target=_above(target))
    await _arrive(_above(target))
    return target


async def _place_point(target: Target) -> Point:
    """Where the held object's center goes. On a named surface (e.g. "tray"), that is the
    surface height plus half the object's height."""
    if not isinstance(target, str):
        return tuple(float(v) for v in target)
    surface = await position(target)
    held = world.get("gripper.holding").value
    scene = world.get("scene.objects").value
    height = scene.get(held, {}).get("height", 2 * CUP_HALF_HEIGHT) if held else 0.0
    return (surface[0], surface[1], surface[2] + height / 2 + 0.003)


@tool
@action(resources={"arm", "gripper"})
async def place(target: Target) -> Point:
    """Put down what the gripper holds, release it, and lift.

    ``target`` is a surface name (e.g. "tray", which must be tracked) or the position where
    the held object's center should end up. Fails at once with NothingHeld if the gripper is
    empty, and with Dropped, stopping the arm, if the object slips out on the way.
    """
    try:
        held = world.get("gripper.holding").value
        if held is None:
            raise NothingHeld("the gripper holds nothing to place")
        point = await _place_point(target)
        for goal in (_above(point), point):
            await send("goal", "arm", target=goal)
            await _arrive(goal, holding=held)
        await _gripper(close=False)
        await send("goal", "arm", target=_above(point))
        await _arrive(_above(point))
        return point
    except asyncio.CancelledError:
        await _stop()
        raise


@tool
@action(resources={"arm", "gripper"})
async def place_blocking(target: Target) -> Point:
    """The same place as one blocking call: the target is fixed when it starts, and it does
    not check what the gripper holds on the way."""
    point = await _place_point(target)
    for goal in (_above(point), point):
        await send("goal", "arm", target=goal)
        await _arrive(goal)
    await _gripper(close=False)
    await send("goal", "arm", target=_above(point))
    await _arrive(_above(point))
    return point
