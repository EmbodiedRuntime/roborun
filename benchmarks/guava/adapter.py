"""Run Guava's benchmark tasks on roborun, with Guava's own tools, prompt, parser, and scorer.

Guava (https://github.com/hdacnw/guava-release) is used read-only from a checkout at
``GUAVA_ROOT``; nothing in it is modified. Two harnesses drive the same model, tools, scene,
and clock (simulated time follows wall time, so the world does not pause while the model
thinks):

- ``blocking``: Guava's interaction loop. A motion tool returns when the motion is over.
- ``roborun``: a motion tool starts a task and returns at once; ``wait`` (with ``wake_on``)
  and ``cancel`` are added. Nothing else changes.

How Guava's tools run on roborun: every Guava motion is a loop over ``RoboEnv.step(action)``.
The tool runs in a worker thread; its ``step`` hands each action to a roborun task, which sends
it through the Gate as a ``setpoint``; the backend applies it on the next control step. So the
tool's own code decides what to do, and roborun decides whether and when it reaches the arm:
cancelling the task stops the tool at its next step, and the arm holds where it is.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PINNED = "027cbc2"  # the Guava commit this adapter was written against


def guava_root() -> Path:
    """Where Guava's checkout is: ``GUAVA_ROOT``, or the location of an importable ``guava``."""
    if os.environ.get("GUAVA_ROOT"):
        root = Path(os.environ["GUAVA_ROOT"]).expanduser().resolve()
    else:
        try:
            import guava  # installed, or on PYTHONPATH

            root = Path(guava.__file__).resolve().parents[1]
        except ImportError:
            raise SystemExit(
                "Guava not found: set GUAVA_ROOT to a checkout of "
                f"https://github.com/hdacnw/guava-release (commit {PINNED}); see benchmarks/guava/README.md"
            ) from None
    if not (root / "guava" / "tools" / "tools.py").is_file() or not (root / "configs").is_dir():
        raise SystemExit(f"GUAVA_ROOT={root} does not look like a guava-release checkout")
    return root


GUAVA = guava_root()
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(GUAVA))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from guava.collection.trajectory import _fmt_tool_result
from guava.inference import sft_frame
from guava.inference.benchmark_engine import build_prompt, decision, observation
from guava.inference.rollout_env import RolloutEngine, _to_data_url
from guava.tools.tools import _gripper_state_text, _refresh
from perception import SegMaskSolver

from roborun import TaskCancelled, TaskFailed, action, now, run_sim, send, sleep, start, world

DT = 0.05  # Guava's control period (robosuite control_freq 20 Hz)
MOTION = {"move", "grasp", "release", "rotate", "align", "close_gripper", "home_pose"}


class Aborted(Exception):
    """Raised inside a Guava tool when its roborun task was cancelled."""


# -- the bridge between a Guava tool (a thread) and roborun (the event loop) -------------


class Bridge:
    """Runs one Guava tool at a time in a worker thread, one simulation step per handshake.

    The worker never steps or renders the simulation itself: ``env.step`` hands the action to
    the roborun task (which sends it through the Gate; the backend applies it), and rendering
    (camera capture, masks) is done on the main thread, which owns the EGL context, while the
    worker waits.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.main = threading.current_thread()
        self.lock = threading.RLock()  # whoever holds it may touch the simulation
        self.orig_step = env.step
        self.tool_thread: threading.Thread | None = None
        self.pending: np.ndarray | None = None  # the action the tool wants next
        self.approved: np.ndarray | None = None  # ... once the Gate accepted it
        self.go = threading.Event()  # main -> worker: the step was applied
        self.wake = threading.Event()  # worker -> main: a step, a render, or the end
        self.request: list | None = None  # [fn, result, error, done-event]
        self.abort = False
        self.result: Any = None
        self.error: BaseException | None = None
        env.step = self._hook

    def _hook(self, act: np.ndarray) -> None:
        if threading.current_thread() is not self.tool_thread:
            return self.orig_step(act)
        self.pending = np.asarray(act, dtype=float).copy()
        self.go.clear()
        self.wake.set()
        self.lock.release()
        self.go.wait()
        self.lock.acquire()
        if self.abort:
            raise Aborted()
        return None

    def on_main(self, fn: Any) -> Any:
        """Run ``fn`` on the main thread (from the worker, wait for it)."""
        if threading.current_thread() is not self.tool_thread:
            return fn()
        req = [fn, None, None, threading.Event()]
        self.request = req
        self.wake.set()
        req[3].wait()
        if self.abort:
            raise Aborted()
        if req[2] is not None:
            raise req[2]
        return req[1]

    def run(self, fn: Any) -> None:
        self.abort, self.result, self.error, self.pending = False, None, None, None
        self.wake.clear()

        def body() -> None:
            with self.lock:
                try:
                    self.result = fn()
                except BaseException as e:  # noqa: BLE001 - reported to the task
                    self.error = e
                finally:
                    self.pending = None
                    self.wake.set()

        self.tool_thread = threading.Thread(target=body, daemon=True)
        self.tool_thread.start()

    def wait_hook(self) -> bool:
        """On the main thread: serve render requests until the tool asks for its next step
        (True) or has finished (False)."""
        while True:
            self.wake.wait()
            self.wake.clear()
            req, self.request = self.request, None
            if req is not None:
                try:
                    req[1] = req[0]()
                except BaseException as e:  # noqa: BLE001 - re-raised in the worker
                    req[2] = e
                req[3].set()
                continue
            if self.pending is not None:
                return True
            if self.tool_thread is None or not self.tool_thread.is_alive():
                return False

    def finish(self) -> None:
        if self.tool_thread is not None:
            self.abort = True
            self.go.set()
            if self.request is not None:
                self.request[3].set()
            self.tool_thread.join(timeout=10)
            self.tool_thread = None


# -- the backend -----------------------------------------------------------------------


@dataclass
class Push:
    """Push the target object once, as a person would, while the gripper descends onto it."""

    obj: str = "can"
    distance: float = 0.12  # more than the gripper's footprint: a grasp aimed at the old spot misses
    done_at: float | None = None
    log: list = field(default_factory=list)


class GuavaBackend:
    actuators = ("arm", "gripper")

    def __init__(self, engine: Any, push: Push | None = None) -> None:
        self.engine, self.env, self.rob = engine, engine.env, engine.env.rob
        self.bridge = Bridge(self.env)
        _route_rendering(self.bridge, engine)
        self.seg: SegMaskSolver = engine.ctx.perception
        self.push = push
        self.rt = None
        self.closed_on_nothing = 0
        self.current: str | None = None  # the Guava tool running now

    def attach(self, rt: Any) -> None:
        self.rt = rt
        self._publish()

    def now(self) -> float:
        return self.rt.clock.now()

    def execute(self, cmd: Any) -> None:
        if cmd.kind == "setpoint":
            self.bridge.approved = np.asarray(cmd.payload["action"], dtype=float)
        elif cmd.kind == "stop":
            self.env._current_joints = self.env.joint_positions().copy()  # hold here

    def step(self, dt: float) -> None:
        b = self.bridge
        with b.lock:
            if b.approved is not None:
                act, b.approved, b.pending = b.approved, None, None
                b.orig_step(act)
                b.go.set()
            else:
                b.orig_step(self.env.build_action(self.env._current_joints))  # hold
            self._disturb()
            self._publish()

    # world: object positions (world frame, ground truth) and the gripper
    def _pos(self, name: str) -> list[float]:
        return [float(v) for v in self.rob.sim.data.xpos[self.seg._body(name)]]

    def _publish(self) -> None:
        for name in self.engine.cfg.sim.object_aliases:
            self.rt.world.write(name, self._pos(name), source="guava.sim")
        self.rt.world.write("arm.joints", self.env.joint_positions().tolist(), source="guava.sim")

    def _disturb(self) -> None:
        p = self.push
        if p is None or p.done_at is not None:
            return
        obj = np.array(self._pos(p.obj))
        ee = np.array(self.env.eef_wxyz_xyz()[4:])
        # during grasp's descent: over the object, fingers still open
        near = np.linalg.norm(ee[:2] - obj[:2]) < 0.05 and 0.02 < ee[2] - obj[2] < 0.10
        if self.current != "grasp" or not near or self.env._gripper_fraction < 0.9:
            return
        sim = self.rob.sim
        j = sim.model.joint_name2id(f"{self.seg.aliases[p.obj]}_joint0")
        adr, vadr = sim.model.jnt_qposadr[j], sim.model.jnt_dofadr[j]
        dx = -p.distance if obj[0] > 0.0 else p.distance  # toward the table's middle
        sim.data.qpos[adr] += dx
        sim.data.qvel[vadr : vadr + 6] = 0.0
        sim.forward()
        p.done_at = self.now()
        p.log.append({"t": round(self.now(), 2), "dx": dx})


def _route_rendering(bridge: Bridge, engine: Any) -> None:
    """Guava's tools render (camera capture, masks); do that on the main thread."""
    from guava.tools import tools

    capture = getattr(tools._capture_with_depth, "__wrapped__", tools._capture_with_depth)
    tools._capture_with_depth = lambda env: bridge.on_main(lambda: capture(env))
    tools._capture_with_depth.__wrapped__ = capture
    seg = engine.ctx.perception
    mask = getattr(seg.get_mask, "__wrapped__", seg.get_mask)
    seg.get_mask = lambda name, rgb: bridge.on_main(lambda: mask(name, rgb))
    seg.get_mask.__wrapped__ = mask


# -- tools as roborun tasks ---------------------------------------------------------------


@action(resources={"arm", "gripper"})
async def guava_motion(backend: GuavaBackend, name: str, fn: Any) -> Any:
    """Run one Guava motion tool; each of its simulation steps passes through the Gate."""
    b = backend.bridge
    backend.current = name
    b.run(fn)
    try:
        while b.wait_hook():
            await send("setpoint", "arm", action=b.pending.tolist())
            await sleep(DT / 2)  # the backend applies it on the next step
        if b.error is not None:
            raise b.error
        return b.result
    except asyncio.CancelledError:
        b.finish()
        await send("stop", "arm")
        await sleep(3 * DT)  # hold still for a few steps: the arm is at rest
        raise
    finally:
        b.tool_thread = None
        backend.current = None


# -- harnesses -------------------------------------------------------------------------------


ASYNC_PROMPT = """

---

## INTERRUPTIBLE MOTIONS

Every motion tool (align, grasp, move, rotate, release, close_gripper, home_pose) takes an
optional argument `wake_on`, a list of object names; a motion that names an object (e.g.
`grasp("can")`) watches that object by default, and `wake_on=[]` turns this off. The call returns
when the motion ends, as usual, or earlier if a watched object moves by more than 2 cm; then the
motion is still running and the result says so, with its task id. You can then:

### `cancel(task_id)`
Stop that motion. Returns once the arm is at rest.

### `wait(task_id, seconds, wake_on)`
Keep waiting for that motion, until it ends, `seconds` pass, or an object in `wake_on` moves.

Only one motion runs at a time.
"""


def _first_call(text: str) -> tuple[str, dict] | None:
    """The first well-formed tool call in ``text``, if the reply had more than one."""
    import re

    for body in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            v = json.loads(body.strip())
        except ValueError:
            return None
        if isinstance(v, dict) and set(v) == {"name", "arguments"} and isinstance(v["arguments"], dict):
            return v["name"], v["arguments"]
        return None
    return None


@dataclass
class Episode:
    variant: str
    seed: int
    push: bool
    log: list = field(default_factory=list)
    decisions: int = 0
    actions: int = 0
    model_seconds: float = 0.0
    end_reason: str = ""
    format_errors: int = 0  # replies Guava's parser rejects (several calls, malformed JSON)
    early_wakes: int = 0


def make_engine(task: str = "can_in_bin") -> Any:
    raw = yaml.safe_load((GUAVA / f"configs/{task}.yaml").read_text())
    raw["sim"]["perception"] = "gt"
    cfg = Path(tempfile.mkdtemp()) / f"{task}_gt.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    cwd = os.getcwd()
    os.chdir(GUAVA)
    try:
        engine = RolloutEngine(cfg, system_prompt=build_prompt("full"))
    finally:
        os.chdir(cwd)
    engine.ctx.perception = SegMaskSolver(engine.env, raw["sim"]["object_aliases"])
    engine.registry = __import__("guava.tools.tools", fromlist=["make_tool_registry"]).make_tool_registry(engine.ctx)
    return engine


async def run_episode(engine: Any, backend: GuavaBackend, model: Any, ep: Episode,
                      max_decisions: int = 25) -> None:
    offset = sft_frame.measured_offset(engine.env)
    names = tuple(engine.cfg.sim.object_aliases)
    system = engine.system_prompt + (ASYNC_PROMPT if ep.variant == "roborun" else "")
    messages: list[dict] = [{"role": "system", "content": system}]
    tasks: dict[str, Any] = {}
    running: Any = None

    def observe(result: str = "", initial: bool = False) -> None:
        with backend.bridge.lock:
            _refresh(engine.ctx)
            gripper = sft_frame.gripper_text(engine.ctx, _gripper_state_text(engine.ctx), offset)
            url = _to_data_url(engine.ctx.last_rgb)
        parts = observation("full", url, gripper, result, initial, engine.cfg.task_name, names)
        messages.append({"role": "user" if initial else "tool", "content": parts})

    def call_now(name: str, kwargs: dict) -> Any:
        phys = sft_frame.physical_arguments(name, kwargs, offset)
        fn = engine.registry[name]
        return lambda: sft_frame.model_result(name, fn(**phys), offset)

    async def motion(name: str, kwargs: dict) -> Any:
        return start(guava_motion(backend, name, call_now(name, kwargs)), name=name)

    async def outcome(task: Any) -> tuple[bool, str]:
        try:
            r = await task
            if isinstance(r, str) and "closed" in r and task.name == "grasp":
                backend.closed_on_nothing += 1
            return True, json.dumps(r) if isinstance(r, (dict, list)) else str(r)
        except TaskCancelled:
            return False, "cancelled; the arm is at rest"
        except TaskFailed as e:
            return False, str(e.root)

    async def until(task: Any, wake: list, seconds: float | None) -> str:
        """Wait until ``task`` ends, ``seconds`` pass, or an object in ``wake`` moves 2 cm."""
        before = {k: world.get(k).value for k in wake if k in names}
        end = now() + seconds if seconds is not None else math.inf
        while not task.state.terminal and now() < end:
            await sleep(DT)
            moved = [k for k in before if math.dist(world.get(k).value, before[k]) > 0.02]
            if moved:
                return f"{', '.join(moved)} moved"
        return "task ended" if task.state.terminal else "time"

    async def describe(task: Any) -> str:
        if not task.state.terminal:
            return f"{task.name} running"
        ok, r = await outcome(task)
        return f"{task.name} {'finished' if ok else 'stopped'}: {r}"

    observe(initial=True)
    t0 = time.time()
    retried = False
    for _ in range(max_decisions):
        m0 = time.time()
        text = await asyncio.to_thread(model, messages)
        ep.model_seconds += time.time() - m0
        ep.decisions += 1
        note = ""
        try:
            kind, value = decision(text, "full")
        except ValueError:
            ep.format_errors += 1
            ep.log.append({"t": round(time.time() - t0, 1), "invalid": text[-600:]})
            first = _first_call(text)
            messages.append({"role": "assistant", "content": text})
            if first is None:  # malformed: say so and let the model answer again, once
                if retried:
                    ep.end_reason = "invalid_model_output"
                    break
                retried = True
                messages.append({"role": "user", "content": "Format error: reply with a brief "
                                 "<think>...</think> and exactly one well-formed <tool_call>, or a "
                                 "terminal declaration."})
                continue
            kind, value = "tool", {"name": first[0], "arguments": first[1]}
            note = "Only the first tool call was executed; send one tool call per turn.\n"
        else:
            messages.append({"role": "assistant", "content": text})
        retried = False
        if kind == "terminal":
            ep.end_reason = "model_terminal"
            break
        name, args = value["name"], dict(value["arguments"])
        ep.log.append({"t": round(time.time() - t0, 1), "call": name, "args": args})
        try:
            if name in MOTION:
                ep.actions += 1
                if running is not None and not running.state.terminal:
                    raise RuntimeError(f"a motion is running ({running.name}); wait for it or cancel it")
                wake = []
                if ep.variant == "roborun":  # a motion that names an object watches it by default
                    named = [args[k] for k in ("target_name", "target") if isinstance(args.get(k), str)]
                    wake = list(args.pop("wake_on")) if "wake_on" in args else named
                running = await motion(name, args)
                tid = str(len(tasks) + 1)
                tasks[tid] = running
                why = await until(running, wake, None)
                if running.state.terminal:
                    ok, raw = await outcome(running)
                    if not ok:
                        raise RuntimeError(raw)
                else:
                    ep.early_wakes += 1
                    raw = (f"{why}; {name} is still running as task {tid}. "
                           f"cancel({tid}) stops it; wait({tid}, ...) keeps waiting.")
            elif name == "wait" and ep.variant == "roborun":
                task = tasks[str(args["task_id"])]
                why = await until(task, list(args.get("wake_on") or []), float(args.get("seconds", 5)))
                raw = json.dumps({"woke": "task ended" if task.state.terminal else why,
                                  "task": await describe(task)})
            elif name == "cancel" and ep.variant == "roborun":
                task = tasks[str(args["task_id"])]
                await task.cancel_and_wait()
                raw = await describe(task)
            elif name in engine.registry:
                with backend.bridge.lock:
                    raw = call_now(name, args)()
            else:
                raise ValueError(f"Unknown tool: {name}")
            text = json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
            if ep.variant == "roborun" and (name in ("wait", "cancel") or "still running" in text):
                feedback = text  # early wakes, wait and cancel results
            else:
                feedback = _fmt_tool_result(name, args, text)
        except Exception as exc:  # noqa: BLE001 - shown to the model, as Guava does
            feedback = f"Error calling {name}: {exc}"
        ep.log[-1]["result"] = feedback[:200]
        observe(note + feedback)
    else:
        ep.end_reason = "decision_limit"
    if running is not None and not running.state.terminal:
        await running.cancel_and_wait()
    await sleep(1.0)  # let the scene settle before scoring


def run(variant: str, model: Any, seed: int, push: bool, engine: Any | None = None) -> dict:
    engine = engine or make_engine()
    cwd = os.getcwd()
    os.chdir(GUAVA)
    try:
        engine.episode(seed, Path(tempfile.mkdtemp())).reset(seed=seed)
    finally:
        os.chdir(cwd)
    p = Push() if push else None
    backend = GuavaBackend(engine, p)
    ep = Episode(variant, seed, push)
    t0 = time.time()
    task = asyncio.run(run_sim(lambda: run_episode(engine, backend, model, ep), backend=backend,
                               dt=DT, until=900, pace=1.0))
    backend.env.step = backend.bridge.orig_step
    score = engine.env.rob.score()
    return {"variant": variant, "seed": seed, "push": push, "success": bool(score["success"]),
            "checks": score["checks"], "seconds": round(time.time() - t0, 1),
            "model_seconds": round(ep.model_seconds, 1), "decisions": ep.decisions,
            "actions": ep.actions, "end_reason": ep.end_reason,
            "format_errors": ep.format_errors, "early_wakes": ep.early_wakes,
            "closed_on_nothing": backend.closed_on_nothing,
            "pushed_at": p.done_at if p else None, "task_state": task.state.value,
            "error": repr(task.error) if task.error else None, "log": ep.log}
