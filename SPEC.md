# roborun semantics — draft v0.12

This document specifies *what* the runtime guarantees, independently of the Python implementation.
Another implementation (C++, Rust, on top of ROS 2) that follows it should be interchangeable.
Background: [Understanding Agentic Robotics from a System Perspective](https://embodiedruntime.github.io/blog/).

Status: draft. Sections marked **Open** are undecided. The appendix is non-normative.

## Design rationale

### Why a robot needs its own runtime

A general-purpose runtime (an operating system, asyncio, the JVM, Erlang/OTP, Ray, Temporal) manages state it **owns**. A robot runtime manages a physical world it **does not own**. Each assumption the general runtimes make breaks on a robot, and each roborun concept answers one of those breaks:

| A general runtime assumes | On a robot | roborun |
|---|---|---|
| The runtime owns the state: a value in memory or on disk is the truth | The world is outside the runtime; the runtime has only delayed estimates | Stamped values and World: every value is a belief from some source at some time |
| Cancellation is instant: after SIGKILL the state is gone | An arm has momentum; stopping is itself a process | `cancelling` is not `cancelled` |
| Time affects performance only | Time affects correctness: data goes stale, a late reaction drops the cup | One time base, `max_age`, watchdogs |
| Access control decides *whether* you may act (may this process write this file?) | Danger lies in *what* the command says: where, how fast, how hard | The Gate checks the physical content of each command |
| Let it crash: a restart restores a clean state | A crashed process leaves the arm moving; a restart cannot restore the world | Pre-verified fallback |
| Recovery by replay (Temporal replays a workflow) | The world cannot be replayed; only a simulation can | Trace records and references; only tasks without resources may be replayed |

roborun borrows from several of these: deadlines from real-time operating systems, the task tree from Erlang supervision, tokens from capability systems, long-running tasks from Temporal. What it adds is the distinction between belief and fact, stopping as a physical process, safety checks on command content, and a causal trace from an agent's decision to its physical outcome.

### Why coding agents do not need it

A coding agent already runs inside a runtime that suits it: Unix, git, and its harness. The four problems this spec addresses appear there in degenerate forms, and the degenerate forms are enough:

| roborun concept | In a coding agent's environment | Why the degenerate form suffices |
|---|---|---|
| Task | A process | Kill is instant and leaves no physical residue |
| Stamped values and World | The file system | The world does not change on its own; reading a file returns the truth, never a stale estimate |
| Gate | Permission prompts and sandboxes | Most actions can be undone (git, undo), so only a few irreversible ones need a check |
| Trace | Terminal output, transcript, git history | Observation is exact and nearly free: read the file, run the tests |

Underneath are four differences. The code base waits while the agent thinks; a cup does not. An edit completes in milliseconds, with no half-done state; a grasp takes seconds. A broken change is undone with `git reset`, so trial and error is a sound strategy; a broken cup is not. Reading a file gives the fact; a camera gives a noisy, delayed estimate.

Coding agents work well largely because their environment takes these problems off the model's hands. roborun tries to make the physical world behave as much like a well-behaved computer as it can: where stopping cannot be instant, `cancelling` makes it explicit; where data goes stale, staleness becomes an error; where actions cannot be undone, the Gate stops them first; where observation is partial, the Trace supplies the causal chain.

The need is a matter of degree, not kind. Coding agents meet the same problems when commands run long (Agent as Policy measured zero overlap of motion and generation when a robot call ran inside a blocking shell command), when actions are irreversible (deploys, production databases, email, payments, where harnesses already add confirmation prompts as a crude gate), and when the environment changes on its own (operations agents on live systems). How much a runtime like this is needed depends on how much the world changes on its own, how irreversible actions are, and how much correctness depends on time. Robots are the extreme on all three. roborun is, in this sense, a runtime for **agents acting in environments they do not control**, with robots as the most demanding case.

### What roborun owns

Agent harnesses above and robot software below already provide many parts: tool schemas, agent loops, sandboxes, action lifecycles, timestamps, clocks, hardware safety, logging. roborun does not repeat them. It owns only what neither layer can do alone, because each needs to know both the decision above and the execution below:

1. a **task tree** that spans from a program down to the motions it started, so cancellation reaches all of them;
2. a **freshness check at decision time**, since only the decision knows how old is too old;
3. **task-level control authority**, since the hardware knows which controller is active but not which task, or why;
4. a **causal trace** from a decision to its physical outcome.

### Four concepts, one physical fact each

| Physical fact | Concept | Without it |
|---|---|---|
| Computation and motion take time, and must be stoppable | **Task** (§1) | Calls block; a reach cannot be redirected or stopped while it runs |
| Information ages | **Stamped values and World** (§2) | Decisions silently use stale data |
| Actions are irreversible, and actuators are exclusive | **Gate** (§3) | Safety depends on the model and the code being correct; tasks fight over the arm |
| The system must be improved over time | **Trace** (§4) | Failures cannot be diagnosed or repaired |

### One structural rule

> **Drivers are reachable only from the Gate process.** User code, agent code, and models run elsewhere and affect the robot only by sending commands to the Gate.

This rule turns safety from a convention into a consequence of the structure. Generated code needs no restricted namespace to protect the robot, and "models never touch actuators" needs no enforcement of its own.

### Spec and library

The spec is the four concepts, the structural rule, and the boundary rules (§5). Everything else is library: roles such as *Action* and *Provider*, change-handler sugar, tool export, and backend adapters (Appendix A). The library may grow and change between versions; the spec should not.

### Invariants (derived)

| Invariant | Follows from |
|---|---|
| Only tasks that hold actuators can move the robot, and only through the Gate | §1.6, §3.1, structural rule |
| Models never touch actuators | A model call holds no resources (§1.6) + structural rule |
| A cancel request is not a stop | §1.2 |
| No task outlives its parent unless detached | §1.5 |
| A cancelled task cannot move the robot, even if its code keeps running | Token revocation (§3.3) |
| Every datum carries its age on one time base | §2.1 |

### Mechanism, not policy

No concept encodes *who decides* or *how*. A model call is ordinary computation that returns a stamped value. Decision logic is ordinary user code: a sequential program, loops over World changes, or a model choosing one step at a time. Therefore all combinations of {sequential, event-driven} × {programmatic, model-mediated} decisions are expressible, and the same concepts serve thin-tool and rich-tool systems alike.

### Deliberately out of scope

High-rate control (kHz force control) stays in controllers below the runtime. Multi-robot and distributed World consistency are not addressed.

---

## 1. Task

A task is any running computation: a program, a motion, a perception loop, a model call, a loop reacting to changes.

### 1.1 States

| State | Meaning | Terminal |
|---|---|---|
| `pending` | Created, not yet started; includes waiting for actuators. A pending task has had no effect | no |
| `running` | Executing; may be updated | no |
| `cancelling` | Cancellation requested; cleanup in progress. For a task holding actuators, the robot may still be moving | no |
| `succeeded` | Finished its goal | yes |
| `failed` | Ended without reaching its goal | yes |
| `cancelled` | Stopped after a cancellation request. For a task holding actuators, the robot is confirmed stopped | yes |

### 1.2 Transitions

```
pending    → running      (start)
pending    → cancelled    (cancel before start, including while waiting for actuators)
pending    → failed       (could not start, e.g. its actuators stayed busy: ResourceBusy)
running    → succeeded | failed
running    → cancelling   (cancel requested)
cancelling → cancelled    (cleanup done; for a task holding actuators, the backend confirms the robot is stopped)
cancelling → succeeded    (the work finished before it could be stopped)
cancelling → failed       (cleanup failed; the Gate fallback takes over)
```

No other transition is allowed. A terminal state never changes.

### 1.3 Operations

- `start(body, resources=..., detached=False, on_busy="wait", wait_timeout=None) -> Task` returns immediately. `on_busy` decides what happens when the resources are owned by another task: `wait` (first come, first served; fails with `ResourceBusy` after `wait_timeout`), `fail` (fail at once), or `take` (preempt the owner, if allowed; §3.3).
- `task.update(**params)` is valid only in `running`; otherwise it raises `InvalidState`.
- `task.cancel()` is a *request*: it moves the task to `cancelling` and returns. It is idempotent and synchronous, so calling it without `await` still sends the request.
- `await task.stopped()` resolves only when the task reaches a terminal state.
- `await task` waits for a terminal state, then:
  - returns the task's result if it `succeeded`;
  - raises `TaskFailed` (carrying the cause) if it `failed`;
  - raises `TaskCancelled` (carrying the reason: `requested`, `timeout`, `preempted`, `rejected`, `watchdog`, `operator` for an operator stop (§3.5), or `parent` when its parent ended or was cancelled) if it was `cancelled`.

  This is the form sequential code uses, and it is why sequential code is safe by default: in `await grasp(cup); await place(on="tray")`, `place` never runs after a grasp that did not succeed, including one that was preempted or rejected by the Gate. It matches asyncio (`await` re-raises, `CancelledError`) and Kotlin (`Deferred.await` rethrows, `CancellationException`).
- `await task.stopped()` waits for a terminal state and **never raises**; it returns the final state and reason. It is for code that inspects outcomes instead of stopping on them: supervisors, retry logic, multi-turn generation loops.
- `await body` (awaiting an unstarted body directly) is equivalent to `await start(body)`: the common sequential case needs no handle.

Cancellation is always these two steps, request and wait, as in asyncio (`cancel`, then `await`) and Kotlin (`cancel`, then `join`). Language-level conveniences may combine them (Appendix A) but must not make the request depend on being awaited.

### 1.4 Timeouts

`async with timeout(s):` bounds the tasks started inside it. On expiry, they are cancelled: they pass through `cancelling` like any other cancellation, and the scope exits only after they are terminal. A timeout is therefore never a way to abandon a moving robot.

### 1.5 Parent and child tasks

A task started from inside another task is its child. Tasks form a tree rooted at the runtime.

1. **Cancellation propagates down.** Cancelling a task requests cancellation of all its non-terminal descendants.
2. **Parents outlive children.** A task does not reach a terminal state until all its children have. Its outcome is decided by its own body; a child's failure reaches the parent body as an exception from `await child`.
3. **No orphans.** If a parent body returns or raises while children are still running, the runtime cancels them and waits for them before the parent becomes terminal.
4. **Detached tasks are explicit.** A task that must outlive its creator (e.g. a perception loop) is started with `detached=True` and becomes a child of the root.
5. **Shared providers are system-level.** A provider meant for every session (e.g. the cup tracker) is started from runtime configuration as a child of the root, not by one session. A provider started inside a session ends with that session (§5.1). A reader that depended on it then gets `StaleData` naming the source, never a silently old value. There is no reference counting: who still reads a key cannot be determined reliably, and it would make a provider's lifetime unpredictable.

### 1.6 Resources

A task may declare the actuators it needs, e.g. `resources={"arm", "gripper"}`.
A task that holds no resources cannot command any actuator. Ownership is granted and arbitrated by the Gate (§3.3). A child may use resources its parent holds without a separate grant.

**Holding across tasks.** `async with hold("arm", "gripper"):` makes the current task the owner of those actuators for the whole scope. Child tasks started inside it use that ownership (§3.3), so no other task can take the arm between two of them (for example, between a grasp and a place while the cup is in the gripper). Ownership is released when the scope exits. If another task takes the actuators (§3.3), the current task is cancelled with reason `preempted`.

Two consequences follow, and implementations may rely on them:
- A task that holds no resources has no physical effect, so it can be cancelled immediately and retried or replayed freely.
- A task that holds resources has irreversible effects, so its `cancelling` state is physical and it can be replayed only in simulation.

### 1.7 Provenance

Every task records the source of its body: a code reference and version, and whether the code was written by a person or generated (and by which model call). This is how a generated program is identified; no separate concept is needed.

### 1.8 Failure information

`failed` and `cancelled` mean different things, and they are reported differently:

- **Cancelled with reason `rejected`**: the Gate *did not allow* a command (§3.3, §3.8). It carries a structured `detail` (`code`, the command, the inputs used).
- **Failed**: the work *could not be done*. Either a skill raised its own error (e.g. `GraspMissed`), or the layer below reported a failure as a `BackendError` (§5.2).

A `BackendError` carries:

- `code`, one of the categories below;
- `message`;
- `source`, the lower-layer component that reported it;
- `raw`, the lower layer's own error, kept unchanged;
- `stamp`.

| Code | Meaning | What a caller can do |
|---|---|---|
| `unreachable` | No IK solution; target out of reach | Choose another target |
| `no_plan` | No feasible path | Choose another target or clear the obstacle |
| `blocked` | Motion obstructed: contact, jammed | Inspect the state; may need perception |
| `tolerance` | Did not reach the required accuracy | Retry, or move more slowly |
| `fault` | The robot entered a fault state (reflex, protective stop) | Recover first (§3.5) |
| `unavailable` | Controller inactive or driver disconnected | System problem |
| `unknown` | The lower layer gave no reason | Inspect the attached state |

The runtime cannot report more than the layer below provides, but it guarantees four things:

1. **Nothing is dropped.** The lower layer's error is kept unchanged in `raw`.
2. **Attribution.** A failure always reaches the right task. The Gate tags every forwarded command with its command id and task id, and a lower-layer error that names a command goes to that command's task. An error that names no command (e.g. a sudden reflex) goes to the task that owned the actuator at that time; there is always exactly one.
3. **State at failure.** The trace records the observed state at the moment of failure (joint states, forces, the last commands, and the stamps of inputs used), as refs (§4). Even a bare `ABORTED` can then be diagnosed.
4. **Causal chain.** A failure propagates up the task tree as nested `TaskFailed` (Python exception chaining). `e.root` is the innermost cause, and `e.path` lists the tasks from the outermost inwards, e.g. `program → grasp → move_to → BackendError(unreachable, source=moveit)`. Concurrent child failures follow Python's `ExceptionGroup`.

The harness interface (§5.1) returns the same information:

```json
{"state": "failed",
 "error": {"code": "unreachable", "source": "moveit",
           "message": "no IK solution for target (0.62, 0.10, 0.05)",
           "path": ["program", "grasp", "move_to"], "trace": "run-0412#span-88"}}
```

Why a failure happened in the world (e.g. perception reported the wrong pose) is not something the runtime can decide. The trace records which inputs were used, their stamps, and which provider produced them, so a person or a coding agent can follow the cause.

## 2. Stamped values and World

### 2.1 Time base

There is one time base per runtime, chosen by the backend. All stamps, `max_age` checks, watchdogs, and trace times use it.

- **Code reads time only through the runtime** (`now()`, `sleep()`, timers). Code that reads the system clock directly (`time.time()`, `time.sleep()`) is an error, because it breaks in simulated time. Program code is therefore identical in simulation and on hardware; only the clock behind the interface differs.
- **The producer stamps.** A stamp is set where the observation is made, normally in the Gate process or a driver. Receivers never re-stamp, which would hide transport delay (§2.2). A device with its own clock (e.g. a camera's hardware timestamps) is converted to the time base at the point of reception.

| Mode | Time base | Use |
|---|---|---|
| Hardware | The OS monotonic clock, shared by all processes on the machine; nothing is transmitted | Real runs |
| Real-time simulation | Same as hardware; the simulator steps at wall rate | Latency, watchdogs, moving targets |
| Lockstep simulation | Simulated time, owned by the process that steps the simulator (the Gate process, §3.1) | Deterministic regression tests and replay of failures |
| On ROS 2 | ROS time (`/clock` when `use_sim_time` is set) | Follows ROS 2 (§5.2) |

**Lockstep protocol.** Each step is a handshake:

1. The Gate process sends `tick(t, observations, events)`. The observations keep their own stamps.
2. The runtime sets its clock to `t`, fires due timers, writes the observations to the World, and lets every task react. Commands and ownership requests made meanwhile are answered by the Gate at once (§3.8). The step is settled when no request is waiting for a reply and a full round of the event loop produced no new request.
3. The runtime sends `ack`. The Gate advances the simulator by `dt` and sends the next tick.

The simulator never advances while user code is still handling a step, so results do not depend on CPU speed. (The same pattern as PX4 lockstep with Gazebo and CARLA's synchronous mode.)

Two consequences:

- **In lockstep, computation takes no simulated time.** A two-second model call is instantaneous in simulated time, and the robot does not drift meanwhile. Behavior that depends on latency must be tested in real-time simulation.
- **Watchdogs never fire in lockstep.** If user code hangs, simulated time stops with it. Watchdog tests run in real-time simulation.

Time stepping exists only in simulated time; nothing else in the runtime has a global tick, and components run at their own rates. Clock synchronization across machines (PTP, chrony) is out of scope.

### 2.2 Stamped values

Every piece of information that crosses a task boundary is a stamped value `{value, stamp, source}`. `stamp` is the time at which the underlying observation was taken, not the time the value was computed or written. A value derived from several inputs carries the oldest of their stamps.

This applies equally to sensor data, perception results, model outputs, maps, language instructions, and human input.

### 2.3 World

World is the runtime's shared, present belief: the latest stamped value for each key. It decouples producers from consumers in time: a producer writes at its own rate, a consumer reads at its own.

- `get(key, max_age=s)` raises `StaleData` if `now - stamp > s`. Without `max_age`, the entry's configured default applies; if there is none, the read is allowed but recorded in the trace as unchecked.
- `StaleData` names the key, its age, the limit, and the source that last wrote it, so the caller can tell which producer is missing or slow.
- `snapshot()` returns a consistent view; each entry keeps its own stamp.
- Each key is either a *state* key (the default), which holds only its latest value, or a *queue* key, which keeps a bounded backlog of messages (§2.4). **World is the present; the Trace is the past.**
- When the backend already keeps stamped data (ROS message headers, tf2), World is a view over it and preserves the original stamps.

### 2.4 Changes

There is no separate event concept. Reacting to the world means reacting to new values in World:

```python
async for cup in changes("cup", while_running=g):
    ...
```

- `changes(key, while_running=task)` is an async iterator that yields each new stamped value written to `key`, and ends when `task` reaches a terminal state. Without `while_running` it runs until the enclosing task ends.
- `await next(key)` returns the next value written to `key` (for a queue key, the next message). It is how a program waits for a producer's first value.
- `get(key)` (§2.3) is synchronous: it reads what is there now and never waits.
- Deciding *which* changes matter is ordinary code (`if moved(cup, target): ...`); the runtime does not define named event kinds.
- **Missing producers fail loudly** (state keys). If no new value arrives and the latest one becomes older than the key's `max_age`, the iterator raises `StaleData`. A loop that depends on a tracker therefore fails visibly when the tracker is not running, instead of waiting forever.
- What a loop sees when values arrive faster than it consumes them depends on the key's kind (§2.5).
- Gate decisions (rejections, preemptions) reach code as task state changes with a reason (§3.3), not as events.

### 2.5 State keys and queue keys

Some information replaces what came before; some adds to it. A cup's pose 50 ms ago is useless once a newer one exists, but a second spoken instruction ("the blue one too") does not cancel the first ("take the red one"). Each World key declares which it is:

```python
world.declare("cup",         kind="state", max_age=0.1)              # the default
world.declare("instruction", kind="queue", maxlen=32, max_age=30.0)
```

| | `state` (default) | `queue` |
|---|---|---|
| A new value | **replaces** the previous one | is **appended** after the previous ones |
| Examples | poses, joint states, perception results, maps | language instructions, button presses, corrections from a person, "new object appeared" reports |
| `changes` when the loop falls behind | yields the newest value and skips older ones; skipped values remain in the trace | yields **every** message, in write order |
| Meaning of `max_age` | the value is too old to use; `get` and `changes` raise `StaleData` (§2.3–§2.4) | the message has expired; expired messages are skipped and recorded in the trace as `expired`. Silence is normal and never an error |
| Too many values | cannot happen | a reader that falls more than `maxlen` messages behind raises `Overflow` |
| `get(key)` | the latest value | the latest message |
| ROS 2 QoS counterpart | `KEEP_LAST`, depth 1 | `KEEP_LAST`, depth `maxlen` (reliable) |

Rules for queue keys:

1. **Producers never block.** Sensors and people are not slowed down by a slow consumer. The backlog is a ring of `maxlen` messages.
2. **Every reader has its own cursor.** Each `changes` loop, and each harness client using `wait(wake_on=...)`, sees every message once. Deciding that a message is handled by only one consumer is application logic.
3. **A new reader starts at the oldest retained, unexpired message**, so an instruction that arrived just before a program started listening is not lost.
4. **Overflow fails loudly.** If a reader falls more than `maxlen` messages behind, its next read raises `Overflow`, which names the key, the number of messages lost, and the reader. Nothing is dropped silently.
5. Each message is a stamped value; ordering is by write order within one key. **Open:** ordering across keys.

## 3. Gate

The Gate does not inspect code. It inspects **every command that code sends to the robot**, the way an operating system checks system calls rather than programs.

The Gate is mechanism only. It decides *who* may command an actuator and *when* checks run, and it enforces what happens when they fail. It does not decide *what is dangerous*. That depends on the scene, and checks are functions supplied by the integrator (§3.4). The Gate also **never computes commands**: controllers (IK, interpolation, braking trajectories) belong to the backend below it.

```
user process ──command──▶ Gate ─────────────────────────────▶ backend (controllers, drivers)
                          1. authority: token, owner (§3.3)
                          2. command checks (§3.4)
                          3. plan check, if the backend previews a plan
                          4. monitors over observed state ◀── observations
                          fallback, watchdog, trace
```

### 3.1 Process boundary

The Gate runs in its own process and is the only process connected to the drivers (structural rule). A crash, hang, or bug in user, agent, or model code cannot disable it.

The backend may run inside the Gate process (e.g. MuJoCo) or in its own process (e.g. ros2_control). Either satisfies the rule, as long as no user process can reach it. A controller running in the Gate process is still part of the backend, not of the Gate.

**Process layout.** There is one runtime process per robot, holding the task tree and the World, and one Gate process. Harnesses are clients of the runtime process, not runtimes of their own. Untrusted programs run in their own processes (§3.6), read the World through the runtime, and send commands with their own tokens.

```
harness A ──MCP/CLI──┐
harness B ──MCP/CLI──┼──▶ runtime process (task tree, World) ◀──▶ Gate process ──▶ backend
untrusted program  ──┘
```

**Who is authoritative.** The Gate is authoritative for ownership, the runtime for task state.

- **Tokens** are unforgeable random IDs (e.g. 128 bits). Only the Gate knows which tokens are valid.
- **Cancellation from the runtime** (`task.cancel()`, a timeout): the runtime moves the task to `cancelling`, then asks the Gate to revoke its token.
- **Cancellation from the Gate** (preemption, rejection, watchdog, operator stop): the Gate **revokes the token first**, so the robot is safe at once, then tells the runtime to cancel the task. It never waits for the runtime to agree. A revoked task may still send `stop` during cleanup, since `stop` is always accepted.
- **A disconnected process.** When a connection to the Gate closes (a crash, a hang detected by the watchdog, a normal exit), the Gate revokes every token held through that connection and runs the fallback. The robot never keeps moving because a user process died.

### 3.2 Command vocabulary

Commands are structured messages from a small, fixed vocabulary. Arbitrary function calls never cross the boundary.

| Command | Example | Meaning |
|---|---|---|
| `goal` | `move_ee(target)` | Reach a target; a controller plans the motion |
| `setpoint` | a stream of poses at 10–50 Hz from a loop in generated code | Track successive targets directly |
| `effector` | `gripper(close=True)` | Command a gripper or other end effector |
| `stop` | `stop()` | Bring the named actuators to rest; always accepted, handled first |

`goal` and `setpoint` may be in joint space or in task space (end-effector poses). A backend declares which spaces each actuator accepts.

Every command carries the capability token of the task that issued it (§3.3).

### 3.3 Control authority

- Each actuator has at most one owning task at a time.
- When the Gate grants ownership, it issues the task a **capability token**. Every command must carry a valid token for the actuators it touches; children use their ancestor's token.
- When a task is cancelled, preempted, or rejected, the Gate **revokes** its token. Commands with a revoked token are rejected, so a cancelled task cannot move the robot even if its code keeps running.
- **Arbitration** has no priorities. When a task requests actuators another task owns, it does what it chose at `start` (`on_busy`):
  - `wait`: requests are served **first come, first served**. A later request never overtakes an earlier one that overlaps it, so no request starves. With `wait_timeout`, the requester fails with `ResourceBusy` when it expires.
  - `fail`: fail at once with `ResourceBusy`.
  - `take`: the owner is cancelled with reason `preempted` and enters `cancelling`. Ownership transfers only after the owner reaches a terminal state, and then goes **directly to the taker**: no other task can acquire the actuators in between.
- **Authority to take.** `take` is allowed only if the requester has authority over the owner. The rule is structural, not numeric:
  - **within one session**, take is always allowed, so a program may redirect its own tasks;
  - **across sessions**, take is allowed only if the runtime configuration names the requester's session a **supervisor** of the owner's session (e.g. a recovery program supervising agent sessions).

  Otherwise `take` fails with `ResourceBusy`. Code cannot raise its own authority, just as it cannot loosen checks (§3.7).
- Operator stops are outside arbitration (§3.5).
- **A rejection cancels the task, not only the command.** Generated code rarely checks the result of every `send`, and steps after a rejected command would build on a false premise. The task's terminal state records the reason (`rejected`, `preempted`, `watchdog`) with a structured detail (§3.8), which `await task` raises as `TaskCancelled` (§1.3). A monitor that aborts a running motion cancels its owner with reason `rejected` and code `monitor:<name>`.
- `hold(...)` (§1.6) requests ownership for a scope rather than for one task's lifetime; it is arbitrated in the same way.

### 3.4 Checks

What counts as dangerous depends on the scene: 0.5 m/s is safe over an empty table but not next to a person or while carrying water. The spec therefore defines **how checks are called, not what they check**. (The same split as Kubernetes admission webhooks: the cluster defines the call, its `timeoutSeconds`, and its `failurePolicy`; users define the policy.)

**Interface.** A check is a function the integrator registers with the Gate:

```python
def check(cmd: Command, view: View) -> Verdict    # command check: before a command is executed
def monitor(view: View) -> Verdict                # monitor: at a fixed rate, whatever is commanded
# Verdict is Accept() or Reject(reason)
```

`view` is a read-only snapshot of the stamped inputs the check declared. Each registration states:

- `kinds` (command checks only): which command kinds it applies to;
- `inputs`: the keys it reads, each with a `max_age`;
- `budget`: the longest the Gate waits for its answer;
- `rate` (monitors only).

```python
gate.add_check(keep_out, kinds={"goal", "setpoint"},
               inputs={"arm.joints": 0.01, "scene.humans": 0.2}, budget=0.5e-3)
gate.add_monitor(human_distance, inputs={"scene.humans": 0.2}, rate=100, budget=2e-3)
```

**What the Gate guarantees** (independent of what checks do):

1. **Every applicable check runs before a command reaches the backend.** A command is forwarded only if every applicable check accepts it.
2. **Fail closed.** A check that exceeds its budget, raises, or cannot run counts as `Reject`. There is no "ignore on failure" setting.
3. **Freshness first.** If any declared input is older than its `max_age`, the check is not called and the result is `Reject` (naming the stale key, as `StaleData` does).
4. **Budgets fit the command.** For `setpoint` streams, the total budget of applicable checks must be less than the stream period; the Gate verifies this at startup. `goal` checks may take longer (e.g. checking a whole plan).
5. **Checks cannot stall the Gate.** Checks run outside the Gate's main loop. `stop`, the watchdog, and the fallback never wait for a check.
6. **Checks only judge.** A check returns a verdict; it cannot issue commands or grant authority.
7. **Every verdict is traced**: check name, inputs with stamps, verdict, and time taken.

A budget bounds how long the Gate waits; it cannot make a slow check fast. A check that is too slow keeps rejecting, so the robot holds still. That is safe, but the task fails, and the trace shows which check timed out.

**Plan check.** A backend whose controller plans (e.g. a motion planner producing a joint trajectory) may expose `preview(goal) -> plan`. Checks for `goal` then receive the plan and can check every point. Without a preview, a check sees only the target, and paths are covered by monitors. Monitors detect a violation only as it develops, so keep-out margins must include braking distance.

**Watchdog.** This is not a user check. If a `setpoint` stream stops, or a task holding actuators stops responding within its declared budget, the Gate runs the fallback.

**Delegation.** Where the backend already enforces a limit (controller joint limits, a robot's safety controller), the integrator may omit that check. The configuration then names the lower-layer component responsible, and the Gate verifies at startup that the component is present; if it is not, the Gate refuses to start. On backends without hardware safety (simulation, low-cost arms) every check must be registered with the Gate.

The Gate core needs no robot model. A check that needs one (e.g. forward kinematics for whole-arm keep-out zones) loads it itself, referencing the backend's model (URDF / `robot_description`, MJCF) rather than copying it. Checks recompute kinematics independently of the controllers, the way industrial safety controllers do: a checker must not rely on the computation it checks.

**Open:** whether a verdict may also *modify* a command (e.g. clamp a speed) rather than only accept or reject it. Clamping silently changes what the program asked for, so v0 allows only `Accept` and `Reject`.

### 3.5 Fallback

A pre-verified behavior (hold or stop) runs when a check rejects a command, a monitor fires, a watchdog expires, or a stop fails. For a rejected `setpoint`, hold means staying at the last accepted point.

The fallback runs only for commands from the actuator's owner. A command rejected as `unauthorized` (no valid token) does not run it: otherwise any task could stop another task's motion by sending commands it may not send.

**Locked state.** After a safety event the Gate is *locked* for the affected actuators: their owners are cancelled, and every non-`stop` command for them is rejected with code `locked`. This follows robot controllers such as Franka's reflex mode and UR's protective stop, which require explicit recovery. Two ways out exist, depending on the cause:

| Cause | Unlocks |
|---|---|
| A monitor fired | Automatically, once that monitor accepts again (e.g. the person has left), and no other cause remains. Tasks waiting for the actuators then get them in order |
| A watchdog expired | No lock is needed: the silent owner's token is revoked and it is cancelled with reason `watchdog`; the actuators are free once it is terminal |
| An operator stop, or a lower-layer `fault` | Only by an explicit `reset`, from the operator or the harness. A `fault` also requires the lower layer's own recovery (e.g. Franka's error recovery), which `reset` triggers |

While actuators are locked, commands for them are rejected with `locked`. A task may wait to acquire actuators locked by a monitor, as it waits for owned ones; acquiring actuators locked until `reset` fails at once with `ResourceBusy`.

Both lock and unlock are recorded in the trace.

**Operator stop affects only the named actuators.** A `stop` names actuators (§3.2). An operator stop:
- cancels the owners of those actuators with reason `operator`, including programs that own them through `hold`;
- rejects tasks waiting to acquire them with `locked`;
- locks them until `reset`.

Tasks that hold no actuators keep running: they have no physical effect (§1.6). A tracker that keeps running means the World is still fresh after `reset`. A parent awaiting a cancelled task receives `TaskCancelled("operator")`; if it catches it and tries to move again, the lock rejects the command, so parents need not be cancelled separately.

Stopping the robot and cancelling a program are different operations:

| | Stop the robot | Cancel a program |
|---|---|---|
| Acts on | Actuators | The task tree |
| Path | Straight to the Gate, bypassing models (§5.1) | The runtime's `cancel` |
| Affects | Tasks owning those actuators; the actuators stay locked | That task and all its descendants |
| For | The robot is about to hit something | The agent keeps generating wrong code |

### 3.6 Protection boundary

| Protection | Holds against |
|---|---|
| Checks, monitors, watchdog, stop channel | Any user, agent, or model code, including buggy or adversarial code: they live in the Gate process and are configured only at startup. Check code itself is trusted integrator code |
| Arbitration between tasks | Bugs and conflicts. Code in the same user process can read other tasks' tokens, so it does not hold against adversarial code |
| Isolation between untrusted programs | Run each untrusted program in its own process with its own token |

The Gate does not replace hardware safety (emergency stops, safety controllers, collision reflexes). It sits above it; hardware safety remains the last line of defense.
The Gate also does not judge whether a command is *correct*: a cup placed within limits but in the wrong spot is a failure, not an unsafe state.

### 3.7 Configuration

Checks, monitors, their budgets, delegations, and supervisor sessions are Gate configuration, set when the Gate starts. Changing them is a configuration change and is recorded in the trace.

**Code can only tighten.** At run time, user, agent, and model code can never remove, replace, or loosen a check. A task may *add* a stricter check scoped to itself (e.g. "this task moves only over the left half of the table"); the Gate removes it when the task ends. This is the same rule as Linux seccomp: a process can add restrictions to itself, never remove them.

### 3.8 Process protocol

Common frameworks follow one pattern:

- What is known *before* execution is answered synchronously: syscalls return `EPERM`; a ROS 2 action goal is accepted or rejected; MAVLink sends `COMMAND_ACK`.
- What happens *during* execution is reported asynchronously: a ROS 2 action is `ABORTED`; libfranka raises; a UR robot reports a protective stop.
- High-rate streams are never acknowledged one by one (ROS 2 topics, MAVLink setpoints, libfranka). Errors surface at the next call, as with TCP's `EPIPE`.

The Gate follows the same pattern:

| Command | Acknowledgement |
|---|---|
| `goal`, `effector` | **Synchronous.** `await send(...)` returns after the Gate's verdict, one round trip (~0.1 ms) |
| `stop` | Synchronous, and always accepted |
| `setpoint` | **Pipelined.** `send` returns without waiting. On a rejection the Gate revokes the token and holds at the last accepted point, then notifies the user process. The task sees `TaskCancelled("rejected")` at its next `send` or `await`, at most one period later |

Either way the program sees the same outcome: the task is cancelled with reason `rejected`.

**Every request gets a reply, and replies carry events.** Ownership requests, releases, and revocations are requests too. The Gate's events for a connection travel with its next reply or tick:
- cancellation orders, with the tokens already revoked;
- grants to requests that were waiting.

The runtime therefore knows when it has heard everything the Gate will say about a step, which keeps lockstep runs deterministic (§2.1).

**Rejection codes.** These are deliberately few, as in gRPC status codes and MAVLink results, so a caller knows what to do. Each code may be followed by the name of the check or key.

| Code | Meaning | Caller |
|---|---|---|
| `unauthorized` | No token, revoked token, or actuator outside the token | Program bug |
| `invalid` | Malformed command: unknown kind, missing field, NaN, a space the backend does not accept | Program bug |
| `check:<name>` | A check or plan check rejected the command | Choose another command |
| `check_timeout:<name>`, `check_error:<name>`, `stale:<key>` | A check could not decide | Temporary: retry once the input or check recovers |
| `stale_setpoint` | A setpoint arrived late or out of order | Transport or timing problem |
| `locked` | The actuators are locked (§3.5) | Wait for unlock, or `reset` |

**Transport.** On one machine the reference implementation uses a Unix domain socket with length-prefixed msgpack messages. A round trip takes tens of microseconds, there are no dependencies, and `SO_PEERCRED` identifies the peer process. On ROS 2, DDS carries the same messages with the same semantics. Large data such as images do not pass through the Gate.

## 4. Trace

The runtime records a trace complete enough for a person or a coding agent to find why a run failed.

- Each record is a span: `{id, parent, kind, name, start, end, status, inputs, outputs, refs}`.
  `kind` is one of `task`, `world`, `gate`; tasks may add a library role (e.g. `action`, `provider`) as a label. `world` spans record writes, including values a `changes` loop skipped.
- Span `parent` follows the task tree (§1.5), so a rejected command links back to the task, program, and decision that issued it.
- A span may name an **external parent**, such as the harness turn or model call that started the task, so the trace joins the harness's own logs.
- Spans record the stamps of the inputs they used and the provenance of task bodies (§1.7).
- Large payloads are **referenced, not copied**: a ref may point to a blob, a time range in a rosbag, or an episode in a LeRobot dataset.
- Traces are JSON Lines, one file per run. The first line describes the run (`kind: run`, format version). **Open:** blob layout.

**What is recorded, and what is kept short.** A trace that records everything is unreadable; one that drops the wrong thing hides the cause.

| Recorded | How |
|---|---|
| Every task | One span when it ends: state, reason, rejection detail, the error chain (`path`, `root`), inputs, result, resources, session, provenance. A task that did not succeed also records the World state at that moment (§1.8) |
| Every command | A `gate` span with its status (`accepted`, `rejected`, `error`) and every check verdict. A command identical to the previous accepted one from the same task is **counted on the task** instead of recorded again |
| World writes by tasks | A `world` span per write, e.g. each value a tracker publishes. Observations from the backend are not recorded one by one; they belong in a rosbag or dataset, referenced |
| Safety events | A `gate` span for each lock, unlock, fallback, watchdog, and reset, including events raised inside the Gate process |

**Reading a trace.** `roborun trace show --failed` prints, for each failure that no failing child explains, the chain of tasks from the top down. It then shows the failed task's inputs, the reason it was rejected or the root error, its last commands, nearby safety events, and the state at the end. `--json` prints the same chains for an agent. Values a `changes` loop skipped are not yet recorded.

## 5. Boundaries

### 5.1 Upper boundary: harnesses, models, people

- **Non-blocking interface.** Whatever a harness uses to drive the robot (CLI, MCP, or Python) must not block on motion: `start` returns a task id immediately; `status`, `wait(id, timeout, wake_on=[keys])`, and `cancel` are separate calls. A harness can therefore keep generating while the robot moves.
- **Waking on changes.** `wait` returns when the task ends, when the timeout expires, or when a key in `wake_on` gets a **different** value, whichever comes first, and reports which (with the new stamped value). A model working one step at a time can thus notice the cup move during a grasp without writing any code. A producer that keeps writing the same value (a tracker publishing an unmoved pose at 20 Hz) wakes no one. If two happen at once, the task ending wins.
- **Sessions.** Each harness connection gets a **session task**, a child of the root. Every task the harness starts is a child of its session task, so closing the connection cancels the whole session and the robot stops. A harness that wants a task to outlive it starts it with `detached=True`.
- **Heartbeat.** A harness may declare a heartbeat interval when it connects and then ping at that rate. A session that misses four heartbeats (at least 0.5 s) is treated as gone: its session task is cancelled with reason `watchdog` and the connection is closed. This catches an agent whose process froze or blocks while its connection stays open, which closing the connection alone does not. The interval is wall time: an agent's liveness is not simulated. A session is also the principal for authority: a detached task stays in the session that created it, and the runtime configuration may name a session the supervisor of others (§3.3).
- **Information is shared; control is not.** Every task, from any session or process, reads the same World, and reading needs no ownership. A harness task can use a tracker another session or the runtime configuration started. Actuators are different: a task gets them only through arbitration (§3.3).
- **Stop bypasses models.** A stop from a person or an operator goes directly to the Gate as a `stop` command. It never waits for a model turn or a harness loop.
- **Task instructions go through the harness.** Language instructions reach the runtime as stamped values like any other input.
- **Delegated upward:** agent loops, multi-turn generation, model-facing tool schemas, conversation context, and host sandboxing. The library exports registered tools to MCP (Appendix A) instead of defining a model-facing format of its own.

### 5.2 Lower boundary: robot software and hardware

| roborun concept | Lower-layer counterpart | Rule |
|---|---|---|
| Task holding actuators | ROS 2 action | **Wrap**: the task drives an action client; ROS 2 goal states map onto §1.1 (`EXECUTING`→`running`, `CANCELING`→`cancelling`, `CANCELED`→`cancelled`, `ABORTED`→`failed`) |
| Time base | ROS time, `/clock`, simulator clock | **Follow** |
| World | Message headers, tf2 | **View**: keep original stamps; tf2 stays the source of truth for frames |
| Gate checks | Controller limits, safety controllers | **Layer**: delegate where enforced, never assume where not (§3.4) |
| Commands → motion | Controllers: IK, interpolation, MoveIt, ros2_control | **Below the Gate**: the Gate checks commands and plans, never computes them |
| Robot model used by checks | URDF / `robot_description`, MJCF | **Reference**: never copied into Gate configuration |
| Lower-layer errors | MoveIt error codes, `FollowJointTrajectory` error codes, libfranka error flags, UR safety status | **Map** to the `BackendError` codes of §1.8, keeping the original in `raw`; each backend documents its mapping |
| Gate ownership | ros2_control claiming command interfaces | **Map**: granting ownership activates the controller that claims those interfaces |
| Trace payloads | rosbag, LeRobot datasets | **Reference** |
| An existing behavior tree or navigation stack | Its own scheduler | **Wrap as one task**; do not mix two scheduling semantics |

## 6. Changing code

Two kinds of change are distinguished:

- **Additive**: defining a function or tool that does not yet exist (e.g. generating a missing function on demand, as in Code as Policies). It takes effect immediately and does not affect running tasks.
- **Replacing**: changing code that running tasks may execute. It takes effect only at a task boundary (a *safe point*); running tasks finish under the version that started them.

**Replacing a running program.** `replace(task, new_version)` marks the program.
- **The step running now finishes.** A grasp in progress is never cut off.
- **The safe point.** The next time the program starts a step, it ends `cancelled` with reason `replaced`, and its remaining children are reaped as usual (§1.5).
- **The new version.** It is a task from the moment of the request, so it can be named and watched. It waits for the old one to end, then runs. Its provenance records the version it replaces and its own version number (§1.7).
- **State passes through the World.** The new version reads what is held and what is done, and continues from there. Nothing else is migrated, because a program's local variables are not a contract.
- **Rollback** is another replacement, with the earlier source from the provenance.
- **No safe point.** A program that never starts another step never reaches one. Cancel it instead.

---

## Appendix A. Library (non-normative)

The library gives names and helpers to common patterns. None of them adds semantics beyond §1–§5.

| Library name | What it is in spec terms |
|---|---|
| `Action` | A task that holds resources (§1.6). Named after ROS 2 actions |
| `Provider` | A task, usually detached, that holds no resources and returns or streams stamped values (§2.2). Deliberately not called *service*: in ROS 2 a service is a single request–response call |
| `@on_change(key)` | Sugar for a child task that loops over `changes(key)` and runs the function for each value |
| `cancel_and_wait()` | `cancel()` followed by `await stopped()`, named after Kotlin's `cancelAndJoin` |
| Key-following skills | A skill may accept a World key instead of a value (`grasp(target="cup")`) and re-read it with `max_age` on every control cycle. This is a convention for writing skills, not a runtime feature: it needs a producer for that key. When the value is stale (occluded, producer gone), the skill does not act on it: it stops the arm and waits for fresh data, and fails with `StaleData` after a bounded wait |
| Supervised skills | A skill that holds an object checks every cycle that it still holds it, and fails at once with a specific error (`Dropped`) instead of finishing a motion that no longer achieves anything. Contact flickers when the object touches a surface, so loss of contact alone is not a slip: the fingers must also have closed on nothing |
| `@tool` | Marks a function, usually a skill, as a tool. A host exports the tools of the modules it is given: to MCP with a JSON schema from the signature, and by name to generated code |
| Program replacement | `replace(task, code)` on the harness interface and `replace_program` over MCP: a new version of a running program takes effect at its next step (§6) |
| Program helper | Runs generated code that defines `async def main():` as a task. The source and the model call that generated it are recorded as provenance (§1.7). The task's trace summary goes back for the next model turn |
| Simulation controls | `sim("push", object=..., to=...)` changes the simulated world, through the runtime to the Gate process that owns the simulator. It is not a robot command: it carries no token and passes no check, and it exists only for simulator backends. Harnesses (`sim_push` over MCP) and operators (`roborun sim push`) use it to test how running tasks react; it is traced as a `world` span |
| Host and `roborun mcp` | The robot's one runtime process (§3.1) serves harness sessions over a Unix socket; `roborun mcp` translates MCP for harnesses such as Claude Code |
| Backends | MuJoCo, LeRobot, ROS 2, each implementing §5.2 |
| Skill convention | An Action succeeds only when its effect is complete *and the robot is at rest*, so the next sequential step never starts while the previous motion is still settling |
| `FakeBackend` | A kinematic stand-in for tests: stopping takes time, objects can be nudged, runs on the stepped clock |
| Common checks | Ready-made check functions for §3.4: workspace box, keep-out zones (with a robot model), speed caps, distance to people. They are examples of policy, not part of the Gate |

A multi-turn Code-as-Policies loop, for example, needs nothing beyond the spec:

```python
feedback = None
while True:
    code = await llm.query(instruction, feedback)           # stamped value
    prog = start(program(code))
    await prog.stopped()
    if prog.state is TaskState.SUCCEEDED:
        break
    feedback = trace.summary(prog)
```

## Appendix B. Programming model (non-normative)

The same API serves people and models. Each form below matches a pattern that mainstream languages already use, so neither needs to learn a new style.

| Need | roborun | Mainstream precedent |
|---|---|---|
| Do it and wait | `await grasp(cup)` | `await f()` in Python, JavaScript, C#, Swift; suspend functions in Kotlin |
| Start without waiting | `g = start(grasp(cup))` | `asyncio.create_task`, Kotlin `launch`, Swift `Task {}`, Go `go`, `tokio::spawn` |
| Cancel | `g.cancel()`, then `await g.stopped()` | asyncio `cancel` then `await`; Kotlin `cancel` then `join` (`cancelAndJoin`) |
| Children stop with their parent | implicit task tree | Kotlin `coroutineScope`, Swift `TaskGroup`, Trio nurseries, asyncio `TaskGroup` |
| React while something runs | `async for x in changes(key, while_running=g)` | RxJS `takeUntil`, Go `select` loop, `for await` / `await foreach` over streams |
| Time limit | `async with timeout(5):` | `asyncio.timeout`, Trio `move_on_after`, Kotlin `withTimeout` |
| Too old to use | `get(key, max_age=0.1)` | HTTP `Cache-Control: max-age` |
| Exclusive use | `@action(resources=...)`, `async with hold(...)` | Java `synchronized` methods; `async with lock:` |
| Wait for whichever comes first (harness) | `wait(id, timeout, wake_on=[keys])` | `poll`/`select`, `asyncio.wait(FIRST_COMPLETED)`, `Promise.race` |

The running example, written by a person or generated by a model:

```python
from roborun import start, changes, world

start(track("cup"), detached=True)          # who keeps "cup" fresh is explicit

cup = await world.next("cup")               # wait for the tracker's first value
g = start(grasp(cup.value))
async for cup in changes("cup", while_running=g):
    if moved(cup, g.target):                # which changes matter is ordinary code
        g.update(target=cup.value)
await g
await place(on="tray")
```

Line by line: start a tracker that keeps writing `cup`; read the cup's pose; start the grasp without waiting; for each new pose of the cup while the grasp is still running, redirect the grasp if the cup actually moved; wait for the grasp to finish; place. If the tracker dies, the loop raises `StaleData` instead of silently following an old pose.

The same task in one-step-at-a-time form, through the harness interface:

```
start(action="grasp", target="cup")              -> {"id": "t7", "state": "running"}
wait(id="t7", timeout=5, wake_on=["cup"])        -> {"state": "running", "woke_on": "cup", "value": {...}, "age": 0.03}
update(id="t7", target="cup")                    -> ok
wait(id="t7", timeout=5)                         -> {"state": "succeeded"}
```
