# Positioning

What roborun is, what it is not, and how it relates to the systems around it.
The normative semantics are in [SPEC.md](../SPEC.md); this document explains where they sit.

## In one sentence

roborun gives programs written by agents (and people) well-defined execution semantics on a robot: tasks that can be redirected and stopped, information that says how old it is, a gate that every command must pass, and a trace from each decision to its physical outcome.

## Where it sits

```
 ┌────────────────────────────────────────────────────────────┐
 │ Above (clients): Claude Code · Codex · pi · LangGraph       │  call roborun through Python, CLI, or MCP
 │                  VLMs, VLAs, people                          │  a VLA runs as a task; a person can always stop
 ├────────────────────────────────────────────────────────────┤
 │ roborun spec:    Task · Stamped values/World · Gate · Trace │
 │         library: Action · Provider · @on_change · tools→MCP │
 ├────────────────────────────────────────────────────────────┤
 │ Below (backends): ROS 2 · dora-rs · LeRobot · MuJoCo         │  used, not replaced
 │                   controllers, ros2_control, hardware safety │
 └────────────────────────────────────────────────────────────┘
```

roborun owns only what neither layer can do alone, because it needs to know both the decision above and the execution below (SPEC, *What roborun owns*):
the task tree from a program down to its motions, freshness checks at decision time, task-level control authority, and a causal trace from decision to outcome.

## An analogy: what coding agents already have

Coding agents work well partly because Unix gives them clear semantics. A robot needs the same, under harder physics:

| Coding agent's environment | roborun | What is harder on a robot |
|---|---|---|
| Process | Task | A cancel request is not a stop (`cancelling` ≠ `cancelled`) |
| Signal | Changes in World | The world produces changes on its own (the cup is nudged) |
| File system | Stamped values and World | Values go stale and must carry their age |
| Permissions, sandbox | Gate | Actions are irreversible, so they must be checked before they happen |
| Logs, strace | Trace | It must let research mode repair the program |

SPEC, *Why coding agents do not need it*, gives the full argument.

## Relationship to other systems

| System | Relationship | What it already has | What roborun adds |
|---|---|---|---|
| ROS 2 (actions, executors, topics) | Backend below | Goal, feedback, cancel; communication; timestamps; `/clock` | A small API agents can write; task tree across layers; freshness at decision time; task-level Gate; causal trace. Writing ROS 2 directly is too verbose for LLMs |
| dora-rs, LeRobot | Backends below | Dataflow; hardware access; LeRobot datasets and policies | Task semantics and the Gate |
| ros2_control | Below | Controllers claim command interfaces exclusively | Arbitration by *task* (who preempts whom, and the preempted task stops first); SPEC §5.2 maps ownership onto controller activation |
| Hardware safety (e-stop, safety controllers, collision reflexes) | Below, last line of defense | Joint-level protection | Task- and scene-level checks above it; the Gate never replaces it |
| Behavior trees (BehaviorTree.CPP, py_trees), Nav2 | Closest prior art; interoperate | Reactive execution, halt | Built for fixed, hand-written structure, not code an agent writes and rewrites; no freshness or causal trace. An existing tree runs as one task |
| Temporal, Ray | Ideas borrowed, not used | Durable long-running workflows, signals, actors | Physical-time semantics and robot safety; Temporal's replay does not apply to the physical world |
| Agent harnesses (Claude Code, Codex, pi, LangGraph) | Clients above | Agent loop, tool schemas, sandbox, context | The same execution semantics whichever harness drives the robot. Agent as Policy found that changing the harness changed robot behavior (zero overlap of motion and generation under a blocking shell call) |
| Research systems (ASPIRE, GaP, TypeGo, Agent as Policy, ReKep) | roborun aims to be their common substrate | Each built part of this: ASPIRE's trace, Agent as Policy's bridge, TypeGo's reflexes and Skill Kernel | One set of semantics they could build on, and compare on |
| Benchmarks (LIBERO, BEHAVIOR-1K) | Complementary | Task success | A disturbance benchmark measuring system properties: reaction latency, safe stopping, recovery from failure |

## Expressiveness check: surveyed systems in roborun terms

| System | Expressed with |
|---|---|
| Guava (one tool per turn) | A harness calling `start` / `wait(wake_on=...)` per step (SPEC §5.1) |
| Code as Policies | A task whose body is generated code (SPEC §1.7) |
| CaP-X multi-turn | Run a program, `await stopped()`, return `trace.summary` to the model (SPEC Appendix A) |
| ASPIRE | Trace of every primitive; programs rewritten between runs from that trace |
| ReKep | A tracker task writing a state key at 20 Hz; an optimizer task at 10 Hz reading it with `max_age` |
| TypeGo | Tasks at different rates; preemption through Gate arbitration; reflexes as tasks with no model on the path |
| HAMSTER | A VLM call per episode returning a stamped path; a VLA running as a task that holds the arm |
| Agent as Policy | Its bridge is the Gate; `frames` and `state` are World reads |

Because no concept encodes who decides or how, all combinations of {sequential, event-driven} × {programmatic, model-mediated} decisions are expressible (SPEC, *Mechanism, not policy*).

## What is new

Each primitive has precedent on its own: ROS 2 actions can be cancelled, ASPIRE has traces, Agent as Policy has a bridge, Erlang has supervision trees, capability systems have tokens.
What is new is (1) unifying them into one set of semantics an agent can write against, (2) fixing those semantics in a spec, and (3) showing with a benchmark that they matter.

To answer "isn't this just ROS 2 actions?", two things are needed: the cup-nudge demo (the same agent with and without these semantics) and the ROS 2 bridge (roborun adds a layer above ROS 2 rather than replacing it).

## Non-goals

Middleware or a message bus; models; agent harnesses; simulators; policy learning; hard real-time control; multi-robot coordination (for now).

## Risks

| Risk | Mitigation |
|---|---|
| Seen as yet another framework | Spec first; adapters instead of replacement; the spec stays at four concepts |
| Seen as a thin asyncio wrapper | The value must show in the Gate, freshness, and trace, and be measured on the benchmark |
| Building code replacement during a run (Pattern D) too early | v0 only reserves versions and safe points (SPEC §6) |
| Hidden dependencies make code look simpler than it is | Producers are started explicitly; a missing producer raises `StaleData`; overflow raises `Overflow` |
