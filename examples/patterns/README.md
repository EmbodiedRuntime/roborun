# Programming patterns from the blog, as running code

[Understanding Agentic Robotics from a System Perspective](https://embodiedruntime.github.io/blog/)
sorts agentic robot systems by how their control flow runs, who decides at run time, and
what triggers the next step. Every pattern it names runs on the same four concepts:
Task, Stamped values/World, Gate, and Trace. Each example below is one file, runs on the
simulated Panda, and has a test (`tests/test_pattern_examples.py`, `tests/test_agent_examples.py`).

| The blog's pattern | Example | What carries it in roborun |
|---|---|---|
| **Sequential, decided by a program** (Code as Policies) | [`agents/code_as_policies.py`](../agents/code_as_policies.py) | a generated `async def main()` runs as a task behind the Gate |
| **Sequential, decided by the model** (ReAct, Guava) | [`agents/react.py`](../agents/react.py) | skills return at once; `wait(wake_on=)`; `update`, `cancel` |
| **Event-driven, decided by a program** (a slip monitor beside the grasp) | [`patterns/parallel_monitor.py`](parallel_monitor.py) | `on_change` handlers and `changes` loops beside a running task; `cancel_and_wait` |
| **Event-driven, decided by the model** (event-triggered replanning) | [`agents/event_replan.py`](../agents/event_replan.py) | instructions as stamped messages on a queue key; cancel, then replan from the real state |
| **Fixed loop / two-level hierarchy** (VoxPoser, ReKep, HAMSTER) | [`patterns/two_rate.py`](two_rate.py) | the model sets a constraint; a 20 Hz task streams setpoints through the Gate's stream checks and watchdog |
| **Thin tools** (Agent as Policy) | [`agents/thin_tools.py`](../agents/thin_tools.py) | the same ReAct loop with only `move_to`, `gripper`, and the World |
| **Program as a graph** (GaP) | [`agents/graph_policy.py`](../agents/graph_policy.py) | each node is a task; outcome edges; `watch` edges leave a node early |
| **Pattern C: regenerate between runs** (ASPIRE) | [`agents/code_as_policies.py`](../agents/code_as_policies.py) | `trace(id)` explains the failure; the next attempt is a new program with provenance |
| **Pattern D: modify while running** | [`agents/live_patch.py`](../agents/live_patch.py) | `replace(task, code)` takes effect at the next task boundary (SPEC §6); the new version reads the World |

```sh
pip install -e ".[mujoco]"
python examples/patterns/parallel_monitor.py
python examples/patterns/two_rate.py
python examples/agents/event_replan.py              # add --model claude-code for a real model
python examples/agents/graph_policy.py
python examples/agents/live_patch.py
python examples/agents/thin_tools.py "put the red cup on the tray"   # needs a model
```

## What the runs show

- **Monitor beside the grasp.** A handler keeps the grasp aimed at the cup while someone pushes it. A monitor watching the gripper cancels the carry when the cup slips, waits for the arm to stop, and goes back for it. Two attempts, about 11 s.
- **Two rates.** While the cup rides a conveyor at 3 cm/s, the 20 Hz loop keeps the gripper within 5 mm (median) of 12 cm above it. All 200 setpoints pass the Gate's `max_jump` check. When the cup is taken away, the loop stops the arm instead of following stale data. The model is asked again and parks the arm over the tray.
- **A new instruction mid-task.** "Put the blue cup first" arrives during the red cup's grasp. The plan is cancelled, the robot comes to rest, and the model plans from what the gripper actually holds; both cups end on the tray, 10 cm apart. With Claude Code as the model, the second instruction arrived while the first plan was still being written. The queue kept it, and the first plan was cancelled as soon as it started.
- **Graph.** The cup slips during `place`. The `Dropped` edge leads back to `grasp`, and the graph finishes without the model.
- **Live patch.** The fix is submitted while version 1 is grasping the red cup. The grasp finishes; at its next step version 1 ends (`replaced`). Version 2 sees the cup in the gripper, puts it on the left of the tray, then puts the blue cup on the right.
- **Thin tools.** With Claude Code as the model, 24 steps: open, approach, descend, close, check, lift, carry, lower, release, verify.

Two of these runs found problems in the simulated robot, now fixed:
- **A grasp declared over too soon.** A grasp ended while one finger was still centering the cup.
- **The grip and its contacts.** The gripper squeezed with about 2 N, and soft contacts let a held cup creep out within seconds. Friction also needed to differ between the table and the finger pads.
