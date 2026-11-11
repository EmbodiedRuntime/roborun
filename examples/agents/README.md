# Agents on roborun: Code as Policies and ReAct

The two common ways a model drives a robot, each as one short file on top of the harness
interface (SPEC §5.1). The agent loop is ordinary code; what makes it short is what the
runtime already provides.

| File | The model… | Loop |
|---|---|---|
| [`code_as_policies.py`](code_as_policies.py) | writes a whole program; it runs as one task; if it fails, the trace is the feedback for the next attempt | 24 lines |
| [`react.py`](react.py) | calls one tool per turn, observes, decides again, while the robot keeps moving | 20 lines, plus 19 mapping tool names to calls |
| [`event_replan.py`](event_replan.py) | replans when a new instruction arrives mid-task (event-driven, decided by the model) | |
| [`thin_tools.py`](thin_tools.py) | has only `move_to`, `gripper`, and the World (Agent as Policy) | the ReAct loop |
| [`graph_policy.py`](graph_policy.py) | writes a graph that an interpreter runs without it (GaP) | ~30 lines |
| [`live_patch.py`](live_patch.py) | fixes a program while it runs; the fix takes effect at a safe point (pattern D) | |
| [`models.py`](models.py) | any text model: Claude Code headless (`claude -p`, default), the Claude API, or a script (tests) | |
| [`robot.py`](robot.py) | connects to a running `roborun serve`, or simulates the table scene in-process | |

```sh
pip install -e ".[mujoco]"
python examples/agents/code_as_policies.py "put both cups on the tray"
python examples/agents/react.py "put the red cup on the tray" --push cup_red 0.45 0.05 12
python examples/agents/react.py "..." --model api      # the Claude API instead of Claude Code
python examples/agents/react.py "..." --view           # watch in the MuJoCo viewer
```

## What a harness needs, and what provides it

| Harness need | Code as Policies | ReAct | roborun piece |
|---|---|---|---|
| Run generated code without trusting it | ✓ | | `program(code)` runs it as a task; every command it sends passes the Gate (ownership, checks) |
| Act without blocking the model | | ✓ | a skill call returns a task id at once; the robot keeps moving while the model thinks |
| Notice the world change during an action | ✓ (skills follow by name) | ✓ (`wait(..., wake_on=[...])`) | key-following skills; `wait` wakes on a changed value |
| Correct course mid-action | | ✓ | `update` redirects a running task; `cancel` stops it, through `cancelling` |
| Know what is on the table | ✓ | ✓ | `scene.objects` in the World, with sizes |
| Know why something failed | ✓ (feedback for the next attempt) | ✓ (in the observation) | `trace(id)`: the chain of tasks, rejection code or root error, state at the end |
| Not damage anything while learning | ✓ | ✓ | Gate checks reject the command and cancel the task; the arm stops |
| Clean up | ✓ | ✓ | closing the session cancels everything the agent started |
| Provenance | ✓ | | each program records its source and the model turn that wrote it |

## What the runs looked like (Claude Code as the model, simulated Panda)

**Code as Policies, "put both cups on the tray"**: succeeded on the first attempt. The model
read the tray's size and the cups' radius from `scene.objects`, placed the cups 5 cm either
side of the tray center so they would not collide, and wrapped each grasp in a retry.

**ReAct, "put the red cup on the tray", someone pushes the cup 12 s in**: nine steps. The model
listed the objects, tracked the cup and the tray, and started `grasp("cup_red")`. The push came
while the grasp was running; the grasp followed the cup by itself, then the model placed it
and verified the cup's position. Each model turn took about 2.5 s, while the robot kept moving.

**In the tests** ([`tests/test_agent_examples.py`](../../tests/test_agent_examples.py)) scripted
models check both loops offline:
- a first program that drives the gripper into the table is rejected (`check:workspace`); the
  trace becomes the feedback, and the second program succeeds;
- a ReAct agent waiting on a grasp is woken by the push (`woke: "cup_red"`), switches the grasp
  to follow the cup by name, and places it;
- calls to tools that do not exist, and replies that are not JSON, become observations.
