"""The Code-as-Policies and ReAct examples, driven by scripted models (examples/agents)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

mj = pytest.importorskip("roborun.backends.mujoco")
if not (mj.menagerie_path() / "franka_emika_panda/panda.xml").exists():
    pytest.skip("MuJoCo Menagerie Panda model not found", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "agents"))
from code_as_policies import code_as_policies
from models import scripted
from react import react
from robot import robot


def in_tray(p):
    return abs(p[0] - 0.3) < 0.08 and abs(p[1] - 0.38) < 0.055 and abs(p[2] - 0.05) < 0.008


INTO_THE_TABLE = '''```python
async def main():
    start(track("cup_red"), detached=True)
    start(track("tray"), detached=True)
    await move_to((0.5, -0.1, -0.02))  # a bug: below the table top
    await grasp("cup_red")
    await place("tray")
```'''

FIXED = '''```python
async def main():
    start(track("cup_red"), detached=True)
    start(track("tray"), detached=True)
    await grasp("cup_red")
    await place("tray")
```'''


def test_code_as_policies_repairs_a_rejected_program_from_the_trace():
    seen = []

    async def model(system, messages):
        seen.append(messages[-1]["content"])
        return INTO_THE_TABLE if len(seen) == 1 else FIXED

    async def go():
        async with robot(pace=None) as client:
            result = await code_as_policies(client, model, "put the red cup on the tray")
            cup = (await client.world("cup_red"))["cup_red"]["value"]
            return result, cup

    result, cup = asyncio.run(go())
    assert "cup_red" in seen[0] and "tray" in seen[0]  # the scene went into the first prompt
    assert "check:workspace" in seen[1] and "move_to" in seen[1]  # the trace became feedback
    assert result["state"] == "succeeded" and in_tray(cup)


def last_observation(messages):
    text = messages[-1]["content"]
    return json.loads(text.split("Observation: ", 1)[1]) if "Observation: " in text else None


def test_react_notices_the_push_while_grasping_and_redirects():
    state = {"step": 0}

    def say(thought, tool=None, **args):
        if tool is None:
            return json.dumps({"thought": thought, "done": args["done"]})
        return json.dumps({"thought": thought, "tool": tool, "args": args})

    async def model(system, messages):
        obs = last_observation(messages)
        state["step"] += 1
        s = state["step"]
        if s == 1:
            return say("Track the cup first.", "track", name="cup_red")
        if s == 2:
            return say("And the tray.", "track", name="tray")
        if s == 3:
            return say("Where is the cup?", "world", keys=["cup_red"])
        if s == 4:
            return say("Grasp it there.", "grasp", target=obs["cup_red"]["value"])
        if s == 5:
            state["grasp"] = obs["id"]
            await state["client"].push("cup_red", (0.45, 0.05))  # someone, meanwhile
            return say("Wait, but wake me if the cup moves.", "wait", task=obs["id"],
                       timeout=10, wake_on=["cup_red"])
        if s == 6:
            state["woke"] = obs["woke"]
            # It is still sliding: follow it by name instead of chasing one position.
            return say("The cup is moving; follow it by name.", "update", task=state["grasp"],
                       params={"target": "cup_red"})
        if s == 7:
            return say("Wait for the grasp.", "wait", task=state["grasp"], timeout=20)
        if s == 8:
            state["grasped"] = obs["state"]
            return say("Now the tray.", "place", target="tray")
        if s == 9:
            return say("Wait for the place.", "wait", task=obs["id"], timeout=20)
        return say("Done.", done=f"place {obs['state']}")

    async def go():
        async with robot(pace=None) as client:
            state["client"] = client
            summary = await react(client, model, "put the red cup on the tray")
            cup = (await client.world("cup_red"))["cup_red"]["value"]
            return summary, cup

    summary, cup = asyncio.run(go())
    assert state["woke"] == "cup_red"
    assert state["grasped"] == "succeeded" and summary == "place succeeded"
    assert in_tray(cup)


def test_react_turns_bad_calls_into_observations():
    replies = [json.dumps({"thought": "fly", "tool": "fly", "args": {}}),
               "not json at all",
               json.dumps({"thought": "stop", "done": "gave up"})]
    seen = []
    model = scripted(replies)

    async def recording(system, messages):
        seen.append(messages[-1]["content"])
        return await model(system, messages)

    async def go():
        async with robot(pace=None) as client:
            return await react(client, recording, "anything")

    assert asyncio.run(go()) == "gave up"
    assert "no tool 'fly'" in seen[1] and "Invalid reply" in seen[2]
