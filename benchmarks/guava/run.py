"""Run Guava's can_in_bin with a blocking harness and with roborun, with and without a push.

    $GUAVA_ROOT/.venv-eval/bin/python benchmarks/guava/run.py --model scripted --seeds 0 1
    OPENROUTER_API_KEY=... $GUAVA_ROOT/.venv-eval/bin/python benchmarks/guava/run.py \\
        --model openai/gpt-5.4 --seeds 0 1 2 3 4 --out benchmarks/guava/results/gpt-5.4.json

``GUAVA_ROOT`` (or ``--guava``) is a checkout of guava-release; see README.md.

Both harnesses use Guava's prompt, tools, parser, and scorer, the same scene for a seed, and
simulated time that follows wall time. The scripted models are for testing the plumbing; they
follow Guava's suggested sequence (align, grasp, lift, align over the box, release).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
adapter = None  # imported in main(), after --guava is applied


def _call(name: str, **arguments: object) -> str:
    return f"<think>next step</think>\n<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


DONE = "<think>The can is in the box.</think>\nTask complete."


def _last_text(messages: list[dict]) -> str:
    content = messages[-1]["content"]
    return content if isinstance(content, str) else content[-1]["text"]


def _gripper(text: str) -> list[float]:
    return [float(v) for v in re.search(r"position \[([^\]]+)\]", text)[1].split(",")]


def scripted_blocking():
    state = {"step": 0}

    def model(messages: list[dict]) -> str:
        text, s = _last_text(messages), state["step"]
        if s == 2 and "no object grasped" in text:  # missed: open, re-align, retry
            state["step"] = 0
            return _call("release")
        plan = [lambda: _call("align", target_name="can", position="top", clearance="small"),
                lambda: _call("grasp", target_name="can"),
                lambda: _call("move", target_position=[*_gripper(text)[:2], _gripper(text)[2] + 0.15]),
                lambda: _call("align", target_name="box", position="top", clearance="medium"),
                lambda: _call("release"),
                lambda: DONE]
        state["step"] = s + 1
        return plan[min(s, len(plan) - 1)]()

    return model


def scripted_roborun():
    state = {"step": 0}

    def model(messages: list[dict]) -> str:
        text = _last_text(messages)
        m = re.search(r"still running as task (\d+)", text)
        if m and "can moved" in text:  # pushed during the grasp: stop, then start over
            state["step"] = 0
            return _call("cancel", task_id=m[1])
        if "no object grasped" in text or "finished: closed" in text:
            state["step"] = 0
            return _call("release")
        s = state["step"]
        plan = [lambda: _call("align", target_name="can", position="top", clearance="small"),
                lambda: _call("grasp", target_name="can", wake_on=["can"]),
                lambda: _call("move", target_position=[*_gripper(text)[:2], _gripper(text)[2] + 0.15]),
                lambda: _call("align", target_name="box", position="top", clearance="medium"),
                lambda: _call("release"),
                lambda: DONE]
        state["step"] = s + 1
        return plan[min(s, len(plan) - 1)]()

    return model


def gpt(model_name: str, backend: str, reasoning: str | None):
    import functools

    from guava.inference.remote_eval_client import RemoteEvalClient

    client = RemoteEvalClient(backend, model_name)
    if reasoning and backend == "openrouter":  # Guava passes reasoning effort only to OpenAI/NVIDIA
        client.provider.complete = functools.partial(client.provider.complete,
                                                     reasoning={"effort": reasoning})

    def model(messages: list[dict]) -> str:
        r = client.create(model=model_name, messages=messages, temperature=0, max_tokens=8192)
        return r.choices[0].message.content

    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scripted",
                    help="scripted, or a hosted model: gpt-5.4 (openai) or openai/gpt-5.4 (openrouter)")
    ap.add_argument("--backend", choices=("openai", "openrouter"), default="openrouter",
                    help="hosted provider; reads OPENAI_API_KEY or OPENROUTER_API_KEY")
    ap.add_argument("--reasoning", default="medium", help="reasoning effort over OpenRouter, or ''")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variants", nargs="+", default=["blocking", "roborun"])
    ap.add_argument("--push", choices=("no", "yes", "both"), default="both")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--guava", type=Path, help="guava-release checkout (default: $GUAVA_ROOT)")
    args = ap.parse_args()
    if args.guava:
        os.environ["GUAVA_ROOT"] = str(args.guava)
    global adapter
    import adapter
    engine = adapter.make_engine()
    pushes = {"no": [False], "yes": [True], "both": [False, True]}[args.push]
    results = []
    for seed in args.seeds:
        for push in pushes:
            for variant in args.variants:
                if args.model == "scripted":
                    model = scripted_blocking() if variant == "blocking" else scripted_roborun()
                else:
                    model = gpt(args.model, args.backend, args.reasoning)
                r = adapter.run(variant, model, seed, push, engine)
                results.append(r)
                print(json.dumps({k: v for k, v in r.items() if k not in ("log", "checks")}), flush=True)
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
