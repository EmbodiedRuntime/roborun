# Guava's benchmark on roborun

Runs [Guava](https://github.com/hdacnw/guava-release)'s tasks with Guava's own prompt, tools,
parser, and scorer, under two harnesses that differ only in how tools run:

- `blocking`: Guava's loop. A motion tool returns when the motion is over.
- `roborun`: a motion tool starts a task and returns at once; `wait` (with `wake_on`) and
  `cancel` are added.

Both run with simulated time following wall time, so the world does not pause while the model
thinks. Guava is used read-only from a separate checkout (`GUAVA_ROOT`); nothing in it changes.

How it works (`adapter.py`): each Guava motion tool runs in a worker thread; every
`RoboEnv.step(action)` it makes is handed to a roborun task, sent through the Gate as a
`setpoint`, and applied by the backend on the next control step. Cancelling the task stops the
tool at its next step and the arm holds where it is. Perception is a perfect segmenter: masks
come from MuJoCo's segmentation buffer for the same camera (`perception.py`), so Guava's
point-cloud pipeline is unchanged and only SAM3 is replaced.

Set up a separate checkout of Guava anywhere (not your working copy), pinned to the commit the
adapter was written against, and point `GUAVA_ROOT` at it:

```sh
export GUAVA_ROOT=/path/to/guava-bench
git clone https://github.com/hdacnw/guava-release "$GUAVA_ROOT"
git -C "$GUAVA_ROOT" checkout 027cbc2
git -C "$GUAVA_ROOT" submodule update --init third_party/robosuite third_party/sam3   # sam3: metadata only
(cd "$GUAVA_ROOT" && UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --locked --no-default-groups --group eval)

# from the roborun checkout
"$GUAVA_ROOT/.venv-eval/bin/python" benchmarks/guava/run.py --model scripted --seeds 0 1 2
OPENROUTER_API_KEY=... "$GUAVA_ROOT/.venv-eval/bin/python" benchmarks/guava/run.py \
    --model openai/gpt-5.4 --seeds 0 1 2 3 4 --out benchmarks/guava/results/gpt-5.4.json
```

`--guava PATH` overrides `GUAVA_ROOT`; if neither is set, an importable `guava` package is used.
Use `--backend openai --model gpt-5.4` with `OPENAI_API_KEY` to call OpenAI directly.

The push (`--push yes`) slides the can 6 cm while `grasp` descends onto it, once per episode.

## Results: GPT-5.4 (OpenRouter, reasoning medium), can_in_bin, 5 seeds

`results/gpt-5.4.json`. Ground-truth masks instead of SAM3; simulated time follows wall time in
both harnesses; a reply with several tool calls runs only the first, malformed JSON gets one retry.

| harness | push | success | median time | median decisions | grasp closed on nothing | format errors |
|---|---|---|---|---|---|---|
| blocking | no | 5/5 | 62.0 s | 7 | 0 | 1 |
| roborun | no | 5/5 | 75.4 s | 9 | 0 | 5 |
| blocking | yes | 5/5 | 87.6 s | 11 | 1 | 0 |
| roborun | yes | 5/5 | 94.8 s | 12 | 1 | 2 |

In every pushed roborun episode the grasp returned early ("can moved; grasp is still running"),
and every time GPT-5.4 chose `wait` over `cancel`. Interruptible tools only help if the model
acts on the interruption; here it did not, and the longer prompt cost format errors and
decisions. Reactions that live in the skill, not in the model, are the next arm to try.

With a 12 cm push (`results/gpt-5.4-push12.json`), more than the gripper's footprint, a grasp
aimed at the old spot always misses:

| harness | success | median time | grasp closed on nothing | model's choice when woken |
|---|---|---|---|---|
| blocking | 5/5 | 105.3 s | 5 | — |
| roborun | 5/5 | 113.6 s | 4 | `cancel` once, `wait` four times |

The one episode where GPT-5.4 cancelled avoided the empty grasp and finished in 85 s against
110 s for blocking on the same seed. The mechanism works when the model uses it; GPT-5.4 rarely
does.
