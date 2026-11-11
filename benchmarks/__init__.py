"""roborun-bench: one suite per part of the runtime, each with one main metric.

Run from the repository root; the suites are not installed with the roborun package.

    python -m benchmarks safety      # Gate: unsafe commands from buggy programs; do they reach the arm?
    python -m benchmarks crash       # the agent process dies mid-motion; does the arm stop?
    python -m benchmarks stopping    # cancel, code that ignores it, two agents, operator stop
    python -m benchmarks concurrency # a slow planner: does the arm wait for every decision?
    python -m benchmarks stale       # World: the data stops; does the robot act on old values?
    python -m benchmarks trace       # a model diagnoses failures from an error line or the trace
    python -m benchmarks ablation    # every suite with one part of roborun off at a time
    python -m benchmarks model-safety  # GPT-5.4 writes the programs; shadow vs enforced Gate
    python -m benchmarks disturbances  # pushes, a moving cup, occlusion, a slip, removal

Every suite runs the same programs twice: once on a baseline (no runtime protections, as a
program calling a robot driver directly would have) and once on roborun. Results are
deterministic (the simulation steps in lockstep) unless a suite says otherwise.

``--ablate gate,task,world`` (or ``all``) adds an arm per part: roborun with only that part
switched off (``benchmarks.ablate``). ``python -m benchmarks ablation`` runs them all as a matrix.
"""

SUITES = ("safety", "crash", "stopping", "concurrency", "stale", "trace")
