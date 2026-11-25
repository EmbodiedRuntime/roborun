"""Is a recorded run of the Gate a behaviour of Gate.tla?   python spec/validate RUN.journal.jsonl ...

The journal is what ``gate.record()`` saves (``roborun.journal``). Its events are turned
into steps of the model (``gate_trace``) and TLC is asked whether the model can take them
in this order, ending in the states the Gate was in (``tlc``).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import gate_trace
import tlc

from roborun import journal


def validate(meta: dict, entries: list[dict], out: Path | None = None, actuator: str = "arm") -> tlc.Result:
    tasks, log, allow = gate_trace.steps(meta, entries, actuator)
    if out is not None:
        return tlc.check(tasks, log, allow, out)
    with tempfile.TemporaryDirectory(prefix="roborun-validate-") as tmp:
        return tlc.check(tasks, log, allow, Path(tmp))


def explain(r: tlc.Result, entries: list[dict]) -> str:
    if r.ok:
        return f"a behaviour of Gate.tla: {r.total} steps"
    f = r.failed
    entry = next(e for e in entries if e["n"] == f["n"])
    short = {k: entry[k] for k in ("n", "event", "args", "result", "raised", "effects") if k in entry}
    return (f"not a behaviour of Gate.tla: the model follows {r.matched} of {r.total} steps, "
            f"then cannot take {sorted(f['a'])} for {f['t']}\n"
            f"  the event: {json.dumps(short)}\n  the Gate's state after it: {json.dumps(entry['state'])}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    bad = 0
    for path in argv:
        meta, entries = journal.load(path)
        try:
            r = validate(meta, entries)
        except gate_trace.Outside as e:
            print(f"{path}: outside the model: {e}")
            bad = 1
            continue
        print(f"{path}: {explain(r, entries)}")
        bad |= not r.ok
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
