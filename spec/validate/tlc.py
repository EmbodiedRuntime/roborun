"""Ask TLC whether a sequence of steps is a behaviour of Gate.tla (Kuppe et al., 2024).

The steps become a module that extends the model with a position in the log. Its next-state
relation is the model's ``Next``, restricted at each position to the logged action and task,
and to the logged state where there is one. TLC explores it breadth first; the run is a
behaviour of the model exactly if some path reaches the end of the log.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
JAR = SPEC / ".tools" / "tla2tools-1.7.4.jar"


@dataclass
class Result:
    ok: bool
    matched: int  # the steps TLC could follow
    total: int
    failed: dict | None  # the first step no behaviour of the model takes
    output: str = ""


def _set(xs) -> str:
    return "{" + ", ".join(f'"{x}"' for x in sorted(xs)) + "}"


def _bool(b: bool) -> str:
    return "TRUE" if b else "FALSE"


def _record(s: dict) -> str:
    o = s["obs"] or {"valid": (), "revoked": (), "locks": (), "failing": False, "watch": False,
                     "lag": ()}
    t = "None" if s["t"] is None else f'"{s["t"]}"'
    return (f'[a |-> {_set(s["a"])}, t |-> {t}, o |-> {_bool(s["obs"] is not None)}, '
            f'v |-> {_set(o["valid"])}, r |-> {_set(o["revoked"])}, l |-> {_set(o["lag"])}, '
            f'k |-> {_set(o["locks"])}, f |-> {_bool(o["failing"])}, w |-> {_bool(o["watch"])}]')


MODULE = """---- MODULE Trace ----
EXTENDS Gate, Sequences

VARIABLE i   \\* the next step of the log

TraceTasks == %(tasks)s

Log == <<
%(log)s
>>

Seen(e) == /\\ \\A t \\in Tasks \\ e.l : /\\ (grant'[t] = "valid") <=> (t \\in e.v)
                                    /\\ (grant'[t] = "revoked") <=> (t \\in e.r)
           /\\ locks' = e.k /\\ failing' = e.f /\\ watch' = e.w

TraceInit == Init /\\ i = 1

TraceNext == /\\ i <= Len(Log)
             /\\ i' = i + 1
             /\\ Next
             /\\ step'.a \\in Log[i].a /\\ step'.t = Log[i].t
             /\\ Log[i].o => Seen(Log[i])

TraceSpec == TraceInit /\\ [][TraceNext]_<<vars, i>>
====
"""

CFG = """CONSTANTS
  None = None
  Tasks <- TraceTasks
  Allow = %(allow)s  Recheck = TRUE  RecheckRelease = TRUE  FaultRejects = TRUE  UnauthFallback = FALSE  StopOnRelease = TRUE
SPECIFICATION TraceSpec
CHECK_DEADLOCK FALSE
"""


def check(tasks: list[str], log: list[dict], allow: bool, out: Path) -> Result:
    if not log:
        return Result(True, 0, 0, None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "Trace.tla").write_text(MODULE % {
        "tasks": _set(tasks), "log": ",\n".join("  " + _record(s) for s in log)})
    (out / "Trace.cfg").write_text(CFG % {"allow": _bool(allow)})
    r = subprocess.run(
        ["java", "-XX:+UseSerialGC", f"-DTLA-Library={SPEC}", "-cp", str(JAR), "tlc2.TLC",
         "Trace", "-config", "Trace.cfg", "-workers", "1", "-metadir", str(out / "meta")],
        check=False, cwd=out, capture_output=True, text=True)
    m = re.search(r"The depth of the complete state graph search is (\d+)", r.stdout)
    if r.returncode != 0 or m is None:
        raise RuntimeError(f"TLC failed (exit {r.returncode}):\n{r.stdout[-3000:]}")
    matched = int(m[1]) - 1
    ok = matched == len(log)
    return Result(ok, matched, len(log), None if ok else log[matched], r.stdout)
