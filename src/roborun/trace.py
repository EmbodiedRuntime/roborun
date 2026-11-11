"""The trace (SPEC §4): enough to find why a run failed, for a person or a coding agent.

Spans are JSON Lines, one file per run. The first line describes the run; every other line
is a span ``{id, parent, kind, name, start, end, status, inputs, outputs, refs, ...}`` with
``kind`` one of ``task``, ``world``, ``gate``. A span's ``parent`` follows the task tree, so
a rejected command links back to the task and program that sent it. Times are the runtime's
time base.

What is recorded, and what is kept short:

- every task, when it ends: state, reason, rejection detail, error chain, inputs, result,
  provenance, and for a task that did not succeed, the World state at that moment;
- every command, with the check verdicts; a command identical to the previous accepted one
  from the same task is counted on the task instead of recorded again (a skill that resends
  its goal every cycle would otherwise fill the trace);
- every World write made by a task (e.g. a tracker); observations from the backend are not
  recorded one by one;
- every safety event (lock, unlock, fallback, watchdog, reset).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import itertools
import json
import math
from pathlib import Path
from typing import Any

from roborun.errors import BackendError, TaskCancelled, TaskFailed

VERSION = 1


def plain(x: Any, depth: int = 0) -> Any:
    """A JSON-friendly copy of ``x``; anything unknown becomes its repr."""
    if depth > 6:
        return repr(x)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, dict):
        return {str(k): plain(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v, depth + 1) for v in x]
    if hasattr(x, "item") and callable(x.item):  # numpy scalars
        with contextlib.suppress(Exception):
            return plain(x.item(), depth + 1)
    return repr(x)


def describe_error(e: BaseException | None) -> dict | None:
    """The error and its causal chain down the task tree (SPEC §1.8)."""
    if e is None:
        return None
    out: dict[str, Any] = {"type": type(e).__name__, "message": str(e)}
    if isinstance(e, TaskFailed):
        out["path"] = e.path
        out["root"] = describe_error(e.root)
    if isinstance(e, BackendError):
        out.update(code=e.code, source=e.source, raw=plain(e.raw), stamp=e.stamp)
    if isinstance(e, TaskCancelled):
        out["reason"] = e.reason
    return out


class Trace:
    def __init__(self, path: str | Path | None = None, *, meta: dict | None = None) -> None:
        self.spans: list[dict] = []
        self._ids = itertools.count(1)
        self._f = None
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open("w")
        self._emit({
            "kind": "run", "version": VERSION,
            "wall_start": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            **(meta or {}),
        }, keep=False)
        self._last_cmd: dict[tuple[str, str, str], Any] = {}

    def _emit(self, span: dict, *, keep: bool = True) -> None:
        span = plain(span)
        if keep:
            self.spans.append(span)
        if self._f is not None:
            self._f.write(json.dumps(span, separators=(",", ":")) + "\n")
            self._f.flush()  # a crash still leaves the trace up to that point

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    # -- recording ---------------------------------------------------------------

    def task(self, task: Any, world: Any) -> None:
        state = task.state.value
        span = {
            "id": task.id, "parent": _parent_id(task), "kind": "task", "name": task.name,
            "role": task.role, "start": task.started_at if task.started_at is not None
            else task.created_at, "end": task.ended_at, "status": state,
            "reason": task.reason, "detail": task.detail,
            "inputs": task.params, "outputs": task.result if state == "succeeded" else None,
            "error": describe_error(task.error) if state == "failed" else None,
            "resources": sorted(task.resources), "session": task.session,
            "provenance": task.provenance, "repeats": task._repeats or None,
            "refs": [],
        }
        if task.trace_parent:
            span["external_parent"] = task.trace_parent
        if state != "succeeded":
            span["state_at_end"] = {
                k: {"value": s.value, "stamp": s.stamp, "source": s.source}
                for k, s in sorted(world.snapshot().items())
            }
        self._emit({k: v for k, v in span.items() if v is not None})

    def command(
        self,
        task: Any,
        cmd: Any,
        t: float,
        *,
        status: str,
        detail: dict | None = None,
        verdicts: list | None = None,
        error: BaseException | None = None,
    ) -> None:
        key = (getattr(task, "id", ""), cmd.actuator, cmd.kind)
        payload = plain(cmd.payload)
        same = _rounded(payload, 3)  # a target that moved less than a millimetre is a repeat
        # Accepted repeats are counted even when checks ran: every check accepted them, and a
        # skill resending its goal each cycle would otherwise fill the trace.
        if status == "accepted" and self._last_cmd.get(key) == same:
            if task is not None:
                task._repeats = (task._repeats or 0) + 1
            return
        if status == "accepted":
            self._last_cmd[key] = same
        span = {
            "id": f"g{next(self._ids)}", "parent": getattr(task, "id", None), "kind": "gate",
            "name": f"{cmd.kind} {cmd.actuator}", "start": t, "end": t, "status": status,
            "inputs": payload, "outputs": detail, "verdicts": verdicts or None,
            "error": describe_error(error),
        }
        self._emit({k: v for k, v in span.items() if v is not None})

    def world(self, task: Any, key: str, sv: Any) -> None:
        self._emit({
            "id": f"w{next(self._ids)}", "parent": task.id, "kind": "world",
            "name": f"write {key}", "start": sv.stamp, "end": sv.stamp, "status": "ok",
            "outputs": {"key": key, "value": sv.value, "stamp": sv.stamp, "source": sv.source},
        })

    def sim(self, parent: Any, action: str, args: dict, result: Any, t: float) -> None:
        """A change to the simulated world (e.g. a push), so the trace explains why things moved."""
        self._emit({
            "id": f"w{next(self._ids)}", "parent": getattr(parent, "id", None), "kind": "world",
            "name": f"sim {action} {args.get('object', '')}".strip(), "start": t, "end": t,
            "status": "ok", "inputs": args, "outputs": result,
        })

    def safety(self, ev: dict) -> None:
        self._emit({
            "id": f"s{next(self._ids)}", "parent": None, "kind": "gate",
            "name": f"{ev['kind']} {','.join(ev['actuators'])}", "start": ev["t"], "end": ev["t"],
            "status": ev["kind"], "outputs": {"cause": ev["cause"], "detail": ev.get("detail", "")},
        })


def _parent_id(task: Any) -> str | None:
    p = task.parent
    return getattr(p, "id", None) or None


# -- reading ------------------------------------------------------------------------------


def load(path: str | Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    meta = lines[0] if lines and lines[0].get("kind") == "run" else {}
    return meta, [s for s in lines if s.get("kind") != "run"]


BAD_REASONS = {"rejected", "watchdog", "operator", "timeout", "preempted"}


def _is_failure(span: dict) -> bool:
    if span.get("kind") != "task":
        return False
    return span["status"] == "failed" or (
        span["status"] == "cancelled" and span.get("reason") in BAD_REASONS
    )


def failures(spans: list[dict]) -> list[list[dict]]:
    """For each failure that no failing child explains, the task path from the top down."""
    by_id = {s["id"]: s for s in spans if s.get("kind") == "task"}
    failing = [s for s in by_id.values() if _is_failure(s)]
    has_failing_child = {s.get("parent") for s in failing}
    leaves = [s for s in failing if s["id"] not in has_failing_child]
    chains = []
    for leaf in sorted(leaves, key=lambda s: (s.get("end") or 0, s["id"])):
        chain, node = [], leaf
        while node is not None:
            chain.append(node)
            node = by_id.get(node.get("parent"))
        chains.append(list(reversed(chain)))
    return chains


def _fmt_t(t: Any) -> str:
    return "?" if t is None else f"{t:.2f}s"


def _rounded(x: Any, digits: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, list):
        return [_rounded(v, digits) for v in x]
    if isinstance(x, dict):
        return {k: _rounded(v, digits) for k, v in x.items()}
    return x


def _short(x: Any, n: int = 100) -> str:
    s = json.dumps(_rounded(x), separators=(",", ":")) if not isinstance(x, str) else x
    return s if len(s) <= n else s[: n - 3] + "..."


def _status(span: dict) -> str:
    s = span["status"]
    if span.get("reason"):
        s += f" ({span['reason']})"
    return s


def show_failed(path: str | Path) -> str:
    """The text ``roborun trace show --failed`` prints: each failure's causal chain."""
    _, spans = load(path)
    return show_failed_spans(spans, title=str(path))[0]


def show_failed_spans(spans: list[dict], *, title: str) -> tuple[str, list[list[dict]]]:
    """Each failure's causal chain in ``spans``, as text, and the chains themselves."""
    chains = failures(spans)
    out = [f"{title}: {len(chains)} failure(s)" if chains else f"{title}: no failures"]
    gates = [s for s in spans if s.get("kind") == "gate"]
    for chain in chains:
        leaf = chain[-1]
        out.append("")
        for depth, s in enumerate(chain):
            out.append(f"{'  ' * depth}{'└ ' if depth else ''}{s['name']} [{s['id']}] "
                       f"{_status(s)}  {_fmt_t(s.get('start'))}–{_fmt_t(s.get('end'))}")
        pad = "  " * len(chain)
        if leaf.get("inputs"):
            out.append(f"{pad}inputs: {_short(leaf['inputs'])}")
        if leaf.get("detail"):
            out.append(f"{pad}why: {leaf['detail'].get('code')}: {leaf['detail'].get('message', '')}")
        err = leaf.get("error")
        if err:
            root = err.get("root") or err
            line = f"{pad}error: {root.get('type')}: {root.get('message')}"
            if root.get("code"):
                line += f" [code {root['code']}, from {root.get('source')}]"
            out.append(line)
        own = [g for g in gates if g.get("parent") == leaf["id"]]
        for g in own[-3:]:
            extra = ""
            if g.get("outputs"):
                extra = f" -> {g['outputs'].get('code')}: {g['outputs'].get('message', '')}"
            out.append(f"{pad}command at {_fmt_t(g['start'])}: {g['name']} "
                       f"{_short(g.get('inputs', {}), 60)} {g['status']}{extra}")
        end = leaf.get("end") or 0
        for g in gates:
            if g.get("parent") is None and end - 1.0 <= (g.get("start") or 0) <= end + 1e-9:
                o = g.get("outputs", {})
                out.append(f"{pad}safety at {_fmt_t(g['start'])}: {g['name']} "
                           f"({o.get('cause')}) {o.get('detail', '')}".rstrip())
        state = leaf.get("state_at_end")
        if state:
            keys = [k for k in state if not k.startswith("sim.")][:6]
            shown = ", ".join(f"{k}={_short(state[k]['value'], 40)}" for k in keys)
            out.append(f"{pad}state at end: {shown}")
    return "\n".join(out), chains


def show_tree(path: str | Path) -> str:
    """Every task as a tree, with its commands; world writes are counted, not listed."""
    _meta, spans = load(path)
    tasks = [s for s in spans if s.get("kind") == "task"]
    children: dict[Any, list[dict]] = {}
    for s in tasks:
        children.setdefault(s.get("parent"), []).append(s)
    cmds: dict[Any, list[dict]] = {}
    writes: dict[Any, int] = {}
    for s in spans:
        if s.get("kind") == "gate" and s.get("parent"):
            cmds.setdefault(s["parent"], []).append(s)
        elif s.get("kind") == "world":
            writes[s["parent"]] = writes.get(s["parent"], 0) + 1
    out = [f"{path}"]

    def walk(parent: Any, depth: int) -> None:
        for s in sorted(children.get(parent, []), key=lambda s: (s.get("start") or 0, s["id"])):
            extra = []
            if writes.get(s["id"]):
                extra.append(f"{writes[s['id']]} writes")
            if s.get("repeats"):
                extra.append(f"{s['repeats']} repeated commands")
            out.append(f"{'  ' * depth}{s['name']} [{s['id']}] {_status(s)}  "
                       f"{_fmt_t(s.get('start'))}–{_fmt_t(s.get('end'))}"
                       + (f"  ({', '.join(extra)})" if extra else ""))
            for g in cmds.get(s["id"], []):
                out.append(f"{'  ' * (depth + 1)}· {g['name']} {_short(g.get('inputs', {}), 50)} "
                           f"{g['status']} at {_fmt_t(g['start'])}")
            walk(s["id"], depth + 1)

    walk(None, 0)
    return "\n".join(out)
