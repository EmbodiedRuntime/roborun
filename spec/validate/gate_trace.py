"""A journal of the Gate's core as a sequence of steps of Gate.tla, for one actuator.

Gate.tla is about one actuator and tasks that each ask for it once and command it with their
own token. So a model task here is a request (``r3``), and a run can be put to the model
only if every request is for the actuator alone and every command is sent by the task that
owns the token; arbitration over several actuators and the use of a parent's token are
Ownership.tla's. ``Outside`` says which entry of the journal leaves that scope.

One event of the core is one step of the model, with these exceptions:

- A ``take`` cancels the owner and then waits: ``Cancel`` and ``Acquire``.
- A request that fails at once because the actuator is owned (``on_busy="fail"``) is no step:
  the model has no such request.
- The checks' answer and what the backend did are one step (``Accepted``), as in the model.
- A rejected command cancels its sender in the model's ``Rejected``. In the runtime the
  token is revoked by a second event, a moment later; until then the token is not observed.
- A command that a monitor's ``allow`` check refuses is the model's ``locked`` at the moment
  it was sent: the model asks ``allow`` then.
- Rest is seen, not signalled: before each event the journal says whether the actuator is
  at rest, and a change is the model's ``Rest`` if the backend was sent a stop since the
  last command it took, and ``Arrive`` otherwise.

After the last step of every event the model's state must agree with the core's: who holds
a valid or a revoked token, the locks, whether the monitor rejects, whether the watchdog is
armed. What a task is doing (``ts``) and the motion are the model's own.
"""

# ruff: noqa: B023 - the helpers inside the loop are used in the iteration that defines them
from __future__ import annotations

MANUAL = {"operator", "fault"}
NOBODY = "nobody"  # a sender with no token at all


class Outside(Exception):
    """The run does something Gate.tla does not describe."""


def _view(s: dict, a: str) -> dict:
    rid = s["owner"].get(a)
    revoked = rid is not None and rid in s["revoked"]
    return {
        "valid": set() if rid is None or revoked else {rid},
        "revoked": {rid} if revoked else set(),
        "locks": {c.split(":")[0] for c in s["locks"].get(a, ())},
        "failing": bool(s["failing"]),
        "watch": a in s["watch"],
    }


EMPTY = {"owner": {}, "revoked": [], "released": [], "queue": [], "locks": {}, "failing": [],
         "pending": [], "watch": {}, "settling": [], "stopped": {}, "last": {}}


def steps(meta: dict, entries: list[dict], actuator: str = "arm") -> tuple[list[str], list[dict], bool]:
    """(the model's tasks, the steps, whether the monitor has an ``allow`` check)."""
    a = actuator
    tasks: list[str] = [NOBODY]
    out: list[dict] = []
    mine: set[str] = set()  # the requests for this actuator
    lag: set[str] = set()  # rejected, and the runtime has not revoked the token yet
    pending: dict[str, str] = {}  # pending -> request
    silent: set[str] = set()  # pendings that were `locked` when sent
    motion = "rest"
    prev = EMPTY

    def fate(k: int, pid: str) -> str | None:
        for e in entries[k + 1:]:
            if e["event"] in ("dropped", "checked", "executed", "fault") and e["args"] \
                    and isinstance(e["args"][0], dict) and e["args"][0].get("pending") == pid:
                return e["event"]
        return None

    for k, e in enumerate(entries):
        s, ev, args = e["state"], e["event"], e["args"]
        effects = e["effects"]
        new: list[tuple[set, str | None]] = []

        def step(labels, t=None):
            new.append(({labels} if isinstance(labels, str) else set(labels), t))

        # -- rest is seen before the event --------------------------------------------
        if e["rest"].get(a, True) and motion != "rest":
            step("arrive" if motion == "moving" else "rest")
            motion = "rest"
            out.append({"a": new[-1][0], "t": None, "n": e["n"], "obs": None})
            new.clear()

        # -- what changed for the requests ----------------------------------------------
        was, now = prev["owner"].get(a), s["owner"].get(a)
        was_revoked = was is not None and (was in prev["revoked"] or was in lag)
        revoked, stopped, gone = [], [], []
        if was is not None and now == was:
            if was in s["released"] and was not in prev["released"]:
                if was_revoked:
                    lag.discard(was)
                else:
                    stopped.append(was)
            elif was in s["revoked"] and not was_revoked:
                revoked.append(was)
            elif was in s["revoked"]:
                lag.discard(was)
        elif was is not None:
            gone.append(was)
            lag.discard(was)
        dequeued = [r for r in prev["queue"] if r in mine and r not in s["queue"] and r != now]
        granted = [now] if now is not None and now != was else []

        def others(skip=()):
            """The steps for what changed, besides the event's own."""
            for r in revoked:
                if r not in skip:
                    step("cancel", r)
            for r in stopped:
                step("release-stop", r)
            for r in gone:
                step({"end", "settle"} if was_revoked else "release", r)
            for r in dequeued:
                if r not in skip:
                    step("cancel", r)

        def dispatch(skip=()):
            for r in granted:
                if r not in skip:
                    step("grant", r)

        def about(x) -> bool:
            return isinstance(x, dict) and x.get("actuator") == a

        if ev == "request":
            resources = set(args[1])
            if a in resources:
                if resources != {a}:
                    raise Outside(f"entry {e['n']}: a request for {sorted(resources)}")
                if "raised" in e:
                    if set(prev["locks"].get(a, ())) & MANUAL:
                        rid = f"x{e['n']}"
                        tasks.append(rid)
                        step("busy", rid)
                else:
                    rid = e["result"]["request"]
                    mine.add(rid)
                    tasks.append(rid)
                    others()
                    step("grant" if rid in granted else "wait", rid)
                    dispatch(skip={rid})
        elif ev in ("expire", "withdraw", "release", "release_all", "revoke", "revoke_grant",
                    "rested"):
            before = len(new)
            others()
            braked = any(lab & {"cancel", "release-stop"} and t in revoked + stopped
                         for lab, t in new[before:])
            if not braked and any(x["effect"] == "Stop" and x["actuator"] == a for x in effects):
                step("stop")
            dispatch()
        elif ev == "command":
            cmd = args[1]
            if about(cmd):
                grant = args[0]
                rid = grant["request"] if grant else NOBODY
                result = e.get("result")
                if cmd["kind"] == "stop":
                    step("stop")
                elif cmd["kind"] not in ("goal", "setpoint"):
                    raise Outside(f"entry {e['n']}: a {cmd['kind']!r} command")
                elif isinstance(result, dict) and "rejected" in result:
                    step(result["rejected"], rid)  # unauthorized, or locked
                else:
                    sender = result["task"]
                    if sender is not None and sender["id"] != grant["task"]["id"]:
                        raise Outside(f"entry {e['n']}: {sender['name']} commands with the "
                                      f"token of {grant['task']['name']}")
                    if rid in pending.values():
                        raise Outside(f"entry {e['n']}: two commands in flight for {rid}")
                    pid = result["pending"]
                    if result["passed"] and fate(k, pid) in ("dropped", None):
                        silent.add(pid)
                        step("locked", rid)
                    else:
                        pending[pid] = rid
                        step("checking", rid)
        elif ev == "dropped":
            pid = args[0]["pending"]
            if pid in pending and pid in prev["pending"]:
                step("drop", pending.pop(pid))
        elif ev == "checked":
            pid = args[0]["pending"]
            if pid in pending:
                rid = pending[pid]
                if args[1] is not None:
                    del pending[pid]
                    step("reject", rid)
                    if rid == now and rid not in s["revoked"]:
                        lag.add(rid)
                elif e.get("result") is not None:
                    del pending[pid]
                    step("unauthorized", rid)
        elif ev in ("executed", "fault"):
            pid = args[0]["pending"]
            if pid in pending:
                step("execute" if ev == "executed" else "fault", pending.pop(pid))
        elif ev == "operator_stop":
            if not args or args[0] is None or a in args[0]:
                step("operator")
        elif ev == "reset":
            if set(prev["locks"].get(a, ())) & MANUAL and (not args or args[0] is None or a in args[0]):
                step("reset")
                dispatch()
        elif ev == "monitor":
            if bool(prev["failing"]) != bool(s["failing"]):
                if len(prev["failing"]) + len(s["failing"]) > 1:
                    raise Outside(f"entry {e['n']}: more than one monitor")
                step("monitor" if s["failing"] else "unlock")
                dispatch()
        elif ev == "watch_expired":
            if args[0] == a and any(x["effect"] == "Safety" and x["kind"] == "watchdog" for x in effects):
                step("watchdog")
        else:
            raise Outside(f"entry {e['n']}: the event {ev!r} is not called from outside the core")

        # What the actuator does is taken from what the backend was sent, whatever steps the
        # event was read as: a stop the model does not take shows when the arm rests.
        if any(labels == {"execute"} for labels, _ in new):
            motion = "moving"
        stop = any(x["effect"] == "Stop" and x["actuator"] == a for x in effects) or (
            ev == "command" and about(args[1]) and args[1]["kind"] == "stop")
        if stop and motion == "moving":
            motion = "stopping"
        for labels, t in new:
            out.append({"a": labels, "t": t, "n": e["n"], "obs": None})
        if new:
            out[-1]["obs"] = {**_view(s, a), "lag": set(lag)}
        prev = s

    allow = any(c.startswith("monitor:") for c in meta.get("soft", ()))
    return tasks, out, allow
