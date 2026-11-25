"""Compare the Gate's core with the TLA+ models:  python spec/compare [-v] [--all | name | prefix* ...]

Without a name it runs the quick set (about four minutes): the two-task Gate instances, the
controls, and three Ownership instances. ``--all`` adds the full-size instances, which take
much longer (``ownership_takers`` alone took 54 minutes and 10 GB on a smaller model).

For each comparison TLC dumps the model's state graph, the core is run through every order
of events in the same instance, and the two are compared (see ``explore``).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import explore
import tla
from gate_env import GateEnv
from mutations import MUTATIONS, mutated
from ownership_env import OwnershipEnv, instance

TWO = {"Tasks = {a, b, c}": "Tasks = {a, b}"}

# name: (module, config, changes to the config, environment, the core's mutation, expected)
COMPARISONS = {
    "gate": ("MCGate", "gate", {}, lambda: GateEnv(), None, "same"),
    "gate_allow": ("MCGate", "gate_allow", {}, lambda: GateEnv(allow=True), None, "same"),
    "gate2": ("MCGate", "gate", TWO, lambda: GateEnv(("a", "b")), None, "same"),
    "gate2_allow": ("MCGate", "gate_allow", TWO, lambda: GateEnv(("a", "b"), allow=True), None, "same"),
}
# The controls: a mechanism switched off in the model only, or in the core only, must show
# as a difference.
for _m in MUTATIONS:
    _two = lambda: GateEnv(("a", "b"))
    COMPARISONS[f"gate2_model_{_m}"] = ("MCGate", f"gate_mut_{_m}", {}, _two, None, "differs")
    COMPARISONS[f"gate2_core_{_m}"] = ("MCGate", "gate", TWO, _two, _m, "differs")

for _cfg in ("ownership", "ownership_takers", "ownership_child", "ownership_deep",
             "ownership_hold", "ownership_hold_take", "ownership_hold_child"):
    COMPARISONS[_cfg] = ("MCOwnership", _cfg, {},
                         lambda _cfg=_cfg: OwnershipEnv(**instance(_cfg)), None, "same")


def run(name: str, out: Path, verbose: bool) -> bool:
    module, cfg, replace, make, mutation, expected = COMPARISONS[name]
    t0 = time.time()
    graph = tla.dump(module, cfg, out, replace=replace)
    with mutated(mutation):
        env = make()
        r = explore.compare(env, graph, inner=env.inner)
    found = "same" if r.ok else "differs"
    ok = found == expected
    print(f"{'ok' if ok else 'FAILED':7}{name:34}{found + (', as expected' if ok and found == 'differs' else ''):22}"
          f"core {r.states} states, {r.steps} steps; model {r.model_states} states, "
          f"{r.model_steps} steps   {time.time() - t0:.0f}s", flush=True)
    if ok and not verbose:
        return ok
    for what, items in (("steps of the core that the model does not have", r.bad),
                        ("states of the model the core never meets", r.missed_states),
                        ("steps of the model that a state of the core cannot take", r.missed_steps)):
        if not items:
            continue
        if len(items[0]) == 3:
            kinds: dict = {}
            for item in items:
                action = item[1][0] if isinstance(item[1], tuple) else item[1]
                kinds[action] = kinds.get(action, 0) + 1
            print(f"  {len(items)} {what}: {kinds}; the first:")
        else:
            print(f"  {len(items)} {what}; the first:")
        for item in items[:3]:
            if len(item) == 3:
                print(f"    from  {explore.show(item[0]) if item[0] else '-'}\n"
                      f"    by    {item[1]}\n    to    {explore.show(item[2])}")
            else:
                print(f"    {explore.show(item)}")
    return ok


QUICK = ("gate2", "gate2_allow", "ownership_child", "ownership_hold", "ownership_hold_child")


def main(argv: list[str]) -> int:
    verbose, everything = "-v" in argv, "--all" in argv
    argv = [a for a in argv if a not in ("-v", "--all")]
    if argv:
        names = [n for n in COMPARISONS
                 if any(n == a or (a.endswith("*") and n.startswith(a[:-1])) for a in argv)]
    else:
        names = [n for n in COMPARISONS
                 if everything or n in QUICK or COMPARISONS[n][5] == "differs"]
    if not names:
        print(f"no comparison named {argv}; there are: {', '.join(COMPARISONS)}")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        ok = [run(n, Path(tmp), verbose) for n in names]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
