"""Write the ablation figure: each failure, with no runtime, with roborun, and with one part off.

    python docs/media/ablation_svg.py docs/media/ablation.svg

Reads examples/results/ablation.json (``python -m benchmarks ablation --out``) and trace.json. A cell
is green when it matches roborun's result and red when it does not.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the benchmarks package
from benchmarks import ablation as ab

ROOT = Path(__file__).resolve().parents[2]
RES = json.loads((ROOT / "examples/results/ablation.json").read_text())
TRACE = json.loads((ROOT / "examples/results/trace.json").read_text())

METRIC = {m: f for _, _, m, f in ab.ROWS}
GROUPS = [  # (the part(s) that prevent it, [(label, metric in ablation.ROWS, suite)])
    ("Gate", [("buggy programs cause harm", "unsafe runs that caused harm", "safety"),
              ("two agents fight over the arm (target switches)", "target switches between two agents", "stopping"),
              ("retries override an operator stop", "operator stop overridden by retries", "stopping")]),
    ("Task", [("arm speed when cancel returns (m/s)", "arm speed when cancel returned", "stopping"),
              ("agent dies, arm goes on to its goal", "agent killed: arm kept going to the goal", "crash")]),
    ("World", [("occluded cup: grasps close on nothing", "occluded cup: grasps closed on nothing", "stale"),
               ("tracker dies: cup placed", "tracker died: cup placed", "stale")]),
    ("Trace", [("model finds the root cause of a failure", None, "trace")]),
    ("Gate + Task", [("cancelled code still moves the arm (setpoints)",
                      "setpoints from cancelled code reaching the arm", "stopping")]),
    ("World + Gate", [("silent sensor: arm moves into a hand", "sensor silent, hand in path: arm moved into it", "stale")]),
]
COLS = [("no runtime", "baseline"), ("roborun", "roborun"), ("no Gate", "roborun-no-gate"),
        ("no Task", "roborun-no-task"), ("no World", "roborun-no-world"), ("no Trace", "roborun-no-trace")]


def cells(metric: str | None, suite: str) -> list[str]:
    if metric is None:
        rs = TRACE["results"]

        def found(fb: str) -> str:
            sel = [r for r in rs if r["feedback"] == fb]
            return f"{sum(r['correct'] for r in sel)}/{len(sel)}"
        return [found("error line"), *[found("trace")] * 4, found("error line")]
    f = METRIC[metric]
    return [f(RES[suite], "roborun" if arm == "roborun-no-trace" else arm).removesuffix(" m/s")
            for _, arm in COLS]


W, GL, LW, CW, RH, HEAD = 990, 132, 330, 86, 30, 54  # columns sized so labels fit in wide system fonts (SF Pro, Segoe UI)
X0 = GL + LW
rows = sum(len(r) for _, r in GROUPS)
H = HEAD + rows * RH + len(GROUPS) * 8 + 38
o: list[str] = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
  'aria-label="Failures by the part of roborun that prevents them, with no runtime, with roborun, and with '
  'one part switched off. Every failure happens with no runtime and none with roborun; switching off one '
  'part brings back exactly the failures that part prevents.">')
a("""<style>
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;fill:#1f2328}
.bg{fill:#ffffff}.h{font-size:13px;font-weight:700;fill:#59636e}.hc{font-size:14px;font-weight:800;fill:#2848d8}
.g{font-size:14px;font-weight:800;fill:#2848d8}.l{font-size:13px}.v{font-size:12.5px;font-variant-numeric:tabular-nums}
.ok{fill:#0ca30c;fill-opacity:.13}.bad{fill:#d03b3b;fill-opacity:.17}
.iok{fill:#0a8a0a;font-size:11px;font-weight:800}.ibad{fill:#c22f2f;font-size:11px;font-weight:800}
.sep{stroke:#d1d9e0}.core{fill:none;stroke:#2848d8;stroke-width:1.5}.foot{font-size:12px;fill:#59636e}
@media (prefers-color-scheme: dark){text{fill:#e6edf3}.bg{fill:#0d1117}.h,.foot{fill:#9198a1}
.hc,.g{fill:#7b8fff}.ok{fill-opacity:.22}.bad{fill-opacity:.28}.iok{fill:#3fb950}.ibad{fill:#ff7b72}
.sep{stroke:#3d444d}.core{stroke:#7b8fff}}
</style>""")
a(f'<rect class="bg" x="0" y="0" width="{W}" height="{H}" rx="10"/>')
a(f'<text class="h" x="{GL + 8}" y="{HEAD - 14}">what goes wrong</text>')
a(f'<text class="h" x="12" y="{HEAD - 14}">prevented by</text>')
for i, (name, arm) in enumerate(COLS):
    cx = X0 + i * CW + CW / 2
    a(f'<text class="{"hc" if arm == "roborun" else "h"}" x="{cx}" y="{HEAD - 14}" text-anchor="middle">{name}</text>')
y = HEAD
core_top = y
for group, items in GROUPS:
    a(f'<line class="sep" x1="8" x2="{W - 8}" y1="{y + 3}" y2="{y + 3}"/>')
    y += 8
    a(f'<text class="g" x="12" y="{y + RH / 2 + 5}">{group}</text>')
    for label, metric, suite in items:
        vals = cells(metric, suite)
        ref = vals[1]
        a(f'<text class="l" x="{GL + 8}" y="{y + RH / 2 + 5}">{label}</text>')
        for i, v in enumerate(vals):
            x = X0 + i * CW
            good = v == ref
            a(f'<g><title>{label} — {COLS[i][0]}: {v}</title>'
              f'<rect class="{"ok" if good else "bad"}" x="{x + 3}" y="{y + 3}" width="{CW - 6}" height="{RH - 6}" rx="5"/>'
              f'<text class="{"iok" if good else "ibad"}" x="{x + 11}" y="{y + RH / 2 + 4}">{"✓" if good else "✕"}</text>'
              f'<text class="v" x="{x + CW / 2 + 6}" y="{y + RH / 2 + 4.5}" text-anchor="middle">{v}</text></g>')
        y += RH
a(f'<rect class="core" x="{X0 + CW + 1}" y="{core_top - 30}" width="{CW - 2}" height="{y - core_top + 34}" rx="7"/>')
a(f'<text class="foot" x="12" y="{y + 26}">Each column after roborun switches off one part and keeps the rest. '
  f'✕ marks a result worse than roborun\'s. Source: python -m benchmarks ablation.</text>')
a("</svg>")
Path(sys.argv[1] if len(sys.argv) > 1 else "ablation.svg").write_text("\n".join(o) + "\n")
