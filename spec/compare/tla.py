"""Read the state graph TLC dumps (``-dump dot,actionlabels``) into Python values.

A TLA+ value becomes: a string or a model value -> ``str``; ``None`` (the model value the
models use for "nothing") -> ``None``; TRUE / FALSE -> ``bool``; a number -> ``int``; a set
-> ``frozenset``; a function or a record -> a sorted tuple of ``(key, value)`` pairs; a
sequence -> ``("seq", (v, ...))``. A state is the function from variable names to values.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent
TLC_VERSION = "1.7.4"  # the version spec/check.sh pins
JAR = SPEC / ".tools" / f"tla2tools-{TLC_VERSION}.jar"

_TOKEN = re.compile(r'\s*(:>|@@|\|->|<<|>>|"(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z0-9_]*|-?\d+|[(){}\[\],])')


def fn(pairs) -> tuple:
    """A TLA+ function or record, as this module represents it."""
    return tuple(sorted(dict(pairs).items(), key=lambda kv: repr(kv[0])))


def parse(text: str):
    tokens = _TOKEN.findall(text)
    if "".join(tokens) != re.sub(r"\s+", "", text):
        raise ValueError(f"cannot read the TLA+ value {text!r}")
    value, rest = _value(tokens, 0)
    if rest != len(tokens):
        raise ValueError(f"cannot read the TLA+ value {text!r}")
    return value


def _value(t: list[str], i: int):
    tok = t[i]
    if tok == "{":
        items, i = _list(t, i + 1, "}")
        return frozenset(items), i
    if tok == "<<":
        items, i = _list(t, i + 1, ">>")
        return ("seq", tuple(items)), i
    if tok == "(":  # (k :> v @@ k :> v)
        pairs, i = [], i + 1
        while True:
            k, i = _value(t, i)
            assert t[i] == ":>", t[i]
            v, i = _value(t, i + 1)
            pairs.append((k, v))
            if t[i] == ")":
                return fn(pairs), i + 1
            assert t[i] == "@@", t[i]
            i += 1
    if tok == "[":  # [f |-> v, g |-> w]
        pairs, i = [], i + 1
        while True:
            k = t[i]
            assert t[i + 1] == "|->", t[i + 1]
            v, i = _value(t, i + 2)
            pairs.append((k, v))
            if t[i] == "]":
                return fn(pairs), i + 1
            assert t[i] == ",", t[i]
            i += 1
    if tok.startswith('"'):
        return tok[1:-1], i + 1
    if tok in ("TRUE", "FALSE"):
        return tok == "TRUE", i + 1
    if tok == "None":
        return None, i + 1
    if re.fullmatch(r"-?\d+", tok):
        return int(tok), i + 1
    return tok, i + 1  # a model value


def _list(t: list[str], i: int, close: str):
    items = []
    if t[i] == close:
        return items, i + 1
    while True:
        v, i = _value(t, i)
        items.append(v)
        if t[i] == close:
            return items, i + 1
        assert t[i] == ",", t[i]
        i += 1


_VAR = re.compile(r"(?:^|\n)/\\ (\w+) = ")


def parse_state(label: str) -> tuple:
    parts = _VAR.split(label)[1:]
    return fn((parts[i], parse(parts[i + 1])) for i in range(0, len(parts), 2))


class Graph:
    """The reachable states of a model and its transitions, labelled with action names.
    A state is known by a number: ``state[i]`` is its value and ``index`` the way back."""

    def __init__(self) -> None:
        self.state: list[tuple] = []
        self.index: dict[tuple, int] = {}
        self.init: set[int] = set()
        self.edges: list[dict[int, frozenset[str]]] = []  # state -> next state -> actions

    def add(self, value: tuple) -> int:
        i = self.index.get(value)
        if i is None:
            i = self.index[value] = len(self.state)
            self.state.append(value)
        return i


_NODE = re.compile(r'^(-?\d+) \[label="(.*)"(,style = filled)?\];?$')
_EDGE = re.compile(r'^(-?\d+) -> (-?\d+) \[label="(\w+)"')


def load(dot: Path) -> Graph:
    g = Graph()
    ids: dict[str, int] = {}
    edges: dict[tuple[int, int], set[str]] = {}
    with open(dot) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if (m := _EDGE.match(line)) is not None:
                edges.setdefault((m[1], m[2]), set()).add(m[3])
            elif (m := _NODE.match(line)) is not None:
                if m[1] not in ids:
                    label = m[2].replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
                    ids[m[1]] = g.add(parse_state(label))
                if m[3]:
                    g.init.add(ids[m[1]])
    g.edges = [{} for _ in g.state]
    shared: dict[frozenset, frozenset] = {}
    for (a, b), actions in edges.items():
        acts = frozenset(actions)
        g.edges[ids[a]][ids[b]] = shared.setdefault(acts, acts)
    return g


def dump(module: str, cfg: str, out: Path, workers: int = 4, replace: dict[str, str] | None = None) -> Graph:
    """Run TLC on ``spec/cfg/<cfg>.cfg`` and return its state graph. ``replace`` changes
    the text of the config first, e.g. to a smaller set of tasks."""
    out.mkdir(parents=True, exist_ok=True)
    dot = out / f"{cfg}.dot"
    text = (SPEC / "cfg" / f"{cfg}.cfg").read_text()
    for old, new in (replace or {}).items():
        if old not in text:
            raise ValueError(f"{cfg}.cfg has no {old!r}")
        text = text.replace(old, new)
    # Only the graph is wanted. The properties are checked by spec/check.sh, and a
    # mutation config would stop TLC at the first violation.
    text = "\n".join(line for line in text.splitlines()
                     if not line.startswith(("INVARIANT", "PROPERT")))
    (out / f"{cfg}.cfg").write_text(text)
    r = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-cp", str(JAR), "tlc2.TLC", module,
         "-config", str(out / f"{cfg}.cfg"), "-workers", str(workers),
         "-metadir", str(out / f"{cfg}.meta"), "-dump", "dot,actionlabels", str(dot)],
        check=False, cwd=SPEC, capture_output=True, text=True)
    if r.returncode != 0 or not dot.exists():
        raise RuntimeError(f"TLC failed on {cfg} (exit {r.returncode}):\n{r.stdout[-2000:]}")
    return load(dot)
