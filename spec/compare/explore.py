"""Run the Gate's core through every order of events, and compare with a model's graph.

An environment (``gate_env``, ``ownership_env``) holds a ``GateCore`` and what the model
describes besides the Gate: the tasks' states and the motion of the actuators. It offers
the events that can happen in a state (each named after the model's action), applies one,
and maps the result to a state of the model (``view``).

``compare`` then checks two things.

Refinement: every step of the core is a step of the model. The core's step, seen through
``view``, must be the named action of the model followed by zero or more of the model's
``inner`` actions (those the core does in the same step, e.g. serving the next request
when an actuator is given up), or leave the model's state unchanged. So every invariant
TLC proved for the model holds for the core, in this instance.

Coverage: the model does nothing the core cannot do. The model takes an inner action in a
step of its own, at any time; the core takes it at once. So the model is followed with
the same haste (``eager``: where an inner action can happen, only inner actions happen).
Then every state of the core must be able to take every step the model has from the state
it is seen as, and every state of the model must be met. The first is asked of each state
of the core, not of each state of the model: two histories that look the same to the
model must both be able to do what the model allows.
"""

from __future__ import annotations

import hashlib
import pickle
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Result:
    states: int = 0  # of the core and its environment
    steps: int = 0
    bad: list = field(default_factory=list)  # steps of the core that the model does not have
    missed_states: list = field(default_factory=list)  # of the model, never met by the core
    missed_steps: list = field(default_factory=list)  # of the model, that some state of the core cannot take
    model_states: int = 0  # of the model followed eagerly
    model_steps: int = 0

    @property
    def ok(self) -> bool:
        return not (self.bad or self.missed_states or self.missed_steps)


def explore(env0, on_state, limit: int = 20_000_000) -> int:
    """Breadth first over everything ``env0`` can do; ``on_state(view, steps)`` for every
    state, with its model state and its steps as ``(event, view after)``. Returns the
    number of distinct states. A state already seen is remembered by a digest of its key."""
    def digest(env) -> bytes:
        return hashlib.blake2b(repr(env.key()).encode(), digest_size=16).digest()

    seen = {digest(env0)}
    queue = deque([pickle.dumps(env0, pickle.HIGHEST_PROTOCOL)])
    while queue:
        blob = queue.popleft()
        env = pickle.loads(blob)
        steps = []
        for event in env.events():
            env2 = pickle.loads(blob)
            env2.apply(event)
            steps.append((event, env2.view()))
            k = digest(env2)
            if k not in seen:
                if len(seen) >= limit:
                    raise RuntimeError(f"more than {limit} states")
                seen.add(k)
                queue.append(pickle.dumps(env2, pickle.HIGHEST_PROTOCOL))
        on_state(env.view(), steps)
    return len(seen)


def _paths(graph, m, action, m2, inner, depth=4):
    """The model's paths from m to m2: ``action``, then up to ``depth`` inner actions."""
    found = []
    frontier = [(x, [(m, action, x)]) for x, acts in graph.edges[m].items() if action in acts]
    for _ in range(depth + 1):
        nxt = []
        for x, path in frontier:
            if x == m2:
                found.append(path)
            for y, acts in graph.edges[x].items():
                for a in acts & inner:
                    nxt.append((y, [*path, (x, a, y)]))
        frontier = nxt
    return found


def eager(graph, inner):
    """The model's states and transitions when inner actions are taken as soon as they can
    be: from a state where one can happen, nothing else does."""
    states, steps = set(graph.init), set()
    queue = deque(graph.init)
    while queue:
        s = queue.popleft()
        out = [(a, s2) for s2, acts in graph.edges[s].items() for a in acts if s2 != s]
        hasty = [(a, s2) for a, s2 in out if a in inner]
        for a, s2 in hasty or out:
            steps.add((s, a, s2))
            if s2 not in states:
                states.add(s2)
                queue.append(s2)
    return states, steps


def compare(env0, graph, *, inner=frozenset()) -> Result:
    """States are compared by their numbers in ``graph``; a state of the core that the
    model does not have gets a number past the model's."""
    r = Result()
    known = len(graph.state)
    states, steps = eager(graph, inner)
    r.model_states, r.model_steps = len(states), len(steps)
    owed: dict[int, frozenset] = {}  # model state -> the steps the model has from it
    for s, a, s2 in steps:
        owed[s] = owed.get(s, frozenset()) | {(a, s2)}
    v0 = graph.add(env0.view())
    if v0 not in graph.init:
        r.bad.append((None, "Init", graph.state[v0]))
    passed = {v0}
    first: dict[tuple, frozenset] = {}  # a step of the core -> the model steps it starts with
    missed: set[tuple] = set()

    def on_state(view, moves):
        m = graph.index[view]
        done: set = set()
        for event, view2 in moves:
            r.steps += 1
            step = (m, event[0], graph.add(view2))
            got = first.get(step)
            if got is None:
                got = first[step] = check(step, event)
            done |= got
        if m < known:
            for a, s2 in owed.get(m, frozenset()) - done:
                missed.add((m, a, s2))

    def check(step, event) -> frozenset:
        m, action, m2 = step
        if m >= known:
            return frozenset()  # reported when the step that led here was checked
        paths = _paths(graph, m, action, m2, inner)
        if not paths and m != m2:  # the model always allows a step that changes nothing
            r.bad.append((graph.state[m], event, graph.state[m2]))
        for path in paths:
            passed.update(s2 for _, _, s2 in path)
        return frozenset((path[0][1], path[0][2]) for path in paths)

    r.states = explore(env0, on_state)
    r.missed_states = [graph.state[s] for s in states if s not in passed]
    r.missed_steps = [(graph.state[s], a, graph.state[s2]) for s, a, s2 in sorted(missed)]
    return r


def show(state: tuple) -> str:
    return "  ".join(f"{k}={_show(v)}" for k, v in state)


def _show(v) -> str:
    if isinstance(v, frozenset):
        return "{" + ",".join(sorted(_show(x) for x in v)) + "}"
    if isinstance(v, tuple) and v[:1] == ("seq",):
        return "<" + ",".join(_show(x) for x in v[1]) + ">"
    if isinstance(v, tuple):
        return "[" + " ".join(f"{k}:{_show(x)}" for k, x in v) + "]"
    return str(v)
