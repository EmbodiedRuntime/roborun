"""World: the shared, present belief (SPEC §2.3–§2.5)."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from roborun.errors import Overflow, StaleData
from roborun.stamped import Stamped

if TYPE_CHECKING:
    from collections.abc import Callable

    from roborun.clock import Clock
    from roborun.task import Task

STATE = "state"
QUEUE = "queue"
_EPS = 1e-9
_DEFAULT = object()


@dataclass
class _Entry:
    kind: str
    max_age: float | None
    maxlen: int | None
    latest: Stamped | None = None
    version: int = 0
    backlog: deque | None = None  # (seq, Stamped) for queue keys
    waiters: list = field(default_factory=list)


class World:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self.on_write: Callable[[str, Stamped], None] | None = None  # the runtime's trace hook

    # -- declaration -------------------------------------------------------

    def declare(
        self, key: str, kind: str = STATE, *, max_age: float | None = None, maxlen: int | None = None
    ) -> None:
        if kind not in (STATE, QUEUE):
            raise ValueError(f"kind must be {STATE!r} or {QUEUE!r}, not {kind!r}")
        if kind == QUEUE and not maxlen:
            raise ValueError(f"queue key {key!r} needs a maxlen")
        old = self._entries.get(key)
        if old is not None and old.version:
            raise ValueError(f"{key!r} already has values; declare keys before writing them")
        self._entries[key] = _Entry(
            kind, max_age, maxlen, backlog=deque(maxlen=maxlen) if kind == QUEUE else None
        )

    def _entry(self, key: str) -> _Entry:
        e = self._entries.get(key)
        if e is None:
            e = self._entries[key] = _Entry(STATE, None, None)
        return e

    # -- writing and reading -----------------------------------------------

    def write(self, key: str, value: Any, *, stamp: float | None = None, source: str = "") -> Stamped:
        if isinstance(value, Stamped):
            sv = value
        else:
            sv = Stamped(value, self._clock.now() if stamp is None else stamp, source)
        e = self._entry(key)
        e.version += 1
        e.latest = sv
        if self.on_write is not None:
            self.on_write(key, sv)
        if e.kind == QUEUE:
            e.backlog.append((e.version, sv))
        waiters, e.waiters = e.waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)
        return sv

    def get(self, key: str, max_age: Any = _DEFAULT) -> Stamped | None:
        """The latest value. State keys raise ``StaleData`` if it is missing or too old."""
        e = self._entries.get(key)
        if e is not None and e.kind == QUEUE:
            return e.latest
        if e is None or e.latest is None:
            raise StaleData(key, None, None, None)
        limit = e.max_age if max_age is _DEFAULT else max_age
        if limit is not None:
            age = e.latest.age(self._clock.now())
            if age > limit + _EPS:
                raise StaleData(key, age, limit, e.latest.source)
        return e.latest

    def latest(self, key: str) -> Stamped | None:
        """The latest value without any freshness check, or None."""
        e = self._entries.get(key)
        return None if e is None else e.latest

    def snapshot(self) -> dict[str, Stamped]:
        return {k: e.latest for k, e in self._entries.items() if e.latest is not None}

    def changes(self, key: str, while_running: Task | None = None) -> Changes:
        return Changes(self, key, while_running)

    async def next(self, key: str) -> Stamped:
        """The next value written to ``key`` (for a queue key, the next message)."""
        return await anext(self.changes(key))


class Changes:
    """Async iterator over new values of one key (SPEC §2.4, §2.5)."""

    def __init__(self, world: World, key: str, while_running: Task | None) -> None:
        self._w = world
        self._key = key
        self._task = while_running
        self._started = world._clock.now()
        e = world._entry(key)
        if e.kind == STATE:
            self._seen = e.version  # only values written from now on
        else:
            # A new reader starts at the oldest retained message (rule 3).
            self._cursor = e.backlog[0][0] if e.backlog else e.version + 1

    def __aiter__(self) -> Changes:
        return self

    async def __anext__(self) -> Stamped:
        e = self._w._entry(self._key)
        clock = self._w._clock
        while True:
            if self._task is not None and self._task.state.terminal:
                raise StopAsyncIteration
            deadline = None
            if e.kind == STATE:
                if e.version > self._seen:
                    self._seen = e.version
                    if e.max_age is not None:
                        age = e.latest.age(clock.now())
                        if age > e.max_age + _EPS:
                            raise StaleData(self._key, age, e.max_age, e.latest.source)
                    return e.latest
                if e.max_age is not None:
                    base = e.latest.stamp if e.latest is not None else self._started
                    deadline = base + e.max_age
                    if clock.now() > deadline + _EPS:
                        age = None if e.latest is None else e.latest.age(clock.now())
                        source = None if e.latest is None else e.latest.source
                        raise StaleData(self._key, age, e.max_age, source)
            else:
                msg = self._next_message(e, clock.now())
                if msg is not None:
                    return msg
            await self._wait(e, deadline)

    def _next_message(self, e: _Entry, now: float) -> Stamped | None:
        if e.backlog and self._cursor < e.backlog[0][0]:
            lost = e.backlog[0][0] - self._cursor
            self._cursor = e.backlog[0][0]
            raise Overflow(self._key, lost)
        for seq, sv in e.backlog:
            if seq < self._cursor:
                continue
            self._cursor = seq + 1
            if e.max_age is not None and sv.age(now) > e.max_age + _EPS:
                continue  # expired; silence and expiry are not errors for queue keys
            return sv
        return None

    async def _wait(self, e: _Entry, deadline: float | None) -> None:
        loop = asyncio.get_running_loop()
        written = loop.create_future()
        e.waiters.append(written)
        waits = [written]
        timer = None
        if deadline is not None:
            timer = self._w._clock.timer(deadline)
            waits.append(timer)
        if self._task is not None:
            waits.append(self._task._done)
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for f in (written, timer):
                if f is not None and not f.done():
                    f.cancel()
