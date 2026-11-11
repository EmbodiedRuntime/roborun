"""Time base (SPEC §2.1).

There is one time base per runtime. On hardware it is monotonic wall time; in
simulation it advances only when the simulator steps. Everything in the runtime
waits through the clock, never through ``asyncio.sleep`` directly, so the same
program has the same timing semantics in both.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools

_EPS = 1e-9


def _resolve(fut: asyncio.Future) -> None:
    if not fut.done():
        fut.set_result(None)


class Clock:
    def now(self) -> float:
        raise NotImplementedError

    def timer(self, deadline: float) -> asyncio.Future:
        """Return a future that resolves when the clock reaches ``deadline``."""
        raise NotImplementedError

    async def sleep(self, seconds: float) -> None:
        await self.timer(self.now() + seconds)


class WallClock(Clock):
    """Monotonic wall time, taken from the running event loop."""

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    def timer(self, deadline: float) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        handle = loop.call_at(deadline, _resolve, fut)
        fut.add_done_callback(lambda _: handle.cancel())
        return fut


class SteppedClock(Clock):
    """Simulated time that moves only when :meth:`advance` is called."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._heap: list[tuple[float, int, asyncio.Future]] = []
        self._seq = itertools.count()

    def now(self) -> float:
        return self._now

    def timer(self, deadline: float) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        if deadline <= self._now + _EPS:
            fut.set_result(None)
        else:
            heapq.heappush(self._heap, (deadline, next(self._seq), fut))
        return fut

    def advance(self, dt: float) -> None:
        self._now += dt
        while self._heap and self._heap[0][0] <= self._now + _EPS:
            _, _, fut = heapq.heappop(self._heap)
            _resolve(fut)
