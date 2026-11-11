import asyncio

import pytest

from roborun import run_sim
from roborun.backends.fake import FakeBackend


@pytest.fixture
def sim():
    """Run an async main on a fresh fake backend; returns (task, backend)."""

    def _run(main, *, until=30.0, **backend_kwargs):
        backend = FakeBackend(**backend_kwargs)
        task = asyncio.run(run_sim(main, backend=backend, until=until))
        return task, backend

    return _run
