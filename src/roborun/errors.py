"""Errors raised by the runtime. Messages should tell the caller what to do next."""

from __future__ import annotations


class RoborunError(Exception):
    """Base class for roborun errors."""


class InvalidState(RoborunError):
    """An operation is not allowed in the task's current state (SPEC §1.3)."""


class StaleData(RoborunError):
    """A World entry is missing or older than the caller's max_age (SPEC §2.3)."""

    def __init__(self, key: str, age: float | None, limit: float | None, source: str | None):
        self.key, self.age, self.limit, self.source = key, age, limit, source
        if age is None:
            msg = f"no value for {key!r} yet; is its producer running?"
        else:
            msg = (
                f"{key!r} is {age:.3f}s old (limit {limit:.3f}s), last written by "
                f"{source or 'an unnamed source'}; is that producer still running?"
            )
        super().__init__(msg)


class Overflow(RoborunError):
    """A reader fell more than maxlen messages behind on a queue key (SPEC §2.5)."""

    def __init__(self, key: str, lost: int):
        self.key, self.lost = key, lost
        super().__init__(
            f"reader of queue {key!r} fell behind and lost {lost} message(s); "
            f"read faster or declare a larger maxlen"
        )


class ResourceBusy(RoborunError):
    """Actuators could not be acquired: owned (on_busy='fail'), no authority to take, or wait_timeout expired."""


class TaskFailed(RoborunError):
    """Raised by ``await task`` when the task ended in ``failed`` (SPEC §1.3)."""

    def __init__(self, task_name: str, cause: BaseException):
        self.task_name = task_name
        self.cause = cause
        super().__init__(f"task {task_name!r} failed: {cause!r}")

    @property
    def path(self) -> list[str]:
        """The failed tasks from this one inwards, e.g. ``["program", "grasp", "move_to"]``."""
        out, e = [self.task_name], self.cause
        while isinstance(e, TaskFailed):
            out.append(e.task_name)
            e = e.cause
        return out

    @property
    def root(self) -> BaseException:
        """The innermost cause: the error that started the chain (SPEC §1.8)."""
        e: BaseException = self.cause
        while isinstance(e, TaskFailed):
            e = e.cause
        return e


class TaskCancelled(RoborunError):
    """Raised by ``await task`` when the task ended in ``cancelled`` (SPEC §1.3).

    ``reason`` is one of: requested, timeout, preempted, rejected, watchdog, parent.
    """

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or f"task cancelled: {reason}")
        self.reason = reason


class BackendError(RoborunError):
    """The layer below could not do what was asked (SPEC §1.8). The task ends ``failed``.

    ``code`` is one of: unreachable, no_plan, blocked, tolerance, fault, unavailable, unknown.
    ``raw`` keeps the lower layer's own error unchanged.
    """

    CODES = frozenset(
        {"unreachable", "no_plan", "blocked", "tolerance", "fault", "unavailable", "unknown"}
    )

    def __init__(
        self,
        code: str,
        message: str,
        *,
        source: str = "",
        raw: object = None,
        stamp: float | None = None,
    ):
        if code not in self.CODES:
            raise ValueError(f"unknown BackendError code {code!r}; expected one of {sorted(self.CODES)}")
        self.code, self.message, self.source, self.raw, self.stamp = code, message, source, raw, stamp
        super().__init__(f"{code}: {message}" + (f" (from {source})" if source else ""))


class Replaced(RoborunError):
    """Raised inside a program at its next safe point after a new version was requested
    (SPEC §6). The program ends ``cancelled`` with reason ``replaced``; the step it was
    running finished under the old version, and the new version starts next."""
