"""Stamped values (SPEC §2.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stamped(Generic[T]):
    """A value with the time its underlying observation was taken, and who produced it."""

    value: T
    stamp: float
    source: str = ""

    def age(self, now: float) -> float:
        return now - self.stamp


def derive(value: Any, *inputs: Stamped, source: str = "") -> Stamped:
    """A value computed from several inputs carries the oldest of their stamps."""
    if not inputs:
        raise ValueError("derive needs at least one stamped input")
    return Stamped(value, min(i.stamp for i in inputs), source)
