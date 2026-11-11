"""Tools (SPEC Appendix A): functions a harness or generated code may start by name.

``@tool`` marks a function, usually a skill, as a tool; it does not change it. A host
collects the tools of the modules it is given, so two backends can both have a
``move_to`` without a global registry clashing. Each tool gets a JSON schema from its
signature, which the MCP adapter exports (SPEC §5.1: tool schemas are the harness's
format, so roborun only translates).
"""

from __future__ import annotations

import importlib
import inspect
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

_MARK = "__roborun_tool__"


def tool(fn: Callable | None = None, *, name: str | None = None,
         description: str | None = None) -> Any:
    """Mark ``fn`` as a tool. Use as ``@tool`` or ``@tool(name=..., description=...)``."""

    def mark(f: Callable) -> Callable:
        setattr(f, _MARK, {"name": name or f.__name__, "description": description})
        return f

    return mark(fn) if fn is not None else mark


@dataclass
class Tool:
    name: str
    fn: Callable
    description: str
    schema: dict


def _json_type(annotation: Any) -> dict:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = typing.get_origin(annotation)
    if origin in (tuple, list) or annotation in (tuple, list):
        args = [a for a in typing.get_args(annotation) if a is not Ellipsis]
        item = _json_type(args[0]) if args else {}
        out = {"type": "array", "items": item}
        if origin is tuple and args and Ellipsis not in typing.get_args(annotation):
            out["minItems"] = out["maxItems"] = len(args)
        return out
    if origin in (typing.Union, types.UnionType):
        options = [_json_type(a) for a in typing.get_args(annotation) if a is not type(None)]
        return options[0] if len(options) == 1 else {"anyOf": options}
    return {str: {"type": "string"}, float: {"type": "number"}, int: {"type": "integer"},
            bool: {"type": "boolean"}, dict: {"type": "object"}}.get(annotation, {})


def describe(fn: Callable, name: str | None = None, description: str | None = None) -> Tool:
    target = inspect.unwrap(fn)
    try:
        hints = typing.get_type_hints(target)
    except Exception:  # noqa: BLE001 - unresolvable annotations: no types in the schema
        hints = {}
    props, required = {}, []
    for p in inspect.signature(target).parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[p.name] = _json_type(hints.get(p.name, p.annotation))
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        else:
            props[p.name]["default"] = p.default
    doc = description or inspect.getdoc(target) or ""
    return Tool(
        name or target.__name__, fn, doc.strip(),
        {"type": "object", "properties": props, "required": required},
    )


def collect(modules: Iterable[str | types.ModuleType]) -> dict[str, Tool]:
    """The tools marked with ``@tool`` in ``modules`` (module objects or dotted names)."""
    out: dict[str, Tool] = {}
    for m in modules:
        mod = importlib.import_module(m) if isinstance(m, str) else m
        for value in vars(mod).values():
            # Only plain functions; proxies such as ``world`` must not be touched.
            mark = vars(value).get(_MARK) if inspect.isfunction(value) else None
            if mark is None:
                continue
            t = describe(value, mark["name"], mark["description"])
            if t.name in out and out[t.name].fn is not value:
                raise ValueError(f"two tools are named {t.name!r}")
            out[t.name] = t
    return out
