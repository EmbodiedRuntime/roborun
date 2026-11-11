"""Model backends for the agent examples: ``await model(system, messages) -> str``.

- ``scripted``: canned replies (a list, or a function of the conversation); offline, for tests.
- ``claude-code``: Claude Code in headless mode (``claude -p``), using its login; no tools.
- ``api``: the Claude API through the ``anthropic`` SDK (``pip install anthropic``).

The examples only need text in, text out. A model is not part of the runtime (SPEC §5.1):
the agent loop, the prompt, and the model choice all belong to the harness.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from typing import Any

Model = Callable[[str, list[dict]], Any]  # async (system, messages) -> str


def scripted(replies: list[str] | Callable[[list[dict]], str]) -> Model:
    """Reply from a list in order, or by calling ``replies(messages)``."""
    queue = list(replies) if isinstance(replies, list) else None

    async def model(system: str, messages: list[dict]) -> str:
        if queue is not None:
            if not queue:
                raise RuntimeError("the scripted model has no more replies")
            return queue.pop(0)
        return replies(messages)

    return model


def claude_code(model_name: str | None = None) -> Model:
    """Claude Code as a plain text model: no tools, no MCP servers, no saved session."""

    async def model(system: str, messages: list[dict]) -> str:
        transcript = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        prompt = (f"{transcript}\n\n[assistant]\n(Reply as the assistant to the last "
                  "message, following the system prompt exactly.)")
        cmd = ["claude", "-p", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--system-prompt", system, "--output-format", "json"]
        if model_name:
            cmd += ["--model", model_name]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate(prompt.encode())
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed: {err.decode()[-500:]}")
        return json.loads(out)["result"]

    return model


def claude_api(model_name: str = "claude-opus-5-5") -> Model:
    """The Claude API. Credentials come from ANTHROPIC_API_KEY or an ``ant auth login`` profile."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()

    async def model(system: str, messages: list[dict]) -> str:
        response = await client.messages.create(
            model=model_name, max_tokens=16000, system=system, messages=messages)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"the model declined: {response.stop_details}")
        return "".join(b.text for b in response.content if b.type == "text")

    return model


def pick(name: str | None = None) -> Model:
    """``claude-code`` (default), ``api``, or set ROBORUN_MODEL."""
    name = name or os.environ.get("ROBORUN_MODEL", "claude-code")
    if name == "claude-code":
        return claude_code()
    if name == "api":
        return claude_api()
    raise ValueError(f"unknown model {name!r}; use claude-code or api")
