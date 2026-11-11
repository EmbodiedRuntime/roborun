"""Message framing between the runtime and the Gate process (SPEC §3.8).

Each message is a msgpack map prefixed by its length as a 4-byte big-endian integer.
Sequences decode as tuples, so poses stay tuples on both sides.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

import msgpack

_LEN = struct.Struct(">I")


def encode(msg: dict[str, Any]) -> bytes:
    body = msgpack.packb(msg, use_bin_type=True)
    return _LEN.pack(len(body)) + body


async def read(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one message, or return None when the peer has closed the connection."""
    try:
        head = await reader.readexactly(_LEN.size)
        body = await reader.readexactly(_LEN.unpack(head)[0])
    except (asyncio.IncompleteReadError, ConnectionResetError):
        return None
    return msgpack.unpackb(body, raw=False, use_list=False, strict_map_key=False)


def write(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    writer.write(encode(msg))
