from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Callable, Optional


class RateLimiter:
    def __init__(self, min_interval: float = 0.2) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    async def wait(self) -> None:
        now = asyncio.get_event_loop().time()
        delay = max(0.0, self.min_interval - (now - self._last))
        if delay:
            await asyncio.sleep(delay)
        self._last = asyncio.get_event_loop().time()


async def stream_text_buffered(
    source: AsyncGenerator[str, None],
    flush_cb: Callable[[str], "asyncio.Future[None] | None"],
    interval: float = 0.5,
    max_buffer: int = 1600,
) -> str:
    """Consume an async generator of text, periodically flushing to callback."""
    buf: list[str] = []
    size = 0
    rl = RateLimiter(interval)
    full_text = []
    async for chunk in source:
        buf.append(chunk)
        full_text.append(chunk)
        size += len(chunk)
        if size >= max_buffer:
            await flush_cb("".join(buf))
            buf.clear()
            size = 0
            await rl.wait()
    if buf:
        await flush_cb("".join(buf))
    return "".join(full_text)

