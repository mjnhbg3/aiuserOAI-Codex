import asyncio
import pytest

from gpt5assistant.utils.streaming import stream_text_buffered


async def agen():
    for tok in ["Hello ", "world", "!"]:
        await asyncio.sleep(0)
        yield tok


@pytest.mark.asyncio
async def test_stream_text_buffered_flushes():
    outs = []

    async def flush(x: str):
        outs.append(x)

    full = await stream_text_buffered(agen(), flush, interval=0.0, max_buffer=6)
    assert "".join(outs) == "Hello world!"
    assert full == "Hello world!"

