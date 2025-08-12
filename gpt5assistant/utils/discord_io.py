from __future__ import annotations

import asyncio
from typing import Optional

import discord

ALLOWED_MENTIONS = discord.AllowedMentions(everyone=False, users=True, roles=False, replied_user=False)


async def send_with_typing(channel: discord.abc.Messageable, content: str, *, ephemeral: bool = False):
    async with channel.typing():
        await asyncio.sleep(0.2)
        return await channel.send(content, allowed_mentions=ALLOWED_MENTIONS)


def sanitize_log(text: str, max_len: int = 2000) -> str:
    if len(text) > max_len:
        return text[: max_len - 20] + "… [truncated]"
    return text

