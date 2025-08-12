from __future__ import annotations

import datetime as _dt
from typing import Dict

import discord


async def format_variables(ctx_message: discord.Message, text: str) -> str:
    guild = ctx_message.guild
    channel = ctx_message.channel
    author = ctx_message.author
    bot = getattr(ctx_message.guild, "me", None) if guild else None

    now = _dt.datetime.now()
    variables: Dict[str, str] = {
        "{botname}": (bot.nick or bot.display_name) if bot else "Bot",
        "{botowner}": (getattr(guild.owner, "display_name", "") if guild and getattr(guild, "owner", None) else ""),
        "{authorname}": author.display_name,
        "{authormention}": author.mention,
        "{servername}": guild.name if guild else "",
        "{channelname}": channel.name if hasattr(channel, "name") else "",
        "{channeltopic}": getattr(channel, "topic", "") or "",
        "{currentdate}": now.strftime("%Y/%m/%d"),
        "{currentweekday}": now.strftime("%A"),
        "{currenttime}": now.strftime("%H:%M"),
        "{randomnumber}": str(__import__("random").randint(0, 100)),
    }
    if guild and guild.emojis:
        variables["{serveremojis}"] = " ".join(str(e) for e in guild.emojis[:50])
    else:
        variables["{serveremojis}"] = ""

    out = text
    for k, v in variables.items():
        out = out.replace(k, v)
    return out
