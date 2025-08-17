from __future__ import annotations

import datetime as _dt
from typing import Dict, Tuple

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


# Variable categorization for caching optimization
DYNAMIC_VARIABLES = {
    "{currenttime}": lambda: _dt.datetime.now().strftime("%H:%M"),
    "{currentdate}": lambda: _dt.datetime.now().strftime("%Y/%m/%d"),
    "{currentweekday}": lambda: _dt.datetime.now().strftime("%A"),
    "{randomnumber}": lambda: str(__import__("random").randint(0, 100)),
}

SEMI_DYNAMIC_VARIABLES = {
    "{authorname}": lambda msg: msg.author.display_name,
    "{authormention}": lambda msg: msg.author.mention,
    "{channelname}": lambda msg: msg.channel.name if hasattr(msg.channel, "name") else "",
    "{channeltopic}": lambda msg: getattr(msg.channel, "topic", "") or "",
    "{servername}": lambda msg: msg.guild.name if msg.guild else "",
    "{botname}": lambda msg: (msg.guild.me.nick or msg.guild.me.display_name) if msg.guild and msg.guild.me else "Bot",
    "{botowner}": lambda msg: (getattr(msg.guild.owner, "display_name", "") if msg.guild and getattr(msg.guild, "owner", None) else ""),
    "{serveremojis}": lambda msg: (" ".join(str(e) for e in msg.guild.emojis[:50]) if msg.guild and msg.guild.emojis else ""),
}


async def separate_template_variables(message: discord.Message, template: str) -> Tuple[str, Dict[str, str]]:
    """Separate static template from dynamic variables for caching optimization
    
    Returns:
        Tuple of (static_template, dynamic_values)
        - static_template: Template with variable placeholders for caching (or original if no variables)
        - dynamic_values: Actual variable values for dynamic context
    """
    
    if not template:
        return template or "", {}
    
    static_template = template
    dynamic_values = {}
    
    try:
        # Extract dynamic variables (time-based, changes frequently)
        for var_name, var_func in DYNAMIC_VARIABLES.items():
            if var_name in static_template:
                try:
                    dynamic_values[var_name] = var_func()
                    # Replace with placeholder for static template
                    placeholder = f"{{{{DYNAMIC_{var_name.strip('{}').upper()}}}}}"
                    static_template = static_template.replace(var_name, placeholder)
                except Exception:
                    # If variable extraction fails, skip this variable
                    continue
        
        # Extract semi-dynamic variables (user/context-based, changes less frequently)
        for var_name, var_func in SEMI_DYNAMIC_VARIABLES.items():
            if var_name in static_template:
                try:
                    dynamic_values[var_name] = var_func(message)
                    # Replace with placeholder for static template
                    placeholder = f"{{{{CONTEXT_{var_name.strip('{}').upper()}}}}}"
                    static_template = static_template.replace(var_name, placeholder)
                except Exception:
                    # If variable extraction fails, skip this variable
                    continue
    
    except Exception:
        # If separation completely fails, return original template
        return template, {}
    
    return static_template, dynamic_values


def build_dynamic_context_message(dynamic_values: Dict[str, str]) -> str:
    """Build dynamic context as separate system message"""
    if not dynamic_values:
        return ""
    
    context_parts = []
    
    # Time-based context (changes frequently)
    time_vars = {k: v for k, v in dynamic_values.items() if k in DYNAMIC_VARIABLES}
    if time_vars:
        time_parts = []
        for var_name, value in time_vars.items():
            clean_name = var_name.strip("{}")
            time_parts.append(f"{clean_name}={value}")
        if time_parts:
            context_parts.append(f"Current time context: {', '.join(time_parts)}")
    
    # User/channel context (changes less frequently)
    context_vars = {k: v for k, v in dynamic_values.items() if k in SEMI_DYNAMIC_VARIABLES}
    if context_vars:
        context_parts_list = []
        for var_name, value in context_vars.items():
            clean_name = var_name.strip("{}")
            if value:  # Only include non-empty values
                context_parts_list.append(f"{clean_name}={value}")
        if context_parts_list:
            context_parts.append(f"Current interaction context: {', '.join(context_parts_list)}")
    
    return "\n".join(context_parts)
