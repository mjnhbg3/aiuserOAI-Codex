from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ChannelOverrides(TypedDict, total=False):
    model: str
    reasoning: str
    tools: Dict[str, bool]
    system_prompt: str
    max_tokens: int
    temperature: float
    ephemeral: bool


class GuildConfig(TypedDict, total=False):
    model: str
    reasoning: str
    tools: Dict[str, bool]
    allowed_channels: List[int]
    system_prompt: str
    max_tokens: int
    temperature: float
    ephemeral: bool
    file_kb_id: Optional[str]
    file_ids: List[str]
    respond_on_mention: bool
    random_autoreply: bool
    random_rate: float
    history_turns: int
    history_chars: int
    include_others: bool
    channel_contexts: Dict[int, str]
    # Opt-in controls
    optin_by_default: bool
    optin: List[int]
    optout: List[int]
    # Reply controls
    reply_percent: float
    reply_to_mentions_replies: bool
    # History backread/age window
    messages_backread: int
    messages_backread_seconds: int
    # Response filters
    removelist_regexes: List[str]
    # Prompt overrides
    role_prompts: Dict[int, str]
    member_prompts: Dict[int, str]


DEFAULT_TOOLS: Dict[str, bool] = {
    "web_search": True,
    "file_search": True,
    "code_interpreter": False,
    "image": True,
}


DEFAULT_GUILD_CONFIG: GuildConfig = {
    "model": "gpt-5",
    "reasoning": "medium",
    "tools": DEFAULT_TOOLS.copy(),
    "allowed_channels": [],
    "system_prompt": (
        "You are GPT-5, a helpful assistant for this Discord server. "
        "Keep replies concise."
    ),
    "max_tokens": 800,
    "temperature": 0.7,
    "ephemeral": False,
    "file_kb_id": None,
    "file_ids": [],
    "respond_on_mention": True,
    "random_autoreply": False,
    "random_rate": 0.02,
    "history_turns": 8,
    "history_chars": 6000,
    "include_others": True,
    "channel_contexts": {},
    "optin_by_default": True,
    "optin": [],
    "optout": [],
    "reply_percent": 0.5,
    "reply_to_mentions_replies": True,
    "messages_backread": 25,
    "messages_backread_seconds": 1800,
    "removelist_regexes": [],
    "role_prompts": {},
    "member_prompts": {},
}
