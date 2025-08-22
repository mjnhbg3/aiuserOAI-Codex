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
    images_backread: int
    images_backread_seconds: int
    # Response filters
    removelist_regexes: List[str]
    # Prompt overrides
    role_prompts: Dict[int, str]
    member_prompts: Dict[int, str]
    # Memory system
    memories_enabled: bool
    memories_max_items_per_call: int
    memories_max_profile_kb: int
    memories_confidence_min: float
    memories_similarity_window_minutes: int
    memories_vector_store_max_files: int
    memories_vector_store_id_by_guild: Dict[str, str]
    memories_consolidation_char_limit: int
    # Token optimization settings
    enable_response_storage: bool


DEFAULT_TOOLS: Dict[str, bool] = {
    "web_search": True,
    "file_search": False,
    "code_interpreter": False,
    "image": True,
    "memories": True,
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
    "images_backread": 3,
    "images_backread_seconds": 1800,
    "removelist_regexes": [],
    # Debug toggles
    "debug_attachments": False,
    "role_prompts": {},
    "member_prompts": {},
    # Memory system defaults
    "memories_enabled": True,
    "memories_max_items_per_call": 50,
    "memories_max_profile_kb": 10,
    "memories_confidence_min": 0.4,
    "memories_similarity_window_minutes": 5,  # Separate from backread time for spam prevention
    "memories_vector_store_max_files": 8000,  # Utilize 1GB quota effectively (~800MB)
    "memories_vector_store_id_by_guild": {},
    "memories_consolidation_char_limit": 400,  # Max characters per consolidated memory
    # Token optimization settings
    "enable_response_storage": True,  # Enable caching by default for token savings
}
