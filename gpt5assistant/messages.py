from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable, Callable, Awaitable

import asyncio



def build_messages(system_prompt: str, history: List[Dict[str, str]], new_user: str, current_user_name: str = None) -> List[Dict[str, Any]]:
    # Note: system_prompt is no longer inserted as a system message.
    # We pass the prompt via Responses API 'instructions' only, to avoid duplication.
    msgs: List[Dict[str, Any]] = []
    msgs.extend(history)
    
    # Include username for current user message
    if current_user_name:
        user_content = f"{current_user_name}: {new_user}"
    else:
        user_content = new_user
    
    msgs.append({"role": "user", "content": user_content})
    return msgs


def _token_counter():
    # Deprecated: token counting removed; retained for backward import safety
    return lambda _m, s: len(s) // 4


async def gather_history(
    *,
    channel,
    bot_user_id: Optional[int],
    before_message,
    include_others: bool,
    backread_limit: int = 25,
    max_seconds_gap: int = 1800,
    optin_set: Iterable[int] = (),
    optout_set: Iterable[int] = (),
    optin_by_default: bool = True,
    earliest_timestamp=None,
    skip_prefixes: Iterable[str] = (),
    is_command_message: Optional[Callable[[Any], Awaitable[bool]]] = None,
) -> List[Dict[str, str]]:
    """Collect recent channel context into chat history list.

    - Maps bot messages -> role assistant, others -> user.
    - Stops when hits max_turns pairs roughly or max_chars budget.
    - If channel has no history() (tests), returns empty list.
    """
    history: List[Dict[str, str]] = []
    if not hasattr(channel, "history"):
        return history
    # Pull up to backread_limit messages and filter
    try:
        # Normalize earliest cutoff if provided (aware datetime)
        from datetime import datetime, timezone
        earliest_dt = None
        if earliest_timestamp is not None:
            if isinstance(earliest_timestamp, (int, float)):
                earliest_dt = datetime.fromtimestamp(float(earliest_timestamp), tz=timezone.utc)
            else:
                earliest_dt = earliest_timestamp
        last_time = before_message.created_at
        async for msg in channel.history(limit=max(1, backread_limit), before=before_message, oldest_first=False):
            if not msg.content:
                continue
            # Respect time gap window
            try:
                if last_time and abs((last_time - msg.created_at).total_seconds()) > max_seconds_gap:
                    break
            except Exception:
                pass
            # Enforce forget/cutoff point (do not include messages older than earliest)
            try:
                if earliest_dt is not None and msg.created_at < earliest_dt:
                    break
            except Exception:
                pass
            if msg.author.bot and bot_user_id and msg.author.id == bot_user_id:
                role = "assistant"
            else:
                if not include_others and (not msg.author.bot or (bot_user_id and msg.author.id != bot_user_id)):
                    continue
                role = "user"
                # Opt-in checks
                uid = msg.author.id
                if uid in optout_set:
                    continue
                if (uid not in optin_set) and not optin_by_default:
                    continue
            # Skip command messages (prefer parser; fallback to prefix)
            raw = (msg.content or "")
            leading = raw.lstrip()
            used_parser = False
            parser_failed = False
            if is_command_message is not None:
                used_parser = True
                try:
                    if await is_command_message(msg):
                        continue
                except Exception:
                    parser_failed = True
            # Fallback to prefix-based skip only if parser not used or failed
            if (not used_parser) or parser_failed:
                try:
                    if any(isinstance(p, str) and p and leading.startswith(p) for p in skip_prefixes):
                        continue
                except Exception:
                    pass
            # Skip bot's own forget-confirmation message so it doesn't pollute context
            if role == "assistant":
                lowered = (msg.content or "").lower()
                if ("forget previous context" in lowered) and ("starting now" in lowered):
                    continue
            text = raw.strip()
            if not text:
                continue
            
            # Include username information for user messages
            if role == "user":
                display_name = msg.author.display_name or msg.author.name
                content_with_user = f"{display_name}: {text}"
                history.append({"role": role, "content": content_with_user})
            else:
                history.append({"role": role, "content": text})
    except Exception:
        return []
    # reverse to chronological oldest->newest for the model
    history.reverse()
    return history
