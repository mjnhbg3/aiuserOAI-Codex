from __future__ import annotations

from typing import Any, Dict, List, Optional, Iterable

import asyncio



def build_messages(system_prompt: str, history: List[Dict[str, str]], new_user: str) -> List[Dict[str, Any]]:
    # Note: system_prompt is no longer inserted as a system message.
    # We pass the prompt via Responses API 'instructions' only, to avoid duplication.
    msgs: List[Dict[str, Any]] = []
    msgs.extend(history)
    msgs.append({"role": "user", "content": new_user})
    return msgs


def _token_counter():
    try:
        import tiktoken

        def count(model: str, text: str) -> int:
            try:
                enc = tiktoken.encoding_for_model(model)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text, disallowed_special=()))

        return count
    except Exception:
        return lambda _m, s: len(s) // 4


async def gather_history(
    *,
    channel,
    bot_user_id: Optional[int],
    before_message,
    max_turns: int,
    max_chars: int,
    include_others: bool,
    model: str = "gpt-5",
    backread_limit: int = 25,
    max_seconds_gap: int = 1800,
    optin_set: Iterable[int] = (),
    optout_set: Iterable[int] = (),
    optin_by_default: bool = True,
    earliest_timestamp=None,
) -> List[Dict[str, str]]:
    """Collect recent channel context into chat history list.

    - Maps bot messages -> role assistant, others -> user.
    - Stops when hits max_turns pairs roughly or max_chars budget.
    - If channel has no history() (tests), returns empty list.
    """
    history: List[Dict[str, str]] = []
    if not hasattr(channel, "history"):
        return history
    count_chars = 0
    count_tokens = 0
    tok = _token_counter()
    # Pull a reasonable number to filter down to pairs
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
        async for msg in channel.history(limit=max(backread_limit, max_turns * 6), before=before_message, oldest_first=False):
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
            text = msg.content.strip()
            if not text:
                continue
            # crude char budgeting
            if count_chars + len(text) > max_chars:
                break
            t = tok(model, text)
            if count_tokens + t > max_chars * 2:
                break
            history.append({"role": role, "content": text})
            count_chars += len(text)
            count_tokens += t
            if len(history) >= max_turns * 2:
                break
    except Exception:
        return []
    # reverse to chronological oldest->newest for the model
    history.reverse()
    return history
