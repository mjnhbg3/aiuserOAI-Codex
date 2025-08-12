from __future__ import annotations

from typing import Iterable, List

MAX_DISCORD = 2000


def _close_open_fence(chunk: str) -> str:
    fences = chunk.count("```")
    if fences % 2 == 1:
        return chunk + "\n```"
    return chunk


def chunk_message(text: str, limit: int = MAX_DISCORD) -> List[str]:
    if not text:
        return []

    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)

        # Try to break at last newline under limit
        slice_text = text[start:end]
        if end < len(text):
            nl = slice_text.rfind("\n")
            if nl > 0 and (start + nl) - start > 50:  # avoid tiny tail
                end = start + nl
                slice_text = text[start:end]

        slice_text = _close_open_fence(slice_text)
        parts.append(slice_text)
        start = end
    return parts

