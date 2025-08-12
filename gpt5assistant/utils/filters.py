from __future__ import annotations

import asyncio
import re
from typing import Iterable, List, Set


async def _compile_and_apply(pattern_str: str, text: str) -> str:
    pattern = re.compile(pattern_str)
    return pattern.sub("", text).strip(" \n")


async def apply_removelist(
    *,
    patterns: List[str],
    text: str,
    botname: str,
    recent_authors: Iterable[str],
    timeout: float = 0.25,
) -> str:
    # Expand placeholders
    expanded: List[str] = []
    for p in patterns:
        p = p.replace(r"{botname}", botname)
        if "{authorname}" in p:
            for a in set(recent_authors):
                expanded.append(p.replace(r"{authorname}", a))
        else:
            expanded.append(p)

    cleaned = text.strip(" \n")
    for pat in expanded:
        try:
            cleaned = await asyncio.wait_for(_compile_and_apply(pat, cleaned), timeout=timeout)
        except asyncio.TimeoutError:
            continue
        except Exception:
            continue
    return cleaned

