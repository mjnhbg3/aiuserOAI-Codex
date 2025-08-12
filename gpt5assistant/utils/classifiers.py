from __future__ import annotations

from typing import Iterable


IMAGE_HINTS = (
    "generate an image",
    "draw",
    "logo",
    "picture of",
    "illustration",
    "photo of",
    "image of",
    "wallpaper",
)


def looks_like_image_request(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in IMAGE_HINTS)

