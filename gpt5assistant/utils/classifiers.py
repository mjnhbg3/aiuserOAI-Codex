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


EDIT_HINTS = (
    "edit",
    "change",
    "modify",
    "make this",
    "turn this",
    "remove",
    "add",
    "replace",
    "inpaint",
    "outpaint",
    "erase",
    "fix",
    "improve",
    "upscale",
    "stylize",
)


def looks_like_image_edit_request(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in EDIT_HINTS)
