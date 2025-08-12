from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class ChatOptions:
    model: str
    tools: Dict[str, bool]
    reasoning: str
    verbosity: str
    max_tokens: int
    temperature: float
    system_prompt: str
    file_ids: Optional[List[str]] = None


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, *, timeout: float = 60.0) -> None:
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def _iter_text_from_stream(self, stream) -> AsyncGenerator[str, None]:
        async for event in stream:
            # Responses API streaming events; we focus on output_text.delta
            if hasattr(event, "type") and event.type == "response.output_text.delta":
                yield event.delta
            # When a tool call is happening via native OpenAI tools, server executes it.
            # We simply relay final text output chunks.

    async def _iter_text_from_chat_stream(self, stream) -> AsyncGenerator[str, None]:
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content
            except Exception:
                continue

    def _tools_array(self, tools: Dict[str, bool]) -> List[Dict[str, Any]]:
        arr: List[Dict[str, Any]] = []
        if tools.get("web_search"):
            arr.append({"type": "web_search"})
        if tools.get("file_search"):
            arr.append({"type": "file_search"})
        if tools.get("code_interpreter"):
            arr.append({"type": "code_interpreter"})
        return arr

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def respond_chat(
        self,
        messages: List[Dict[str, Any]],
        options: ChatOptions,
    ) -> AsyncGenerator[str, None]:
        # Prefer Responses API if available; otherwise fall back to Chat Completions streaming
        if hasattr(self.client, "responses"):
            attachments = None
            if options.file_ids:
                attachments = [
                    {"file_id": fid, "tools": [{"type": "file_search"}]}
                    for fid in options.file_ids
                ]

            stream = await self.client.responses.stream(
                model=options.model,
                messages=messages,
                tools=self._tools_array(options.tools),
                reasoning={"effort": options.reasoning},
                text={"verbosity": options.verbosity},
                max_output_tokens=options.max_tokens,
                temperature=options.temperature,
                attachments=attachments,
            )
            async with stream as s:
                async for text in self._iter_text_from_stream(s):
                    yield text
        else:
            # Compatibility path: Chat Completions streaming (no native tools)
            stream = await self.client.chat.completions.create(
                model=options.model,
                messages=messages,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                stream=True,
            )
            async for text in self._iter_text_from_chat_stream(stream):
                yield text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def generate_image(
        self, prompt: str, *, size: str = "1024x1024", seed: Optional[int] = None
    ) -> bytes:
        resp = await self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            seed=seed,
        )
        # SDK returns b64 JSON
        b64 = resp.data[0].b64_json
        return base64.b64decode(b64)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        size: str = "1024x1024",
        mask: Optional[bytes] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        # The SDK expects file-like objects
        from io import BytesIO

        img = BytesIO(image_bytes)
        kwargs: Dict[str, Any] = {"model": "gpt-image-1", "image": img, "prompt": prompt, "size": size}
        if mask is not None:
            kwargs["mask"] = BytesIO(mask)
        if seed is not None:
            kwargs["seed"] = seed
        resp = await self.client.images.edits(**kwargs)
        b64 = resp.data[0].b64_json
        return base64.b64decode(b64)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def index_files(self, files: List[bytes], filenames: List[str]) -> List[str]:
        # Upload files to OpenAI file store; return file IDs
        file_ids: List[str] = []
        from io import BytesIO

        for content, name in zip(files, filenames):
            fobj = BytesIO(content)
            # Provide filename to avoid multipart quirks
            try:
                fobj.name = name  # type: ignore[attr-defined]
            except Exception:
                pass
            uploaded = await self.client.files.create(file=(name, fobj), purpose="assistants")
            file_ids.append(uploaded.id)
        return file_ids
