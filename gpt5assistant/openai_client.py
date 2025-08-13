from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import time


@dataclass
class ChatOptions:
    model: str
    tools: Dict[str, bool]
    reasoning: str
    max_tokens: int
    temperature: float
    system_prompt: str
    file_ids: Optional[List[str]] = None
    vector_store_id: Optional[str] = None
    # Attachments from the current message
    inline_file_ids: Optional[List[str]] = None
    inline_image_ids: Optional[List[str]] = None
    # Code interpreter container type (model/environment specific)
    code_container_type: Optional[str] = None


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, *, timeout: float = 180.0) -> None:
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    async def _iter_text_from_stream(self, stream) -> AsyncGenerator[str, None]:
        async for event in stream:
            etype = getattr(event, "type", None)
            if etype == "response.output_text.delta":
                yield getattr(event, "delta", "")
            elif etype == "response.output_text.done":
                # no-op; end of a text block
                continue

    async def _iter_text_from_chat_stream(self, stream) -> AsyncGenerator[str, None]:
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content
            except Exception:
                continue

    def _tools_array(
        self,
        tools: Dict[str, bool],
        *,
        vector_store_id: Optional[str],
        code_container_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        arr: List[Dict[str, Any]] = []
        if tools.get("web_search"):
            arr.append({"type": "web_search"})
        if tools.get("file_search") and vector_store_id:
            arr.append({"type": "file_search", "vector_store_ids": [vector_store_id]})
        if tools.get("code_interpreter"):
            ctype = (code_container_type or "auto").strip()
            arr.append({"type": "code_interpreter", "container": {"type": ctype}})
        if tools.get("image"):
            # Allow the model to call image generation natively via Responses tools
            arr.append({"type": "image_generation"})
        return arr

    def _to_input(
        self,
        messages: List[Dict[str, Any]],
        *,
        file_ids: Optional[List[str]] = None,
        enable_file_search: bool = False,
        inline_file_ids: Optional[List[str]] = None,
        inline_image_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        files_added = False
        inline_files_added = False
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts: List[Dict[str, Any]]
            if role in ("system", "user"):
                if isinstance(content, str):
                    parts = [{"type": "input_text", "text": content}]
                else:
                    parts = content  # assume already parts
                if (
                    enable_file_search
                    and not files_added
                    and role == "user"
                    and file_ids
                ):
                    for fid in file_ids:
                        parts.append({"type": "input_file", "file_id": fid})
                    files_added = True
                # Always allow attaching current message files/images on first user message
                if (not inline_files_added) and role == "user":
                    if inline_file_ids:
                        for fid in inline_file_ids:
                            parts.append({"type": "input_file", "file_id": fid})
                    if inline_image_ids:
                        for fid in inline_image_ids:
                            parts.append({"type": "input_image", "image_file": {"file_id": fid}})
                    inline_files_added = True
                formatted.append({"type": "message", "role": role, "content": parts})
            elif role == "assistant":
                # Encode prior assistant replies as output messages
                text = content if isinstance(content, str) else ""
                out_parts = [{"type": "output_text", "text": text}]
                formatted.append({
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": out_parts,
                })
        return formatted

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def respond_chat(
        self,
        messages: List[Dict[str, Any]],
        options: ChatOptions,
    ) -> AsyncGenerator[str, None]:
        if not hasattr(self.client, "responses"):
            raise RuntimeError(
                "OpenAI SDK does not support Responses API. Install 'openai>=1.99.0' and restart Red."
            )

        attachments = None
        if options.file_ids:
            attachments = [
                {"file_id": fid, "tools": [{"type": "file_search"}]}
                for fid in options.file_ids
            ]

        # Only enable file_search input_file parts when we actually have a vector store
        enable_fs = bool(options.tools.get("file_search") and options.vector_store_id)

        # For reliability with tool calls, use non-streaming create and yield the final text.
        resp = await self.client.responses.create(
            model=options.model,
            input=self._to_input(
                messages,
                file_ids=options.file_ids,
                enable_file_search=enable_fs,
                inline_file_ids=options.inline_file_ids,
                inline_image_ids=options.inline_image_ids,
            ),
            tools=self._tools_array(
                options.tools,
                vector_store_id=options.vector_store_id,
                code_container_type=options.code_container_type,
            ),
            reasoning={"effort": options.reasoning},
            max_output_tokens=options.max_tokens,
            tool_choice="auto",
            instructions=options.system_prompt,
        )
        # Extract output text robustly across SDK variants
        text = None
        try:
            ot = getattr(resp, "output_text", None)
            if isinstance(ot, str) and ot.strip():
                text = ot
        except Exception:
            text = None
        if not text:
            try:
                parts: list[str] = []
                out = getattr(resp, "output", None)
                if isinstance(out, list):
                    for msg in out:
                        content = getattr(msg, "content", None)
                        if content is None and isinstance(msg, dict):
                            content = msg.get("content")
                        if not isinstance(content, list):
                            continue
                        for c in content:
                            ctype = getattr(c, "type", None)
                            if ctype is None and isinstance(c, dict):
                                ctype = c.get("type")
                            if ctype == "output_text":
                                txt = getattr(c, "text", None)
                                if txt is None and isinstance(c, dict):
                                    txt = c.get("text")
                                if txt:
                                    parts.append(txt)
                text = "".join(parts)
            except Exception:
                text = ""
        if text:
            # Yield as a single chunk
            yield text

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def respond_collect(
        self,
        messages: List[Dict[str, Any]],
        options: ChatOptions,
    ) -> Dict[str, Any]:
        if not hasattr(self.client, "responses"):
            raise RuntimeError(
                "OpenAI SDK does not support Responses API. Install 'openai>=1.99.0' and restart Red."
            )

        attachments = None
        if options.file_ids:
            attachments = [
                {"file_id": fid, "tools": [{"type": "file_search"}]}
                for fid in options.file_ids
            ]

        enable_fs = bool(options.tools.get("file_search") and options.vector_store_id)

        resp = await self.client.responses.create(
            model=options.model,
            input=self._to_input(
                messages,
                file_ids=options.file_ids,
                enable_file_search=enable_fs,
                inline_file_ids=options.inline_file_ids,
                inline_image_ids=options.inline_image_ids,
            ),
            tools=self._tools_array(
                options.tools,
                vector_store_id=options.vector_store_id,
                code_container_type=options.code_container_type,
            ),
            reasoning={"effort": options.reasoning},
            max_output_tokens=options.max_tokens,
            tool_choice="auto",
            instructions=options.system_prompt,
        )

        # If the response is still running tools, poll until completed or timeout
        status = getattr(resp, "status", None)
        if status not in {"completed", "failed", "cancelled"}:
            deadline = time.monotonic() + 170.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.75)
                try:
                    resp = await self.client.responses.get(resp.id)
                    status = getattr(resp, "status", None)
                    if status in {"completed", "failed", "cancelled"}:
                        break
                except Exception:
                    break

        # Text
        text = None
        try:
            ot = getattr(resp, "output_text", None)
            if isinstance(ot, str) and ot.strip():
                text = ot
        except Exception:
            text = None
        if text is None:
            text = ""
        if not text:
            try:
                parts: list[str] = []
                out = getattr(resp, "output", None)
                if isinstance(out, list):
                    for msg in out:
                        content = getattr(msg, "content", None)
                        if content is None and isinstance(msg, dict):
                            content = msg.get("content")
                        if not isinstance(content, list):
                            continue
                        for c in content:
                            ctype = getattr(c, "type", None)
                            if ctype is None and isinstance(c, dict):
                                ctype = c.get("type")
                            if ctype == "output_text":
                                txt = getattr(c, "text", None)
                                if txt is None and isinstance(c, dict):
                                    txt = c.get("text")
                                if txt:
                                    parts.append(txt)
                text = "".join(parts)
            except Exception:
                text = ""

        # Images
        images: list[bytes] = []
        image_urls: list[str] = []
        file_ids_to_fetch: list[str] = []
        try:
            out = getattr(resp, "output", None)
            if isinstance(out, list):
                for msg in out:
                    m = msg if isinstance(msg, dict) else getattr(msg, "__dict__", {})
                    mtype = m.get("type") or getattr(msg, "type", None)
                    # Direct image generation call
                    if mtype == "image_generation_call":
                        res = m.get("result") or getattr(msg, "result", None)
                        if isinstance(res, str) and res.strip():
                            try:
                                images.append(base64.b64decode(res))
                            except Exception:
                                pass
                        url = m.get("url") or getattr(msg, "url", None)
                        if isinstance(url, str):
                            image_urls.append(url)
                        fid = m.get("id") or m.get("file_id") or getattr(msg, "id", None)
                        if isinstance(fid, str):
                            file_ids_to_fetch.append(fid)
                        continue
                    # Message content list
                    content = getattr(msg, "content", None)
                    if content is None and isinstance(msg, dict):
                        content = msg.get("content")
                    if isinstance(content, list):
                        for c in content:
                            cdict = c if isinstance(c, dict) else getattr(c, "__dict__", {})
                            ctype = cdict.get("type")
                            if ctype in ("output_image", "image"):
                                imgobj = cdict.get("image") or {}
                                b64 = imgobj.get("b64_json") or cdict.get("b64_json")
                                if b64:
                                    try:
                                        images.append(base64.b64decode(b64))
                                    except Exception:
                                        pass
                                url = imgobj.get("url") or cdict.get("url")
                                if isinstance(url, str):
                                    image_urls.append(url)
                                fid = imgobj.get("id") or imgobj.get("file_id") or cdict.get("id")
                                if isinstance(fid, str):
                                    file_ids_to_fetch.append(fid)
                            elif ctype in ("tool_output", "tool_result"):
                                data = cdict.get("content") or cdict.get("output")
                                if isinstance(data, dict):
                                    b64 = data.get("b64_json")
                                    if b64:
                                        try:
                                            images.append(base64.b64decode(b64))
                                        except Exception:
                                            pass
        except Exception:
            pass

        # Fetch any image URLs
        if image_urls:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=30) as http:
                    for u in image_urls:
                        try:
                            r = await http.get(u)
                            if r.status_code == 200:
                                images.append(r.content)
                        except Exception:
                            continue
            except Exception:
                pass

        # Fetch any file ids via OpenAI files.content
        for fid in file_ids_to_fetch:
            try:
                file_resp = await self.client.files.content(fid)
                # file_resp is an httpx.Response-like object with bytes in .read()
                chunk = await file_resp.aread() if hasattr(file_resp, "aread") else file_resp.read()
                if chunk:
                    images.append(chunk)
            except Exception:
                continue

        return {"text": text or "", "images": images}

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(4))
    async def generate_image(
        self, prompt: str, *, size: str = "1024x1024", seed: Optional[int] = None
    ) -> bytes:
        # Only include 'seed' if provided to avoid SDK TypeError on unsupported versions
        kwargs: Dict[str, Any] = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "size": size,
        }
        if seed is not None:
            kwargs["seed"] = seed
        resp = await self.client.images.generate(**kwargs)
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
        # Only include 'seed' if provided to avoid SDK TypeError on unsupported versions
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

    async def ensure_vector_store(self, *, name: str, current_id: Optional[str] = None) -> str:
        if current_id:
            return current_id
        vs = await self.client.vector_stores.create(name=name)
        return vs.id

    async def add_files_to_vector_store(self, vector_store_id: str, file_ids: List[str]) -> None:
        for fid in file_ids:
            await self.client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=fid)
