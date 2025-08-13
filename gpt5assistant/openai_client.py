from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from hashlib import sha256
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
    inline_image_urls: Optional[List[str]] = None
    # Code interpreter container type (model/environment specific)
    code_container_type: Optional[str] = None


class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None, *, timeout: float = 180.0) -> None:
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._api_key = api_key
        self._base_url = "https://api.openai.com/v1"

    async def _fetch_container_file(self, container_id: str, file_id: str) -> Optional[bytes]:
        """Download bytes for a container file via the containers API.

        Uses direct HTTP call to ensure compatibility even if the SDK surface differs.
        """
        if not self._api_key:
            return None
        try:
            import httpx
            url = f"{self._base_url}/containers/{container_id}/files/{file_id}/content"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                # Some endpoints require this beta header for container access
                "OpenAI-Beta": "assistants=v2",
            }
            async with httpx.AsyncClient(timeout=60) as http:
                r = await http.get(url, headers=headers)
                if r.status_code == 200:
                    return r.content
        except Exception:
            return None
        return None

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
        inline_image_urls: Optional[List[str]] = None,
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
                            # Per API Reference, provide file IDs under 'image_file'
                            parts.append({"type": "input_image", "image_file": {"file_id": fid}})
                    if inline_image_urls:
                        for url in inline_image_urls:
                            # API expects image_url as a string, not an object
                            parts.append({"type": "input_image", "image_url": url})
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
                inline_image_urls=options.inline_image_urls,
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
        *,
        debug: bool = False,
    ) -> Dict[str, Any]:
        if not hasattr(self.client, "responses"):
            raise RuntimeError(
                "OpenAI SDK does not support Responses API. Install 'openai>=1.99.0' and restart Red."
            )

        attachments = None
        dbg: List[str] = []
        def _dbg(msg: str) -> None:
            if debug:
                try:
                    dbg.append(msg)
                except Exception:
                    pass
        if options.file_ids:
            attachments = [
                {"file_id": fid, "tools": [{"type": "file_search"}]}
                for fid in options.file_ids
            ]

        enable_fs = bool(options.tools.get("file_search") and options.vector_store_id)

        _dbg("responses.create: sending request")
        resp = await self.client.responses.create(
            model=options.model,
            input=self._to_input(
                messages,
                file_ids=options.file_ids,
                enable_file_search=enable_fs,
                inline_file_ids=options.inline_file_ids,
                inline_image_ids=options.inline_image_ids,
                inline_image_urls=options.inline_image_urls,
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
        _dbg(f"responses.create: status={status}")
        if status not in {"completed", "failed", "cancelled"}:
            deadline = time.monotonic() + 170.0
            while time.monotonic() < deadline:
                await asyncio.sleep(0.75)
                try:
                    resp = await self.client.responses.get(resp.id)
                    status = getattr(resp, "status", None)
                    _dbg(f"responses.get: status={status}")
                    if status in {"completed", "failed", "cancelled"}:
                        break
                except Exception:
                    _dbg("responses.get: exception while polling; stopping early")
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
        image_names: list[Optional[str]] = []
        image_urls: list[str] = []
        file_ids_to_fetch: list[str] = []
        all_file_ids: set[str] = set()
        filename_to_fileid: dict[str, str] = {}
        id_to_filename: dict[str, str] = {}
        mentioned_filenames: list[str] = []
        container_citations: list[Dict[str, str]] = []  # {container_id, file_id, filename}
        files: list[Dict[str, Any]] = []  # {name, bytes}
        container_ids: set[str] = set()
        cfile_ids: set[str] = set()
        cfile_container: dict[str, str] = {}
        # Extract any filenames from the visible text for correlation (e.g., parabola.png, random_numbers.csv)
        try:
            import re
            if text:
                pattern = r"([A-Za-z0-9_\- .]+\.(?:png|jpg|jpeg|gif|bmp|webp|tif|tiff|csv|pdf|zip|xlsx|xls|json|txt|md|html|docx|pptx|xml|tar))"
                for m in re.findall(pattern, text, flags=re.IGNORECASE):
                    # Normalize spacing and case
                    fname = m.strip().lower()
                    if fname not in mentioned_filenames:
                        mentioned_filenames.append(fname)
            _dbg(f"text filenames: {mentioned_filenames}")
        except Exception:
            pass
        try:
            out = getattr(resp, "output", None)
            if isinstance(out, list):
                for msg in out:
                    m = msg if isinstance(msg, dict) else getattr(msg, "__dict__", {})
                    mtype = m.get("type") or getattr(msg, "type", None)
                    # Message-level annotations
                    mans = m.get("annotations") or getattr(msg, "annotations", None)
                    if isinstance(mans, list):
                        for ann in mans:
                            ad = ann if isinstance(ann, dict) else getattr(ann, "__dict__", {})
                            atype = str(ad.get("type", "")).lower().replace('.', '_').replace('-', '_')
                            if ("container" in atype and "file" in atype and "citation" in atype):
                                fid = ad.get("file_id")
                                cid = ad.get("container_id")
                                fname = (ad.get("filename") or ad.get("name") or "").strip()
                                if isinstance(fid, str) and isinstance(cid, str):
                                    container_citations.append({
                                        "container_id": cid,
                                        "file_id": fid,
                                        "filename": fname,
                                    })
                                    container_ids.add(cid)
                                    if str(fid).startswith("cfile_"):
                                        cfile_ids.add(fid)
                                        cfile_container[fid] = cid
                    # Direct image generation call
                    if mtype == "image_generation_call":
                        res = m.get("result") or getattr(msg, "result", None)
                        if isinstance(res, str) and res.strip():
                            try:
                                images.append(base64.b64decode(res))
                                image_names.append(None)
                            except Exception:
                                pass
                        url = m.get("url") or getattr(msg, "url", None)
                        if isinstance(url, str):
                            image_urls.append(url)
                        fid = m.get("file_id") or m.get("id") or getattr(msg, "id", None)
                        if isinstance(fid, str):
                            file_ids_to_fetch.append(fid)
                            all_file_ids.add(fid)
                            if fid.startswith("cfile_"):
                                cfile_ids.add(fid)
                            _dbg(f"image_generation_call: file_id={fid}")
                        # Continue on to also scan nested structures on this message, if any
                    # Message content list
                    content = getattr(msg, "content", None)
                    if content is None and isinstance(msg, dict):
                        content = msg.get("content")
                    if isinstance(content, list):
                        for c in content:
                            cdict = c if isinstance(c, dict) else getattr(c, "__dict__", {})
                            ctype = cdict.get("type")
                            # Capture container file citations from annotations on any content item
                            anns = cdict.get("annotations") or getattr(c, "annotations", None)
                            if isinstance(anns, list):
                                for ann in anns:
                                    ad = ann if isinstance(ann, dict) else getattr(ann, "__dict__", {})
                                    atype = str(ad.get("type", "")).lower().replace('.', '_').replace('-', '_')
                                    # Accept any annotation type that looks like a container file citation
                                    if ("container" in atype and "file" in atype and "citation" in atype):
                                        fid = ad.get("file_id")
                                        cid = ad.get("container_id")
                                        fname = (ad.get("filename") or ad.get("name") or "").strip()
                                        if isinstance(fid, str) and isinstance(cid, str):
                                            container_citations.append({
                                                "container_id": cid,
                                                "file_id": fid,
                                                "filename": fname,
                                            })
                                            container_ids.add(cid)
                                            if str(fid).startswith("cfile_"):
                                                cfile_ids.add(fid)
                                                cfile_container[fid] = cid
                            if ctype in ("output_image", "image"):
                                imgobj = cdict.get("image") or {}
                                b64 = imgobj.get("b64_json") or cdict.get("b64_json")
                                if b64:
                                    try:
                                        images.append(base64.b64decode(b64))
                                        image_names.append(None)
                                    except Exception:
                                        pass
                                url = imgobj.get("url") or cdict.get("url")
                                if isinstance(url, str):
                                    image_urls.append(url)
                                fid = imgobj.get("file_id") or imgobj.get("id") or cdict.get("id")
                                if isinstance(fid, str):
                                    file_ids_to_fetch.append(fid)
                                    all_file_ids.add(fid)
                                    if fid.startswith("cfile_"):
                                        cfile_ids.add(fid)
                                    _dbg(f"output_image: file_id={fid}")
                                # Filename mapping if present
                                fname = (imgobj.get("filename") or cdict.get("filename") or cdict.get("name") or "").strip().lower()
                                if fname and isinstance(fid, str) and fname not in filename_to_fileid:
                                    filename_to_fileid[fname] = fid
                                    id_to_filename[fid] = fname
                            elif ctype in ("tool_output", "tool_result"):
                                data = cdict.get("content") or cdict.get("output")
                                # Data might be dict, list, or string
                                if isinstance(data, dict):
                                    b64 = data.get("b64_json")
                                    if b64:
                                        try:
                                            images.append(base64.b64decode(b64))
                                        except Exception:
                                            pass
                                    # Look for nested image/file references
                                    fid = data.get("file_id") or data.get("id")
                                    if isinstance(fid, str):
                                        file_ids_to_fetch.append(fid)
                                        all_file_ids.add(fid)
                                    fname = (data.get("filename") or data.get("name") or "").strip().lower()
                                    if fname and isinstance(fid, str) and fname not in filename_to_fileid:
                                        filename_to_fileid[fname] = fid
                                        id_to_filename[fid] = fname
                                    # Some SDKs place images under 'images'
                                    imgs = data.get("images")
                                    if isinstance(imgs, list):
                                        for it in imgs:
                                            if isinstance(it, dict):
                                                if isinstance(it.get("b64_json"), str):
                                                    try:
                                                        images.append(base64.b64decode(it["b64_json"]))
                                                    except Exception:
                                                        pass
                                                fid2 = it.get("file_id") or it.get("id")
                                                if isinstance(fid2, str):
                                                    file_ids_to_fetch.append(fid2)
                                                    all_file_ids.add(fid2)
                                                    if str(fid2).startswith("cfile_"):
                                                        cfile_ids.add(fid2)
                                                fname2 = (it.get("filename") or it.get("name") or "").strip().lower()
                                                if fname2 and isinstance(fid2, str) and fname2 not in filename_to_fileid:
                                                    filename_to_fileid[fname2] = fid2
                                                    id_to_filename[fid2] = fname2
                                elif isinstance(data, list):
                                    for it in data:
                                        if isinstance(it, dict):
                                            b64 = it.get("b64_json")
                                            if isinstance(b64, str):
                                                try:
                                                    images.append(base64.b64decode(b64))
                                                except Exception:
                                                    pass
                                            fid = it.get("file_id") or it.get("id")
                                            if isinstance(fid, str):
                                                file_ids_to_fetch.append(fid)
                                                all_file_ids.add(fid)
                                                if fid.startswith("cfile_"):
                                                    cfile_ids.add(fid)
                                            fname = (it.get("filename") or it.get("name") or "").strip().lower()
                                            if fname and isinstance(fid, str) and fname not in filename_to_fileid:
                                                filename_to_fileid[fname] = fid
                                                id_to_filename[fid] = fname
                                        elif isinstance(it, str):
                                            try:
                                                s = it.strip()
                                                # If the tool returned a filename as a plain string (common), capture it
                                                import re
                                                m = re.match(r"^[A-Za-z0-9_\- .]+\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff|csv|pdf|zip|xlsx|xls|json|txt|md|html|docx|pptx|xml|tar)$", s, flags=re.IGNORECASE)
                                                if m:
                                                    low = s.lower()
                                                    if low not in mentioned_filenames:
                                                        mentioned_filenames.append(low)
                                            except Exception:
                                                pass
                                elif isinstance(data, str):
                                    try:
                                        s = data.strip()
                                        import re
                                        m = re.match(r"^[A-Za-z0-9_\- .]+\.(png|jpg|jpeg|gif|bmp|webp|tif|tiff|csv|pdf|zip|xlsx|xls|json|txt|md|html|docx|pptx|xml|tar)$", s, flags=re.IGNORECASE)
                                        if m:
                                            low = s.lower()
                                            if low not in mentioned_filenames:
                                                mentioned_filenames.append(low)
                                    except Exception:
                                        pass
                            elif ctype in ("output_file", "file"):
                                fid = cdict.get("file_id") or cdict.get("id")
                                fname_raw = (cdict.get("filename") or cdict.get("name") or "").strip()
                                fname = fname_raw.lower() if isinstance(fname_raw, str) else ""
                                mime = cdict.get("mime_type") or cdict.get("mime")
                                is_img = False
                                if isinstance(mime, str) and mime.startswith("image/"):
                                    is_img = True
                                if not is_img and isinstance(fname, str):
                                    low = fname.lower()
                                    if any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")):
                                        is_img = True
                                if isinstance(fid, str):
                                    # Prioritize fetching this file id regardless of type; we'll sniff bytes later
                                    file_ids_to_fetch.append(fid)
                                    all_file_ids.add(fid)
                                    # Track filename mappings for nicer names on attachment
                                    if fname:
                                        filename_to_fileid.setdefault(fname, fid)
                                        id_to_filename[fid] = fname
                                    # If this looks like a container file id, remember it for container fetch fallback
                                    if str(fid).startswith("cfile_"):
                                        cfile_ids.add(fid)
                                    _dbg(f"output_file: file_id={fid} name={fname_raw} mime={mime}")
            # Final safety pass: recursively scan the entire output for any stray image/file refs
            def _scan(obj: Any) -> None:
                if isinstance(obj, dict):
                    # Base64 blobs
                    b64 = obj.get("b64_json")
                    if isinstance(b64, str):
                        try:
                            images.append(base64.b64decode(b64))
                        except Exception:
                            pass
                    # URLs
                    u = obj.get("url")
                    if isinstance(u, str):
                        image_urls.append(u)
                    # Container id anywhere present
                    cid = obj.get("container_id")
                    if isinstance(cid, str) and cid:
                        container_ids.add(cid)
                    # File IDs (favor likely image mime)
                    fid = obj.get("file_id") or obj.get("id")
                    mime = obj.get("mime_type") or obj.get("mime")
                    fname = obj.get("filename") or obj.get("name")
                    if isinstance(fid, str):
                        if fid.startswith("cfile_"):
                            cfile_ids.add(fid)
                            if isinstance(cid, str):
                                cfile_container[fid] = cid
                        is_img = False
                        if isinstance(mime, str) and mime.startswith("image/"):
                            is_img = True
                        if not is_img and isinstance(fname, str):
                            low = fname.lower()
                            if any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")):
                                is_img = True
                        if not is_img and ("image" in str(obj.get("type", "")).lower()):
                            is_img = True
                        if is_img:
                            file_ids_to_fetch.append(fid)
                            try:
                                if isinstance(fname, str) and fname:
                                    low = fname.strip().lower()
                                    if low not in filename_to_fileid:
                                        filename_to_fileid[low] = fid
                            except Exception:
                                pass
                        # Track all file ids regardless for sniffing if needed
                        all_file_ids.add(fid)
                    # Consider filename-like strings as hints
                    try:
                        for key in ("filename", "name", "path"):
                            v = obj.get(key)
                            if isinstance(v, str):
                                base = v
                                # For paths, take basename
                                if key == "path":
                                    try:
                                        import os
                                        base = os.path.basename(v)
                                    except Exception:
                                        base = v
                                low = base.strip().lower()
                                if any(low.endswith(ext) for ext in (".png",".jpg",".jpeg",".gif",".bmp",".webp",".tif",".tiff",".csv",".pdf",".zip",".xlsx",".xls",".json",".txt",".md",".html",".docx",".pptx",".xml",".tar")):
                                    if low not in mentioned_filenames:
                                        mentioned_filenames.append(low)
                    except Exception:
                        pass
                    for v in obj.values():
                        _scan(v)
                elif isinstance(obj, list):
                    for v in obj:
                        _scan(v)
            try:
                if out is not None:
                    _scan(out)
            except Exception:
                pass
            # Also scan the entire response object for any container/file hints
            try:
                _scan(getattr(resp, "__dict__", {}))
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
                                try:
                                    from urllib.parse import urlparse
                                    import os
                                    path = urlparse(u).path
                                    image_names.append(os.path.basename(path) or None)
                                except Exception:
                                    image_names.append(None)
                                _dbg(f"image_url fetched: {u} ({len(r.content)} bytes)")
                        except Exception:
                            _dbg(f"image_url fetch failed: {u}")
                            continue
            except Exception:
                pass

        # Try listing response files (if SDK supports it) to discover image assets
        try:
            # Some SDK versions expose responses.files.list(response_id)
            files_api = getattr(self.client, "responses", None)
            if files_api is not None and hasattr(files_api, "files") and hasattr(files_api.files, "list"):
                try:
                    # Try keyword first, then positional for older SDKs
                    try:
                        flist = await files_api.files.list(response_id=resp.id)
                    except Exception:
                        flist = await files_api.files.list(resp.id)
                    items = getattr(flist, "data", None) or getattr(flist, "__dict__", {}).get("data")
                    if isinstance(items, list):
                        _dbg(f"responses.files.list: {len(items)} items")
                        for it in items:
                            obj = it if isinstance(it, dict) else getattr(it, "__dict__", {})
                            fid = obj.get("id") or obj.get("file_id")
                            fname = obj.get("filename") or obj.get("display_name") or obj.get("name")
                            mime = obj.get("mime_type") or obj.get("mime")
                            is_image = False
                            if isinstance(mime, str) and mime.startswith("image/"):
                                is_image = True
                            if not is_image and isinstance(fname, str):
                                low = fname.lower()
                                if any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff")):
                                    is_image = True
                            if is_image and isinstance(fid, str):
                                file_ids_to_fetch.append(fid)
                                low = (fname or "").strip().lower()
                                if low and low not in filename_to_fileid:
                                    filename_to_fileid[low] = fid
                                    id_to_filename[fid] = low
                            elif isinstance(fid, str):
                                # non-image assets
                                all_file_ids.add(fid)
                                low = (fname or "").strip().lower()
                                if low:
                                    id_to_filename[fid] = low
                except Exception:
                    pass
        except Exception:
            pass

        # Fetch any container-cited files first (highest confidence)
        for cit in container_citations:
            try:
                cid = cit.get("container_id", "")
                fid = cit.get("file_id", "")
                _dbg(f"container_citation fetch: cid={cid} fid={fid}")
                chunk = await self._fetch_container_file(cid, fid)
                if chunk:
                    # Map filename
                    fname = (cit.get("filename") or "").strip().lower()
                    if fname and isinstance(cit.get("file_id"), str):
                        filename_to_fileid[fname] = cit["file_id"]
                        id_to_filename[cit["file_id"]] = fname
                    # Route based on content
                    def _looks_like_image(buf: bytes) -> bool:
                        try:
                            if len(buf) < 12:
                                return False
                            b = buf[:12]
                            if b.startswith(b"\x89PNG\r\n\x1a\n"):
                                return True
                            if b.startswith(b"\xff\xd8\xff"):
                                return True
                            if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
                                return True
                            if b[:4] == b"RIFF" and buf[8:12] == b"WEBP":
                                return True
                            if b[:2] == b"BM":
                                return True
                        except Exception:
                            return False
                        return False
                    if _looks_like_image(chunk):
                        images.append(chunk)
                        # Use display filename if available
                        try:
                            fname_disp = (cit.get("filename") or "").strip() or None
                            image_names.append(fname_disp)
                        except Exception:
                            image_names.append(None)
                    else:
                        name = fname or f"{cit.get('file_id','cfile')}.bin"
                        files.append({"name": name, "bytes": chunk})
                else:
                    _dbg("container_citation fetch: no bytes returned")
            except Exception:
                _dbg("container_citation fetch: exception")
                continue

        # Fetch any file ids via OpenAI files.content
        # If the text mentioned specific filenames, prioritize fetching those by mapped ids
        prioritized_ids: list[str] = []
        for fname in mentioned_filenames:
            fid = filename_to_fileid.get(fname)
            if isinstance(fid, str) and fid not in prioritized_ids:
                prioritized_ids.append(fid)
        # Add remaining discovered ids
        for fid in file_ids_to_fetch:
            if fid not in prioritized_ids:
                prioritized_ids.append(fid)
        # Finally, add any other file ids (we'll sniff content type after download)
        for fid in all_file_ids:
            if fid not in prioritized_ids:
                prioritized_ids.append(fid)

        def _looks_like_image(buf: bytes) -> bool:
            try:
                if len(buf) < 12:
                    return False
                b = buf[:12]
                # PNG
                if b.startswith(b"\x89PNG\r\n\x1a\n"):
                    return True
                # JPEG
                if b.startswith(b"\xff\xd8\xff"):
                    return True
                # GIF
                if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
                    return True
                # WEBP (RIFF....WEBP)
                if b[:4] == b"RIFF" and buf[8:12] == b"WEBP":
                    return True
                # BMP
                if b[:2] == b"BM":
                    return True
            except Exception:
                return False
            return False

        _dbg(f"files.content: prioritized_ids={prioritized_ids}")
        for fid in prioritized_ids:
            try:
                file_resp = await self.client.files.content(fid)
                # file_resp is an httpx.Response-like object with bytes in .read()
                chunk = await file_resp.aread() if hasattr(file_resp, "aread") else file_resp.read()
                if chunk:
                    if _looks_like_image(chunk):
                        images.append(chunk)
                        # record name if known
                        image_names.append(id_to_filename.get(fid))
                    else:
                        name = id_to_filename.get(fid) or f"{fid}.bin"
                        files.append({"name": name, "bytes": chunk})
                    _dbg(f"files.content: fetched fid={fid} bytes={len(chunk)}")
            except Exception:
                _dbg(f"files.content: failed fid={fid}")
                continue

        # Attempt to fetch any referenced cfile_* ids directly via known containers
        fetched_cfiles: set[str] = set()
        _dbg(f"container fallback: cfile_ids={list(cfile_ids)}")
        for cf in list(cfile_ids):
            try:
                cid = cfile_container.get(cf)
                if cid:
                    _dbg(f"container fetch: cid={cid} fid={cf}")
                    chunk = await self._fetch_container_file(cid, cf)
                    if chunk:
                        if _looks_like_image(chunk):
                            images.append(chunk)
                        else:
                            name = id_to_filename.get(cf) or f"{cf}.bin"
                            files.append({"name": name, "bytes": chunk})
                        fetched_cfiles.add(cf)
                        _dbg(f"container fetch: got bytes={len(chunk)} for {cf}")
                else:
                    _dbg(f"container fetch: no cid mapped for {cf}")
            except Exception:
                _dbg(f"container fetch: exception for {cf}")
                continue

        # Final fallback A: if we have container ids and mentioned filenames not yet attached, list container files and fetch by basename match
        try:
            import httpx, os
            if mentioned_filenames and self._api_key:
                # Calculate names already present
                present = set(f.get("name") for f in files if isinstance(f.get("name"), str))
                needed = [n for n in mentioned_filenames if n not in present]
                if needed:
                    headers = {"Authorization": f"Bearer {self._api_key}", "OpenAI-Beta": "assistants=v2"}
                    async with httpx.AsyncClient(timeout=30) as http:
                        # If we don't have a container id, list recent containers and search within them
                        search_containers: list[str] = []
                        if container_ids:
                            search_containers = list(container_ids)
                        else:
                            try:
                                rc = await http.get(f"{self._base_url}/containers", headers=headers)
                                if rc.status_code == 200:
                                    lst = rc.json().get("data") or []
                                    # take up to the 3 most recent
                                    for it in lst[:3]:
                                        cid = it.get("id")
                                        if isinstance(cid, str):
                                            search_containers.append(cid)
                                _dbg(f"containers list: searched={len(search_containers)}")
                            except Exception:
                                pass
                        for cid in search_containers:
                            try:
                                r = await http.get(f"{self._base_url}/containers/{cid}/files", headers=headers)
                                if r.status_code != 200:
                                    _dbg(f"containers/{cid}/files: status={r.status_code}")
                                    continue
                                data = r.json()
                                items = data.get("data") or []
                                _dbg(f"containers/{cid}/files: {len(items)} items")
                                for it in items:
                                    try:
                                        fid = it.get("id")
                                        path = it.get("path") or ""
                                        base = os.path.basename(path).lower() if path else ""
                                        # If we still have unknown cfile ids without container mapping, try to fetch by id
                                        if isinstance(fid, str) and fid.startswith("cfile_") and fid in cfile_ids and fid not in fetched_cfiles:
                                            _dbg(f"container list fetch by id: cid={cid} fid={fid}")
                                            chunk = await self._fetch_container_file(cid, fid)
                                            if chunk:
                                                if _looks_like_image(chunk):
                                                    images.append(chunk)
                                                else:
                                                    name = id_to_filename.get(fid) or base or f"{fid}.bin"
                                                    files.append({"name": name, "bytes": chunk})
                                                fetched_cfiles.add(fid)
                                                _dbg(f"container list fetch: got bytes={len(chunk)} for {fid}")
                                        for want in list(needed):
                                            if base == want:
                                                _dbg(f"container list fetch by name: want={want} cid={cid} fid={fid}")
                                                chunk = await self._fetch_container_file(cid, fid)
                                                if chunk:
                                                    if _looks_like_image(chunk):
                                                        images.append(chunk)
                                                    else:
                                                        files.append({"name": want, "bytes": chunk})
                                                    needed.remove(want)
                                                    _dbg(f"container list fetch: satisfied {want} bytes={len(chunk)}")
                                                break
                                    except Exception:
                                        _dbg("containers/{cid}/files: item loop exception")
                                        continue
                            except Exception:
                                _dbg(f"containers/{cid}/files: exception")
                                continue
        except Exception:
            pass

        _dbg(f"final counts: images={len(images)} files={len(files)}")

        # De-duplicate images by content (hash) and keep aligned names
        try:
            if images:
                # Ensure image_names length matches images length
                if len(image_names) < len(images):
                    image_names.extend([None] * (len(images) - len(image_names)))
                seen_hashes: set[str] = set()
                dedup_images: list[bytes] = []
                dedup_names: list[Optional[str]] = []
                for img, name in zip(images, image_names):
                    try:
                        h = sha256(img).hexdigest()
                    except Exception:
                        # If hashing fails, fall back to length-based key
                        h = f"len:{len(img)}"
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    dedup_images.append(img)
                    dedup_names.append(name)
                images = dedup_images
                image_names = dedup_names
                _dbg(f"dedup images: kept={len(images)}")
        except Exception:
            # On any error, keep originals
            pass

        # De-duplicate files by content (hash) with name preservation
        try:
            if files:
                seen_hashes: set[str] = set()
                dedup_files: list[Dict[str, Any]] = []
                for f in files:
                    data = f.get("bytes")
                    if not isinstance(data, (bytes, bytearray)):
                        continue
                    try:
                        h = sha256(data).hexdigest()
                    except Exception:
                        h = f"len:{len(data)}"
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    dedup_files.append(f)
                files = dedup_files
                _dbg(f"dedup files: kept={len(files)}")
        except Exception:
            pass

        result: Dict[str, Any] = {"text": text or "", "images": images, "image_names": image_names, "files": files}
        if debug:
            result["debug"] = dbg
        return result

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
