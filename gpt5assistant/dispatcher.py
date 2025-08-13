from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from io import BytesIO
import random
import inspect

import discord
from redbot.core import commands, Config

from .config_schemas import DEFAULT_GUILD_CONFIG
from .messages import build_messages, gather_history
from .openai_client import ChatOptions, OpenAIClient
from .utils.chunking import chunk_message
from .utils.streaming import stream_text_buffered
from .utils.classifiers import looks_like_image_request, looks_like_image_edit_request
from .utils.variables import format_variables
from .utils.filters import apply_removelist


class Dispatcher:
    def __init__(self, bot: commands.Bot, config: Config, client: OpenAIClient) -> None:
        self.bot = bot
        self.config = config
        self.client = client
        self._locks: dict[int, asyncio.Lock] = {}
        self._last_intent_warn: dict[int, float] = {}

    def _get_lock(self, channel_id: int) -> asyncio.Lock:
        if channel_id not in self._locks:
            self._locks[channel_id] = asyncio.Lock()
        return self._locks[channel_id]

    async def handle_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        guild = message.guild
        gconf = await self.config.guild(guild).all()
        allowed = gconf.get("allowed_channels") or []
        if allowed and message.channel.id not in allowed:
            return

        # Skip if message starts with a command prefix (let Red command parser handle)
        # Robustly get valid prefixes (sync or async depending on Red version)
        prefixes = []
        try:
            res = self.bot.get_valid_prefixes(guild)
            if inspect.isawaitable(res):
                res = await res
            prefixes = list(res) if isinstance(res, (list, tuple)) else []
        except Exception:
            prefixes = []
        if any(message.content.startswith(p) for p in prefixes if isinstance(p, str)):
            return

        # Decide whether to reply: mention, replies, RNG, or reply_percent
        respond_on_mention = gconf.get("respond_on_mention", True)
        random_autoreply = gconf.get("random_autoreply", False)
        random_rate = float(gconf.get("random_rate", 0.0) or 0.0)
        reply_percent = float(gconf.get("reply_percent", 0.5) or 0.0)
        reply_to_mentions_replies = bool(gconf.get("reply_to_mentions_replies", True))

        mentioned = False
        if respond_on_mention and getattr(message, "mentions", None) and self.bot.user:
            mentioned = any(m.id == self.bot.user.id for m in message.mentions)

        is_reply_to_bot = False
        if reply_to_mentions_replies and message.reference and self.bot.user:
            ref = message.reference
            try:
                replied = ref.cached_message or await self.bot.get_channel(ref.channel_id).fetch_message(ref.message_id)
                is_reply_to_bot = replied.author.id == self.bot.user.id
            except Exception:
                is_reply_to_bot = False

        prob = random.random()
        should_reply = (
            mentioned
            or is_reply_to_bot
            or (random_autoreply and (prob < max(0.0, min(1.0, random_rate))))
            or (prob < max(0.0, min(1.0, reply_percent)))
        )
        if not should_reply:
            return

        content = (message.content or "").strip()
        if not content:
            # Likely Message Content Intent not enabled; warn sparingly
            now = asyncio.get_event_loop().time()
            last = self._last_intent_warn.get(guild.id, 0.0)
            if now - last > 300:  # 5 minutes
                self._last_intent_warn[guild.id] = now
                try:
                    await message.channel.send(
                        "I can’t read message text. Enable the 'Message Content Intent' "
                        "for this bot in the Discord Developer Portal and in Red."
                    )
                except Exception:
                    pass

        # Strip leading bot mention from content for cleaner prompts
        if mentioned and self.bot.user:
            mention_strs = {f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"}
            for mstr in mention_strs:
                if content.startswith(mstr):
                    content = content[len(mstr):].strip()
                    break

        # Route all requests through chat; allow the model to select tools, including image generation.
        await self._chat_path(message, content, gconf)

    async def _chat_path(self, message: discord.Message, content: str, gconf: Dict[str, Any]) -> None:
        lock = self._get_lock(message.channel.id)
        if lock.locked():
            return
        async with lock:
            # Single global system prompt with dynamic variables
            try:
                sys_prompt_global = await self.config.system_prompt()
            except Exception:
                sys_prompt_global = gconf.get("system_prompt", "")
            system_prompt = await format_variables(message, sys_prompt_global or "")

            # Gather recent channel history
            # Apply per-channel forget cutoff if set
            earliest_ts = None
            try:
                earliest_ts = await self.config.channel(message.channel).forget_after_ts()
                if not earliest_ts:
                    earliest_ts = None
            except Exception:
                earliest_ts = None

            history: List[Dict[str, str]] = await gather_history(
                channel=message.channel,
                bot_user_id=getattr(self.bot.user, "id", None) if self.bot else None,
                before_message=message,
                include_others=bool(gconf.get("include_others", True)),
                backread_limit=int(gconf.get("messages_backread", 25)),
                max_seconds_gap=int(gconf.get("messages_backread_seconds", 1800)),
                optin_set=set(gconf.get("optin", []) or []),
                optout_set=set(gconf.get("optout", []) or []),
                optin_by_default=bool(gconf.get("optin_by_default", True)),
                earliest_timestamp=earliest_ts,
            )
            msgs = build_messages(system_prompt, history, content)
            # Determine vector store availability; disable file_search tool if none
            vector_store_id = gconf.get("file_kb_id") or None
            effective_tools = dict(gconf["tools"])  # shallow copy
            if effective_tools.get("file_search"):
                # Do not use vector stores for files; only read current attachments
                effective_tools["file_search"] = False

            # Collect and upload current message attachments for inline reading/vision
            inline_file_ids: list[str] = []
            inline_image_ids: list[str] = []
            if getattr(message, "attachments", None):
                file_bytes: list[bytes] = []
                fnames: list[str] = []
                kinds: list[str] = []
                for a in message.attachments:
                    ctype = a.content_type or ""
                    # Skip very large files (>20MB) to avoid timeouts
                    try:
                        if a.size and a.size > 20 * 1024 * 1024:
                            continue
                    except Exception:
                        pass
                    try:
                        data = await a.read()
                    except Exception:
                        continue
                    file_bytes.append(data)
                    fnames.append(a.filename or "attachment")
                    kinds.append(ctype)
                if file_bytes:
                    try:
                        ids = await self.client.index_files(file_bytes, fnames)
                        for fid, ctype in zip(ids, kinds):
                            if isinstance(ctype, str) and ctype.startswith("image/"):
                                inline_image_ids.append(fid)
                            else:
                                inline_file_ids.append(fid)
                    except Exception:
                        inline_file_ids = []
                        inline_image_ids = []

            options = ChatOptions(
                model=gconf["model"],
                tools=effective_tools,
                reasoning=gconf["reasoning"],
                max_tokens=gconf["max_tokens"],
                temperature=gconf["temperature"],
                system_prompt=system_prompt,
                file_ids=None,
                vector_store_id=None,
                inline_file_ids=inline_file_ids or None,
                inline_image_ids=inline_image_ids or None,
            )

            sent_msg: Optional[discord.Message] = None

            async def flush_cb(buf: str):
                nonlocal sent_msg
                chunks = chunk_message(buf)
                if not chunks:
                    return
                # First chunk: reply; subsequent edits append
                if sent_msg is None:
                    sent_msg = await message.channel.send(chunks[0])
                    for ch in chunks[1:]:
                        await message.channel.send(ch)
                else:
                    for ch in chunks:
                        await message.channel.send(ch)

            try:
                patterns = gconf.get("removelist_regexes", []) or []
                # If filters present, disable streaming and send once after cleanup
                if patterns:
                    async with message.channel.typing():
                        stream = self.client.respond_chat(msgs, options)
                        full_chunks: list[str] = []
                        async for t in stream:
                            full_chunks.append(t)
                        text = "".join(full_chunks)
                        # recent authors for {authorname}
                        authors = []
                        if hasattr(message.channel, "history"):
                            async for m in message.channel.history(limit=10):
                                if m.author != message.guild.me:
                                    authors.append(m.author.display_name)
                        botname = (message.guild.me.nick or message.guild.me.display_name) if message.guild else ""
                        cleaned = await apply_removelist(
                            patterns=patterns, text=text, botname=botname, recent_authors=authors
                        )
                        if cleaned.strip():
                            for ch in chunk_message(cleaned):
                                await message.channel.send(ch)
                        else:
                            await message.channel.send("I didn’t generate any text for that. Try rephrasing or mention me directly.")
                else:
                    # Try with tools first
                    async with message.channel.typing():
                        stream = self.client.respond_chat(msgs, options)
                        combined = await stream_text_buffered(stream, flush_cb, interval=0.4, max_buffer=1500)
                        if (sent_msg is None) and (not combined.strip()):
                            # Retry with a simplified prompt (short system, no history) and tools enabled
                            simple_msgs = build_messages("You are a helpful assistant.", [], content)
                            stream_simple = self.client.respond_chat(simple_msgs, options)
                            combined_simple = await stream_text_buffered(stream_simple, flush_cb, interval=0.4, max_buffer=1500)
                            if (sent_msg is None) and (not combined_simple.strip()):
                                # Retry once without tools as fallback
                                tooly = dict(options.tools)
                                options.tools = {k: False for k in tooly}
                                stream2 = self.client.respond_chat(simple_msgs, options)
                                combined2 = await stream_text_buffered(stream2, flush_cb, interval=0.4, max_buffer=1500)
                                options.tools = tooly
                                if (sent_msg is None) and (not combined2.strip()):
                                    await message.channel.send("Sorry, I couldn’t produce a reply. Try again or use [p]gpt5 ask …")
            except Exception as e:
                await message.channel.send(f"Sorry, I hit an error: {e}")

    async def _image_path(self, message: discord.Message, content: str, gconf: Dict[str, Any]) -> None:
        lock = self._get_lock(message.channel.id)
        if lock.locked():
            return
        async with lock:
            # Find candidate base image from current message or the referenced message
            base_image_bytes: Optional[bytes] = None
            # Current message attachment
            attachment = next((a for a in message.attachments if a.content_type and a.content_type.startswith("image/")), None)
            if attachment is None and message.reference:
                try:
                    ref = message.reference
                    replied = ref.cached_message or await self.bot.get_channel(ref.channel_id).fetch_message(ref.message_id)
                    attachment = next(
                        (a for a in getattr(replied, "attachments", []) if a.content_type and a.content_type.startswith("image/")),
                        None,
                    )
                except Exception:
                    attachment = None

            # Decide whether this is an edit request or a fresh generation
            is_edit = bool(attachment and looks_like_image_edit_request(content))
            try:
                async with message.channel.typing():
                    if is_edit and attachment is not None:
                        base_image_bytes = await attachment.read()
                        img = await self.client.edit_image(base_image_bytes, content)
                        caption = "Sure — I updated the image as requested."
                    else:
                        img = await self.client.generate_image(content)
                        # Construct a friendly caption
                        preview = content.strip()
                        if len(preview) > 80:
                            preview = preview[:77] + "…"
                        caption = f"Sure — here’s an image for: {preview}"
            except Exception as e:
                await message.channel.send(f"Image generation failed: {e}")
                return
            file = discord.File(BytesIO(img), filename="image.png")
            await message.channel.send(file=file, content=caption)
