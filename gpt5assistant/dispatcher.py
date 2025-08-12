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
from .utils.classifiers import looks_like_image_request
from .utils.variables import format_variables
from .utils.filters import apply_removelist


class Dispatcher:
    def __init__(self, bot: commands.Bot, config: Config, client: OpenAIClient) -> None:
        self.bot = bot
        self.config = config
        self.client = client
        self._locks: dict[int, asyncio.Lock] = {}

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

        content = message.content or ""
        content = content.strip()
        if not content:
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

        # Strip leading bot mention from content for cleaner prompts
        if mentioned and self.bot.user:
            mention_strs = {f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"}
            for mstr in mention_strs:
                if content.startswith(mstr):
                    content = content[len(mstr):].strip()
                    break

        if looks_like_image_request(content) and (gconf["tools"].get("image", True)):
            await self._image_path(message, content, gconf)
        else:
            await self._chat_path(message, content, gconf)

    async def _chat_path(self, message: discord.Message, content: str, gconf: Dict[str, Any]) -> None:
        lock = self._get_lock(message.channel.id)
        if lock.locked():
            return
        async with lock:
            # Build system/context prompt including optional channel context
            channel_contexts = gconf.get("channel_contexts", {}) or {}
            ch_ctx = channel_contexts.get(message.channel.id, "") or channel_contexts.get(str(message.channel.id), "")
            # Prompt override resolution: member > role > channel context > guild/system
            system_prompt = gconf["system_prompt"]
            role_prompts = gconf.get("role_prompts", {}) or {}
            member_prompts = gconf.get("member_prompts", {}) or {}
            # member
            mp = member_prompts.get(message.author.id) or member_prompts.get(str(message.author.id))
            if mp:
                system_prompt = mp
            else:
                for r in getattr(message.author, "roles", []) or []:
                    rp = role_prompts.get(r.id) or role_prompts.get(str(r.id))
                    if rp:
                        system_prompt = rp
                        break
            if ch_ctx:
                system_prompt = f"{system_prompt}\n\nChannel context: {ch_ctx}"
            # Dynamic variables
            system_prompt = await format_variables(message, system_prompt)

            # Gather recent channel history
            history: List[Dict[str, str]] = await gather_history(
                channel=message.channel,
                bot_user_id=getattr(self.bot.user, "id", None) if self.bot else None,
                before_message=message,
                max_turns=int(gconf.get("history_turns", 8)),
                max_chars=int(gconf.get("history_chars", 6000)),
                include_others=bool(gconf.get("include_others", True)),
                model=gconf.get("model", "gpt-5"),
                backread_limit=int(gconf.get("messages_backread", 25)),
                max_seconds_gap=int(gconf.get("messages_backread_seconds", 1800)),
                optin_set=set(gconf.get("optin", []) or []),
                optout_set=set(gconf.get("optout", []) or []),
                optin_by_default=bool(gconf.get("optin_by_default", True)),
            )
            msgs = build_messages(system_prompt, history, content)
            options = ChatOptions(
                model=gconf["model"],
                tools=gconf["tools"],
                reasoning=gconf["reasoning"],
                verbosity=gconf["verbosity"],
                max_tokens=gconf["max_tokens"],
                temperature=gconf["temperature"],
                system_prompt=gconf["system_prompt"],
                file_ids=gconf.get("file_ids") or None,
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
                        full_text = []
                        async for t in stream:
                            full_text.append(t)
                        text = "".join(full_text)
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
                        for ch in chunk_message(cleaned):
                            await message.channel.send(ch)
                else:
                    # Show typing while we stream
                    async with message.channel.typing():
                        stream = self.client.respond_chat(msgs, options)
                        await stream_text_buffered(stream, flush_cb, interval=0.4, max_buffer=1500)
            except Exception as e:
                await message.channel.send(f"Sorry, I hit an error: {e}")

    async def _image_path(self, message: discord.Message, content: str, gconf: Dict[str, Any]) -> None:
        lock = self._get_lock(message.channel.id)
        if lock.locked():
            return
        async with lock:
            # If there is an attachment, try edit
            attachment = next((a for a in message.attachments if a.content_type and a.content_type.startswith("image/")), None)
            try:
                if attachment is not None:
                    data = await attachment.read()
                    img = await self.client.edit_image(data, content)
                else:
                    img = await self.client.generate_image(content)
            except Exception as e:
                await message.channel.send(f"Image generation failed: {e}")
                return
            file = discord.File(BytesIO(img), filename="image.png")
            await message.channel.send(file=file, content=f"Prompt: {content}")
