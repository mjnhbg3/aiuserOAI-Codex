from __future__ import annotations

from typing import Any, Dict, List, Optional

import discord
from redbot.core import checks, commands, Config

from .config_schemas import DEFAULT_GUILD_CONFIG
from .dispatcher import Dispatcher
OpenAIClient = None  # lazy import to avoid load-time failures if deps missing


class GPT5Assistant(commands.Cog):
    """GPT-5 Assistant using OpenAI Responses API and native tools."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xA15E1157, force_registration=True)
        self.config.register_guild(**DEFAULT_GUILD_CONFIG)
        # Global system-wide prompt
        self.config.register_global(
            system_prompt="You are GPT-5, a helpful assistant. Keep replies concise.",
            diag_plain="Diagnostic ping: reply with the single word PONG.",
            diag_tools="What is one major headline today? Provide a short sentence.",
            code_container_type="auto",
        )
        # Per-channel cutoff for forget
        self.config.register_channel(forget_after_ts=0)
        self._client: Optional[OpenAIClient] = None
        self._dispatcher: Optional[Dispatcher] = None

    async def _ensure_client(self) -> OpenAIClient:
        if self._client is None:
            tokens = await self.bot.get_shared_api_tokens("openai")
            api_key = tokens.get("api_key") if tokens else None
            try:
                from .openai_client import OpenAIClient as _Client
            except Exception as e:
                raise RuntimeError(
                    "OpenAI client unavailable. Ensure 'openai>=1.99.0' is installed via [p]pipinstall and reload."
                ) from e
            self._client = _Client(api_key=api_key)
        return self._client

    async def _ensure_dispatcher(self) -> Dispatcher:
        if self._dispatcher is None:
            client = await self._ensure_client()
            self._dispatcher = Dispatcher(self.bot, self.config, client)
        return self._dispatcher

    @commands.group(name="gpt5")
    async def gpt5(self, ctx: commands.Context) -> None:
        """GPT-5 Assistant commands."""
        pass

    @gpt5.command(name="status")
    async def gpt5_status(self, ctx: commands.Context) -> None:
        g = await self.config.guild(ctx.guild).all()
        tools = ", ".join([k for k, v in g["tools"].items() if v]) or "none"
        await ctx.send(
            f"Model: {g['model']}\nReasoning: {g['reasoning']}\n"
            f"Tools: {tools}\nMax tokens: {g['max_tokens']} Temp: {g['temperature']}\n"
            f"Ephemeral: {g['ephemeral']} Allowed channels: {len(g['allowed_channels'])}\n"
            f"Respond on mention: {g.get('respond_on_mention', True)}\n"
            f"Random autoreply: {g.get('random_autoreply', False)} rate={g.get('random_rate', 0.0)}\n"
            f"History: backread_msgs={g.get('messages_backread', 25)} backread_images={g.get('images_backread', 0)} backread_seconds={g.get('messages_backread_seconds', 1800)} backread_images_seconds={g.get('images_backread_seconds', 1800)} include_others={g.get('include_others', True)}"
        )

    @gpt5.group(name="config")
    @checks.admin_or_permissions(manage_guild=True)
    async def gpt5_config(self, ctx: commands.Context) -> None:
        """Configure GPT-5 Assistant."""
        pass

    @gpt5_config.group(name="diag")
    async def gpt5_config_diag(self, ctx: commands.Context) -> None:
        """Configure gpt5 diag test prompts."""
        pass

    @gpt5_config.group(name="debug")
    @checks.admin_or_permissions(manage_guild=True)
    async def gpt5_config_debug(self, ctx: commands.Context) -> None:
        """Debug toggles for troubleshooting (admin only)."""
        pass

    @gpt5_config_debug.command(name="attachments")
    async def gpt5_config_debug_attachments(self, ctx: commands.Context, value: str) -> None:
        """Enable or disable debug notes when sending attachments in normal replies.

        Usage: [p]gpt5 config debug attachments on|off
        """
        flag = value.lower() in {"on", "true", "1", "enable", "enabled"}
        await self.config.guild(ctx.guild).debug_attachments.set(flag)
        await ctx.send(f"Debug attachments {'enabled' if flag else 'disabled'}.")

    @gpt5_config_diag.command(name="plain")
    async def gpt5_config_diag_plain(self, ctx: commands.Context, *, text: str) -> None:
        await self.config.diag_plain.set(text)
        await ctx.send("Set diag plain prompt.")

    @gpt5_config_diag.command(name="tools")
    async def gpt5_config_diag_tools(self, ctx: commands.Context, *, text: str) -> None:
        await self.config.diag_tools.set(text)
        await ctx.send("Set diag tools prompt.")

    @gpt5_config.group(name="code")
    async def gpt5_config_code(self, ctx: commands.Context) -> None:
        """Configure code interpreter tool settings."""
        pass

    @gpt5_config_code.command(name="container")
    async def gpt5_config_code_container(self, ctx: commands.Context, container_type: str) -> None:
        """Set code interpreter container type (provider-specific). Use 'off' to disable.

        Example: [p]gpt5 config code container default
        """
        if container_type.lower() in {"off", "none", ""}:
            await self.config.code_container_type.set("")
            await ctx.send("Code interpreter container cleared; tool will be omitted.")
        else:
            await self.config.code_container_type.set(container_type)
            await ctx.send(f"Code interpreter container type set to '{container_type}'.")

    @gpt5_config.command(name="model")
    async def gpt5_config_model(self, ctx: commands.Context, *, name: str) -> None:
        await self.config.guild(ctx.guild).model.set(name)
        await ctx.send(f"Model set to {name}.")

    # Verbosity setting removed; Responses API no longer uses this parameter.

    @gpt5_config.command(name="reasoning")
    async def gpt5_config_reasoning(self, ctx: commands.Context, effort: str) -> None:
        effort = effort.lower()
        if effort not in {"minimal", "low", "medium", "high"}:
            await ctx.send("Reasoning must be minimal|low|medium|high.")
            return
        await self.config.guild(ctx.guild).reasoning.set(effort)
        await ctx.send(f"Reasoning effort set to {effort}.")

    @gpt5_config.command(name="tools")
    async def gpt5_config_tools(self, ctx: commands.Context, action: str, tool: str) -> None:
        action = action.lower()
        tool = tool.lower()
        if tool not in {"web_search", "file_search", "code_interpreter", "image"}:
            await ctx.send("Unknown tool. Choose web_search|file_search|code_interpreter|image.")
            return
        current = await self.config.guild(ctx.guild).tools()
        val = action == "enable"
        current[tool] = val
        await self.config.guild(ctx.guild).tools.set(current)
        await ctx.send(f"Tool {tool} {'enabled' if val else 'disabled'}.")

    @gpt5_config.command(name="channels")
    async def gpt5_config_channels(self, ctx: commands.Context, action: str, channel: Optional[discord.TextChannel] = None) -> None:
        if channel is None:
            channel = ctx.channel
        allowed = await self.config.guild(ctx.guild).allowed_channels()
        if action == "allow":
            if channel.id not in allowed:
                allowed.append(channel.id)
        elif action == "deny":
            if channel.id in allowed:
                allowed.remove(channel.id)
        else:
            await ctx.send("Use allow|deny.")
            return
        await self.config.guild(ctx.guild).allowed_channels.set(allowed)
        await ctx.send(f"Channel {channel.mention} {action}ed.")

    @gpt5_config.group(name="system", invoke_without_command=True)
    async def gpt5_config_system(self, ctx: commands.Context, *, prompt: Optional[str] = None) -> None:
        """Set or show the system prompt.

        Usage:
          [p]gpt5 config system <prompt>  -> set prompt
          [p]gpt5 config system show      -> show current prompt
        """
        if prompt is None:
            # Show global system prompt
            current = await self.config.system_prompt()
            shown = current or "(empty)"
            if len(shown) <= 1900:
                await ctx.send(f"Current system prompt (global):\n```\n{shown}\n```")
            else:
                await ctx.send("Current system prompt is long; showing first 1900 chars:")
                await ctx.send(f"```\n{shown[:1900]}\n```")
            return
        await self.config.system_prompt.set(prompt)
        await ctx.send("System prompt (global) updated.")

    @gpt5_config_system.command(name="show")
    async def gpt5_config_system_show(self, ctx: commands.Context) -> None:
        """Show the current system prompt."""
        current = await self.config.system_prompt()
        shown = current or "(empty)"
        if len(shown) <= 1900:
            await ctx.send(f"Current system prompt (global):\n```\n{shown}\n```")
        else:
            await ctx.send("Current system prompt is long; showing first 1900 chars:")
            await ctx.send(f"```\n{shown[:1900]}\n```")

    @gpt5_config.command(name="max_tokens")
    async def gpt5_config_max_tokens(self, ctx: commands.Context, n: int) -> None:
        if n <= 0:
            await ctx.send("Must be > 0.")
            return
        await self.config.guild(ctx.guild).max_tokens.set(n)
        await ctx.send(f"max_tokens set to {n}.")

    @gpt5_config.command(name="temperature")
    async def gpt5_config_temperature(self, ctx: commands.Context, t: float) -> None:
        if not (0.0 <= t <= 2.0):
            await ctx.send("Temperature must be between 0 and 2.")
            return
        await self.config.guild(ctx.guild).temperature.set(t)
        await ctx.send(f"temperature set to {t}.")

    @gpt5_config.command(name="privacy")
    async def gpt5_config_privacy(self, ctx: commands.Context, mode: str) -> None:
        val = mode.lower() == "ephemeral" or mode.lower() == "on"
        await self.config.guild(ctx.guild).ephemeral.set(val)
        await ctx.send(f"Ephemeral responses {'enabled' if val else 'disabled'}.")

    @gpt5.command(name="optin")
    async def gpt5_optin(self, ctx: commands.Context) -> None:
        """Opt-in to allow the bot to use your messages for context."""
        optin = await self.config.guild(ctx.guild).optin()
        optout = await self.config.guild(ctx.guild).optout()
        uid = ctx.author.id
        if uid not in optin:
            optin.append(uid)
            await self.config.guild(ctx.guild).optin.set(optin)
        if uid in optout:
            optout.remove(uid)
            await self.config.guild(ctx.guild).optout.set(optout)
        try:
            await ctx.message.add_reaction("✅")
        except Exception:
            pass

    @gpt5.command(name="optout")
    async def gpt5_optout(self, ctx: commands.Context) -> None:
        """Opt-out from having your messages used for context."""
        optout = await self.config.guild(ctx.guild).optout()
        optin = await self.config.guild(ctx.guild).optin()
        uid = ctx.author.id
        if uid not in optout:
            optout.append(uid)
            await self.config.guild(ctx.guild).optout.set(optout)
        if uid in optin:
            optin.remove(uid)
            await self.config.guild(ctx.guild).optin.set(optin)
        try:
            await ctx.message.add_reaction("✅")
        except Exception:
            pass

    @gpt5_config.command(name="optindefault")
    async def gpt5_config_optindefault(self, ctx: commands.Context, value: str) -> None:
        flag = value.lower() in {"on", "true", "1", "enable", "enabled"}
        await self.config.guild(ctx.guild).optin_by_default.set(flag)
        await ctx.send(f"Opt-in by default set to {'on' if flag else 'off'}.")

    @gpt5_config.command(name="replypercent")
    async def gpt5_config_replypercent(self, ctx: commands.Context, p: float) -> None:
        # Accept 0-100 and store as 0.0-1.0
        if not (0.0 <= p <= 100.0):
            await ctx.send("Percent must be between 0 and 100")
            return
        frac = p / 100.0
        await self.config.guild(ctx.guild).reply_percent.set(frac)
        await ctx.send(f"Reply percent set to {p:.0f}% ({frac:.2f}).")

    @gpt5_config.command(name="replymentions")
    async def gpt5_config_replymentions(self, ctx: commands.Context, value: str) -> None:
        flag = value.lower() in {"on", "true", "1", "enable", "enabled"}
        await self.config.guild(ctx.guild).reply_to_mentions_replies.set(flag)
        await ctx.send(f"Reply to mentions/replies set to {'on' if flag else 'off'}.")

    @gpt5_config.group(name="backread")
    async def gpt5_config_backread(self, ctx: commands.Context) -> None:
        """Configure history backread limits."""
        pass

    @gpt5_config_backread.command(name="messages")
    async def gpt5_config_backread_messages(self, ctx: commands.Context, n: int) -> None:
        await self.config.guild(ctx.guild).messages_backread.set(max(0, n))
        await ctx.send(f"Messages backread set to {n}.")

    @gpt5_config_backread.command(name="images")
    async def gpt5_config_backread_images(self, ctx: commands.Context, n: int) -> None:
        """Set how many prior images to include from history.

        Example: [p]gpt5 config backread images 3
        """
        await self.config.guild(ctx.guild).images_backread.set(max(0, n))
        await ctx.send(f"Images backread set to {n}.")

    @gpt5_config_backread.command(name="seconds")
    async def gpt5_config_backread_seconds(self, ctx: commands.Context, n: int) -> None:
        await self.config.guild(ctx.guild).messages_backread_seconds.set(max(0, n))
        await ctx.send(f"Backread seconds gap set to {n}.")

    @gpt5_config_backread.command(name="imageseconds")
    async def gpt5_config_backread_imageseconds(self, ctx: commands.Context, n: int) -> None:
        """Set how many seconds old images can be before being excluded from history.

        Example: [p]gpt5 config backread imageseconds 600
        """
        await self.config.guild(ctx.guild).images_backread_seconds.set(max(0, n))
        await ctx.send(f"Images backread seconds set to {n}.")

    @gpt5_config.command(name="filters")
    async def gpt5_config_filters(self, ctx: commands.Context, action: str, *, pattern: str = "") -> None:
        """Manage remove-list regex filters: add|remove <pattern>, list."""
        action = action.lower()
        gl = await self.config.guild(ctx.guild).removelist_regexes()
        if action == "add" and pattern:
            gl.append(pattern)
            await self.config.guild(ctx.guild).removelist_regexes.set(gl)
            await ctx.send("Pattern added.")
        elif action == "remove" and pattern:
            try:
                gl.remove(pattern)
                await self.config.guild(ctx.guild).removelist_regexes.set(gl)
            except ValueError:
                pass
            await ctx.send("Pattern removed.")
        elif action == "list":
            await ctx.send("\n".join(gl) or "(none)")
        else:
            await ctx.send("Use add <pattern> | remove <pattern> | list")

    # Role/member/channel prompts removed in favor of a single global prompt.

    @gpt5.command(name="add")
    @checks.admin_or_permissions(manage_guild=True)
    async def gpt5_add_channel(self, ctx: commands.Context, channel: discord.TextChannel) -> None:
        allowed = await self.config.guild(ctx.guild).allowed_channels()
        if channel.id not in allowed:
            allowed.append(channel.id)
        await self.config.guild(ctx.guild).allowed_channels.set(allowed)
        await ctx.send(f"Whitelisted {channel.mention}.")

    @gpt5.command(name="remove")
    @checks.admin_or_permissions(manage_guild=True)
    async def gpt5_remove_channel(self, ctx: commands.Context, channel: discord.TextChannel) -> None:
        allowed = await self.config.guild(ctx.guild).allowed_channels()
        if channel.id in allowed:
            allowed.remove(channel.id)
        await self.config.guild(ctx.guild).allowed_channels.set(allowed)
        await ctx.send(f"Removed {channel.mention} from whitelist.")

    @gpt5_config.command(name="show")
    async def gpt5_config_show(self, ctx: commands.Context) -> None:
        g = await self.config.guild(ctx.guild).all()
        color = await ctx.embed_color()
        embed = discord.Embed(title="GPT-5 Assistant Settings", color=color)

        # Core
        embed.add_field(name="Model", value=f"`{g.get('model')}`", inline=True)
        embed.add_field(name="Reasoning", value=f"`{g.get('reasoning')}`", inline=True)
        embed.add_field(name="Max Tokens", value=f"`{g.get('max_tokens')}`", inline=True)

        # Reply behavior
        embed.add_field(
            name="Replying",
            value=(
                f"reply_percent=`{g.get('reply_percent', 0.0):.2f}`\n"
                f"reply_mentions=`{g.get('reply_to_mentions_replies', True)}`\n"
                f"respond_on_mention=`{g.get('respond_on_mention', True)}`\n"
                f"random_autoreply=`{g.get('random_autoreply', False)}` rate=`{g.get('random_rate', 0.0):.2f}`"
            ),
            inline=False,
        )

        # History and backread
        embed.add_field(
            name="History Limits",
            value=(
                f"backread_msgs=`{g.get('messages_backread', 0)}`\n"
                f"backread_images=`{g.get('images_backread', 0)}`\n"
                f"backread_seconds=`{g.get('messages_backread_seconds', 0)}`\n"
                f"backread_images_seconds=`{g.get('images_backread_seconds', 1800)}`\n"
                f"include_others=`{g.get('include_others', True)}`"
            ),
            inline=False,
        )

        # Tools
        tools_enabled = ", ".join([k for k, v in g.get("tools", {}).items() if v]) or "none"
        embed.add_field(name="Tools Enabled", value=tools_enabled, inline=False)

        # File Search / KB summary
        kb_id = g.get("file_kb_id")
        file_ids = g.get("file_ids", [])
        vs_status = "set" if kb_id else "none"
        embed.add_field(
            name="Knowledge Base",
            value=(
                f"vector_store_id={vs_status}\n"
                f"file_ids_count={len(file_ids)}"
            ),
            inline=True,
        )

        # Privacy / filters
        patterns = g.get("removelist_regexes", []) or []
        embed.add_field(
            name="Privacy/Safety",
            value=(
                f"ephemeral=`{g.get('ephemeral', False)}`\n"
                f"optin_by_default=`{g.get('optin_by_default', True)}`\n"
                f"remove_regex_count=`{len(patterns)}`"
            ),
            inline=True,
        )

        # Whitelisted channels
        wl = g.get("allowed_channels", []) or []
        chans = " ".join(f"<#{cid}>" for cid in wl) if wl else "`None`"
        embed.add_field(name="Whitelisted Channels", value=chans, inline=False)

        # System prompt preview
        sys_prompt = (await self.config.system_prompt()) or "(empty)"
        preview = sys_prompt if len(sys_prompt) <= 200 else sys_prompt[:200] + "…"
        embed.add_field(name="System Prompt (global)", value=f"```\n{preview}\n```", inline=False)

        await ctx.send(embed=embed)

    @commands.command(name="forget")
    async def gpt5_forget(self, ctx: commands.Context) -> None:
        """Forget all prior context in this channel from now on.

        Sets a cutoff so message history before this point is not read.
        """
        try:
            import time
            now = int(time.time())
            await self.config.channel(ctx.channel).forget_after_ts.set(now)
            await ctx.send("Okay, I’ll forget previous context in this channel starting now.")
        except Exception as e:
            await ctx.send(f"Could not set forget point: {e}")

    @gpt5_config.command(name="triggers")
    async def gpt5_config_triggers(self, ctx: commands.Context, which: str, value: str) -> None:
        """Set triggers: mention on/off, random on/off.

        Example: [p]gpt5 config triggers mention on
                 [p]gpt5 config triggers random off
        """
        which = which.lower()
        value = value.lower()
        if which not in {"mention", "random"}:
            await ctx.send("Use which=mention|random and value=on|off.")
            return
        flag = value in {"on", "true", "1", "enable", "enabled"}
        if which == "mention":
            await self.config.guild(ctx.guild).respond_on_mention.set(flag)
        else:
            await self.config.guild(ctx.guild).random_autoreply.set(flag)
        await ctx.send(f"Trigger {which} set to {'on' if flag else 'off'}.")

    @gpt5_config.command(name="rngrate")
    async def gpt5_config_rngrate(self, ctx: commands.Context, rate: float) -> None:
        """Set random autoreply rate between 0.0 and 1.0."""
        if not (0.0 <= rate <= 1.0):
            await ctx.send("Rate must be between 0.0 and 1.0")
            return
        await self.config.guild(ctx.guild).random_rate.set(rate)
        await ctx.send(f"Random autoreply rate set to {rate}.")

    @gpt5_config.group(name="history")
    async def gpt5_config_history(self, ctx: commands.Context) -> None:
        """Configure chat history window (backread only)."""
        pass

    @gpt5_config_history.command(name="includeothers")
    async def gpt5_config_history_includeothers(self, ctx: commands.Context, value: str) -> None:
        flag = value.lower() in {"on", "true", "1", "enable", "enabled"}
        await self.config.guild(ctx.guild).include_others.set(flag)
        await ctx.send(f"Include others in history set to {'on' if flag else 'off'}.")

    @gpt5_config.command(name="channelcontext")
    async def gpt5_config_channelcontext(self, ctx: commands.Context, action: str, *, text: str = "") -> None:
        """Set or clear context for this channel.

        Examples:
          [p]gpt5 config channelcontext set You are the #help-desk assistant.
          [p]gpt5 config channelcontext clear
        """
        channel = ctx.channel
        data = await self.config.guild(ctx.guild).channel_contexts()
        if action.lower() == "set":
            if not text:
                await ctx.send("Provide the context text.")
                return
            data[str(channel.id)] = text
            await self.config.guild(ctx.guild).channel_contexts.set(data)
            await ctx.send("Channel context set.")
        elif action.lower() == "clear":
            data.pop(str(channel.id), None)
            await self.config.guild(ctx.guild).channel_contexts.set(data)
            await ctx.send("Channel context cleared.")
        else:
            await ctx.send("Use set <text> or clear.")

    @gpt5.command(name="ask")
    async def gpt5_ask(self, ctx: commands.Context, *, text: str) -> None:
        dispatcher = await self._ensure_dispatcher()
        # fabricate a message-like wrapper so we reuse dispatcher path
        fake = ctx.message
        fake.content = text
        await dispatcher._chat_path(fake, text, await self.config.guild(ctx.guild).all())

    @gpt5.command(name="image")
    async def gpt5_image(self, ctx: commands.Context, *, prompt: str) -> None:
        dispatcher = await self._ensure_dispatcher()
        fake = ctx.message
        fake.content = prompt
        g = await self.config.guild(ctx.guild).all()
        await dispatcher._image_path(fake, prompt, g)

    @gpt5.command(name="upload")
    async def gpt5_upload(self, ctx: commands.Context) -> None:
        if not ctx.message.attachments:
            await ctx.send("Attach files to upload.")
            return
        client = await self._ensure_client()
        contents = [await a.read() for a in ctx.message.attachments]
        names = [a.filename for a in ctx.message.attachments]
        ids = await client.index_files(contents, names)
        g = await self.config.guild(ctx.guild).all()
        file_ids = g.get("file_ids") or []
        file_ids.extend(ids)
        await self.config.guild(ctx.guild).file_ids.set(file_ids)
        # Ensure vector store exists and attach files
        vs_id = g.get("file_kb_id") or None
        vs_id = await client.ensure_vector_store(name=f"guild-{ctx.guild.id}-kb", current_id=vs_id)
        await client.add_files_to_vector_store(vs_id, ids)
        await self.config.guild(ctx.guild).file_kb_id.set(vs_id)
        await ctx.send(f"Uploaded {len(ids)} files and added to knowledge base.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return
        dispatcher = await self._ensure_dispatcher()
        await dispatcher.handle_message(message)

    @gpt5.command(name="diag")
    @checks.admin_or_permissions(manage_guild=True)
    async def gpt5_diag(self, ctx: commands.Context, *, prompt: Optional[str] = None) -> None:
        """Run a verbose Responses API diagnostic and show tool settings/payloads.

        Optionally provide a one-off tools test prompt: [p]gpt5 diag <prompt>
        """
        client = await self._ensure_client()
        g = await self.config.guild(ctx.guild).all()
        tools = g.get("tools", {}) or {}
        model = g.get("model", "gpt-5")
        kb = g.get("file_kb_id") or None
        # Load configurable diag prompts (global)
        d_plain = await self.config.diag_plain()
        d_tools = prompt or (await self.config.diag_tools())
        try:
            import openai as _oai
            sdk_ver = getattr(_oai, "__version__", "unknown")
        except Exception:
            sdk_ver = "unknown"

        # Prepare a minimal input
        from .openai_client import ChatOptions
        messages = [
            {"role": "user", "content": d_plain or "Diagnostic ping: reply with the single word PONG."},
        ]

        code_container_type = await self.config.code_container_type()
        opts = ChatOptions(
            model=model,
            tools=tools,
            reasoning=g.get("reasoning", "medium"),
            max_tokens=g.get("max_tokens", 500),
            temperature=g.get("temperature", 0.7),
            system_prompt=g.get("system_prompt", ""),
            file_ids=g.get("file_ids") or None,
            vector_store_id=kb,
            code_container_type=code_container_type or None,
        )

        # Try a quick no-tool ping first
        opts_no_tools = ChatOptions(
            model=model,
            tools={"web_search": False, "file_search": False, "code_interpreter": False, "image": False},
            reasoning=opts.reasoning,
            max_tokens=opts.max_tokens,
            temperature=opts.temperature,
            system_prompt=opts.system_prompt,
            file_ids=None,
            vector_store_id=None,
        )

        async def _collect(gen):
            out = []
            async for ch in gen:
                out.append(ch)
            return "".join(out)

        ok_plain = ""
        ok_tools = ""
        err_plain = ""
        err_tools = ""
        tools_payload_str = ""
        try:
            ok_plain = await _collect(client.respond_chat(messages, opts_no_tools))
        except Exception as e:
            # Extract HTTP info if present
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            body = None
            resp = getattr(e, "response", None)
            try:
                if resp and hasattr(resp, "json"):
                    body = resp.json()
            except Exception:
                body = None
            err_plain = f"{type(e).__name__}: {e} (status={status})\n{str(body)[:500] if body else ''}"
        # If the diag command message has attachments, include them inline for the tools test
        inline_file_ids: list[str] = []
        inline_image_ids: list[str] = []
        inline_image_urls: list[str] = []
        if getattr(ctx.message, "attachments", None):
            contents: list[bytes] = []
            names: list[str] = []
            kinds: list[str] = []
            need_image_file_ids = bool(tools.get("code_interpreter"))
            for a in ctx.message.attachments:
                ctype = a.content_type or ""
                # Prefer URL for images
                if isinstance(ctype, str) and ctype.startswith("image/"):
                    try:
                        inline_image_urls.append(a.url)  # type: ignore[attr-defined]
                        if not need_image_file_ids:
                            names.append(a.filename or "attachment")
                            kinds.append(ctype)
                            continue
                    except Exception:
                        pass
                try:
                    data = await a.read()
                except Exception:
                    continue
                contents.append(data)
                fname = a.filename or "attachment"
                names.append(fname)
                if not ctype and "." in fname:
                    ext = fname.lower().rsplit(".", 1)[-1]
                    if ext in {"png","jpg","jpeg","gif","webp","bmp","tif","tiff","svg"}:
                        ctype = f"image/{'jpeg' if ext=='jpg' else ext}"
                kinds.append(ctype)
            if contents:
                try:
                    ids = await client.index_files(contents, names)
                    for fid, ctype in zip(ids, kinds):
                        if isinstance(ctype, str) and ctype.startswith("image/"):
                            inline_image_ids.append(fid)
                            if need_image_file_ids:
                                inline_file_ids.append(fid)
                        else:
                            inline_file_ids.append(fid)
                except Exception:
                    pass

        try:
            # Short tool test prompt
            messages[-1]["content"] = d_tools or "What is one major headline today? Provide a short sentence."
            # Compose the actual tools payload we will send
            eff_tools = {}
            eff_tools.update(tools)
            if eff_tools.get("file_search") and not kb:
                eff_tools["file_search"] = False
            # Recreate the payload array similar to client _tools_array
            payload = []
            if eff_tools.get("web_search"):
                payload.append({"type": "web_search"})
            if eff_tools.get("file_search") and kb:
                payload.append({"type": "file_search", "vector_store_ids": [kb]})
            if eff_tools.get("code_interpreter"):
                # Note: In actual chat, the two-call sentinel approach is used to avoid unnecessary charges
                ctype = await self.config.code_container_type()
                ctype = (ctype or "auto").strip()
                payload.append({"type": "code_interpreter", "container": {"type": ctype}, "note": "two-call optimization in chat"})
            if eff_tools.get("image"):
                payload.append({"type": "image_generation"})
            tools_payload_str = str(payload)

            # Use the client path and collect final output including images
            # Include inline attachments for this diag run
            opts.inline_file_ids = inline_file_ids or None
            opts.inline_image_ids = inline_image_ids or None
            opts.inline_image_urls = inline_image_urls or None
            result = await client.respond_collect(messages, opts, debug=True)
            ok_tools = result.get("text", "") or (f"[images: {len(result.get('images') or [])}]" if result.get("images") else "")
        except Exception as e:
            # Unwrap RetryError to original HTTP error where possible
            orig = e
            last_attempt = getattr(e, "last_attempt", None)
            if last_attempt and hasattr(last_attempt, "exception"):
                try:
                    orig = last_attempt.exception()
                except Exception:
                    orig = e
            status = getattr(orig, "status_code", None) or getattr(orig, "status", None)
            body = None
            resp = getattr(orig, "response", None)
            try:
                if resp and hasattr(resp, "json"):
                    body = resp.json()
            except Exception:
                body = None
            err_tools = f"{type(orig).__name__}: {orig} (status={status})\n{str(body)[:700] if body else ''}"

        embed = discord.Embed(title="gpt5 diag", color=await ctx.embed_color())
        embed.add_field(name="SDK", value=f"openai {sdk_ver}", inline=True)
        embed.add_field(name="Model", value=f"`{model}`", inline=True)
        enabled = ", ".join(k for k, v in tools.items() if v) or "none"
        embed.add_field(name="Tools Enabled", value=enabled, inline=False)
        if kb:
            embed.add_field(name="Vector Store", value=f"set ({kb[:10]}…)", inline=True)
        else:
            embed.add_field(name="Vector Store", value="not set", inline=True)
        if tools_payload_str:
            preview = tools_payload_str if len(tools_payload_str) <= 250 else tools_payload_str[:250] + "…"
            embed.add_field(name="Tools Payload", value=f"```json\n{preview}\n```", inline=False)
        # Show code container type if set
        ctype = await self.config.code_container_type()
        if ctype:
            embed.add_field(name="Code Container", value=f"`{ctype}`", inline=True)
        else:
            embed.add_field(name="Code Container", value="(not set)", inline=True)

        if ok_plain:
            embed.add_field(name="Plain Test", value=(ok_plain[:200] + ("…" if len(ok_plain) > 200 else "")), inline=False)
        if err_plain:
            embed.add_field(name="Plain Error", value=err_plain[:500], inline=False)
        if ok_tools:
            embed.add_field(name="Tools Test", value=(ok_tools[:400] + ("…" if len(ok_tools) > 400 else "")), inline=False)
        if err_tools:
            embed.add_field(name="Tools Error", value=err_tools[:500], inline=False)

        # Send embed first
        await ctx.send(embed=embed)

        # If tools run produced images/files, attach them so users can see them inline
        try:
            if 'result' in locals():
                imgs = result.get('images') or []
                for idx, img in enumerate(imgs):
                    try:
                        file = discord.File(BytesIO(img), filename=f"diag_image_{idx+1}.png")
                        await ctx.send(file=file)
                    except Exception:
                        continue
                fitems = result.get('files') or []
                for item in fitems:
                    try:
                        name = item.get('name') or 'attachment.bin'
                        data = item.get('bytes')
                        if not isinstance(name, str) or not isinstance(data, (bytes, bytearray)):
                            continue
                        if len(data) > 7_900_000:
                            await ctx.send(f"Generated a file '{name}' (~{len(data)//1024} KB), but it's too large to attach here.")
                            continue
                        from io import BytesIO
                        file = discord.File(BytesIO(data), filename=name)
                        await ctx.send(file=file)
                    except Exception:
                        continue
                # Attach debug trace as a text file if present
                dbg = result.get('debug')
                if isinstance(dbg, list) and dbg:
                    try:
                        from io import BytesIO
                        text = "\n".join(str(x) for x in dbg)
                        buf = BytesIO(text.encode('utf-8', errors='replace'))
                        file = discord.File(buf, filename='gpt5_diag_debug.txt')
                        await ctx.send(file=file)
                    except Exception:
                        pass
        except Exception:
            pass
