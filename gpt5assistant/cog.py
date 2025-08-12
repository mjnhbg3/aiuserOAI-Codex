from __future__ import annotations

from typing import Any, Dict, List, Optional

import discord
from redbot.core import checks, commands, Config

from .config_schemas import DEFAULT_GUILD_CONFIG
from .dispatcher import Dispatcher
from .openai_client import OpenAIClient


class GPT5Assistant(commands.Cog):
    """GPT-5 Assistant using OpenAI Responses API and native tools."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xA15E1157, force_registration=True)
        self.config.register_guild(**DEFAULT_GUILD_CONFIG)
        self._client: Optional[OpenAIClient] = None
        self._dispatcher: Optional[Dispatcher] = None

    async def _ensure_client(self) -> OpenAIClient:
        if self._client is None:
            tokens = await self.bot.get_shared_api_tokens("openai")
            api_key = tokens.get("api_key") if tokens else None
            self._client = OpenAIClient(api_key=api_key)
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
            f"History: turns={g.get('history_turns', 8)} chars={g.get('history_chars', 6000)} include_others={g.get('include_others', True)}"
        )

    @gpt5.group(name="config")
    @checks.admin_or_permissions(manage_guild=True)
    async def gpt5_config(self, ctx: commands.Context) -> None:
        """Configure GPT-5 Assistant."""
        pass

    @gpt5_config.command(name="model")
    async def gpt5_config_model(self, ctx: commands.Context, *, name: str) -> None:
        await self.config.guild(ctx.guild).model.set(name)
        await ctx.send(f"Model set to {name}.")

    # Verbosity setting removed; Responses API no longer uses this parameter.

    @gpt5_config.command(name="reasoning")
    async def gpt5_config_reasoning(self, ctx: commands.Context, effort: str) -> None:
        effort = effort.lower()
        if effort not in {"minimal", "medium", "high"}:
            await ctx.send("Reasoning must be minimal|medium|high.")
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

    @gpt5_config.command(name="system")
    async def gpt5_config_system(self, ctx: commands.Context, *, prompt: str) -> None:
        await self.config.guild(ctx.guild).system_prompt.set(prompt)
        await ctx.send("System prompt updated.")

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
        if not (0.0 <= p <= 1.0):
            await ctx.send("Percent must be between 0.0 and 1.0")
            return
        await self.config.guild(ctx.guild).reply_percent.set(p)
        await ctx.send(f"Reply percent set to {p}.")

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

    @gpt5_config_backread.command(name="seconds")
    async def gpt5_config_backread_seconds(self, ctx: commands.Context, n: int) -> None:
        await self.config.guild(ctx.guild).messages_backread_seconds.set(max(0, n))
        await ctx.send(f"Backread seconds gap set to {n}.")

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

    @gpt5_config.command(name="roleprompt")
    async def gpt5_config_roleprompt(self, ctx: commands.Context, role: discord.Role, *, prompt: str) -> None:
        data = await self.config.guild(ctx.guild).role_prompts()
        data[str(role.id)] = prompt
        await self.config.guild(ctx.guild).role_prompts.set(data)
        await ctx.send(f"Role prompt set for {role.name}.")

    @gpt5_config.command(name="memberprompt")
    async def gpt5_config_memberprompt(self, ctx: commands.Context, member: discord.Member, *, prompt: str) -> None:
        data = await self.config.guild(ctx.guild).member_prompts()
        data[str(member.id)] = prompt
        await self.config.guild(ctx.guild).member_prompts.set(data)
        await ctx.send(f"Member prompt set for {member.display_name}.")

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
        embed = discord.Embed(title="GPT-5 Assistant Settings", color=await ctx.embed_color())
        embed.add_field(name="Model", value=f"`{g['model']}`", inline=True)
        embed.add_field(name="Reply Percent", value=f"`{g.get('reply_percent', 0.0):.2f}`", inline=True)
        embed.add_field(name="Opt-in by default", value=f"`{g.get('optin_by_default', True)}`", inline=True)
        embed.add_field(name="Backread", value=f"`{g.get('messages_backread', 0)} msgs / {g.get('messages_backread_seconds', 0)} sec`", inline=False)
        tools = ", ".join([k for k, v in g["tools"].items() if v]) or "none"
        embed.add_field(name="Tools", value=tools, inline=False)
        wl = g.get('allowed_channels', [])
        chans = " ".join(f"<#{cid}>" for cid in wl) if wl else "`None`"
        embed.add_field(name="Whitelisted Channels", value=chans, inline=False)
        await ctx.send(embed=embed)

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
        """Configure chat history window."""
        pass

    @gpt5_config_history.command(name="turns")
    async def gpt5_config_history_turns(self, ctx: commands.Context, n: int) -> None:
        if n < 0:
            await ctx.send("Turns must be >= 0")
            return
        await self.config.guild(ctx.guild).history_turns.set(n)
        await ctx.send(f"History turns set to {n}.")

    @gpt5_config_history.command(name="chars")
    async def gpt5_config_history_chars(self, ctx: commands.Context, n: int) -> None:
        if n < 0:
            await ctx.send("Chars must be >= 0")
            return
        await self.config.guild(ctx.guild).history_chars.set(n)
        await ctx.send(f"History char budget set to {n}.")

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
        await ctx.send(f"Uploaded {len(ids)} files and attached to knowledge base.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if not message.guild or message.author.bot:
            return
        dispatcher = await self._ensure_dispatcher()
        await dispatcher.handle_message(message)
