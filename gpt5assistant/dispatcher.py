from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from io import BytesIO
import random
import inspect
import hashlib

import discord
from redbot.core import commands, Config

from .config_schemas import DEFAULT_GUILD_CONFIG
from .messages import build_messages, gather_history, build_messages_with_separated_prompts
from .openai_client import ChatOptions, OpenAIClient
from .utils.chunking import chunk_message
from .utils.streaming import stream_text_buffered
from .utils.classifiers import looks_like_image_request, looks_like_image_edit_request
from .utils.variables import format_variables, separate_template_variables, build_dynamic_context_message
from .utils.filters import apply_removelist
from .memory_storage import Memory
from .vector_store_manager import MemoryManager
from .function_handlers import FunctionCallHandler


class Dispatcher:
    def __init__(self, bot: commands.Bot, config: Config, client: OpenAIClient, cog_instance=None) -> None:
        self.bot = bot
        self.config = config
        self.client = client
        self._locks: dict[int, asyncio.Lock] = {}
        self._last_intent_warn: dict[int, float] = {}
        self.memory_manager = MemoryManager(client, config, cog_instance)
        self.function_handler = FunctionCallHandler(self.memory_manager)
        self._memory_initialized = False
    
    def build_collision_resistant_cache_key(
        self, 
        static_template: str, 
        tools_config: Dict[str, bool],
        model: str,
        channel_context: str = "",
        attachment_flags: str = "",
        memory_flags: str = ""
    ) -> str:
        """Build collision-resistant cache key with proper granularity"""
        
        try:
            # 1. Template hash with collision resistance (16 chars = 2^64 possibilities)
            template_hash = hashlib.sha256(static_template.encode('utf-8')).hexdigest()[:16]
            
            # 2. Tool signature with collision resistance
            enabled_tools = sorted([k for k, v in tools_config.items() if v])
            tools_string = ",".join(enabled_tools)
            tool_hash = hashlib.sha256(tools_string.encode('utf-8')).hexdigest()[:12]
            
            # 3. Channel context hash (for channel-specific caching)
            if channel_context and len(channel_context.strip()) > 0:
                # Normalize channel context for better cache grouping
                normalized_context = self._normalize_channel_context(channel_context)
                ctx_hash = hashlib.sha256(normalized_context.encode('utf-8')).hexdigest()[:8]
            else:
                ctx_hash = "default"
            
            # 4. State flags (highly cacheable binary states)
            state_sig = f"{attachment_flags},{memory_flags}" if attachment_flags or memory_flags else "clean"
            
            # 5. Build compact cache key (must be ≤64 chars for OpenAI)
            cache_version = "v3"
            cache_key = f"gpt5:{cache_version}:{template_hash[:8]}:{tool_hash[:6]}:{ctx_hash[:6]}:{model[:6]}"
            
            # 6. Enforce OpenAI's 64-character limit
            if len(cache_key) > 64:
                # Hash entire key to fit within limit
                full_hash = hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:32]
                cache_key = f"gpt5:v3:{full_hash}"
            
            return cache_key
            
        except Exception as e:
            # Fallback to basic key on any error
            return f"gpt5assistant:fallback:{hashlib.sha256(str(e).encode()).hexdigest()[:16]}"

    def _normalize_channel_context(self, context: str) -> str:
        """Normalize channel context for better cache grouping"""
        context_lower = context.lower().strip()
        
        # Group similar contexts for better cache hit rates
        if any(keyword in context_lower for keyword in ["code", "programming", "dev", "technical"]):
            return "coding_context"
        elif any(keyword in context_lower for keyword in ["support", "help", "assistance"]):
            return "support_context"
        elif any(keyword in context_lower for keyword in ["game", "gaming", "play"]):
            return "gaming_context"
        elif any(keyword in context_lower for keyword in ["general", "chat", "casual"]):
            return "general_context"
        elif len(context_lower) > 100:
            # Hash very long contexts
            return f"custom_{hashlib.sha1(context.encode()).hexdigest()[:8]}"
        else:
            # Use context directly if short and doesn't match patterns
            return context_lower.replace(" ", "_")[:30]

    def _get_lock(self, channel_id: int) -> asyncio.Lock:
        if channel_id not in self._locks:
            self._locks[channel_id] = asyncio.Lock()
        return self._locks[channel_id]

    async def _is_command_message(self, msg: discord.Message) -> bool:
        try:
            ctx = await self.bot.get_context(msg)
            # Support both d.py styles
            if hasattr(ctx, "valid"):
                return bool(getattr(ctx, "valid"))
            return bool(getattr(ctx, "command", None))
        except Exception:
            return False

    async def handle_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return
        guild = message.guild
        gconf = await self.config.guild(guild).all()
        allowed = gconf.get("allowed_channels") or []
        if allowed and message.channel.id not in allowed:
            return

        # Skip only real commands (across all cogs) using Red's parser
        try:
            ctx = await self.bot.get_context(message)
            is_cmd = bool(getattr(ctx, "valid", False)) or bool(getattr(ctx, "command", None))
            if is_cmd:
                return
        except Exception:
            # On parser failure, fall back to old behavior: skip prefix-starting messages
            prefixes = []
            try:
                res = self.bot.get_valid_prefixes(guild)
                if inspect.isawaitable(res):
                    res = await res
                prefixes = list(res) if isinstance(res, (list, tuple)) else []
            except Exception:
                prefixes = []
            if any(isinstance(p, str) and p and (message.content or "").startswith(p) for p in prefixes):
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

    async def _process_memory_function_calls(self, result: dict, messages: list, options, guild_id: str, channel_id: str, user_id: str) -> dict:
        """Process memory function calls and continue conversation if needed."""
        debug_info = []
        try:
            # Check if there are memory function calls in the raw response
            raw_resp = result.get("_raw_response")
            if not raw_resp:
                return result
                
            memory_function_calls = []
            # Primary: top-level function_call items
            debug_info.append(f"Checking for function calls in raw_resp.output: {len(getattr(raw_resp, 'output', []))}")
            for item in getattr(raw_resp, 'output', []) or []:
                itype = getattr(item, 'type', None)
                if isinstance(item, dict):
                    itype = item.get('type')
                debug_info.append(f"Found output item type: {itype}")
                if itype in ("function_call", "tool_call"):
                    name = getattr(item, 'name', None) if not isinstance(item, dict) else item.get('name')
                    debug_info.append(f"Found function call: {name}")
                    if name in ("propose_memories", "save_memories"):
                        raw_args = getattr(item, 'arguments', {}) if not isinstance(item, dict) else item.get('arguments', {})
                        if isinstance(raw_args, str):
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                parsed_args = {}
                        else:
                            parsed_args = raw_args
                        # Prefer call_id if present, else fall back to id
                        call_id = (
                            (getattr(item, 'call_id', None) if not isinstance(item, dict) else item.get('call_id'))
                            or (getattr(item, 'id', None) if not isinstance(item, dict) else item.get('id'))
                        )
                        debug_info.append(f"Function call details: name={name}, call_id={call_id}, args={parsed_args}")
                        memory_function_calls.append({
                            "call_id": call_id,
                            "name": name,
                            "arguments": parsed_args,
                        })
            # Fallback: nested in message content lists
            if not memory_function_calls:
                for item in getattr(raw_resp, 'output', []) or []:
                    content = getattr(item, 'content', None)
                    if content is None and isinstance(item, dict):
                        content = item.get('content')
                    if isinstance(content, list):
                        for c in content:
                            ctype = getattr(c, 'type', None) if not isinstance(c, dict) else c.get('type')
                            if ctype == 'function_call':
                                name = getattr(c, 'name', None) if not isinstance(c, dict) else c.get('name')
                                if name in ("propose_memories", "save_memories"):
                                    raw_args = getattr(c, 'arguments', {}) if not isinstance(c, dict) else c.get('arguments', {})
                                    if isinstance(raw_args, str):
                                        try:
                                            parsed_args = json.loads(raw_args)
                                        except json.JSONDecodeError:
                                            parsed_args = {}
                                    else:
                                        parsed_args = raw_args
                                    call_id = (
                                        (getattr(c, 'call_id', None) if not isinstance(c, dict) else c.get('call_id'))
                                        or (getattr(c, 'id', None) if not isinstance(c, dict) else c.get('id'))
                                    )
                                    memory_function_calls.append({
                                        "call_id": call_id,
                                        "name": name,
                                        "arguments": parsed_args,
                                    })
            
            if not memory_function_calls:
                return result
                
            # If Responses requires action, parse tool_calls list
            try:
                req = getattr(raw_resp, 'required_action', None)
                if req is None and isinstance(raw_resp, dict):
                    req = raw_resp.get('required_action')
                rtype = getattr(req, 'type', None) if req is not None else None
                if rtype is None and isinstance(req, dict):
                    rtype = req.get('type')
                if req and rtype == 'submit_tool_outputs':
                    sto = getattr(req, 'submit_tool_outputs', None)
                    if sto is None and isinstance(req, dict):
                        sto = req.get('submit_tool_outputs')
                    tool_calls = getattr(sto, 'tool_calls', None) if sto is not None else None
                    if tool_calls is None and isinstance(sto, dict):
                        tool_calls = sto.get('tool_calls')
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            # SDK shape: {id, type, function: {name, arguments}}
                            if isinstance(tc, dict):
                                tcid = tc.get('id')
                                fn = tc.get('function') or {}
                                tname = (fn.get('name') if isinstance(fn, dict) else None) or tc.get('name')
                                targs = (fn.get('arguments') if isinstance(fn, dict) else None) or tc.get('arguments')
                            else:
                                tcid = getattr(tc, 'id', None)
                                fn = getattr(tc, 'function', None)
                                tname = getattr(fn, 'name', None) if fn else getattr(tc, 'name', None)
                                targs = getattr(fn, 'arguments', None) if fn else getattr(tc, 'arguments', None)
                            if tname in ("propose_memories", "save_memories"):
                                if isinstance(targs, str):
                                    try:
                                        parsed_args = json.loads(targs)
                                    except json.JSONDecodeError:
                                        parsed_args = {}
                                else:
                                    parsed_args = targs or {}
                                memory_function_calls.append({
                                    "call_id": tcid,
                                    "name": tname,
                                    "arguments": parsed_args,
                                })
            except Exception:
                pass

            # Process each memory function call
            function_outputs = []
            for call in memory_function_calls:
                try:
                    # Execute the memory function
                    debug_info.append(f"Executing {call['name']} with args: {call['arguments']}")
                    output = await self.function_handler.handle_function_call(
                        call["name"], 
                        call["arguments"], 
                        guild_id, 
                        channel_id, 
                        user_id
                    )
                    debug_info.append(f"Function output: {output}")
                    
                    # Format function output for Responses API
                    try:
                        import json as _json
                        out_str = _json.dumps(output, ensure_ascii=False)
                    except Exception:
                        out_str = str(output)
                    function_outputs.append({
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": out_str
                    })
                except Exception as e:
                    # Return error output for this function call
                    debug_info.append(f"Error executing {call['name']}: {str(e)}")
                    function_outputs.append({
                        "type": "function_call_output", 
                        "call_id": call["call_id"],
                        "output": f"Error: {str(e)}"
                    })
            
            # If we have function outputs, continue the conversation
            if function_outputs:
                # Submit tool outputs using the native endpoint if available
                toolouts = [{"call_id": f.get("call_id"), "output": f.get("output", "")} for f in function_outputs]
                debug_info.append(f"Submitting tool outputs: {len(toolouts)}")
                try:
                    final_result = await self.client.submit_tool_outputs(
                        previous_response_id=getattr(raw_resp, 'id', None),
                        tool_outputs=toolouts,
                        debug=False,
                    )
                    # Add debug info to final result for diag display
                    if debug_info:
                        if 'debug' not in final_result:
                            final_result['debug'] = []
                        final_result['debug'].extend(debug_info)
                    
                    # Check if the follow-up response has more memory function calls
                    return await self._process_memory_function_calls(final_result, messages, options, guild_id, channel_id, user_id)
                except Exception as e:
                    debug_info.append(f"submit_tool_outputs failed: {e}; falling back to create")
                    # Fallback to create-based continuation
                    updated_messages = list(messages)  # Start with original conversation
                    # Include reasoning items if present to satisfy reasoning model requirements
                    try:
                        for item in getattr(raw_resp, 'output', []) or []:
                            itype = getattr(item, 'type', None)
                            if itype is None and isinstance(item, dict):
                                itype = item.get('type')
                            if itype == 'reasoning':
                                # Pass through the reasoning block as-is
                                updated_messages.append(item if isinstance(item, dict) else getattr(item, '__dict__', {}))
                    except Exception:
                        pass
                    # Append function_call_output items
                    for t in toolouts:
                        if t.get("call_id") and t.get("output") is not None:
                            updated_messages.append({
                                "type": "function_call_output",
                                "call_id": t["call_id"],
                                "output": t["output"],
                            })
                    from .openai_client import ChatOptions
                    continue_options = ChatOptions(
                        model=options.model,
                        tools=options.tools,
                        reasoning=options.reasoning,
                        max_tokens=options.max_tokens,
                        temperature=options.temperature,
                        system_prompt=options.system_prompt,
                        file_ids=options.file_ids,
                        vector_store_id=options.vector_store_id,
                        inline_file_ids=options.inline_file_ids,
                        inline_image_ids=options.inline_image_ids,
                        inline_image_urls=options.inline_image_urls,
                        code_container_type=options.code_container_type,
                        previous_response_id=getattr(raw_resp, 'id', None)
                    )
                    final_result = await self.client.respond_collect(updated_messages, continue_options)
                    
                    # Add debug info to final result for diag display
                    if debug_info:
                        if 'debug' not in final_result:
                            final_result['debug'] = []
                        final_result['debug'].extend(debug_info)
                    
                    # Check if the follow-up response has more memory function calls
                    return await self._process_memory_function_calls(final_result, messages, continue_options, guild_id, channel_id, user_id)
                
        except Exception as e:
            debug_info.append(f"Error processing memory function calls: {e}")
            
        # Add debug info to result for diag display
        if debug_info:
            if 'debug' not in result:
                result['debug'] = []
            result['debug'].extend(debug_info)
            
        return result

    async def _chat_path(self, message: discord.Message, content: str, gconf: Dict[str, Any]) -> None:
        lock = self._get_lock(message.channel.id)
        async with lock:
            # Determine guild prefixes to filter command messages from context
            prefixes: list[str] = []
            try:
                res = self.bot.get_valid_prefixes(message.guild)
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, (list, tuple)):
                    prefixes = [p for p in res if isinstance(p, str)]
            except Exception:
                prefixes = []
            # Separate static template from dynamic variables for caching optimization
            try:
                sys_prompt_global = await self.config.system_prompt()
            except Exception:
                sys_prompt_global = gconf.get("system_prompt", "")
            
            # Check if variable separation would be beneficial (only if there are actually variables to separate)
            try:
                static_template, dynamic_values = await separate_template_variables(message, sys_prompt_global or "")
                
                # Only use separation if it actually found variables to separate AND the original template is substantial
                # This prevents token overhead when there's no caching benefit
                if dynamic_values and len(sys_prompt_global or "") > 200:
                    dynamic_context = build_dynamic_context_message(dynamic_values)
                else:
                    # Use traditional formatting for better token efficiency
                    static_template = await format_variables(message, sys_prompt_global or "")
                    dynamic_context = ""
            except Exception:
                # Fallback to old behavior if separation fails
                static_template = await format_variables(message, sys_prompt_global or "")
                dynamic_context = ""

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
                skip_prefixes=prefixes,
                is_command_message=self._is_command_message,
            )
            # Include current user's name in the message
            current_user_name = message.author.display_name or message.author.name
            # Use separated prompts if we have dynamic context, otherwise use original behavior
            if dynamic_context:
                msgs = build_messages_with_separated_prompts(static_template, dynamic_context, history, content, current_user_name)
            else:
                # Use original behavior - static_template should be the same as formatted prompt when no variables
                system_prompt = static_template  # Use static_template as system_prompt
                msgs = build_messages(system_prompt, history, content, current_user_name)
            # Collect a limited number of image attachments from recent history
            history_image_limit = int(gconf.get("images_backread", 0) or 0)
            history_image_urls: list[str] = []
            if history_image_limit > 0 and hasattr(message.channel, "history"):
                try:
                    from datetime import datetime, timezone
                    earliest_dt = None
                    if earliest_ts is not None:
                        if isinstance(earliest_ts, (int, float)):
                            earliest_dt = datetime.fromtimestamp(float(earliest_ts), tz=timezone.utc)
                        else:
                            earliest_dt = earliest_ts
                    last_time = message.created_at
                    images_time_limit = int(gconf.get("images_backread_seconds", 1800))
                    # Iterate most recent first, collect image URLs from allowed authors
                    async for msg in message.channel.history(limit=int(gconf.get("messages_backread", 25)), before=message, oldest_first=False):
                        # Respect image-specific time window
                        try:
                            if last_time and abs((last_time - msg.created_at).total_seconds()) > images_time_limit:
                                break
                        except Exception:
                            pass
                        # Skip command messages (use parser if possible)
                        try:
                            if await self._is_command_message(msg):
                                continue
                        except Exception:
                            rawc = (msg.content or "")
                            if any(isinstance(p, str) and p and rawc.lstrip().startswith(p) for p in prefixes):
                                continue
                        # Enforce forget cutoff
                        try:
                            if earliest_dt is not None and msg.created_at < earliest_dt:
                                break
                        except Exception:
                            pass
                        # Author filters similar to text history
                        role_is_assistant = bool(msg.author.bot and self.bot and getattr(self.bot, "user", None) and msg.author.id == self.bot.user.id)
                        if not role_is_assistant:
                            if not bool(gconf.get("include_others", True)):
                                continue
                            uid = msg.author.id
                            if uid in set(gconf.get("optout", []) or []):
                                continue
                            if (uid not in set(gconf.get("optin", []) or [])) and not bool(gconf.get("optin_by_default", True)):
                                continue
                        # Scan attachments for images
                        if getattr(msg, "attachments", None):
                            for a in msg.attachments:
                                ctype = a.content_type or ""
                                is_image = False
                                if isinstance(ctype, str) and ctype.startswith("image/"):
                                    is_image = True
                                else:
                                    # Infer from filename
                                    fname = a.filename or ""
                                    if "." in fname:
                                        ext = fname.lower().rsplit(".", 1)[-1]
                                        if ext in {"png","jpg","jpeg","gif","webp","bmp","tif","tiff","svg"}:
                                            is_image = True
                                if is_image:
                                    try:
                                        history_image_urls.append(a.url)  # type: ignore[attr-defined]
                                    except Exception:
                                        continue
                                    if len(history_image_urls) >= history_image_limit:
                                        break
                        if len(history_image_urls) >= history_image_limit:
                            break
                except Exception:
                    history_image_urls = []
            # Determine vector store availability
            vector_store_id = gconf.get("file_kb_id") or None
            effective_tools = dict(gconf["tools"])  # shallow copy
            
            # Initialize memory system if needed
            memory_init_failed = False
            if not self._memory_initialized and effective_tools.get("memories"):
                try:
                    await self.memory_manager.initialize()
                    self._memory_initialized = True
                except Exception as e:
                    # If memory initialization fails, disable memories for this session
                    print(f"Memory system initialization failed: {e}")
                    effective_tools["memories"] = False
                    memory_init_failed = True
            
            # Enable file_search for memories if memories tool is enabled AND memories exist
            # This ensures file_search is only used when there are actually memories to retrieve
            if effective_tools.get("memories") and gconf.get("memories_enabled", True) and not memory_init_failed:
                try:
                    memory_vector_store_id = await self.memory_manager.get_vector_store_id(str(message.guild.id))
                    if memory_vector_store_id:
                        # Check if there are any memories in the database before enabling file_search
                        stats = await self.memory_manager.storage.get_guild_stats(str(message.guild.id))
                        if stats.get("total", 0) > 0:
                            effective_tools["file_search"] = True
                            vector_store_id = memory_vector_store_id
                        # If no memories exist, don't enable file_search to avoid unnecessary API calls
                except Exception:
                    # If memory operations fail, continue without memories
                    effective_tools["memories"] = False
            elif effective_tools.get("file_search"):
                # Do not use vector stores for files; only read current attachments
                effective_tools["file_search"] = False

            # Collect and upload current message attachments for inline reading/vision
            inline_file_ids: list[str] = []
            inline_image_ids: list[str] = []
            inline_image_urls: list[str] = []
            if getattr(message, "attachments", None):
                file_bytes: list[bytes] = []
                fnames: list[str] = []
                kinds: list[str] = []
                # Don't preload files for code interpreter to avoid unnecessary container charges
                # Let the model decide when to use code interpreter - files will be available via URLs
                need_image_file_ids = False
                for a in message.attachments:
                    ctype = a.content_type or ""
                    # Skip very large files (>20MB) to avoid timeouts
                    try:
                        if a.size and a.size > 20 * 1024 * 1024:
                            continue
                    except Exception:
                        pass
                    # For images, prefer using the Discord CDN URL directly
                    if isinstance(ctype, str) and ctype.startswith("image/"):
                        try:
                            inline_image_urls.append(a.url)  # type: ignore[attr-defined]
                            # Also upload image bytes when code_interpreter is enabled so container can access the file
                            if not need_image_file_ids:
                                fnames.append(a.filename or "attachment")
                                kinds.append(ctype)
                                continue
                        except Exception:
                            pass
                    try:
                        data = await a.read()
                    except Exception:
                        continue
                    file_bytes.append(data)
                    fname = a.filename or "attachment"
                    fnames.append(fname)
                    # Fallback to filename extension when content_type is missing
                    if not ctype and "." in fname:
                        ext = fname.lower().rsplit(".", 1)[-1]
                        if ext in {"png","jpg","jpeg","gif","webp","bmp","tif","tiff","svg"}:
                            ctype = f"image/{'jpeg' if ext=='jpg' else ext}"
                    kinds.append(ctype)
                if file_bytes:
                    try:
                        ids = await self.client.index_files(file_bytes, fnames)
                        for fid, ctype in zip(ids, kinds):
                            if isinstance(ctype, str) and ctype.startswith("image/"):
                                inline_image_ids.append(fid)
                                # Also provide as input_file so code_interpreter container receives it
                                if need_image_file_ids:
                                    inline_file_ids.append(fid)
                            else:
                                inline_file_ids.append(fid)
                    except Exception:
                        inline_file_ids = []
                        inline_image_ids = []

            # Seed inline image URLs with recent historical images (limited)
            if history_image_urls:
                inline_image_urls.extend(history_image_urls)

            # Include attachments from the replied-to message (one hop) so the model can edit/inspect them
            if getattr(message, "reference", None):
                try:
                    ref = message.reference
                    replied = ref.cached_message or await self.bot.get_channel(ref.channel_id).fetch_message(ref.message_id)
                except Exception:
                    replied = None
                if replied and getattr(replied, "attachments", None):
                    file_bytes_r: list[bytes] = []
                    fnames_r: list[str] = []
                    kinds_r: list[str] = []
                    # Don't preload replied files for code interpreter to avoid unnecessary charges
                    need_image_file_ids_r = False
                    for a in replied.attachments:
                        ctype = a.content_type or ""
                        # Skip very large files (>20MB)
                        try:
                            if a.size and a.size > 20 * 1024 * 1024:
                                continue
                        except Exception:
                            pass
                        # Prefer URL for images
                        if isinstance(ctype, str) and ctype.startswith("image/"):
                            try:
                                inline_image_urls.append(a.url)  # type: ignore[attr-defined]
                                if not need_image_file_ids_r:
                                    fnames_r.append(a.filename or "attachment")
                                    kinds_r.append(ctype)
                                    continue
                            except Exception:
                                pass
                        try:
                            data = await a.read()
                        except Exception:
                            continue
                        file_bytes_r.append(data)
                        fnames_r.append(a.filename or "attachment")
                        kinds_r.append(ctype)
                    if file_bytes_r:
                        try:
                            ids_r = await self.client.index_files(file_bytes_r, fnames_r)
                            for fid, ctype in zip(ids_r, kinds_r):
                                if isinstance(ctype, str) and ctype.startswith("image/"):
                                    inline_image_ids.append(fid)
                                    if need_image_file_ids_r:
                                        inline_file_ids.append(fid)
                                else:
                                    inline_file_ids.append(fid)
                        except Exception:
                            pass

            # Code interpreter container type (optional, global)
            try:
                code_container_type = await self.config.code_container_type()
            except Exception:
                code_container_type = None

            # Build augmented system prompt (for backwards compatibility and additional instructions)
            # Use static template as base, since it's more cacheable
            base_prompt = static_template
            
            additional_instructions = []
            
            # If current or replied-to attachments are present, add attachment instructions
            if (inline_image_urls or inline_image_ids or inline_file_ids):
                additional_instructions.append(
                    "You have access to the user's current attachments in this turn. "
                    "If the user asks to describe or edit an image, use the provided input_image parts "
                    "as your source rather than referencing prior response or image IDs."
                )
            
            # If file_search is enabled for memories, add memory instructions
            if effective_tools.get("file_search") and vector_store_id:
                additional_instructions.append(
                    "You have access to a memory system via file_search. When users ask about their preferences, "
                    "past conversations, or personal information they've shared, use the file_search tool to "
                    "retrieve relevant memories before responding. IMPORTANT: When using retrieved memories, "
                    "respond naturally as if recalling information about a friend. Don't quote the raw memory "
                    "format or say 'you told me' - instead, integrate the knowledge into conversational responses "
                    "that sound personal and natural. For example, if asked about their name and you retrieve "
                    "'Legal name: Miles; nickname: Duke', respond like 'Your name is Duke, though your legal "
                    "name is Miles' rather than quoting the stored format."
                )
            
            # If code interpreter is enabled, add code instructions
            if effective_tools.get("code_interpreter"):
                additional_instructions.append(
                    "When creating computational outputs or files, refer to files by name only; I will attach the actual files to the chat."
                )
            
            # Build final system prompt
            if additional_instructions:
                sys_prompt_aug = f"{base_prompt}\n\n" + "\n\n".join(additional_instructions)
            else:
                sys_prompt_aug = base_prompt

            # Get channel-specific context for cache key
            channel_context = gconf.get("channel_contexts", {}).get(str(message.channel.id), "")
            
            # Build state flags for cache differentiation
            attachment_flags = f"att={1 if (inline_image_urls or inline_image_ids or inline_file_ids) else 0}"
            memory_flags = f"mem={1 if (effective_tools.get('file_search') and vector_store_id) else 0}"
            
            # Build collision-resistant cache key using static template
            prompt_cache_key = self.build_collision_resistant_cache_key(
                static_template=static_template,
                tools_config=gconf.get("tools", {}),
                model=gconf["model"],
                channel_context=channel_context,
                attachment_flags=attachment_flags,
                memory_flags=memory_flags
            )
            
            # Get storage setting for caching optimization
            storage_enabled = gconf.get("enable_response_storage", True)
            
            options = ChatOptions(
                model=gconf["model"],
                tools=effective_tools,
                reasoning=gconf["reasoning"],
                max_tokens=gconf["max_tokens"],
                temperature=gconf["temperature"],
                system_prompt=sys_prompt_aug,
                file_ids=None,
                vector_store_id=vector_store_id,
                inline_file_ids=inline_file_ids or None,
                inline_image_ids=inline_image_ids or None,
                inline_image_urls=inline_image_urls or None,
                code_container_type=code_container_type or None,
                prompt_cache_key=prompt_cache_key,
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
                async with message.channel.typing():
                    # Implement two-call approach for code_interpreter billing optimization
                    # Call 1: If code_interpreter is enabled, first try without it but with sentinel function
                    if effective_tools.get("code_interpreter"):
                        # Create options for first call without code_interpreter but with sentinel
                        first_call_tools = dict(effective_tools)
                        first_call_tools["code_interpreter"] = False  # Disable CI for first call
                        
                        # Add instruction to use the sentinel function
                        enhanced_prompt = (options.system_prompt + 
                            "\n\nIMPORTANT: If you need to execute Python code, create plots, or perform any computational task, "
                            "you MUST call the request_python function first. Do NOT provide code directly in your response.")
                        
                        first_options = ChatOptions(
                            model=options.model,
                            tools=first_call_tools,
                            reasoning=options.reasoning,
                            max_tokens=options.max_tokens,
                            temperature=options.temperature,
                            system_prompt=enhanced_prompt,
                            file_ids=options.file_ids,
                            vector_store_id=options.vector_store_id,
                            inline_file_ids=options.inline_file_ids,
                            inline_image_ids=options.inline_image_ids,
                            inline_image_urls=options.inline_image_urls,
                            code_container_type=options.code_container_type,
                            include_python_sentinel=True,  # Include sentinel function
                        )
                        
                        # Make first call
                        first_result = await self.client.respond_collect(
                            msgs, first_options, 
                            guild_id=str(message.guild.id), 
                            storage_enabled=storage_enabled
                        )
                        
                        # Check if model called the sentinel function
                        python_requested = False
                        try:
                            # Check the raw response for function calls (using the correct format from docs)
                            raw_resp = first_result.get("_raw_response")
                            if raw_resp and hasattr(raw_resp, 'output'):
                                for item in raw_resp.output:
                                    # Look for function call items exactly as shown in documentation
                                    if hasattr(item, 'type') and item.type == "function_call":
                                        if hasattr(item, 'name') and item.name == "request_python":
                                            python_requested = True
                                            break
                        except Exception:
                            pass
                        
                        if python_requested:
                            # Model requested Python - make second call with code_interpreter enabled
                            # Completely ignore the first result and only use the second call
                            result = await self.client.respond_collect(
                                msgs, options, 
                                guild_id=str(message.guild.id), 
                                storage_enabled=storage_enabled
                            )
                        else:
                            # Model didn't request Python - use first result (no container charge)
                            result = first_result
                    else:
                        # code_interpreter not enabled, normal call
                        result = await self.client.respond_collect(
                            msgs, options, 
                            guild_id=str(message.guild.id), 
                            storage_enabled=storage_enabled
                        )
                
                # Process memory function calls if present
                result = await self._process_memory_function_calls(result, msgs, options, str(message.guild.id), str(message.channel.id), str(message.author.id))
                
                text = result.get("text", "")
                images = result.get("images") or []
                image_names = result.get("image_names") or [None] * len(images)
                files = result.get("files") or []
                # Optional debug note for admins
                try:
                    if bool(gconf.get("debug_attachments", False)):
                        await message.channel.send(
                            f"[debug] result: text={len(text or '')}B images={len(images)} files={len(files)}"
                        )
                        # Check if images array contains duplicates
                        if len(images) > 1:
                            from hashlib import sha256
                            image_hashes = [sha256(img).hexdigest()[:8] for img in images]
                            await message.channel.send(f"[debug] image hashes: {image_hashes}")
                        # Check refs detection
                        refs_preview = str(refs)[:100] if refs else "empty"
                        await message.channel.send(f"[debug] refs: {refs_preview}")
                except Exception:
                    pass
                # Identify any sandbox container links like [name](sandbox:/mnt/data/...) to replace with attachment URLs
                # Track both the label and the filename extracted from the link target.
                refs: list[tuple[str, str]] = []  # (label, filename)
                if text:
                    i = 0
                    n = len(text)
                    while i < n:
                        lb = text.find('[', i)
                        if lb == -1:
                            break
                        rb = text.find(']', lb + 1)
                        if rb == -1 or rb + 1 >= n or text[rb + 1] != '(':
                            i = lb + 1
                            continue
                        rp = text.find(')', rb + 2)
                        if rp == -1:
                            break
                        label = text[lb + 1:rb]
                        link = text[rb + 2:rp]
                        if (link.startswith('sandbox:') or link.startswith('/mnt/data')):
                            # Extract filename from the path
                            fname = link
                            try:
                                # strip scheme and split on /
                                if link.startswith('sandbox:'):
                                    path = link.split(':', 1)[1]
                                else:
                                    path = link
                                parts = path.split('/')
                                if parts:
                                    fname = parts[-1] or fname
                            except Exception:
                                pass
                            refs.append((label, fname))
                        i = rp + 1
                if patterns:
                    # recent authors for {authorname}
                    authors = []
                    if hasattr(message.channel, "history"):
                        async for m in message.channel.history(limit=10):
                            if m.author != message.guild.me:
                                authors.append(m.author.display_name)
                    botname = (message.guild.me.nick or message.guild.me.display_name) if message.guild else ""
                    text = await apply_removelist(
                        patterns=patterns, text=text, botname=botname, recent_authors=authors
                    )
                # If we need to replace sandbox links, post attachments first to get working URLs
                name_to_url: dict[str, str] = {}
                if refs and (images or files):
                    # Send images with filenames matching referenced names when possible
                    # Track which images we've sent to avoid duplicates
                    sent_image_hashes = set()
                    for idx, img in enumerate(images):
                        try:
                            # Check if this image was already sent
                            from hashlib import sha256
                            img_hash = sha256(img).hexdigest()
                            if img_hash in sent_image_hashes:
                                continue
                            
                            # Prefer a referenced filename from sandbox links
                            fname = None
                            for (label, target) in refs:
                                low = target.lower()
                                if any(low.endswith(ext) for ext in ('.png','.jpg','.jpeg','.gif','.webp','.bmp','.tif','.tiff')):
                                    # if this referenced filename hasn't been mapped yet, use it
                                    if target not in name_to_url:
                                        fname = target
                                        break
                            # Otherwise fallback to image_names provided by client
                            if not fname and idx < len(image_names) and isinstance(image_names[idx], str) and image_names[idx]:
                                fname = image_names[idx]  # type: ignore[index]
                            if not fname:
                                fname = f"image_{idx+1}.png"
                            msg_img = await message.channel.send(file=discord.File(BytesIO(img), filename=fname))
                            if msg_img.attachments:
                                att = msg_img.attachments[0]
                                name_to_url[att.filename] = att.url
                            sent_image_hashes.add(img_hash)
                        except Exception:
                            continue
                    # Send files and capture URLs
                    for item in files:
                        try:
                            name = item.get("name") or "attachment.bin"
                            data = item.get("bytes")
                            if not isinstance(name, str) or not isinstance(data, (bytes, bytearray)):
                                continue
                            if len(data) > 7_900_000:
                                await message.channel.send(f"Generated a file '{name}' (~{len(data)//1024} KB), but it's too large to attach here.")
                                continue
                            # Check if this file contains image data that we've already sent
                            is_image_file = False
                            try:
                                if isinstance(name, str):
                                    low = name.lower()
                                    if any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")):
                                        is_image_file = True
                                        file_hash = sha256(data).hexdigest()
                                        if file_hash in sent_image_hashes:
                                            # Skip this file as we've already sent the same image
                                            continue
                                        sent_image_hashes.add(file_hash)
                            except Exception:
                                pass
                            
                            msg_file = await message.channel.send(file=discord.File(BytesIO(data), filename=name))
                            try:
                                if bool(gconf.get("debug_attachments", False)):
                                    await message.channel.send(f"[debug] pre-sent file: {name} -> {(msg_file.attachments[0].url if msg_file.attachments else 'no attachment')}" )
                            except Exception:
                                pass
                            if msg_file.attachments:
                                att = msg_file.attachments[0]
                                name_to_url[att.filename] = att.url
                        except Exception:
                            try:
                                if bool(gconf.get("debug_attachments", False)):
                                    await message.channel.send(f"[debug] failed to pre-send file: {name}")
                            except Exception:
                                pass
                            continue
                    # Replace sandbox links with working attachment URLs
                    if name_to_url and text:
                        out = []
                        i = 0
                        n = len(text)
                        while i < n:
                            lb = text.find('[', i)
                            if lb == -1:
                                out.append(text[i:])
                                break
                            rb = text.find(']', lb + 1)
                            if rb == -1 or rb + 1 >= n or text[rb + 1] != '(':
                                out.append(text[i:lb+1])
                                i = lb + 1
                                continue
                            rp = text.find(')', rb + 2)
                            if rp == -1:
                                out.append(text[i:])
                                break
                            label = text[lb + 1:rb]
                            link = text[rb + 2:rp]
                            # Replace sandbox links with attachment URL based on target filename
                            if (link.startswith('sandbox:') or link.startswith('/mnt/data')):
                                # extract filename
                                target = link
                                if link.startswith('sandbox:'):
                                    target = link.split(':', 1)[1]
                                target_name = target.split('/')[-1] if '/' in target else target
                                # find a matching mapping by exact filename
                                url = name_to_url.get(target_name)
                                if not url:
                                    # try case-insensitive match
                                    for k, v in name_to_url.items():
                                        if k.lower() == target_name.lower():
                                            url = v
                                            break
                                if url:
                                    out.append(text[i:lb])
                                    out.append(f"[{label}]({url})")
                                    i = rp + 1
                                    continue
                                # If no mapping, drop the sandbox link and leave label
                                out.append(text[i:lb])
                                out.append(label)
                                i = rp + 1
                                continue
                            else:
                                out.append(text[i:rp+1])
                                i = rp + 1
                        text = ''.join(out)
                # Send text after any replacements
                if text and text.strip():
                    for ch in chunk_message(text):
                        await message.channel.send(ch)
                # If we didn't pre-send attachments, send them now
                if not refs:
                    for idx, img in enumerate(images):
                        file = discord.File(BytesIO(img), filename=f"image_{idx+1}.png")
                        await message.channel.send(file=file)
                    for item in files:
                        try:
                            name = item.get("name") or "attachment.bin"
                            data = item.get("bytes")
                            if not isinstance(name, str) or not isinstance(data, (bytes, bytearray)):
                                continue
                            if len(data) > 7_900_000:
                                await message.channel.send(f"Generated a file '{name}' (~{len(data)//1024} KB), but it's too large to attach here.")
                                continue
                            file = discord.File(BytesIO(data), filename=name)
                            msg = await message.channel.send(file=file)
                            try:
                                if bool(gconf.get("debug_attachments", False)):
                                    await message.channel.send(f"[debug] sent file: {name} -> {(msg.attachments[0].url if msg.attachments else 'no attachment')}" )
                            except Exception:
                                pass
                        except Exception:
                            try:
                                if bool(gconf.get("debug_attachments", False)):
                                    await message.channel.send(f"[debug] failed to send file: {item.get('name', 'attachment.bin')}")
                            except Exception:
                                pass
                            continue
                # Minimal fallback: if absolutely nothing, inform user
                if not text.strip() and not images and not files:
                    await message.channel.send("I couldn't produce a result for that. If you asked for an image, ensure the image tool is enabled: [p]gpt5 config tools enable image.")
            except Exception as e:
                # Try to extract body/status for clearer diagnostics
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
                if body:
                    preview = str(body)
                    if len(preview) > 700:
                        preview = preview[:700] + "…"
                    await message.channel.send(f"Sorry, I hit an error: {type(orig).__name__} (status={status})\n{preview}")
                else:
                    await message.channel.send(f"Sorry, I hit an error: {orig}")

    async def _image_path(self, message: discord.Message, content: str, gconf: Dict[str, Any]) -> None:
        lock = self._get_lock(message.channel.id)
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
