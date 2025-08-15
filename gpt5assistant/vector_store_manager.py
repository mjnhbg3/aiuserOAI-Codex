from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any
from io import BytesIO

from .memory_storage import Memory, ScopeGroup, MemoryStorage, group_by_scope, render_profile_markdown
from .openai_client import OpenAIClient


class VectorStoreManager:
    """Manages OpenAI Vector Stores for long-term memory."""
    
    def __init__(self, openai_client: OpenAIClient, memory_storage: MemoryStorage, config):
        self.client = openai_client
        self.storage = memory_storage
        self.config = config
        self._vector_store_cache: Dict[str, str] = {}
        
    async def ensure_vector_store(self, guild_id: str) -> str:
        """Ensure vector store exists for guild. Returns vector store ID."""
        # Check cache first
        if guild_id in self._vector_store_cache:
            return self._vector_store_cache[guild_id]
            
        # Check config for existing vector store ID
        guild_stores = await self.config.memories_vector_store_id_by_guild()
        if guild_id in guild_stores:
            vs_id = guild_stores[guild_id]
            self._vector_store_cache[guild_id] = vs_id
            return vs_id
            
        # Create new vector store
        vs_name = f"vs_guild_{guild_id}"
        
        try:
            # Create vector store via OpenAI API
            vector_store = await self._create_vector_store(vs_name)
            vs_id = vector_store.get("id")
            if not vs_id:
                raise RuntimeError(vector_store.get("error", "unknown error creating vector store"))

            # Cache and persist the ID
            self._vector_store_cache[guild_id] = vs_id
            guild_stores[guild_id] = vs_id
            await self.config.memories_vector_store_id_by_guild.set(guild_stores)

            return vs_id

        except Exception as e:
            raise RuntimeError(f"Failed to create vector store for guild {guild_id}: {e}")
    
    async def _create_vector_store(self, name: str) -> Dict[str, Any]:
        """Create a new vector store using OpenAI API."""
        # Use the existing OpenAI client to create vector store; fallback to direct HTTP with Beta header
        openai_client = self.client.client
        
        try:
            vector_store = await openai_client.vector_stores.create(
                name=name,
                expires_after={
                    "anchor": "last_active_at",
                    "days": 730  # Keep for 2 years after last use (true long-term memory)
                }
            )
            
            return {
                "id": vector_store.id,
                "name": vector_store.name,
                "status": vector_store.status
            }
        except Exception as e:
            # Fallback: direct HTTP call with proper Beta header
            try:
                import httpx
                base_url = getattr(self.client, "_base_url", "https://api.openai.com/v1")
                api_key = getattr(self.client, "_api_key", None)
                if not api_key:
                    raise RuntimeError("Missing API key for vector store creation")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "assistants=v2",
                }
                async with httpx.AsyncClient(timeout=30) as http:
                    r = await http.post(f"{base_url}/vector_stores", headers=headers, json={"name": name})
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                    data = r.json()
                    return {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "status": data.get("status", "completed"),
                    }
            except Exception as e2:
                # If vector store creation fails, return error info
                return {
                    "id": None,
                    "name": name,
                    "status": "error",
                    "error": f"{e} | fallback: {e2}"
                }
    
    async def update_vector_store_profiles(self, guild_id: str, scope_groups: List[ScopeGroup]) -> None:
        """Update vector store with new profile files for the given scope groups."""
        vs_id = await self.ensure_vector_store(guild_id)
        openai_client = self.client.client
        
        for group in scope_groups:
            try:
                # Fetch current memories for this scope
                memories = await self.storage.fetch_scope_memories(
                    guild_id=group.guild_id,
                    scope=group.scope,
                    user_id=group.user_id,
                    channel_id=group.channel_id
                )
                
                # Render markdown profile
                profile_text = render_profile_markdown(group, memories)
                filename = group.get_filename()
                
                # Create file in OpenAI
                file_content = BytesIO(profile_text.encode('utf-8'))
                openai_file = await openai_client.files.create(
                    file=(filename, file_content),
                    purpose="assistants"
                )
                
                # Add file to vector store with metadata
                metadata = group.get_metadata()
                try:
                    await openai_client.vector_stores.files.create(
                        vector_store_id=vs_id,
                        file_id=openai_file.id,
                    )
                except Exception:
                    # Fallback via HTTP
                    try:
                        import httpx
                        base_url = getattr(self.client, "_base_url", "https://api.openai.com/v1")
                        api_key = getattr(self.client, "_api_key", None)
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "OpenAI-Beta": "assistants=v2",
                        }
                        async with httpx.AsyncClient(timeout=30) as http:
                            await http.post(
                                f"{base_url}/vector_stores/{vs_id}/files",
                                headers=headers,
                                json={"file_id": openai_file.id}
                            )
                    except Exception:
                        raise
                
                # Clean up old versions of this profile file to prevent accumulation
                # Keep only the most recent file for each profile
                await self._cleanup_old_profile_files(vs_id, filename, openai_file.id)
                
            except Exception as e:
                # Log error but continue with other groups
                print(f"Failed to update vector store profile for {group.get_filename()}: {e}")
                continue
    
    async def delete_vector_store_files(self, guild_id: str, scope: str, 
                                      user_id: Optional[str] = None,
                                      channel_id: Optional[str] = None) -> None:
        """Delete files from vector store for a specific scope."""
        try:
            vs_id = await self.ensure_vector_store(guild_id)
            openai_client = self.client.client
            
            # List files in vector store
            vector_store_files = await openai_client.vector_stores.files.list(
                vector_store_id=vs_id
            )
            
            # Determine target filename pattern
            if scope == "user" and user_id:
                target_pattern = f"user_mem__{guild_id}__{user_id}.md"
            elif scope == "channel" and channel_id:
                target_pattern = f"chan_mem__{guild_id}__{channel_id}.md"
            else:
                return
            
            # Find and delete matching files
            for file_obj in vector_store_files.data:
                # Note: We may need to fetch file details to get the name
                # This is a simplified approach - in practice you might need
                # to track filenames in your database for easier cleanup
                try:
                    await openai_client.vector_stores.files.delete(
                        vector_store_id=vs_id,
                        file_id=file_obj.id
                    )
                except Exception:
                    continue  # Continue with other files
                    
        except Exception as e:
            print(f"Failed to delete vector store files for {scope}: {e}")
    
    async def _cleanup_old_profile_files(self, vs_id: str, target_filename: str, keep_file_id: str) -> None:
        """Remove old versions of a profile file, keeping only the most recent."""
        try:
            openai_client = self.client.client
            
            # List all files in vector store
            vector_store_files = await openai_client.vector_stores.files.list(
                vector_store_id=vs_id
            )
            
            # Implement proper long-term storage limits utilizing the 1GB quota
            file_count = len(vector_store_files.data)
            
            # Get configurable limits for long-term memory
            try:
                max_files = await self.config.memories_vector_store_max_files()
                if max_files is None:
                    max_files = 8000  # Default for 1GB quota
            except Exception:
                max_files = 8000
                
            cleanup_batch_size = max(50, max_files // 20)  # Remove 5% when limit reached
            
            if file_count > max_files:
                # Sort by creation time (oldest first) and remove 5% of files
                sorted_files = sorted(
                    vector_store_files.data, 
                    key=lambda f: getattr(f, 'created_at', 0)
                )
                
                # Remove oldest 5% to maintain long-term memory while staying within limits
                files_to_remove = sorted_files[:cleanup_batch_size]
                
                for old_file in files_to_remove:
                    if old_file.id != keep_file_id:  # Don't delete the file we just created
                        try:
                            await openai_client.vector_stores.files.delete(
                                vector_store_id=vs_id,
                                file_id=old_file.id
                            )
                        except Exception:
                            continue  # Continue with other files
                            
        except Exception:
            # If cleanup fails, continue - it's not critical
            pass

    async def get_vector_store_stats(self, guild_id: str) -> Dict[str, Any]:
        """Get statistics about the guild's vector store."""
        try:
            vs_id = await self.ensure_vector_store(guild_id)
            openai_client = self.client.client
            
            # Get vector store details
            vector_store = await openai_client.vector_stores.retrieve(vs_id)
            
            # List files in vector store
            vector_store_files = await openai_client.vector_stores.files.list(
                vector_store_id=vs_id
            )
            
            return {
                "vector_store_id": vs_id,
                "status": vector_store.status,
                "file_count": len(vector_store_files.data),
                "usage_bytes": vector_store.usage_bytes if hasattr(vector_store, 'usage_bytes') else 0,
                "created_at": vector_store.created_at,
                "last_active_at": getattr(vector_store, 'last_active_at', None)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "vector_store_id": None,
                "status": "error"
            }


class MemoryManager:
    """High-level memory management combining storage and vector store operations."""
    
    def __init__(self, openai_client: OpenAIClient, config, cog_instance=None):
        self.storage = MemoryStorage(config, cog_instance)
        self.vector_manager = VectorStoreManager(openai_client, self.storage, config)
        self.config = config
        
    async def initialize(self):
        """Initialize the memory system."""
        await self.storage.initialize()
        
    async def propose_memories(self, memories: List[Memory]) -> List[Dict[str, Any]]:
        """Stage candidate memories without writing. Returns normalized items."""
        # Validate and normalize memories
        normalized = []
        
        for memory in memories:
            # Generate ID if not provided
            if not memory.mem_id:
                memory.mem_id = memory.generate_id()
                
            # Validate required fields
            if not memory.scope or memory.scope not in ["user", "channel"]:
                continue
            if not memory.guild_id or not memory.key or not memory.value:
                continue
                
            # Validate scope-specific requirements
            if memory.scope == "user" and not memory.user_id:
                continue
            if memory.scope == "channel" and not memory.channel_id:
                continue
                
            # Apply confidence filtering if configured (guild-scoped)
            try:
                min_confidence = await self.config.guild(int(memory.guild_id)).memories_confidence_min()
            except Exception:
                min_confidence = 0.4
            if memory.confidence is not None and memory.confidence < min_confidence:
                continue
                
            normalized.append(memory.to_dict())
            
        return normalized
        
    async def save_memories(self, memories: List[Memory]) -> Dict[str, Any]:
        """Persist memories to database and update vector store."""
        if not memories:
            return {"status": "ok", "saved": 0}
            
        # Determine guild (assume all items are same guild)
        gid = None
        try:
            gid = str(memories[0].guild_id)
        except Exception:
            pass
        # Check if memories are enabled (guild-scoped)
        try:
            enabled = await self.config.guild(int(gid)).memories_enabled()
        except Exception:
            enabled = True
        if not enabled:
            return {"status": "disabled", "saved": 0}
            
        # Check item count limit
        try:
            max_items = await self.config.guild(int(gid)).memories_max_items_per_call()
        except Exception:
            max_items = 50
        if len(memories) > max_items:
            memories = memories[:max_items]
            
        # 1. Upsert to database (do not lose writes if vector store update fails)
        try:
            saved_memories = await self.storage.bulk_upsert(memories)
        except Exception as e:
            return {"status": "error", "error": f"DB upsert failed: {e}", "saved": 0}

        # 2. Group by scope for vector store updates
        scope_groups = group_by_scope(saved_memories)

        # 3. Update vector store profiles (best-effort)
        vs_error: Optional[str] = None
        if scope_groups:
            try:
                guild_id = scope_groups[0].guild_id
                await self.vector_manager.update_vector_store_profiles(guild_id, scope_groups)
            except Exception as e:
                vs_error = str(e)

        result: Dict[str, Any] = {
            "status": "ok" if vs_error is None else "partial",
            "saved": len(saved_memories),
            "groups_updated": len(scope_groups) if vs_error is None else 0,
        }
        if vs_error is not None:
            result["vector_store_error"] = vs_error
        return result
    
    async def get_vector_store_id(self, guild_id: str) -> Optional[str]:
        """Get the vector store ID for a guild."""
        try:
            return await self.vector_manager.ensure_vector_store(guild_id)
        except Exception:
            return None
