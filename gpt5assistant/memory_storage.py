from __future__ import annotations

import sqlite3
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

try:
    import aiosqlite
except ImportError:
    aiosqlite = None

from redbot.core import Config


@dataclass
class Memory:
    """A single memory item."""
    scope: str  # "user" or "channel"
    guild_id: str
    channel_id: Optional[str] = None
    user_id: Optional[str] = None
    key: str = ""
    value: str = ""
    source: str = "user_input"  # "user_input" or "web"
    confidence: Optional[float] = None
    mem_id: Optional[str] = None
    updated_at: Optional[datetime] = None

    def generate_id(self) -> str:
        """Generate deterministic memory ID for deduplication."""
        id_parts = [
            self.guild_id,
            self.scope,
            self.user_id or "",
            self.channel_id or "",
            self.key,
            self.value.strip().lower()
        ]
        id_string = "|".join(id_parts)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create Memory from dictionary."""
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class MemoryStorage:
    """Handles database operations for memories."""
    
    def __init__(self, config: Config):
        self.config = config
        self._db_path: Optional[str] = None
        
    async def initialize(self):
        """Initialize the database."""
        if aiosqlite is None:
            raise RuntimeError("aiosqlite is required for memory storage. Please install it with: pip install aiosqlite")
            
        # Use Red's data directory for storing the database
        from redbot.core.data_manager import cog_data_path
        try:
            data_dir = cog_data_path(cog_instance=None)  # Get base data directory
            self._db_path = str(data_dir / "gpt5assistant" / "memories.db")
            # Ensure directory exists
            (data_dir / "gpt5assistant").mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to current directory if data manager fails
            self._db_path = "memories.db"
        
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    guild_id TEXT NOT NULL,
                    scope TEXT CHECK(scope IN ('user','channel')) NOT NULL,
                    user_id TEXT,
                    channel_id TEXT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL,
                    mem_id TEXT NOT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(guild_id, scope, COALESCE(user_id,''), COALESCE(channel_id,''), key)
                )
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_lookup 
                ON memories(guild_id, scope, user_id, channel_id)
            ''')
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_mem_id 
                ON memories(mem_id)
            ''')
            await db.commit()

    async def bulk_upsert(self, memories: List[Memory]) -> List[Memory]:
        """Bulk upsert memories to database. Returns list with generated IDs."""
        if not memories:
            return []
            
        # Generate IDs and timestamps for memories that don't have them
        now = datetime.now(timezone.utc)
        for memory in memories:
            if not memory.mem_id:
                memory.mem_id = memory.generate_id()
            if not memory.updated_at:
                memory.updated_at = now
        
        async with aiosqlite.connect(self._db_path) as db:
            # Use INSERT OR REPLACE for upsert behavior
            for memory in memories:
                await db.execute('''
                    INSERT OR REPLACE INTO memories 
                    (guild_id, scope, user_id, channel_id, key, value, source, confidence, mem_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    memory.guild_id,
                    memory.scope,
                    memory.user_id,
                    memory.channel_id,
                    memory.key,
                    memory.value,
                    memory.source,
                    memory.confidence,
                    memory.mem_id,
                    memory.updated_at.isoformat()
                ))
            await db.commit()
            
        return memories

    async def fetch_scope_memories(self, guild_id: str, scope: str, 
                                 user_id: Optional[str] = None, 
                                 channel_id: Optional[str] = None) -> List[Memory]:
        """Fetch all memories for a specific scope."""
        async with aiosqlite.connect(self._db_path) as db:
            query = '''
                SELECT guild_id, scope, user_id, channel_id, key, value, source, confidence, mem_id, updated_at
                FROM memories 
                WHERE guild_id = ? AND scope = ?
            '''
            params = [guild_id, scope]
            
            if scope == "user" and user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            elif scope == "channel" and channel_id:
                query += " AND channel_id = ?"
                params.append(channel_id)
                
            query += " ORDER BY updated_at DESC"
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
        memories = []
        for row in rows:
            memory = Memory(
                guild_id=row[0],
                scope=row[1],
                user_id=row[2],
                channel_id=row[3],
                key=row[4],
                value=row[5],
                source=row[6],
                confidence=row[7],
                mem_id=row[8],
                updated_at=datetime.fromisoformat(row[9]) if row[9] else None
            )
            memories.append(memory)
            
        return memories

    async def delete_memories(self, guild_id: str, scope: str,
                            user_id: Optional[str] = None,
                            channel_id: Optional[str] = None,
                            key: Optional[str] = None) -> int:
        """Delete memories matching criteria. Returns count of deleted rows."""
        async with aiosqlite.connect(self._db_path) as db:
            query = "DELETE FROM memories WHERE guild_id = ? AND scope = ?"
            params = [guild_id, scope]
            
            if scope == "user" and user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            elif scope == "channel" and channel_id:
                query += " AND channel_id = ?"
                params.append(channel_id)
                
            if key:
                query += " AND key = ?"
                params.append(key)
                
            cursor = await db.execute(query, params)
            deleted_count = cursor.rowcount
            await db.commit()
            
        return deleted_count

    async def get_guild_stats(self, guild_id: str) -> Dict[str, int]:
        """Get memory statistics for a guild."""
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute('''
                SELECT scope, COUNT(*) 
                FROM memories 
                WHERE guild_id = ? 
                GROUP BY scope
            ''', (guild_id,)) as cursor:
                rows = await cursor.fetchall()
                
        stats = {"user": 0, "channel": 0, "total": 0}
        for scope, count in rows:
            stats[scope] = count
            stats["total"] += count
            
        return stats


@dataclass  
class ScopeGroup:
    """Groups memories by scope target for vector store operations."""
    guild_id: str
    scope: str  # "user" or "channel"
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    
    @property
    def kind(self) -> str:
        return self.scope
        
    def get_filename(self) -> str:
        """Generate filename for vector store."""
        if self.scope == "user":
            return f"user_mem__{self.guild_id}__{self.user_id}.md"
        else:
            return f"chan_mem__{self.guild_id}__{self.channel_id}.md"
            
    def get_metadata(self) -> Dict[str, str]:
        """Generate metadata attributes for vector store."""
        metadata = {
            "guild_id": self.guild_id,
            "kind": self.scope
        }
        if self.user_id:
            metadata["user_id"] = self.user_id
        if self.channel_id:
            metadata["channel_id"] = self.channel_id
        return metadata


def group_by_scope(memories: List[Memory]) -> List[ScopeGroup]:
    """Group memories by scope target."""
    groups = {}
    
    for memory in memories:
        if memory.scope == "user":
            key = (memory.guild_id, "user", memory.user_id, None)
        else:
            key = (memory.guild_id, "channel", None, memory.channel_id)
            
        if key not in groups:
            groups[key] = ScopeGroup(
                guild_id=memory.guild_id,
                scope=memory.scope,
                user_id=memory.user_id if memory.scope == "user" else None,
                channel_id=memory.channel_id if memory.scope == "channel" else None
            )
            
    return list(groups.values())


def render_profile_markdown(group: ScopeGroup, memories: List[Memory]) -> str:
    """Render a markdown profile for a scope group."""
    if group.scope == "user":
        title = f"# User Profile: <@{group.user_id}>\n\n"
        subtitle = f"Guild: {group.guild_id}\n\n"
    else:
        title = f"# Channel Profile: <#{group.channel_id}>\n\n"
        subtitle = f"Guild: {group.guild_id}\n\n"
        
    content = title + subtitle
    
    if not memories:
        content += "No memories stored.\n"
        return content
        
    # Group by key for better organization
    by_key = {}
    for memory in memories:
        if memory.key not in by_key:
            by_key[memory.key] = []
        by_key[memory.key].append(memory)
        
    for key, key_memories in by_key.items():
        content += f"## {key}\n\n"
        
        for memory in key_memories:
            confidence_str = f" (confidence: {memory.confidence:.2f})" if memory.confidence else ""
            source_str = f" [source: {memory.source}]" if memory.source else ""
            updated_str = memory.updated_at.strftime("%Y-%m-%d") if memory.updated_at else ""
            
            content += f"- {memory.value}{confidence_str}{source_str}"
            if updated_str:
                content += f" _{updated_str}_"
            content += "\n"
            
        content += "\n"
        
    # Keep profile under size limit (~10KB)
    if len(content.encode()) > 10000:
        content = content[:9900] + "\n\n... (truncated for size)"
        
    return content