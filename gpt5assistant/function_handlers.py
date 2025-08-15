from __future__ import annotations

import json
from typing import Dict, List, Any, Optional

from .memory_storage import Memory
from .vector_store_manager import MemoryManager


class FunctionCallHandler:
    """Handles function calls from OpenAI responses."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
    async def handle_function_call(self, function_name: str, arguments: str, 
                                 guild_id: str, channel_id: str, user_id: str) -> Dict[str, Any]:
        """Handle a function call and return the result."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}
            
        if function_name == "propose_memories":
            return await self._handle_propose_memories(args, guild_id, channel_id, user_id)
        elif function_name == "save_memories":
            return await self._handle_save_memories(args, guild_id, channel_id, user_id)
        else:
            return {"error": f"Unknown function: {function_name}"}
    
    async def _handle_propose_memories(self, args: Dict[str, Any], 
                                     guild_id: str, channel_id: str, user_id: str) -> Dict[str, Any]:
        """Handle propose_memories function call."""
        items = args.get("items", [])
        if not items:
            return {"error": "No items provided"}
            
        # Convert to Memory objects and validate
        memories = []
        for item in items:
            try:
                # Fill in context information if missing
                if not item.get("guild_id"):
                    item["guild_id"] = guild_id
                if not item.get("channel_id") and item.get("scope") == "channel":
                    item["channel_id"] = channel_id
                if not item.get("user_id") and item.get("scope") == "user":
                    item["user_id"] = user_id
                    
                memory = Memory.from_dict(item)
                memories.append(memory)
            except Exception as e:
                continue  # Skip invalid items
                
        # Stage memories (validate without writing)
        try:
            normalized_items = await self.memory_manager.propose_memories(memories)
            return {
                "status": "staged",
                "items": normalized_items,
                "count": len(normalized_items)
            }
        except Exception as e:
            return {"error": f"Failed to stage memories: {e}"}
    
    async def _handle_save_memories(self, args: Dict[str, Any], 
                                  guild_id: str, channel_id: str, user_id: str) -> Dict[str, Any]:
        """Handle save_memories function call."""
        items = args.get("items", [])
        if not items:
            return {"error": "No items provided"}
            
        # Convert to Memory objects
        memories = []
        for item in items:
            try:
                # Fill in context information if missing
                if not item.get("guild_id"):
                    item["guild_id"] = guild_id
                if not item.get("channel_id") and item.get("scope") == "channel":
                    item["channel_id"] = channel_id
                if not item.get("user_id") and item.get("scope") == "user":
                    item["user_id"] = user_id
                    
                memory = Memory.from_dict(item)
                memories.append(memory)
            except Exception as e:
                continue  # Skip invalid items
                
        # Save memories to database and vector store
        try:
            result = await self.memory_manager.save_memories(memories)
            return result
        except Exception as e:
            return {"error": f"Failed to save memories: {e}"}


def extract_function_calls(response_output) -> List[Dict[str, Any]]:
    """Extract function calls from OpenAI response output."""
    function_calls = []
    
    if not hasattr(response_output, '__iter__'):
        return function_calls
        
    for item in response_output:
        if hasattr(item, 'type') and item.type == "function_call":
            function_call = {
                "id": getattr(item, 'id', None),
                "name": getattr(item, 'name', None),
                "arguments": getattr(item, 'arguments', None)
            }
            if function_call["name"]:
                function_calls.append(function_call)
                
    return function_calls


def is_memory_function(function_name: str) -> bool:
    """Check if a function name is a memory-related function."""
    return function_name in ["propose_memories", "save_memories"]