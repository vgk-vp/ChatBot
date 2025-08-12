"""Redis-based memory management for conversation history and context."""

import json
import redis
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os

class Config:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "50"))

config = Config()

class RedisMemory:
    def __init__(self):
        self.client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Serialize a LangChain message to a dictionary."""
        return {
            "type": message.__class__.__name__,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        }
    
    def _deserialize_message(self, data: Dict[str, Any]) -> BaseMessage:
        """Deserialize a dictionary to a LangChain message."""
        if data["type"] == "HumanMessage":
            return HumanMessage(content=data["content"])
        elif data["type"] == "AIMessage":
            return AIMessage(content=data["content"])
        else:
            return HumanMessage(content=data["content"])  # fallback
    
    def save_conversation(self, session_id: str, messages: List[BaseMessage]):
        """Save conversation messages to Redis."""
        key = f"conversation:{session_id}"
        serialized_messages = [self._serialize_message(msg) for msg in messages]
        
        # Store as a list in Redis
        self.client.delete(key)  # Clear existing
        for msg in serialized_messages:
            self.client.rpush(key, json.dumps(msg))
        
        # Set expiration (30 days)
        self.client.expire(key, 30 * 24 * 3600)
    
    def get_conversation(self, session_id: str) -> List[BaseMessage]:
        """Retrieve conversation messages from Redis."""
        key = f"conversation:{session_id}"
        raw_messages = self.client.lrange(key, 0, -1)
        
        messages = []
        for raw_msg in raw_messages:
            try:
                data = json.loads(raw_msg)
                messages.append(self._deserialize_message(data))
            except json.JSONDecodeError:
                continue
        
        return messages
    
    def add_message(self, session_id: str, message: BaseMessage):
        """Add a single message to the conversation."""
        key = f"conversation:{session_id}"
        serialized_msg = self._serialize_message(message)
        self.client.rpush(key, json.dumps(serialized_msg))
        
        # Trim conversation if too long
        self.client.ltrim(key, -config.MAX_MEMORY_MESSAGES, -1)
        
        # Reset expiration
        self.client.expire(key, 30 * 24 * 3600)
    
    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[BaseMessage]:
        """Get the most recent messages from a conversation."""
        key = f"conversation:{session_id}"
        raw_messages = self.client.lrange(key, -limit, -1)
        
        messages = []
        for raw_msg in raw_messages:
            try:
                data = json.loads(raw_msg)
                messages.append(self._deserialize_message(data))
            except json.JSONDecodeError:
                continue
        
        return messages
    
    def save_user_context(self, session_id: str, context: Dict[str, Any]):
        """Save user context information."""
        key = f"context:{session_id}"
        context["updated_at"] = datetime.now().isoformat()
        self.client.set(key, json.dumps(context), ex=30 * 24 * 3600)
    
    def get_user_context(self, session_id: str) -> Dict[str, Any]:
        """Retrieve user context information."""
        key = f"context:{session_id}"
        raw_context = self.client.get(key)
        
        if raw_context:
            try:
                return json.loads(raw_context)
            except json.JSONDecodeError:
                pass
        
        return {}
    
    def save_task(self, task_id: str, task_data: Dict[str, Any]):
        """Save task information."""
        key = f"task:{task_id}"
        task_data["created_at"] = datetime.now().isoformat()
        self.client.set(key, json.dumps(task_data), ex=7 * 24 * 3600)  # 7 days
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task information."""
        key = f"task:{task_id}"
        raw_task = self.client.get(key)
        
        if raw_task:
            try:
                return json.loads(raw_task)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def get_user_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a user session."""
        pattern = f"task:*"
        tasks = []
        
        for key in self.client.scan_iter(match=pattern):
            raw_task = self.client.get(key)
            if raw_task:
                try:
                    task_data = json.loads(raw_task)
                    if task_data.get("session_id") == session_id:
                        task_data["task_id"] = key.split(":", 1)[1]
                        tasks.append(task_data)
                except json.JSONDecodeError:
                    continue
        
        return sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        key = f"conversation:{session_id}"
        self.client.delete(key)
    
    def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """Get conversation summary if available."""
        key = f"summary:{session_id}"
        return self.client.get(key)
    
    def save_conversation_summary(self, session_id: str, summary: str):
        """Save conversation summary."""
        key = f"summary:{session_id}"
        self.client.set(key, summary, ex=30 * 24 * 3600)