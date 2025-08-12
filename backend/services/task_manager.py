"""AI Task Manager service for converting natural language to structured tasks."""

import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import os

class Config:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

config = Config()

from memory.redis_memory import RedisMemory

class TaskManager:
    def __init__(self):
        self.llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.1)
        self.memory = RedisMemory()
        
    def parse_natural_language_task(self, text: str, session_id: str) -> Dict[str, Any]:
        """Convert natural language input into structured task data."""
        
        system_prompt = """You are a task management AI. Convert natural language input into structured task data.

Extract the following information and respond with ONLY a JSON object:
{
    "title": "Brief task title",
    "description": "Detailed description",
    "priority": "high|medium|low",
    "category": "work|personal|health|learning|other",
    "due_date": "YYYY-MM-DD or null if not specified",
    "estimated_duration": "duration in minutes or null",
    "subtasks": ["list", "of", "subtasks"],
    "tags": ["relevant", "tags"],
    "status": "pending"
}

Examples:
Input: "I need to finish the quarterly report by Friday and send it to my manager"
Output: {"title": "Complete quarterly report", "description": "Finish the quarterly report and send it to manager", "priority": "high", "category": "work", "due_date": "2024-01-19", "estimated_duration": 240, "subtasks": ["Finish quarterly report", "Send report to manager"], "tags": ["report", "quarterly", "manager"], "status": "pending"}

Input: "Buy groceries tomorrow - milk, bread, eggs"
Output: {"title": "Buy groceries", "description": "Purchase milk, bread, and eggs", "priority": "medium", "category": "personal", "due_date": "2024-01-16", "estimated_duration": 60, "subtasks": ["Buy milk", "Buy bread", "Buy eggs"], "tags": ["groceries", "shopping"], "status": "pending"}

Now process this input:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=text)
            ]
            
            response = self.llm.invoke(messages)
            
            # Try to parse JSON from response
            try:
                task_data = json.loads(response.content.strip())
            except json.JSONDecodeError:
                # Fallback parsing
                task_data = self._fallback_task_parsing(text)
            
            # Add metadata
            task_id = str(uuid.uuid4())
            task_data.update({
                "task_id": task_id,
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            
            # Save to memory
            self.memory.save_task(task_id, task_data)
            
            return task_data
            
        except Exception as e:
            # Fallback task creation
            return self._fallback_task_parsing(text, session_id)
    
    def _fallback_task_parsing(self, text: str, session_id: str = None) -> Dict[str, Any]:
        """Fallback method for task parsing when LLM fails."""
        task_id = str(uuid.uuid4())
        
        # Simple heuristics for priority
        priority = "medium"
        if any(word in text.lower() for word in ["urgent", "asap", "immediately", "critical"]):
            priority = "high"
        elif any(word in text.lower() for word in ["later", "sometime", "eventually"]):
            priority = "low"
        
        # Simple category detection
        category = "other"
        if any(word in text.lower() for word in ["work", "office", "meeting", "report", "project"]):
            category = "work"
        elif any(word in text.lower() for word in ["buy", "shop", "grocery", "personal"]):
            category = "personal"
        elif any(word in text.lower() for word in ["learn", "study", "read", "course"]):
            category = "learning"
        elif any(word in text.lower() for word in ["exercise", "health", "doctor", "gym"]):
            category = "health"
        
        task_data = {
            "task_id": task_id,
            "session_id": session_id,
            "title": text[:50] + "..." if len(text) > 50 else text,
            "description": text,
            "priority": priority,
            "category": category,
            "due_date": None,
            "estimated_duration": None,
            "subtasks": [],
            "tags": [],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        if session_id:
            self.memory.save_task(task_id, task_data)
        
        return task_data
    
    def get_user_tasks(self, session_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all tasks for a user, optionally filtered by status."""
        tasks = self.memory.get_user_tasks(session_id)
        
        if status:
            tasks = [task for task in tasks if task.get("status") == status]
        
        return tasks
    
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        task = self.memory.get_task(task_id)
        if task:
            task["status"] = status
            task["updated_at"] = datetime.now().isoformat()
            self.memory.save_task(task_id, task)
            return True
        return False
    
    def get_task_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of user's tasks."""
        tasks = self.get_user_tasks(session_id)
        
        summary = {
            "total_tasks": len(tasks),
            "pending": len([t for t in tasks if t.get("status") == "pending"]),
            "in_progress": len([t for t in tasks if t.get("status") == "in_progress"]),
            "completed": len([t for t in tasks if t.get("status") == "completed"]),
            "high_priority": len([t for t in tasks if t.get("priority") == "high"]),
            "categories": {}
        }
        
        # Count by category
        for task in tasks:
            category = task.get("category", "other")
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
        
        return summary
    
    def suggest_task_improvements(self, task_data: Dict[str, Any]) -> List[str]:
        """Suggest improvements for a task."""
        suggestions = []
        
        if not task_data.get("due_date"):
            suggestions.append("Consider adding a due date to help with prioritization")
        
        if not task_data.get("estimated_duration"):
            suggestions.append("Estimate how long this task might take")
        
        if len(task_data.get("subtasks", [])) == 0 and len(task_data.get("description", "")) > 100:
            suggestions.append("Break this task down into smaller subtasks")
        
        if task_data.get("priority") == "high" and not task_data.get("due_date"):
            suggestions.append("High priority tasks should have due dates")
        
        return suggestions