"""Multi-step workflow agent using LangGraph for conditional flows."""

import json
import re
from typing import Dict, Any, List, Literal, TypedDict, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import os

class Config:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

config = Config()

from tools.agent_tools import TOOLS, get_tool_descriptions
from memory.redis_memory import RedisMemory

class WorkflowState(TypedDict, total=False):
    messages: List[Any]
    session_id: str
    current_step: str
    workflow_type: str
    context: Dict[str, Any]
    tool_results: Dict[str, Any]
    user_feedback: Optional[str]
    steps_completed: int
    max_steps: int

class WorkflowAgent:
    def __init__(self):
        self.llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.1)
        self.memory = RedisMemory()
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the workflow graph with conditional routing."""
        builder = StateGraph(WorkflowState)
        
        # Add nodes
        builder.add_node("classifier", self._classify_request)
        builder.add_node("research_workflow", self._research_workflow)
        builder.add_node("task_workflow", self._task_workflow)
        builder.add_node("qa_workflow", self._qa_workflow)
        builder.add_node("general_chat", self._general_chat)
        builder.add_node("tool_executor", self._execute_tool)
        builder.add_node("feedback_handler", self._handle_feedback)
        builder.add_node("summarizer", self._summarize_results)
        
        # Set entry point
        builder.set_entry_point("classifier")
        
        # Add conditional edges
        builder.add_conditional_edges(
            "classifier",
            self._route_workflow,
            {
                "research": "research_workflow",
                "task": "task_workflow", 
                "qa": "qa_workflow",
                "chat": "general_chat"
            }
        )
        
        # Add edges for each workflow
        builder.add_conditional_edges(
            "research_workflow",
            self._should_continue_research,
            {"tool": "tool_executor", "feedback": "feedback_handler", "end": "summarizer"}
        )
        
        builder.add_conditional_edges(
            "task_workflow", 
            self._should_continue_task,
            {"tool": "tool_executor", "feedback": "feedback_handler", "end": "summarizer"}
        )
        
        builder.add_conditional_edges(
            "qa_workflow",
            self._should_continue_qa, 
            {"tool": "tool_executor", "feedback": "feedback_handler", "end": "summarizer"}
        )
        
        builder.add_edge("general_chat", END)
        builder.add_edge("tool_executor", "classifier")
        builder.add_edge("feedback_handler", "classifier")
        builder.add_edge("summarizer", END)
        
        return builder.compile()
    
    def _classify_request(self, state: WorkflowState) -> WorkflowState:
        """Classify the user request to determine workflow type."""
        last_message = state["messages"][-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        classification_prompt = f"""Classify this user request into one of these categories:

1. "research" - User wants to search for information, get current data, or research a topic
2. "task" - User wants to create, manage, or organize tasks and to-dos
3. "qa" - User wants to ask questions about uploaded documents or PDFs
4. "chat" - General conversation, greetings, or simple questions

User request: "{user_input}"

Respond with only the category name: research, task, qa, or chat"""

        try:
            response = self.llm.invoke([HumanMessage(content=classification_prompt)])
            workflow_type = response.content.strip().lower()
            
            if workflow_type not in ["research", "task", "qa", "chat"]:
                workflow_type = "chat"
                
        except Exception:
            workflow_type = "chat"
        
        state["workflow_type"] = workflow_type
        state["current_step"] = "classified"
        state["steps_completed"] = 1
        state["max_steps"] = 10
        
        return state
    
    def _route_workflow(self, state: WorkflowState) -> str:
        """Route to appropriate workflow based on classification."""
        return state.get("workflow_type", "chat")
    
    def _research_workflow(self, state: WorkflowState) -> WorkflowState:
        """Handle research-oriented requests."""
        user_input = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
        
        research_prompt = f"""You are a research assistant. Analyze this request and determine what tools to use:

Available tools: {', '.join(TOOLS.keys())}

User request: "{user_input}"

Respond with JSON in this format:
{{
    "action": "tool",
    "tool_name": "tool_name_here",
    "tool_input": "input_for_tool",
    "reasoning": "why this tool is needed"
}}

OR if you have enough information to provide a final answer:
{{
    "action": "final",
    "final_answer": "your complete response"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=research_prompt)])
            action_data = self._parse_json_response(response.content)
            
            if action_data.get("action") == "tool":
                state["context"]["pending_tool"] = {
                    "name": action_data.get("tool_name"),
                    "input": action_data.get("tool_input"),
                    "reasoning": action_data.get("reasoning")
                }
                state["current_step"] = "tool_needed"
            else:
                state["context"]["final_answer"] = action_data.get("final_answer", "I'm not sure how to help with that.")
                state["current_step"] = "complete"
                
        except Exception as e:
            state["context"]["final_answer"] = f"I encountered an error processing your research request: {str(e)}"
            state["current_step"] = "complete"
        
        state["steps_completed"] += 1
        return state
    
    def _task_workflow(self, state: WorkflowState) -> WorkflowState:
        """Handle task management requests."""
        from services.task_manager import TaskManager
        
        user_input = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
        session_id = state.get("session_id", "default")
        
        task_manager = TaskManager()
        
        # Determine if this is task creation or task management
        if any(word in user_input.lower() for word in ["create", "add", "new task", "todo", "need to", "have to"]):
            # Create new task
            task_data = task_manager.parse_natural_language_task(user_input, session_id)
            suggestions = task_manager.suggest_task_improvements(task_data)
            
            response = f"✅ Task created successfully!\n\n"
            response += f"**{task_data['title']}**\n"
            response += f"Priority: {task_data['priority'].title()}\n"
            response += f"Category: {task_data['category'].title()}\n"
            
            if task_data.get('due_date'):
                response += f"Due: {task_data['due_date']}\n"
            
            if task_data.get('subtasks'):
                response += f"\nSubtasks:\n"
                for subtask in task_data['subtasks']:
                    response += f"• {subtask}\n"
            
            if suggestions:
                response += f"\n💡 Suggestions:\n"
                for suggestion in suggestions:
                    response += f"• {suggestion}\n"
            
            state["context"]["final_answer"] = response
            
        elif any(word in user_input.lower() for word in ["list", "show", "my tasks", "what tasks"]):
            # List tasks
            tasks = task_manager.get_user_tasks(session_id)
            summary = task_manager.get_task_summary(session_id)
            
            response = f"📋 **Your Tasks Summary**\n\n"
            response += f"Total: {summary['total_tasks']} | "
            response += f"Pending: {summary['pending']} | "
            response += f"Completed: {summary['completed']}\n\n"
            
            if tasks:
                for task in tasks[:10]:  # Show first 10 tasks
                    status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(task.get("status"), "❓")
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.get("priority"), "⚪")
                    
                    response += f"{status_emoji} {priority_emoji} **{task['title']}**\n"
                    if task.get('due_date'):
                        response += f"   Due: {task['due_date']}\n"
                    response += "\n"
            else:
                response += "No tasks found. Create your first task by saying something like 'I need to buy groceries tomorrow'."
            
            state["context"]["final_answer"] = response
        
        else:
            # General task-related query
            state["context"]["final_answer"] = "I can help you create and manage tasks. Try saying 'I need to...' or 'Add a task to...' or 'Show my tasks'."
        
        state["current_step"] = "complete"
        state["steps_completed"] += 1
        return state
    
    def _qa_workflow(self, state: WorkflowState) -> WorkflowState:
        """Handle PDF QA requests."""
        from services.pdf_service import PDFService
        
        user_input = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
        
        pdf_service = PDFService()
        available_pdfs = pdf_service.get_available_pdfs()
        
        if not available_pdfs:
            state["context"]["final_answer"] = "📄 No PDFs have been uploaded yet. Please upload a PDF first to ask questions about it."
        else:
            # For now, use the first available PDF collection
            collection_name = available_pdfs[0]
            
            try:
                results = pdf_service.query_pdf(collection_name, user_input, k=3)
                
                if results:
                    response = f"📄 **Based on the uploaded PDF:**\n\n"
                    
                    # Use the most relevant chunks to generate an answer
                    context = "\n\n".join([r["content"] for r in results[:2]])
                    
                    answer_prompt = f"""Based on the following context from a PDF document, answer the user's question.

Context:
{context}

Question: {user_input}

Provide a clear, accurate answer based only on the information in the context. If the context doesn't contain enough information to answer the question, say so."""

                    try:
                        answer_response = self.llm.invoke([HumanMessage(content=answer_prompt)])
                        response += answer_response.content
                        
                        # Add source information
                        response += f"\n\n**Sources:**\n"
                        for i, result in enumerate(results[:2], 1):
                            filename = result["metadata"].get("filename", "Unknown")
                            response += f"{i}. {filename} (similarity: {result['similarity_score']:.2f})\n"
                            
                    except Exception as e:
                        response += f"Error generating answer: {str(e)}"
                        
                else:
                    response = "I couldn't find relevant information in the uploaded PDF to answer your question."
                
                state["context"]["final_answer"] = response
                
            except Exception as e:
                state["context"]["final_answer"] = f"Error querying PDF: {str(e)}"
        
        state["current_step"] = "complete"
        state["steps_completed"] += 1
        return state
    
    def _general_chat(self, state: WorkflowState) -> WorkflowState:
        """Handle general conversation."""
        user_input = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
        session_id = state.get("session_id", "default")
        
        # Get conversation history for context
        recent_messages = self.memory.get_recent_messages(session_id, limit=6)
        
        # Build context from recent messages
        context_parts = []
        for msg in recent_messages[-4:]:  # Last 4 messages for context
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        
        conversation_context = "\n".join(context_parts) if context_parts else ""
        
        chat_prompt = f"""You are a helpful AI assistant. Respond naturally to the user's message.

Previous conversation context:
{conversation_context}

Current message: {user_input}

Provide a helpful, friendly response. If the user is asking for specific information that would require web search, calculations, or document analysis, suggest they ask me to help with research, math, or document questions."""

        try:
            response = self.llm.invoke([HumanMessage(content=chat_prompt)])
            final_response = response.content
        except Exception as e:
            final_response = f"I'm sorry, I encountered an error: {str(e)}"
        
        # Add the response to messages and save to memory
        state["messages"].append(AIMessage(content=final_response))
        self.memory.add_message(session_id, HumanMessage(content=user_input))
        self.memory.add_message(session_id, AIMessage(content=final_response))
        
        return state
    
    def _execute_tool(self, state: WorkflowState) -> WorkflowState:
        """Execute the requested tool."""
        pending_tool = state["context"].get("pending_tool")
        
        if not pending_tool:
            state["current_step"] = "complete"
            return state
        
        tool_name = pending_tool["name"]
        tool_input = pending_tool["input"]
        
        if tool_name in TOOLS:
            try:
                tool_func = TOOLS[tool_name]
                result = tool_func.invoke(tool_input) if hasattr(tool_func, "invoke") else tool_func(tool_input)
                
                state["tool_results"][tool_name] = result
                state["context"]["last_tool_result"] = result
                state["current_step"] = "tool_completed"
                
            except Exception as e:
                state["tool_results"][tool_name] = f"Error: {str(e)}"
                state["context"]["last_tool_result"] = f"Error: {str(e)}"
                state["current_step"] = "tool_error"
        else:
            state["current_step"] = "tool_error"
            state["context"]["last_tool_result"] = f"Unknown tool: {tool_name}"
        
        state["steps_completed"] += 1
        return state
    
    def _handle_feedback(self, state: WorkflowState) -> WorkflowState:
        """Handle user feedback and routing."""
        # This would handle user feedback in interactive scenarios
        state["current_step"] = "feedback_processed"
        state["steps_completed"] += 1
        return state
    
    def _summarize_results(self, state: WorkflowState) -> WorkflowState:
        """Summarize and format final results."""
        workflow_type = state.get("workflow_type", "unknown")
        
        if "final_answer" in state["context"]:
            final_response = state["context"]["final_answer"]
        elif "last_tool_result" in state["context"]:
            # Format tool result into a nice response
            result = state["context"]["last_tool_result"]
            final_response = f"Here's what I found:\n\n{result}"
        else:
            final_response = "I completed the workflow but don't have a specific result to show."
        
        # Add workflow completion message
        state["messages"].append(AIMessage(content=final_response))
        
        # Save to memory if we have a session
        if state.get("session_id"):
            user_msg = state["messages"][0] if state["messages"] else HumanMessage(content="")
            self.memory.add_message(state["session_id"], user_msg)
            self.memory.add_message(state["session_id"], AIMessage(content=final_response))
        
        return state
    
    def _should_continue_research(self, state: WorkflowState) -> str:
        """Determine next step for research workflow."""
        if state["steps_completed"] >= state["max_steps"]:
            return "end"
        
        current_step = state.get("current_step", "")
        
        if current_step == "tool_needed":
            return "tool"
        elif current_step == "complete":
            return "end"
        elif state.get("user_feedback"):
            return "feedback"
        else:
            return "end"
    
    def _should_continue_task(self, state: WorkflowState) -> str:
        """Determine next step for task workflow."""
        return "end"  # Task workflow is usually single-step
    
    def _should_continue_qa(self, state: WorkflowState) -> str:
        """Determine next step for QA workflow."""
        return "end"  # QA workflow is usually single-step
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]*"action"[^}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            # Fallback
            return {"action": "final", "final_answer": response}
    
    def process_request(self, user_input: str, session_id: str = "default") -> str:
        """Process a user request through the workflow."""
        initial_state = WorkflowState(
            messages=[HumanMessage(content=user_input)],
            session_id=session_id,
            current_step="start",
            workflow_type="",
            context={},
            tool_results={},
            user_feedback=None,
            steps_completed=0,
            max_steps=10
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Extract final response
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return "I processed your request but don't have a response to show."
                
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"