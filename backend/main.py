"""
Enhanced ChatBot with multiple AI capabilities:
- PDF QA Bot with embeddings
- AI Task Manager 
- Agent with Tools (Search + Calculate + Summarize)
- Multi-step Workflow with LangGraph
- Memory Chatbot with conversation history
"""

import os
import uuid
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Configuration
import os

class Config:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
    MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "50"))
    MEMORY_SUMMARY_THRESHOLD = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "20"))

config = Config()

# Import our services and agents
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.redis_memory import RedisMemory
from services.pdf_service import PDFService
from services.task_manager import TaskManager
from agents.workflow_agent import WorkflowAgent
from tools.agent_tools import get_tool_descriptions

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced AI ChatBot",
    description="Multi-capability AI system with PDF QA, Task Management, Agent Tools, Workflows, and Memory",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
memory = RedisMemory()
pdf_service = PDFService()
task_manager = TaskManager()
workflow_agent = WorkflowAgent()

# Ensure upload directory exists
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: Optional[str] = "auto"  # auto, chat, research, task, pdf_qa

class ChatResponse(BaseModel):
    response: str
    session_id: str
    mode: str
    context: Optional[Dict[str, Any]] = None

class TaskRequest(BaseModel):
    task_description: str
    session_id: Optional[str] = None

class TaskResponse(BaseModel):
    task: Dict[str, Any]
    suggestions: List[str]

class PDFQueryRequest(BaseModel):
    question: str
    collection_name: Optional[str] = None

class PDFQueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Redis connection
        memory.client.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "model": config.OLLAMA_MODEL,
        "redis": redis_status,
        "services": ["chat", "pdf_qa", "task_manager", "agent_tools", "workflow"]
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with multi-modal capabilities."""
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Process through workflow agent for intelligent routing
        response = workflow_agent.process_request(request.message, session_id)
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            mode="workflow",
            context={"processed": True}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

# PDF upload and processing
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF for QA."""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    if file.size > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {config.MAX_FILE_SIZE} bytes")
    
    try:
        # Save uploaded file
        file_path = os.path.join(config.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF
        collection_name = pdf_service.process_pdf(file_path, file.filename)
        
        return {
            "message": "PDF uploaded and processed successfully",
            "collection_name": collection_name,
            "filename": file.filename,
            "size": len(content)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# PDF QA endpoint
@app.post("/pdf-qa", response_model=PDFQueryResponse)
async def pdf_qa(request: PDFQueryRequest):
    """Ask questions about uploaded PDFs."""
    
    try:
        # Get available PDFs if no collection specified
        if not request.collection_name:
            available_pdfs = pdf_service.get_available_pdfs()
            if not available_pdfs:
                raise HTTPException(status_code=404, detail="No PDFs available. Please upload a PDF first.")
            collection_name = available_pdfs[0]  # Use first available
        else:
            collection_name = request.collection_name
        
        # Query the PDF
        results = pdf_service.query_pdf(collection_name, request.question, k=3)
        
        if not results:
            return PDFQueryResponse(
                answer="I couldn't find relevant information in the PDF to answer your question.",
                sources=[]
            )
        
        # Generate answer using the most relevant chunks
        from langchain_community.chat_models import ChatOllama
        from langchain_core.messages import HumanMessage
        
        llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.1)
        context = "\n\n".join([r["content"] for r in results[:2]])
        
        answer_prompt = f"""Based on the following context from a PDF document, answer the user's question clearly and accurately.

Context:
{context}

Question: {request.question}

Provide a comprehensive answer based on the information in the context. If the context doesn't contain enough information, say so clearly."""

        response = llm.invoke([HumanMessage(content=answer_prompt)])
        
        # Format sources
        sources = []
        for result in results:
            sources.append({
                "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                "metadata": result["metadata"],
                "similarity_score": result["similarity_score"]
            })
        
        return PDFQueryResponse(
            answer=response.content,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF QA error: {str(e)}")

# Task management endpoints
@app.post("/create-task", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Create a structured task from natural language."""
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        task_data = task_manager.parse_natural_language_task(request.task_description, session_id)
        suggestions = task_manager.suggest_task_improvements(task_data)
        
        return TaskResponse(
            task=task_data,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task creation error: {str(e)}")

@app.get("/tasks/{session_id}")
async def get_tasks(session_id: str, status: Optional[str] = None):
    """Get tasks for a session."""
    
    try:
        tasks = task_manager.get_user_tasks(session_id, status)
        summary = task_manager.get_task_summary(session_id)
        
        return {
            "tasks": tasks,
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving tasks: {str(e)}")

@app.put("/tasks/{task_id}/status")
async def update_task_status(task_id: str, status: str):
    """Update task status."""
    
    valid_statuses = ["pending", "in_progress", "completed", "cancelled"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
    
    try:
        success = task_manager.update_task_status(task_id, status)
        if success:
            return {"message": "Task status updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating task: {str(e)}")

# Memory and conversation endpoints
@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str, limit: int = 20):
    """Get conversation history for a session."""
    
    try:
        messages = memory.get_recent_messages(session_id, limit)
        context = memory.get_user_context(session_id)
        
        # Convert messages to serializable format
        serialized_messages = []
        for msg in messages:
            serialized_messages.append({
                "type": msg.__class__.__name__,
                "content": msg.content,
                "timestamp": "recent"  # You could add timestamps to the memory system
            })
        
        return {
            "messages": serialized_messages,
            "context": context,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    
    try:
        memory.clear_conversation(session_id)
        return {"message": "Conversation cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

# Available tools endpoint
@app.get("/tools")
async def get_available_tools():
    """Get list of available agent tools."""
    
    return {
        "tools": get_tool_descriptions(),
        "count": len(get_tool_descriptions())
    }

# Available PDFs endpoint
@app.get("/pdfs")
async def get_available_pdfs():
    """Get list of available PDF collections."""
    
    try:
        collections = pdf_service.get_available_pdfs()
        return {
            "collections": collections,
            "count": len(collections)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDFs: {str(e)}")

# System info endpoint
@app.get("/info")
async def get_system_info():
    """Get system information and capabilities."""
    
    return {
        "name": "Enhanced AI ChatBot",
        "version": "2.0.0",
        "capabilities": [
            "🧠 Memory Chatbot - Remembers conversations and builds context",
            "📄 PDF QA Bot - Ask questions from uploaded PDFs using embeddings", 
            "📅 AI Task Manager - Convert natural language to structured tasks",
            "🛠️ Agent with Tools - Search, calculate, summarize, and more",
            "🔄 Multi-step Workflow - Conditional flows with user feedback routing"
        ],
        "model": config.OLLAMA_MODEL,
        "tools": list(get_tool_descriptions().keys()),
        "endpoints": [
            "/chat - Main chat interface",
            "/upload-pdf - Upload PDF for QA",
            "/pdf-qa - Ask questions about PDFs",
            "/create-task - Create tasks from natural language",
            "/tasks/{session_id} - Get user tasks",
            "/conversation/{session_id} - Get conversation history",
            "/tools - List available tools"
        ]
    }

# Serve frontend
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )