# Enhanced AI ChatBot

A comprehensive AI-powered chatbot system with multiple capabilities including memory management, PDF QA, task management, agent tools, and smart workflows.

## 🚀 Features

### 🧠 Memory Chatbot
- Persistent conversation history using Redis
- Context-aware responses that build on previous interactions
- Session management with automatic cleanup
- Conversation summarization for long chats

### 📄 PDF QA Bot
- Upload and process PDF documents
- Ask questions about uploaded PDFs using embeddings
- Semantic search through document content
- Source attribution with similarity scores

### 📅 AI Task Manager
- Convert natural language descriptions into structured tasks
- Automatic priority and category detection
- Task suggestions and improvements
- Task status tracking and management

### 🛠️ Agent with Tools
- **Web Search**: Real-time information retrieval using DuckDuckGo
- **Calculator**: Safe mathematical expression evaluation
- **Text Summarizer**: AI-powered content summarization
- **Weather Info**: Location-based weather information
- **News Search**: Latest news on specific topics
- **Unit Converter**: Convert between different units
- **Time/Date**: Current time and date information

### 🔄 Multi-step Workflow
- Intelligent request classification and routing
- Conditional workflow execution using LangGraph
- Context preservation across workflow steps
- Error handling and recovery mechanisms

## 🏗️ Architecture

```
ChatBot/
├── backend/
│   ├── agents/
│   │   └── workflow_agent.py      # Multi-step workflow orchestration
│   ├── memory/
│   │   └── redis_memory.py        # Conversation and context storage
│   ├── services/
│   │   ├── pdf_service.py         # PDF processing and QA
│   │   └── task_manager.py        # Task creation and management
│   ├── tools/
│   │   └── agent_tools.py         # Search, calc, summarize tools
│   ├── config.py                  # Configuration management
│   ├── main.py                    # FastAPI application
│   └── requirements.txt           # Python dependencies
├── frontend/
│   └── index.html                 # Web interface
└── README.md
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework
- **LangChain**: LLM application framework
- **LangGraph**: Workflow orchestration
- **Ollama**: Local LLM inference
- **Redis**: Memory and session storage
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: Text embeddings

### Frontend
- **HTML5/CSS3/JavaScript**: Modern web interface
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Dynamic status indicators

## 📦 Installation

### Prerequisites
- Python 3.8+
- Redis server
- Ollama with a language model (e.g., llama3, mistral)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ChatBot
   ```

2. **Install Python dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Start Redis server**
   ```bash
   redis-server
   ```

4. **Install and start Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for instructions)
   ollama pull llama3  # or your preferred model
   ```

5. **Configure environment variables** (optional)
   ```bash
   export OLLAMA_MODEL="llama3"
   export REDIS_HOST="localhost"
   export REDIS_PORT="6379"
   ```

6. **Start the application**
   ```bash
   cd backend
   python main.py
   ```

7. **Access the web interface**
   Open your browser and go to `http://localhost:8000`

## 🎯 Usage Examples

### Chat Interface
```
User: "Search for the latest Python developments and summarize key updates"
Bot: [Searches web, finds information, provides summary with sources]

User: "I need to buy groceries tomorrow - milk, bread, eggs"
Bot: [Creates structured task with priority, category, and subtasks]

User: "Calculate compound interest: $5000 at 3.5% for 10 years"
Bot: [Performs calculation and shows result]
```

### PDF QA
1. Upload a PDF document
2. Ask questions like:
   - "What is the main topic of this document?"
   - "Summarize the key findings"
   - "What does it say about [specific topic]?"

### Task Management
- "Schedule dentist appointment next week"
- "Finish project presentation by Friday"
- "Exercise for 30 minutes daily"

## 🔧 API Endpoints

### Core Endpoints
- `POST /chat` - Main chat interface
- `POST /upload-pdf` - Upload PDF documents
- `POST /pdf-qa` - Ask questions about PDFs
- `POST /create-task` - Create tasks from natural language
- `GET /tasks/{session_id}` - Get user tasks
- `GET /conversation/{session_id}` - Get conversation history

### Utility Endpoints
- `GET /health` - System health check
- `GET /tools` - Available agent tools
- `GET /pdfs` - Available PDF collections
- `GET /info` - System information

## ⚙️ Configuration

### Environment Variables
- `OLLAMA_MODEL`: Language model to use (default: "llama3")
- `OLLAMA_BASE_URL`: Ollama server URL (default: "http://localhost:11434")
- `REDIS_HOST`: Redis server host (default: "localhost")
- `REDIS_PORT`: Redis server port (default: "6379")
- `REDIS_DB`: Redis database number (default: "0")
- `CHROMA_PERSIST_DIR`: ChromaDB storage directory (default: "./chroma_db")
- `EMBEDDING_MODEL`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `MAX_FILE_SIZE`: Maximum PDF file size in bytes (default: "10485760")
- `UPLOAD_DIR`: File upload directory (default: "./uploads")

### Model Configuration
The system uses Ollama for local LLM inference. Supported models include:
- llama3 (recommended)
- mistral
- qwen2.5
- codellama
- And other Ollama-compatible models

## 🔒 Security Considerations

- File upload validation and size limits
- Input sanitization for calculations
- Session isolation and cleanup
- Rate limiting (recommended for production)
- CORS configuration for production deployment

## 🚀 Deployment

### Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production
1. Use a production WSGI server (Gunicorn + Uvicorn)
2. Configure reverse proxy (Nginx)
3. Set up SSL/TLS certificates
4. Use managed Redis service
5. Configure proper logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis server is running: `redis-server`
   - Check Redis configuration and port

2. **Ollama Model Not Found**
   - Pull the model: `ollama pull llama3`
   - Check available models: `ollama list`

3. **PDF Processing Errors**
   - Ensure PDF is not password-protected
   - Check file size limits
   - Verify ChromaDB permissions

4. **Memory Issues**
   - Monitor Redis memory usage
   - Adjust conversation history limits
   - Clear old sessions periodically

### Performance Optimization

- Use SSD storage for ChromaDB
- Increase Redis memory allocation
- Configure Ollama with appropriate GPU settings
- Implement caching for frequent queries

## 📊 Monitoring

The system provides several monitoring endpoints:
- `/health` - Overall system health
- Redis metrics through Redis CLI
- Application logs for debugging
- Performance metrics for response times

## 🔮 Future Enhancements

- [ ] Multi-user authentication and authorization
- [ ] Advanced workflow templates
- [ ] Integration with external APIs (Google Calendar, Slack, etc.)
- [ ] Voice input/output capabilities
- [ ] Mobile application
- [ ] Advanced analytics and insights
- [ ] Custom tool creation interface
- [ ] Workflow sharing and templates

---

For more information, support, or feature requests, please open an issue in the repository.