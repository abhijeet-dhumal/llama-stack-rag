# 🦙 **RAG LlamaStack - Streamlit Edition**

> **A modern, intelligent RAG application with real-time document processing**  
> Built with LlamaStack orchestration and Streamlit for seamless AI interactions

## ⚡ **30-Second Start**

```bash
git clone https://github.com/yourusername/rag-llama-stack.git
cd rag-llama-stack
make setup                    # Step 1: Create virtual environment
source venv/bin/activate      # Step 2: Activate virtual environment
make install                  # Step 3: Install Python dependencies
make setup-mcp               # Step 4: Setup MCP server
make start                   # Step 5: Start all services + frontend
# Open: http://localhost:8501
```

**Alternative Flow (Manual Service Control):**
```bash
# Terminal 1: Start Ollama
make ollama

# Terminal 2: Start LlamaStack  
make llamastack

# Terminal 3: Start Streamlit frontend
source venv/bin/activate
make start-frontend
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LlamaStack](https://img.shields.io/badge/LlamaStack-Latest-green.svg)](https://github.com/meta-llama/llama-stack)

---

## 📋 **Table of Contents**

1. [🎯 Features](#-features)
2. [🏛️ System Architecture](#-system-architecture)
3. [🔄 Data Flow & Processing](#-data-flow--processing)
4. [🚀 Quick Start](#-quick-start)
5. [📊 Performance & Monitoring](#-performance--monitoring)
6. [🔧 Configuration](#-configuration)
7. [🐛 Troubleshooting](#-troubleshooting)

---

## 🎯 **Features**

### 🔥 **Core Capabilities**
- **📄 Multi-format Document Processing** - PDF, DOCX, PPTX, TXT, MD (up to 50MB)
- **🌐 Real-time Web Content Extraction** - Process any web URL using MCP server with Mozilla Readability
- **🤖 Intelligent Q&A** - Context-aware responses with source citations
- **🔍 Semantic Search** - Advanced embedding-based document retrieval
- **⚡ Real-time Processing** - Live progress tracking and performance metrics
- **🎨 Modern UI** - Dark/light theme with responsive design

### 🛠️ **Advanced Features**
- **📊 System Status Monitoring** - Real-time LlamaStack and Ollama health checks
- **🩺 Connection Diagnostics** - Smart endpoint detection and troubleshooting
- **📈 Performance Analytics** - Detailed processing metrics and statistics
- **🔄 Upload State Management** - Interrupt-resistant file processing
- **🔍 Debug Tools** - Comprehensive system diagnostics and logging

### 🧠 **AI Integration**
- **🦙 LlamaStack Orchestration** - Unified API for inference and embeddings
- **🏠 Local Model Support** - Ollama integration for privacy-focused AI
- **🧮 Sentence Transformers** - High-quality embeddings with all-MiniLM-L6-v2
- **🔀 Fallback Systems** - Multiple AI provider support with auto-switching
- **📋 Smart Model Filtering** - Only shows models actually available in your Ollama installation
- **🔧 Enhanced Error Handling** - Robust numpy array handling and type conversion

### 🌐 **Web Content Processing (NEW!)**
- **🔧 MCP Server Integration** - Uses @just-every/mcp-read-website-fast for clean content extraction
- **📝 Mozilla Readability** - Advanced web content parsing with readability optimization
- **🔄 Smart Fallback** - BeautifulSoup + requests backup when MCP server unavailable
- **⚡ Real-time Embedding** - URLs processed and vectorized instantly for immediate search
- **🎯 Multiple Sources** - Mix file uploads and web URLs in the same knowledge base
- **🔍 Debug Logging** - Comprehensive logging for troubleshooting web processing issues
- **🛠️ MCP Command Integration** - Automatic setup and testing of web extraction tools

---

## 📸 **Screenshots**

### **Main Application Interface**
![Main Interface](data/images/Screenshot_2025-07-20_at_11.11.29 PM.png)
*Clean, modern interface with sidebar controls and main chat area*

### **Document Processing & Chat**
![Document Processing](/data/images/Screenshot_2025-07-20_at_11.12.41 PM.png) 
*Real-time document processing with performance metrics and intelligent chat responses*

### **System Status & Diagnostics**
![System Status](/data/images/Screenshot_2025-07-20_at_11.13.52 PM.png)
*Comprehensive system monitoring with LlamaStack and Ollama status indicators*

---

## 🏛️ **System Architecture**

### **Current Streamlit-Only Architecture**

```mermaid
graph TB
    subgraph "🎨 Frontend Layer (Streamlit)"
        UI[📱 Main Interface<br/>Port: 8501<br/>Config: .streamlit/config.toml]
        SIDEBAR[🔧 Control Sidebar<br/>Theme: Dark/Light<br/>Upload: 50MB max]
        CHAT[💬 Chat Interface<br/>History: Session State<br/>Context: 6000 chars]
        UPLOAD[📁 File Upload<br/>Formats: PDF,DOCX,PPTX,TXT,MD<br/>Chunk: 3000+600 overlap]
        WEB_URL[🌐 Web URL Input<br/>Timeout: 30s<br/>Size: 50MB limit]
        STATUS[🔌 System Status<br/>Health: Real-time<br/>Models: Ollama + LlamaStack]
        DIAG[🩺 Diagnostics<br/>Connection: Auto-detect<br/>Endpoints: 8321, 11434]
    end
    
    subgraph "🧠 Core Processing"
        DOC_HANDLER[📄 Document Handler<br/>Embedding: all-MiniLM-L6-v2<br/>Similarity: Cosine + 0.25 threshold]
        WEB_PROC[🌐 Web Content Processor<br/>MCP: just-every mcp-read-website-fast<br/>Fallback: BeautifulSoup]
        CHAT_ENGINE[💬 Chat Engine<br/>Temperature: 0.4<br/>Max Tokens: 1024]
        MODEL_MGR[🤖 Model Manager<br/>Filter: Ollama-only models<br/>Status: Real-time polling]
        THEME_MGR[🎨 Theme Manager<br/>CSS: Custom styles<br/>Responsive: Mobile-friendly]
    end
    
    subgraph "🌐 Web Content Processing"
        MCP_SERVER[🔧 MCP Server<br/>Command: fetch<br/>Output: markdown json<br/>Timeout: 30s]
        BEAUTIFUL_SOUP[🍲 BeautifulSoup Fallback<br/>Parser: html parser<br/>Clean: markdownify]
        WEB_EXTRACT[📝 Content Extraction<br/>Readability: Mozilla<br/>Metadata: Title, URL]
        URL_VALIDATE[✅ URL Validation<br/>Schemes: http https<br/>Format: urlparse]
    end
    
    subgraph "🦙 LlamaStack Integration"
        LS_CLIENT[🔗 LlamaStack Client<br/>Port: 8321<br/>Config: llamastack-config.yaml]
        EMBED_API[🧮 Embeddings API<br/>Model: all-MiniLM-L6-v2<br/>Dimensions: 384]
        CHAT_API[💬 Chat Completion API<br/>Provider: ollama<br/>Fallback: demo]
        HEALTH_API[💓 Health Check API<br/>Endpoint: /v1/health<br/>Status: 200 OK]
    end
    
    subgraph "🏠 Local AI (Ollama)"
        OLLAMA[🦙 Ollama Server<br/>Port: 11434<br/>Config: ollama-example.yaml]
        LOCAL_LLM[🤖 Local LLM Models<br/>Default: llama3.2:1b<br/>Alternative: llama3.2:3b]
        MODEL_PULL[⬇️ Model Management<br/>Command: ollama pull<br/>Cache: .ollama]
    end
    
    subgraph "💾 Storage & State"
        SESSION[🔄 Session State<br/>Persistence: Browser<br/>Cleanup: Auto]
        VECTOR_DB[🗄️ Vector Storage<br/>Format: JSON<br/>Location: Session]
        DOC_STORE[📊 Document Storage<br/>Backup: Auto-save<br/>Restore: On reload]
        CACHE[⚡ Performance Cache<br/>TTL: Session<br/>Size: <50MB]
    end
    
    subgraph "🔍 Debug & Monitoring"
        DEBUG_LOG[📝 Debug Logging<br/>Level: INFO/DEBUG<br/>Output: Console]
        ERROR_HANDLE[🛠️ Error Handling<br/>Type: Auto-convert<br/>Fallback: Graceful]
        TYPE_CONVERT[🔄 Type Conversion<br/>Numpy to List<br/>Array to Scalar]
    end
    
    %% User Interactions
    UI --> SIDEBAR
    SIDEBAR --> STATUS
    SIDEBAR --> UPLOAD
    SIDEBAR --> WEB_URL
    SIDEBAR --> DIAG
    UI --> CHAT
    
    %% Core Processing Flow
    UPLOAD --> DOC_HANDLER
    WEB_URL --> WEB_PROC
    CHAT --> CHAT_ENGINE
    STATUS --> MODEL_MGR
    DIAG --> MODEL_MGR
    
    %% Web Processing Flow
    WEB_PROC --> MCP_SERVER
    MCP_SERVER --> WEB_EXTRACT
    WEB_PROC --> BEAUTIFUL_SOUP
    WEB_PROC --> URL_VALIDATE
    WEB_EXTRACT --> DOC_HANDLER
    
    %% LlamaStack Integration
    DOC_HANDLER --> LS_CLIENT
    CHAT_ENGINE --> LS_CLIENT
    MODEL_MGR --> LS_CLIENT
    LS_CLIENT --> EMBED_API
    LS_CLIENT --> CHAT_API
    LS_CLIENT --> HEALTH_API
    
    %% Ollama Fallback
    LS_CLIENT -.->|Fallback| OLLAMA
    MODEL_MGR --> OLLAMA
    OLLAMA --> LOCAL_LLM
    OLLAMA --> MODEL_PULL
    
    %% Storage Layer
    DOC_HANDLER --> SESSION
    CHAT_ENGINE --> SESSION
    SESSION --> VECTOR_DB
    SESSION --> DOC_STORE
    SESSION --> CACHE
    
    %% Debug Layer
    CHAT_ENGINE --> DEBUG_LOG
    DOC_HANDLER --> ERROR_HANDLE
    CHAT_ENGINE --> TYPE_CONVERT
    
    style UI fill:#e3f2fd
    style LS_CLIENT fill:#fff3e0
    style OLLAMA fill:#e8f5e8
    style SESSION fill:#fce4ec
    style MCP_SERVER fill:#fff8e1
    style DEBUG_LOG fill:#f3e5f5
```

### **Technology Stack**

```mermaid
graph LR
    subgraph "🖥️ Frontend"
        ST[Streamlit 1.28+]
        CSS[Custom CSS/JS]
        PD[Pandas DataFrames]
    end
    
    subgraph "🌐 Web Processing"
        MCP[MCP Server]
        BS[BeautifulSoup]
        REQ[Requests]
        MD[Markdownify]
    end
    
    subgraph "🤖 AI/ML"
        LS[LlamaStack API]
        ST_EMB[Sentence Transformers]
        OL[Ollama]
        HF[Hugging Face]
        NP[NumPy]
    end
    
    subgraph "📊 Data Processing"
        DOC[Docling]
        PDF[PyPDF2]
        JSON[JSON Storage]
        CHUNK[Text Chunking]
    end
    
    subgraph "🔧 Infrastructure"
        PY[Python 3.12+]
        SUB[Subprocess]
        OS[OS Integration]
        NODE[Node.js]
    end
    
    ST --> LS
    ST --> CSS
    ST --> PD
    ST --> MCP
    MCP --> BS
    MCP --> REQ
    BS --> MD
    LS --> ST_EMB
    LS --> OL
    ST_EMB --> NP
    DOC --> PDF
    DOC --> CHUNK
    ST --> DOC
    MCP --> NODE
```

---

## 🔄 **Data Flow & Processing**

### **Document Processing Pipeline**

```mermaid
flowchart TD
    START([📁 File Upload<br/>Port: 8501<br/>Max: 50MB<br/>Formats: PDF,DOCX,PPTX,TXT,MD]) --> VALIDATE{🔍 Validation<br/>Size: <50MB<br/>Type: Allowed formats<br/>Config: .streamlit/config.toml}
    
    VALIDATE -->|✅ Valid| STATE_TRACK[🔄 State Tracking<br/>Mark as uploading<br/>Session: st.session_state<br/>Backup: Auto-save]
    VALIDATE -->|❌ Invalid| ERROR[❌ Error Display<br/>Size/type limits<br/>User feedback]
    
    STATE_TRACK --> EXTRACT[📄 Content Extraction<br/>Multi-format support<br/>Docling: PDF processing<br/>PyPDF2: PDF parsing<br/>python-docx: DOCX]
    EXTRACT --> OPTIMIZE[🚀 Performance Optimization<br/>Large file handling<br/>Batch: 10 chunks<br/>Memory: <50MB session]
    
    OPTIMIZE --> CHUNK[✂️ Smart Chunking<br/>3000 chars + 600 overlap<br/>Config: config.py<br/>CHARS_PER_CHUNK: 3000<br/>CHUNK_OVERLAP: 600]
    CHUNK --> BATCH[📦 Batch Processing<br/>Optimized embedding generation<br/>Parallel: 3 requests<br/>Timeout: 30s per batch]
    
    BATCH --> EMBED[🧮 Generate Embeddings<br/>all-MiniLM-L6-v2<br/>Port: 8321<br/>Dimensions: 384<br/>Model: sentence-transformers]
    EMBED --> QUALITY[🎯 Quality Check<br/>Validate embeddings<br/>Type: List/Array<br/>Conversion: Numpy to List<br/>Error: Graceful fallback]
    
    QUALITY -->|✅ Success| STORE[💾 Store Document<br/>Session state + backup<br/>Location: st.session_state<br/>Format: JSON<br/>Persistence: Browser]
    QUALITY -->|⚠️ Partial| FALLBACK[🧪 Dummy Embeddings<br/>Ensure functionality<br/>Random: 384 dimensions<br/>Status: Demo mode]
    
    STORE --> METRICS[📊 Performance Metrics<br/>Speed, quality, stats<br/>Processing time<br/>Chunk count<br/>Embedding success rate]
    FALLBACK --> METRICS
    METRICS --> COMPLETE([✅ Processing Complete<br/>Status: Ready for chat<br/>Storage: Session state])
    
    STATE_TRACK -.->|Interruption| RETRY[🔄 Mark for Retry<br/>State management<br/>Failed uploads set<br/>Retry mechanism]
    RETRY -.-> STATE_TRACK
    
    style START fill:#e1f5fe
    style COMPLETE fill:#e8f5e8
    style EMBED fill:#fff3e0
    style METRICS fill:#f3e5f5
    style ERROR fill:#ffebee
```

### **Chat & Query Processing**

```mermaid
flowchart TD
    QUERY([💬 User Question<br/>Interface: Streamlit Chat<br/>Port: 8501<br/>History: Session state]) --> CHECK{📊 Documents Available?<br/>Session: st.session_state<br/>Collections: documents + uploaded_documents<br/>Count: Real-time check}
    
    CHECK -->|❌ No| NO_DOCS[📝 No Documents Message<br/>Upload prompt<br/>UI: Sidebar guidance<br/>Status: Ready for upload]
    CHECK -->|✅ Yes| EMBED_Q[🧮 Query Embedding<br/>all-MiniLM-L6-v2<br/>Port: 8321<br/>Dimensions: 384<br/>Model: sentence-transformers]
    
    EMBED_Q --> SEARCH[🔍 Similarity Search<br/>Cosine similarity<br/>Algorithm: dot product magnitudes<br/>Type: Auto-convert arrays<br/>Debug: Logged similarity scores]
    SEARCH --> FILTER[🎯 Relevance Filtering<br/>Threshold: 0.25<br/>Config: MIN_SIMILARITY_THRESHOLD<br/>Chunks: Top 4 relevant<br/>Diversity: Source balancing]
    
    FILTER --> RERANK[📊 Chunk Reranking<br/>Diversity + relevance<br/>Sources: Files + Web URLs<br/>Priority: Recent uploads<br/>Limit: 4 chunks max]
    RERANK --> CONTEXT[📝 Context Building<br/>6000 char limit<br/>Config: LLM_MAX_TOKENS<br/>Format: System + Context + Query<br/>Citations: Source attribution]
    
    CONTEXT --> PROMPT[📋 Prompt Engineering<br/>System + context + query<br/>Template: config.py<br/>Temperature: 0.4<br/>Max tokens: 1024]
    PROMPT --> TRY_LS[🦙 Try LlamaStack<br/>Multiple endpoints<br/>Port: 8321<br/>Config: llamastack-config.yaml<br/>Provider: ollama]
    
    TRY_LS -->|✅ Success| RESPONSE[📤 AI Response<br/>With citations<br/>Format: Markdown<br/>Sources: Top 3 documents<br/>History: Session persistence]
    TRY_LS -->|❌ Failed| TRY_OLLAMA[🏠 Try Ollama Fallback<br/>Local processing<br/>Port: 11434<br/>Config: ollama-example.yaml<br/>Model: llama3.2:1b/3b]
    
    TRY_OLLAMA -->|✅ Success| RESPONSE
    TRY_OLLAMA -->|❌ Failed| FALLBACK_RESP[🧪 Demo Response<br/>Educational content<br/>Status: Demo mode<br/>No backend required<br/>Sample responses]
    
    RESPONSE --> HISTORY[💾 Chat History<br/>Session persistence<br/>Storage: st.session_state<br/>Format: List of dicts<br/>Cleanup: Auto on session end]
    FALLBACK_RESP --> HISTORY
    NO_DOCS --> HISTORY
    
    style QUERY fill:#e1f5fe
    style RESPONSE fill:#e8f5e8
    style EMBED_Q fill:#fff3e0
    style SEARCH fill:#f3e5f5
    style NO_DOCS fill:#ffebee
```

### **Web Content Processing Pipeline**

```mermaid
flowchart TD
    URL_INPUT([🌐 Web URL Input<br/>Interface: Sidebar tab<br/>Port: 8501<br/>Validation: urlparse]) --> URL_VALIDATE{🔍 URL Validation<br/>Schemes: http https<br/>Format: urlparse<br/>Domain: netloc check<br/>Config: WebContentProcessor}
    
    URL_VALIDATE -->|❌ Invalid| URL_ERROR[❌ Invalid URL Error<br/>Format check<br/>User feedback<br/>UI: Error message]
    URL_VALIDATE -->|✅ Valid| MCP_TRY[🔧 Try MCP Server<br/>just-every mcp-read-website-fast<br/>Command: fetch<br/>Output: markdown<br/>Timeout: 30s]
    
    MCP_TRY -->|✅ Success| MCP_EXTRACT[📝 MCP Extraction<br/>Mozilla Readability<br/>Clean: markdown output<br/>Metadata: title, url<br/>Size: <50MB limit]
    MCP_TRY -->|❌ Failed| BEAUTIFUL_SOUP[🍲 BeautifulSoup Fallback<br/>Direct HTML parsing<br/>Parser: html.parser<br/>Clean: markdownify<br/>Timeout: 30s]
    
    MCP_EXTRACT --> CONTENT_CHECK{📄 Content Quality?<br/>Length: >100 chars<br/>Text: Readable content<br/>Blocked: robots.txt<br/>Empty: No content}
    BEAUTIFUL_SOUP --> CONTENT_CHECK
    
    CONTENT_CHECK -->|❌ Poor| CONTENT_ERROR[❌ Poor Content Error<br/>Empty/blocked<br/>User feedback<br/>Status: Processing failed]
    CONTENT_CHECK -->|✅ Good| SIZE_CHECK{📏 Size Check<br/>Limit: 50MB<br/>Config: MAX_CONTENT_SIZE<br/>Memory: Session state<br/>Chunk: If large}
    
    SIZE_CHECK -->|❌ Too Large| SIZE_ERROR[❌ Size Limit Error<br/>>50MB<br/>User feedback<br/>Suggestion: Try smaller URL]
    SIZE_CHECK -->|✅ OK| WEB_CHUNK[✂️ Web Content Chunking<br/>Smart segmentation<br/>Config: CHARS_PER_CHUNK: 3000<br/>Overlap: 600 chars<br/>Format: Markdown]
    
    WEB_CHUNK --> WEB_EMBED[🧮 Web Embedding Generation<br/>all-MiniLM-L6-v2<br/>Port: 8321<br/>Dimensions: 384<br/>Model: sentence-transformers]
    WEB_EMBED --> WEB_STORE[💾 Store Web Content<br/>Unified with files<br/>Session: st.session_state<br/>Collection: uploaded_documents<br/>Type: WEB]
    
    WEB_STORE --> WEB_METRICS[📊 Web Processing Metrics<br/>Speed, quality, source<br/>Processing time<br/>Content size<br/>Source: MCP/BeautifulSoup]
    WEB_METRICS --> WEB_COMPLETE([✅ Web Processing Complete<br/>Status: Ready for chat<br/>Storage: Session state<br/>Search: Unified retrieval])
    
    URL_ERROR --> WEB_COMPLETE
    CONTENT_ERROR --> WEB_COMPLETE
    SIZE_ERROR --> WEB_COMPLETE
    
    style URL_INPUT fill:#e1f5fe
    style WEB_COMPLETE fill:#e8f5e8
    style MCP_EXTRACT fill:#fff8e1
    style WEB_EMBED fill:#fff3e0
    style WEB_STORE fill:#f3e5f5
    style URL_ERROR fill:#ffebee
    style CONTENT_ERROR fill:#ffebee
    style SIZE_ERROR fill:#ffebee
```

### **System Configuration & Ports**

```mermaid
graph TB
    subgraph "🌐 User Interface"
        BROWSER[🌐 Browser<br/>URL: http://localhost:8501<br/>Interface: Streamlit Web App<br/>Theme: Dark/Light toggle]
    end
    
    subgraph "📱 Streamlit Frontend"
        STREAMLIT[📱 Streamlit Server<br/>Port: 8501<br/>Config: .streamlit/config.toml<br/>Max Upload: 50MB<br/>Theme: Custom CSS]
    end
    
    subgraph "🦙 LlamaStack Backend"
        LLAMASTACK[🦙 LlamaStack Server<br/>Port: 8321<br/>Config: llamastack/config/llamastack-config.yaml<br/>APIs: inference, embeddings, health<br/>Provider: ollama]
    end
    
    subgraph "🏠 Ollama Local AI"
        OLLAMA_SERVER[🦙 Ollama Server<br/>Port: 11434<br/>Config: ollama-example.yaml<br/>Models: llama3.2:1b, llama3.2:3b<br/>Cache: .ollama]
    end
    
    subgraph "🔧 MCP Web Processing"
        MCP_SERVER[🔧 MCP Server<br/>Command: npx just-every mcp-read-website-fast<br/>Package: package.json<br/>Setup: make setup-mcp<br/>Node.js: 16.0.0+]
        BEAUTIFUL_SOUP[🍲 BeautifulSoup Fallback<br/>Parser: html parser<br/>Clean: markdownify]
        WEB_EXTRACT[📝 Content Extraction<br/>Readability: Mozilla<br/>Metadata: Title, URL]
        URL_VALIDATE[✅ URL Validation<br/>Schemes: http https<br/>Format: urlparse]
    end
    
    subgraph "💾 Data Storage"
        SESSION_STATE[💾 Session State<br/>Location: Browser<br/>Persistence: Session<br/>Format: JSON<br/>Cleanup: Auto]
        VECTOR_STORE[🗄️ Vector Storage<br/>Format: JSON arrays<br/>Dimensions: 384<br/>Model: all-MiniLM-L6-v2<br/>Location: Session]
    end
    
    subgraph "📁 Configuration Files"
        ST_CONFIG[📄 .streamlit/config.toml<br/>Port: 8501<br/>Max Upload: 50MB<br/>Theme: Custom<br/>CORS: Enabled]
        LS_CONFIG[📄 llamastack/config/llamastack-config.yaml<br/>Port: 8321<br/>Provider: ollama<br/>APIs: inference, embeddings<br/>Health: /v1/health]
        OLLAMA_CONFIG[📄 ollama-example.yaml<br/>Port: 11434<br/>Models: llama3.2:1b<br/>Provider: local<br/>Cache: .ollama]
        PKG_CONFIG[📄 package.json<br/>MCP: just-every mcp-read-website-fast<br/>Version: 0.1.13<br/>Node: 16.0.0+]
    end
    
    %% Connections
    BROWSER -->|HTTP/WebSocket| STREAMLIT
    STREAMLIT -->|HTTP API| LLAMASTACK
    STREAMLIT -->|HTTP API| OLLAMA_SERVER
    STREAMLIT -->|Subprocess| MCP_SERVER
    STREAMLIT -->|Session| SESSION_STATE
    STREAMLIT -->|Session| VECTOR_STORE
    
    %% Configuration
    STREAMLIT -->|Read| ST_CONFIG
    LLAMASTACK -->|Read| LS_CONFIG
    OLLAMA_SERVER -->|Read| OLLAMA_CONFIG
    MCP_SERVER -->|Read| PKG_CONFIG
    
    %% Fallback connections
    LLAMASTACK -.->|Fallback| OLLAMA_SERVER
    
    style BROWSER fill:#e3f2fd
    style STREAMLIT fill:#fff3e0
    style LLAMASTACK fill:#e8f5e8
    style OLLAMA_SERVER fill:#f3e5f5
    style MCP_SERVER fill:#fff8e1
    style SESSION_STATE fill:#fce4ec
    style VECTOR_STORE fill:#f1f8e9
```

### **System Health & Diagnostics**

```mermaid
flowchart TD
    MONITOR([🔌 System Monitor]) --> CHECK_LS[🦙 Check LlamaStack<br/>Health endpoint]
    MONITOR --> CHECK_OL[🏠 Check Ollama<br/>Model list]
    
    CHECK_LS -->|✅ Online| LS_DIAG[🩺 LlamaStack Diagnostics<br/>Endpoint discovery]
    CHECK_LS -->|❌ Offline| LS_ERROR[❌ Connection Issues<br/>Show recommendations]
    
    CHECK_OL -->|✅ Running| OL_MODELS[📦 List Models<br/>Available models]
    CHECK_OL -->|❌ Offline| OL_ERROR[❌ Ollama Down<br/>Installation guide]
    
    LS_DIAG --> TEST_ENDPOINTS[📡 Test Endpoints<br/>Models, chat, embeddings]
    TEST_ENDPOINTS --> RECOMMEND[💡 Recommendations<br/>Fix suggestions]
    
    OL_MODELS --> MODEL_STATUS[📊 Model Status<br/>Local vs remote]
    
    RECOMMEND --> STATUS_UI[📱 Status Display<br/>Real-time indicators]
    MODEL_STATUS --> STATUS_UI
    LS_ERROR --> STATUS_UI
    OL_ERROR --> STATUS_UI
    
    style MONITOR fill:#e3f2fd
    style STATUS_UI fill:#e8f5e8
    style LS_ERROR fill:#ffebee
    style OL_ERROR fill:#ffebee
```

---

## 🚀 **Quick Start**

### **Prerequisites**
- **Python 3.12+** (recommended)
- **Git** for cloning
- **8GB+ RAM** for local models
- **Optional**: Ollama for local AI processing

### **Installation & Setup**

```bash
git clone https://github.com/yourusername/rag-llama-stack.git
cd rag-llama-stack
make setup                    # Step 1: Create virtual environment
source venv/bin/activate      # Step 2: Activate virtual environment
make install                  # Step 3: Install Python dependencies
make setup-mcp               # Step 4: Setup MCP server
make start                   # Step 5: Start the application
```

### **Alternative: Manual Setup**

```bash
# Start LlamaStack server
llamastack run ./llamastack/config/llamastack-config.yaml

# In another terminal, start Streamlit
streamlit run frontend/streamlit/app.py --server.port 8501
```

### **First Time Usage**

1. **Open** http://localhost:8501
2. **Check System Status** in the top-left sidebar
3. **Add Content Sources:**
   - 📄 **Upload Documents** using the file uploader (PDF, DOCX, PPTX, TXT, MD)
   - 🌐 **Process Web URLs** by entering any web link for real-time content extraction
4. **Start Chatting** with your documents and web content!

---

## 📊 **Performance & Monitoring**

### **Real-time System Status**

The application provides comprehensive monitoring:

- **🟢 LlamaStack** - Connection and endpoint health
- **🟢 Ollama** - Local model availability  
- **📊 Performance Metrics** - Processing speed and quality
- **🔍 Debug Information** - Configuration and state details

### **Document Processing Metrics**

Each upload provides detailed analytics:

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| **Processing Speed** | MB/second throughput | 0.2-1.0 MB/s |
| **Embedding Quality** | Success rate percentage | 95-100% |
| **Chunk Efficiency** | Characters per chunk | 2500-3500 |
| **Memory Usage** | Session state size | <50MB |

### **Performance Optimization**

The system automatically optimizes for:
- **Large Files**: Batch processing and content filtering
- **Slow Networks**: Fallback systems and local processing
- **Memory**: Efficient chunk management and cleanup
- **Speed**: Parallel operations and smart caching

---

## 🔧 **Configuration**

### **Main Configuration** (`frontend/streamlit/core/config.py`)

```python
# Model Configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.2:1b"

# Processing Configuration  
CHARS_PER_CHUNK = 3000
CHUNK_OVERLAP = 600
MAX_RELEVANT_CHUNKS = 4

# Performance Configuration
MIN_SIMILARITY_THRESHOLD = 0.25
LLM_TEMPERATURE = 0.4
LLM_MAX_TOKENS = 1024
```

### **Streamlit Configuration** (`.streamlit/config.toml`)

```toml
[server]
maxUploadSize = 50
port = 8501

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"

[browser]
gatherUsageStats = false
```

### **LlamaStack Configuration** (`llamastack/config/llamastack-config.yaml`)

```yaml
built_at: '2024-12-XX'
image_type: conda

apis:
  - inference
  - safety  
  - agents
  - memory
  - telemetry

providers:
  inference:
    - provider_id: ollama
      provider_type: remote::ollama
      config:
        url: http://localhost:11434
```

---

## 🐛 **Troubleshooting**

### **Common Issues & Solutions**

#### 🔴 **Virtual Environment Issues**
```bash
# Error: "Virtual environment not activated!"
# Solution: Activate the virtual environment
source venv/bin/activate

# Error: "Streamlit not found!"
# Solution: Install dependencies in activated environment
make install

# Error: "make: streamlit: No such file or directory"
# Solution: Ensure virtual environment is activated and dependencies installed
source venv/bin/activate && make install
```

#### 🔴 **Setup Process Issues**
```bash
# If make setup fails, run steps manually:
make venv                    # Create virtual environment
source venv/bin/activate     # Activate it
make install                 # Install Python dependencies
make setup-mcp              # Setup MCP server
make start                  # Start application

# Check if virtual environment is active:
echo $VIRTUAL_ENV           # Should show path to venv
which python               # Should show venv/bin/python
which streamlit            # Should show venv/bin/streamlit
```

#### 🔴 **LlamaStack Connection Failed**
```bash
# Check if LlamaStack is running
curl http://localhost:8321/v1/health

# Restart LlamaStack
make restart

# Check configuration
cat llamastack/config/llamastack-config.yaml
```

#### 🔴 **Services Not Running**
```bash
# Error: "LlamaStack offline" or "Ollama not found"
# Solution: Start all services with one command
make start

# Or start services manually:
make ollama      # Terminal 1: Start Ollama
make llamastack  # Terminal 2: Start LlamaStack  
make start-frontend  # Terminal 3: Start Streamlit

# Check service status
make status
```

#### 🔴 **Ollama Not Found**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.2:1b
```

#### 🔴 **Web URL Processing Issues**
```bash
# Test MCP server installation
npx @just-every/mcp-read-website-fast fetch https://example.com --output markdown

# Reinstall MCP server if needed
npm install @just-every/mcp-read-website-fast

# Check Node.js version (requires >=16.0.0)
node --version
```

#### 🔴 **"unknown command read-website" Error**
- **Cause**: Incorrect MCP command format
- **Fix**: ✅ **Resolved** - Updated to use `fetch` command
- **Correct Usage**: `npx @just-every/mcp-read-website-fast fetch <url> --output markdown`

#### 🔴 **"truth value of an array with more than one element is ambiguous" Error**
- **Cause**: Numpy array type mismatch in similarity calculation
- **Fix**: ✅ **Resolved** - Automatic type conversion implemented
- **Status**: No longer occurs - arrays are converted to lists automatically

#### 🔴 **Too Many Models in Dropdown**
- **Cause**: Showing models from LlamaStack that aren't available locally
- **Fix**: ✅ **Resolved** - Only shows models actually available in Ollama
- **Status**: Dropdown now filters to your actual installed models

#### 🔴 **File Upload Fails**
- Check file size (max 50MB)
- Verify file format (PDF, DOCX, PPTX, TXT, MD)
- Don't switch models during upload
- Use "Retry" if interrupted

#### 🔴 **Poor Response Quality**
- Upload more relevant documents
- Check embedding quality in performance metrics
- Verify model configuration
- Use connection diagnostics

### **Debug Mode**

Enable detailed logging:

```bash
# Set debug environment
export STREAMLIT_LOGGER_LEVEL=debug

# Run with verbose output
streamlit run frontend/streamlit/app.py --logger.level debug
```

### **Web Processing Debug**

The application includes comprehensive debug logging:
- **Query Processing**: Track embedding generation and retrieval
- **Similarity Calculation**: Monitor cosine similarity computations  
- **Web Processing**: Log MCP server calls and fallback usage
- **Error Handling**: Detailed error messages with context

### **Getting Help**

1. **Connection Diagnostics** - Use the built-in diagnostic tools
2. **Performance Metrics** - Check the detailed performance tables
3. **Debug Information** - Use the debug panel in the sidebar
4. **Logs** - Check `logs/` directory for detailed error logs

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ **Star History**

If you find this project useful, please consider giving it a star! ⭐

---

*Built with ❤️ using LlamaStack, Streamlit, and modern AI technologies*