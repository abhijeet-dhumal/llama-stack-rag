"""
Configuration and Constants for RAG LlamaStack Application
Central location for app settings and configuration
"""

# LlamaStack Configuration
LLAMASTACK_BASE = "http://localhost:8321/v1"

# File Upload Configuration
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_FILE_TYPES = ['pdf', 'txt', 'md', 'docx', 'pptx']

# Model Configuration
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_MODELS = ["llama3.2:3b"]  # Remove nomic-embed-text since we're using sentence-transformers

# Processing Configuration
CHARS_PER_FILE_MULTIPLIER = 2.5  # Estimated characters per byte
CHARS_PER_CHUNK = 3000  # Optimal size for focused context (reduced from 4500)
CHUNK_OVERLAP = 600     # 20% overlap for better continuity
MAX_RELEVANT_CHUNKS = 4  # Fewer, more relevant chunks (reduced from 8)
TOP_SOURCES_COUNT = 3    # Focused source references

# Performance Configuration for Large Files
LARGE_FILE_THRESHOLD_MB = 5      # Files larger than this get optimized processing
MAX_CHUNKS_PER_FILE = 150        # Limit chunks for very large files
BATCH_SIZE_LARGE_FILES = 10      # Larger batches for embedding generation
CONTENT_OPTIMIZATION_THRESHOLD = 500000  # Characters threshold for content optimization

# Response Quality Configuration
MIN_SIMILARITY_THRESHOLD = 0.5  # Higher threshold for more relevant chunks (was 0.35)
MAX_CONTEXT_LENGTH = 6000        # Optimized context length for focused responses
ENABLE_CHUNK_RERANKING = True    # Re-rank chunks by relevance

# LLM Generation Parameters
LLM_TEMPERATURE = 0.4            # Lower temperature for more focused responses
LLM_MAX_TOKENS = 1024           # Longer responses (was 512)
LLM_TOP_P = 0.9                 # Nucleus sampling parameter

# UI Configuration
APP_TITLE = "RAG LlamaStack"
APP_ICON = "🦙"
APP_DESCRIPTION = "Retrieval-Augmented Generation with LlamaStack & Ollama"

# Theme Configuration
DEFAULT_THEME = "dark"  # "dark" or "light"

# Chat Configuration
MAX_CHAT_HISTORY = 100
CHAT_INPUT_PLACEHOLDER = "What would you like to know?"

# Progress Configuration
PROCESSING_STEPS = [
    "📥 Reading content",
    "🔍 Extracting text", 
    "✂️ Creating chunks",
    "🧠 Generating embeddings",
    "💾 Storing data"
]

# Error Messages
ERROR_MESSAGES = {
    "file_too_large": "File is too large. Maximum size is {max_size}MB.",
    "no_documents": "No documents uploaded yet. Upload some documents to get started!",
    "llamastack_offline": "LlamaStack is offline. Please check your connection.",
    "processing_error": "An error occurred while processing your request.",
    "embedding_error": "Failed to generate embeddings for the query.",
    "llm_error": "LLM is currently unavailable. Using fallback response.",
}

# Success Messages
SUCCESS_MESSAGES = {
    "file_uploaded": "Files uploaded successfully!",
    "processing_complete": "Processing complete! Ready for Q&A and search!",
    "model_pulled": "Model pulled successfully!",
    "documents_cleared": "All documents cleared successfully!",
}

# Help Text
HELP_TEXT = {
    "file_upload": "Drag and drop files here (PDF, TXT, MD, DOCX, PPTX • Max 50MB per file)",
    "embedding_model": "Model used for creating document embeddings and search",
    "llm_model": "Model used for generating responses and chat",
    "ollama_models": "Enter any model available in Ollama registry",
    "theme_toggle": "Switch between dark and light themes",
}

# Page Configuration
PAGE_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Streamlit Configuration Override
STREAMLIT_CONFIG = {
    "server.maxUploadSize": MAX_FILE_SIZE_MB,
    "theme.base": "dark",
    "server.enableCORS": False,
    "server.enableXsrfProtection": False,
}

# CSS Classes for Styling
CSS_CLASSES = {
    "chat_message": "chat-message",
    "user_message": "user-message", 
    "assistant_message": "assistant-message",
    "source_citation": "source-citation",
    "document_card": "document-card",
    "upload_area": "upload-area",
    "main_header": "main-header",
}

# Model Status Labels
MODEL_STATUS = {
    "local_ollama": "🏠 Local (Ollama)",
    "cloud_api": "☁️ Cloud API",
    "demo_mode": "🧪 Demo",
    "offline": "❌ Offline",
    "loading": "⏳ Loading...",
}

# File Type Icons
FILE_TYPE_ICONS = {
    "pdf": "📄",
    "txt": "📝", 
    "md": "📋",
    "docx": "📘",
    "pptx": "📊",
    "default": "📎"
}

# Processing Time Estimates (seconds per MB)
PROCESSING_TIME_ESTIMATES = {
    "reading": 0.01,
    "extracting": 0.02,
    "chunking": 0.01,
    "embedding": 0.03,
    "storing": 0.01,
}

# Vector Search Configuration
VECTOR_CONFIG = {
    "embedding_dimension": 384,  # all-MiniLM-L6-v2 dimensions
    "similarity_threshold": 0.1,
    "max_chunks_per_document": 1000,
}

# Development Configuration
DEV_CONFIG = {
    "debug_mode": False,
    "verbose_logging": False,
    "show_performance_metrics": True,
    "enable_experimental_features": False,
} 