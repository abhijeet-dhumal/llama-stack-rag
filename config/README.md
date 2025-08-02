# Configuration Settings

This directory contains configuration settings for the Feast RAG Pipeline.

## Environment Variables

Settings can be configured using environment variables with the `RAG_` prefix:

### Model Settings
- `RAG_EMBEDDING_MODEL` - SentenceTransformer embedding model (default: `all-MiniLM-L6-v2`)
- `RAG_LLM_MODEL` - Ollama LLM model (default: `llama3.2:3b`)

### Feast Settings
- `RAG_FEAST_REPO_PATH` - Path to Feast repository (default: `feast_feature_repo`)
- `RAG_FEAST_FEATURE_VIEW` - Feast feature view name (default: `document_embeddings`)

### Milvus Settings
- `RAG_MILVUS_URI` - Milvus-lite database file path (default: `feast_feature_repo/data/online_store.db`)
- `RAG_MILVUS_COLLECTION` - Milvus collection name (default: `rag_document_embeddings`)

### API Settings
- `RAG_API_HOST` - FastAPI host (default: `0.0.0.0`)
- `RAG_API_PORT` - FastAPI port (default: `8000`)
- `RAG_API_RELOAD` - Enable FastAPI auto-reload (default: `false`)
- `RAG_API_TITLE` - API title (default: `Feast RAG Pipeline`)

### Processing Settings
- `RAG_MAX_CONTEXT_CHUNKS` - Maximum number of chunks for context (default: `5`)
- `RAG_DEFAULT_TOP_K` - Default number of documents to retrieve (default: `5`)
- `RAG_SIMILARITY_THRESHOLD` - Minimum similarity threshold (default: `0.3`)
- `RAG_CHUNK_SIZE` - Maximum chunk size in characters (default: `1000`)
- `RAG_CHUNK_OVERLAP` - Character overlap between chunks (default: `200`)

### Ollama Settings
- `RAG_OLLAMA_HOST` - Ollama server host (default: `localhost`)
- `RAG_OLLAMA_PORT` - Ollama server port (default: `11434`)
- `RAG_OLLAMA_TIMEOUT` - Ollama request timeout in seconds (default: `120`)

### Security Settings
- `RAG_ENABLE_CORS` - Enable CORS (default: `true`)

### Performance Settings
- `RAG_DEBUG_MODE` - Enable debug mode (default: `false`)
- `RAG_ENABLE_METRICS` - Enable performance metrics (default: `true`)
- `RAG_ENABLE_ASYNC_PROCESSING` - Enable async document processing (default: `true`)

### File Upload Settings
- `RAG_MAX_FILE_SIZE` - Maximum file size in bytes (default: `104857600` = 100MB)
- `RAG_UPLOAD_TIMEOUT` - Upload timeout in seconds (default: `300`)

### Logging Settings
- `RAG_LOG_LEVEL` - Logging level (default: `INFO`)

## Example .env File

```bash
# Model Settings
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_LLM_MODEL=llama3.2:3b

# API Settings
RAG_API_PORT=8001
RAG_DEBUG_MODE=true

# Processing Settings
RAG_MAX_CONTEXT_CHUNKS=10
RAG_SIMILARITY_THRESHOLD=0.4
```

## Usage in Code

```python
from config.settings import get_settings

settings = get_settings()
print(f"API running on {settings.api_host}:{settings.api_port}")
```

## Utility Functions

- `get_settings()` - Get application settings
- `get_ollama_url()` - Get complete Ollama URL
- `get_milvus_uri()` - Get Milvus URI with proper path resolution
- `get_feast_repo_path()` - Get Feast repository path with proper path resolution
- `update_settings(**kwargs)` - Update settings with new values
- `load_settings_from_env()` - Reload settings from environment variables