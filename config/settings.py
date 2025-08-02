"""
Configuration settings for the Feast RAG Pipeline
"""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings for Feast RAG Pipeline"""

    # Model settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="SentenceTransformer embedding model"
    )
    llm_model: str = Field(default="llama3.2:3b", description="Ollama LLM model")

    # Feast settings
    feast_repo_path: str = Field(
        default="feast_feature_repo", description="Path to Feast repository"
    )
    feast_feature_view: str = Field(
        default="document_embeddings", description="Feast feature view name"
    )

    # Milvus settings
    milvus_uri: str = Field(
        default="feast_feature_repo/data/online_store.db",
        description="Milvus-lite database file path",
    )
    milvus_collection: str = Field(
        default="rag_document_embeddings", description="Milvus collection name"
    )

    # Chunking settings
    chunk_size: int = Field(
        default=1000, description="Maximum chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200, description="Character overlap between chunks"
    )
    semantic_chunking: bool = Field(
        default=True, description="Enable semantic chunking"
    )
    similarity_threshold: float = Field(
        default=0.3, description="Minimum similarity threshold for retrieval"
    )

    # API settings
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    api_reload: bool = Field(default=False, description="Enable FastAPI auto-reload")
    api_title: str = Field(default="Feast RAG Pipeline", description="API title")

    # Processing settings
    max_context_chunks: int = Field(
        default=5, description="Maximum number of chunks for context"
    )
    default_top_k: int = Field(
        default=5, description="Default number of documents to retrieve"
    )
    batch_size: int = Field(default=10, description="Batch size for processing")
    max_workers: int = Field(default=4, description="Maximum worker threads")

    # Ollama settings
    ollama_host: str = Field(default="localhost", description="Ollama server host")
    ollama_port: int = Field(default=11434, description="Ollama server port")
    ollama_base_url: Optional[str] = Field(
        default=None, description="Custom Ollama base URL"
    )
    ollama_timeout: int = Field(
        default=120, description="Ollama request timeout in seconds"
    )

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )

    # Security settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allowed_methods: List[str] = Field(
        default=["*"], description="Allowed HTTP methods"
    )
    allowed_headers: List[str] = Field(
        default=["*"], description="Allowed HTTP headers"
    )

    # File upload settings
    max_file_size: int = Field(
        default=100 * 1024 * 1024, description="Maximum file size in bytes (100MB)"
    )
    allowed_file_types: List[str] = Field(
        default=[".pdf", ".txt", ".md", ".docx"], description="Allowed file extensions"
    )
    upload_timeout: int = Field(default=300, description="Upload timeout in seconds")

    # Performance settings
    embedding_cache_size: int = Field(default=1000, description="Embedding cache size")
    enable_async_processing: bool = Field(
        default=True, description="Enable async document processing"
    )

    # Development settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "RAG_",  # Environment variables will be prefixed with RAG_
        "case_sensitive": False,
    }


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def get_ollama_url() -> str:
    """Get complete Ollama URL"""
    if settings.ollama_base_url:
        return settings.ollama_base_url
    return f"http://{settings.ollama_host}:{settings.ollama_port}"


def get_milvus_uri() -> str:
    """Get Milvus URI with proper path resolution"""
    if os.path.isabs(settings.milvus_uri):
        return settings.milvus_uri
    return os.path.abspath(settings.milvus_uri)


def get_feast_repo_path() -> str:
    """Get Feast repository path with proper path resolution"""
    if os.path.isabs(settings.feast_repo_path):
        return settings.feast_repo_path
    return os.path.abspath(settings.feast_repo_path)


def update_settings(**kwargs) -> Settings:
    """Update settings with new values"""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def load_settings_from_env() -> Settings:
    """Reload settings from environment variables"""
    global settings
    settings = Settings()
    return settings
