"""
Configuration settings for the RAG Pipeline
"""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Model settings
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2:3b"
    
    # Vector store settings
    vector_db_path: str = "./chroma_db"
    collection_name: str = "documents"
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_chunking: bool = True
    similarity_threshold: float = 0.7
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Processing settings
    max_context_chunks: int = 5
    batch_size: int = 10
    max_workers: int = 4
    
    # Ollama settings
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_base_url: Optional[str] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Security settings
    enable_cors: bool = True
    allowed_origins: list = ["*"]
    
    # File upload settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = [".pdf"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values"""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings 