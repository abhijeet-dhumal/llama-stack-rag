"""
Local RAG Pipeline Package
"""

from .rag_pipeline import RAGPipeline, BatchProcessor
from .api import app

__version__ = "1.0.0"
__all__ = ["RAGPipeline", "BatchProcessor", "app"] 