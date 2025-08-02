"""
Local RAG Pipeline Package
"""

from .feast_rag_pipeline import FeastRAGPipeline
from .api import app

__version__ = "1.0.0"
__all__ = ["FeastRAGPipeline", "app"] 