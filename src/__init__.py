"""
Local RAG Pipeline Package
"""

from .api import app
from .feast_rag_pipeline import FeastRAGPipeline

__version__ = "1.0.0"
__all__ = ["FeastRAGPipeline", "app"]
