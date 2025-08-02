"""
Feast-powered RAG document retriever using file-based milvus-lite
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    from feast import FeatureStore, FeatureService
    from feast.vector_store import FeastVectorStore
    from feast.infra.registry.registry import Registry
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logging.warning("Feast not available. FeastRagRetriever will not work.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Settings import
from config.settings import get_settings

logger = logging.getLogger(__name__)


class FeastRagRetriever:
    """
    RAG document retriever using Feast's file-based milvus-lite capabilities
    """
    
    def __init__(self, repo_path: Optional[str] = None):
        if not FEAST_AVAILABLE:
            raise ImportError("Feast is required for FeastRagRetriever. Install with: pip install feast")
        
        self.settings = get_settings()
        repo_path = repo_path or self.settings.feast_repo_path
        self.repo_path = Path(repo_path)
        self.store = None
        self.vector_store = None
        self.embedding_model = None
        self._initialize_store()
    
    def _initialize_store(self) -> None:
        """Initialize the Feast feature store and vector store"""
        try:
            if self.repo_path.exists():
                self.store = FeatureStore(repo_path=str(self.repo_path))
                
                # Initialize embedding model for vector operations
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.embedding_model = SentenceTransformer(self.settings.embedding_model)
                    logger.info("Initialized embedding model")
                
                # Find document embeddings feature view
                document_fv = None
                for fv in self.store.list_feature_views():
                    if fv.name == "document_embeddings":
                        document_fv = fv
                        break
                
                if document_fv:
                    # Initialize Feast vector store (file-based milvus-lite)
                    try:
                        self.vector_store = FeastVectorStore(
                            repo_path=str(self.repo_path),
                            rag_view=document_fv,
                            features=["vector", "chunk_text", "document_title", "chunk_index", "file_path"]
                        )
                        logger.info("Initialized Feast vector store")
                    except Exception as e:
                        logger.warning(f"Could not initialize Feast vector store: {e}")
            else:
                logger.error(f"Feast repository not found at {self.repo_path}")
                raise FileNotFoundError(f"Feast repository not found at {self.repo_path}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Feast store: {e}")
            raise
    

    async def retrieve_documents(
        self, 
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filters: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents using Feast vector store (file-based milvus-lite)
        
        Args:
            query: The query string
            top_k: Number of top documents to retrieve
            similarity_threshold: Minimum similarity score threshold
            filters: Optional filter expression (not used in current implementation)
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            # Use Feast vector store with file-based milvus-lite
            if self.vector_store and self.embedding_model:
                # Generate query embedding
                query_embedding = self.embedding_model.encode([query])[0]
                
                # Query Feast vector store
                response = self.vector_store.query(
                    query_vector=query_embedding,
                    top_k=top_k
                )
                
                # Convert to dictionary format
                result_dict = response.to_dict()
                
                # Format results to match our interface
                documents = []
                chunk_texts = result_dict.get("chunk_text", [])
                document_titles = result_dict.get("document_title", [])
                chunk_indices = result_dict.get("chunk_index", [])
                file_paths = result_dict.get("file_path", [])
                
                for i in range(len(chunk_texts)):
                    # Calculate similarity score (simple rank-based approximation)
                    similarity_score = 1.0 - (i * 0.1)  # Simple approximation based on rank
                    
                    if similarity_score >= similarity_threshold:
                        doc_data = {
                            "document_id": f"doc_{i}",
                            "text": chunk_texts[i] if i < len(chunk_texts) else "",
                            "similarity_score": float(similarity_score),
                            "chunk_text": chunk_texts[i] if i < len(chunk_texts) else "",
                            "document_title": document_titles[i] if i < len(document_titles) else "Unknown",
                            "chunk_index": chunk_indices[i] if i < len(chunk_indices) else 0,
                            "file_path": file_paths[i] if i < len(file_paths) else "",
                            "metadata": {
                                "document_title": document_titles[i] if i < len(document_titles) else "Unknown",
                                "chunk_index": chunk_indices[i] if i < len(chunk_indices) else 0,
                                "file_path": file_paths[i] if i < len(file_paths) else ""
                            }
                        }
                        documents.append(doc_data)
                
                logger.info(f"Retrieved {len(documents)} documents using Feast vector store")
                return documents
            
            logger.error("Feast vector store or embedding model not available")
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    

    async def retrieve_with_feast_features(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        include_similarity_features: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using Feast's on-demand feature computation
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of top documents to retrieve
            include_similarity_features: Whether to compute on-demand similarity features
            
        Returns:
            List of documents with computed features
        """
        # First get candidate documents
        candidate_docs = await self.retrieve_documents(query_embedding, top_k * 2)  # Get more candidates
        
        if not candidate_docs or not include_similarity_features:
            return candidate_docs[:top_k]
        
        try:
            # Prepare entity dataframe for Feast
            entity_data = []
            for doc in candidate_docs:
                entity_data.append({
                    "document_id": doc["document_id"],
                    "query_embedding": query_embedding,
                    "event_timestamp": pd.Timestamp.now()
                })
            
            entity_df = pd.DataFrame(entity_data)
            
            # Get features including on-demand similarity computation
            feature_vector = self.store.get_historical_features(
                entity_df=entity_df,
                features=[
                    "document_embeddings:embedding",
                    "document_embeddings:chunk_text", 
                    "document_embeddings:document_title",
                    "document_embeddings:chunk_index",
                    "document_embeddings:file_path",
                    "similarity_features:similarity_score",
                    "similarity_features:is_relevant"
                ]
            ).to_df()
            
            # Merge with original document data
            enhanced_docs = []
            for i, doc in enumerate(candidate_docs):
                if i < len(feature_vector):
                    row = feature_vector.iloc[i]
                    doc["similarity_score"] = float(row.get("similarity_score", doc["similarity_score"]))
                    doc["is_relevant"] = row.get("is_relevant", "unknown")
                    doc["feast_features"] = row.to_dict()
                
                enhanced_docs.append(doc)
            
            # Sort by similarity and return top_k
            enhanced_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            return enhanced_docs[:top_k]
        
        except Exception as e:
            logger.error(f"Error computing Feast features: {e}")
            # Fallback to original documents
            return candidate_docs[:top_k]
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored features"""
        try:
            if not self.store:
                return {}
            
            stats = {
                "feature_views": [],
                "entities": [],
                "total_features": 0
            }
            
            # Get feature views
            for fv in self.store.list_feature_views():
                fv_stats = {
                    "name": fv.name,
                    "features": [f.name for f in fv.schema],
                    "entities": [e.name for e in fv.entities],
                    "online": fv.online,
                    "ttl_days": fv.ttl.days if fv.ttl else None
                }
                stats["feature_views"].append(fv_stats)
                stats["total_features"] += len(fv.schema)
            
            # Get entities
            for entity in self.store.list_entities():
                stats["entities"].append({
                    "name": entity.name,
                    "description": entity.description,
                    "value_type": str(entity.dtype)
                })
            
            # File-based milvus-lite stats handled by Feast internally
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting feature statistics: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, bool]:
        """Check the health of the Feast components"""
        health = {
            "feast_store": False,
            "feast_vector_store": False,
            "embedding_model": False,
            "feature_views_available": False
        }
        
        try:
            # Check Feast store
            if self.store:
                health["feast_store"] = True
                
                # Check if feature views are available
                fvs = self.store.list_feature_views()
                health["feature_views_available"] = len(fvs) > 0
                
                # Check Feast vector store (includes file-based milvus-lite)
                if self.vector_store:
                    health["feast_vector_store"] = True
                
                # Check embedding model
                if self.embedding_model:
                    health["embedding_model"] = True
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return health
    
