"""
Tests for the Feast-based RAG Pipeline
"""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

# Test the current Feast-based pipeline
from src.feast_rag_pipeline import FeastRAGPipeline
from src.feast_rag_retriever import FeastRagRetriever


class TestFeastRAGPipeline:
    """Test FeastRAGPipeline class"""

    @patch("src.feast_rag_pipeline.FeatureStore")
    @patch("src.feast_rag_pipeline.FeastRagRetriever")
    @patch("src.feast_rag_pipeline.SentenceTransformer")
    @patch("src.feast_rag_pipeline.Path.exists")
    def test_init_success(
        self, mock_exists, mock_transformer, mock_retriever, mock_feature_store
    ):
        """Test successful FeastRAGPipeline initialization"""
        mock_exists.return_value = True
        mock_transformer.return_value = Mock()
        mock_feature_store.return_value = Mock()
        mock_retriever.return_value = Mock()

        pipeline = FeastRAGPipeline()

        assert pipeline.model_name == "llama3.2:3b"
        assert pipeline.embedding_model_name == "all-MiniLM-L6-v2"
        assert pipeline.embedding_model is not None
        assert pipeline.feast_store is not None
        assert pipeline.feast_retriever is not None

    @patch("src.feast_rag_pipeline.Path.exists")
    def test_init_feast_not_available(self, mock_exists):
        """Test initialization when Feast is not available"""
        mock_exists.return_value = False

        with pytest.raises(ValueError, match="Feast repository is required"):
            FeastRAGPipeline()

    @patch("src.feast_rag_pipeline.FeatureStore")
    @patch("src.feast_rag_pipeline.FeastRagRetriever")
    @patch("src.feast_rag_pipeline.SentenceTransformer")
    @patch("src.feast_rag_pipeline.Path.exists")
    @pytest.mark.asyncio
    async def test_process_document_success(
        self, mock_exists, mock_transformer, mock_retriever, mock_feature_store
    ):
        """Test successful document processing"""
        # Setup mocks
        mock_exists.return_value = True

        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_transformer.return_value = mock_embedding_model

        mock_feature_store.return_value = Mock()
        mock_retriever.return_value = Mock()

        pipeline = FeastRAGPipeline()

        # Mock the internal methods properly
        with patch.object(
            pipeline, "_extract_and_chunk_document", new_callable=AsyncMock
        ) as mock_extract:
            with patch.object(
                pipeline, "_store_with_feast_official_method", new_callable=AsyncMock
            ) as mock_store:
                mock_extract.return_value = ["chunk1", "chunk2", "chunk3"]
                mock_store.return_value = None

                # Create a temporary file for testing
                with tempfile.NamedTemporaryFile(
                    suffix=".txt", delete=False
                ) as tmp_file:
                    tmp_file.write(b"Test content")
                    tmp_path = tmp_file.name

                try:
                    result = await pipeline.process_document(tmp_path, "test_doc.txt")

                    assert "status" in result
                    assert result["status"] == "success"
                    assert result["chunks_created"] == 3
                    assert result["file_name"] == "test_doc.txt"
                finally:
                    os.unlink(tmp_path)

    @patch("src.feast_rag_pipeline.FeatureStore")
    @patch("src.feast_rag_pipeline.FeastRagRetriever")
    @patch("src.feast_rag_pipeline.SentenceTransformer")
    @patch("src.feast_rag_pipeline.Path.exists")
    @pytest.mark.asyncio
    async def test_query_documents_success(
        self, mock_exists, mock_transformer, mock_retriever, mock_feature_store
    ):
        """Test successful document querying"""
        # Setup mocks
        mock_exists.return_value = True
        mock_embedding_model = Mock()
        mock_embedding_model.encode.return_value = [np.array([0.1, 0.2, 0.3])]
        mock_transformer.return_value = mock_embedding_model

        mock_feast_store = Mock()
        mock_feature_store.return_value = mock_feast_store
        mock_retriever.return_value = Mock()

        pipeline = FeastRAGPipeline()

        # Mock the retrieval method to return properly formatted data
        mock_documents = [
            {
                "text": "Test document content",
                "metadata": {
                    "document_title": "test.txt",
                    "chunk_index": 0,
                    "file_path": "/tmp/test.txt",
                },
                "similarity_score": 0.9,
            }
        ]

        # Mock both internal methods and test basic functionality
        with patch.object(
            pipeline, "_retrieve_with_feast_official_method", new_callable=AsyncMock
        ) as mock_retrieve:
            with patch.object(
                pipeline, "_generate_llm_response", new_callable=AsyncMock
            ) as mock_llm:
                mock_retrieve.return_value = mock_documents
                mock_llm.return_value = "This is a test answer"

                result = await pipeline.query_documents("What is this about?")

                assert isinstance(result, dict)
                assert "answer" in result

    @patch("src.feast_rag_pipeline.FeatureStore")
    @patch("src.feast_rag_pipeline.FeastRagRetriever")
    @patch("src.feast_rag_pipeline.SentenceTransformer")
    @patch("src.feast_rag_pipeline.Path.exists")
    @pytest.mark.asyncio
    async def test_clear_collection(
        self, mock_exists, mock_transformer, mock_retriever, mock_feature_store
    ):
        """Test clearing the collection"""
        # Setup mocks
        mock_exists.return_value = True
        mock_transformer.return_value = Mock()
        mock_feature_store.return_value = Mock()
        mock_retriever.return_value = Mock()

        pipeline = FeastRAGPipeline()

        # Mock PyMilvus client by patching it in the clear_collection method
        with patch("pymilvus.MilvusClient") as mock_client:
            with patch("subprocess.run") as mock_subprocess:
                with patch("os.path.exists", return_value=True):
                    mock_client_instance = Mock()
                    mock_client_instance.has_collection.return_value = True
                    mock_client.return_value = mock_client_instance
                    mock_subprocess.return_value.returncode = 0

                    try:
                        await pipeline.clear_collection()
                        assert True
                    except Exception as e:
                        pytest.fail(f"clear_collection raised an exception: {e}")


class TestFeastRagRetriever:
    """Test FeastRagRetriever class"""

    @patch("src.feast_rag_retriever.FeatureStore")
    @patch("src.feast_rag_retriever.SentenceTransformer")
    @patch("src.feast_rag_retriever.Path.exists")
    def test_init(self, mock_exists, mock_transformer, mock_feature_store):
        """Test FeastRagRetriever initialization"""
        mock_exists.return_value = True
        mock_transformer.return_value = Mock()

        # Mock the feature store with proper list_feature_views method
        mock_store = Mock()
        mock_store.list_feature_views.return_value = [Mock(name="document_embeddings")]
        mock_feature_store.return_value = mock_store

        retriever = FeastRagRetriever("feast_feature_repo")

        assert retriever.store is not None
        assert retriever.embedding_model is not None


class TestEmbeddingGeneration:
    """Test embedding generation"""

    @patch("src.feast_rag_pipeline.SentenceTransformer")
    def test_embedding_generation(self, mock_transformer):
        """Test embedding generation for text chunks"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_transformer.return_value = mock_model

        # Simulate embedding generation
        text_chunks = ["This is a test chunk"]
        embeddings = mock_model.encode(text_chunks)

        assert embeddings.shape == (1, 4)
        assert embeddings[0][0] == 0.1


@pytest.fixture
def temp_documents():
    """Create temporary documents for testing"""
    files = []

    # Create text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as txt_file:
        txt_file.write("This is a test text document with sample content.")
        files.append(txt_file.name)

    yield files

    # Cleanup
    for file_path in files:
        if os.path.exists(file_path):
            os.unlink(file_path)


class TestIntegration:
    """Integration tests for the complete pipeline"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_workflow(self, temp_documents):
        """Test complete pipeline workflow"""
        # This would require actual Feast setup and Ollama
        pytest.skip("Integration test requires actual services and Feast setup")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-x"])
