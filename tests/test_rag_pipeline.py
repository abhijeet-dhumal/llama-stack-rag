"""
Tests for the RAG Pipeline
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import asyncio

from src.rag_pipeline import (
    RAGPipeline, 
    DocumentProcessor, 
    OllamaEmbedder, 
    SemanticChunker, 
    VectorStore,
    BatchProcessor
)


class TestDocumentProcessor:
    """Test DocumentProcessor class"""
    
    def test_init(self):
        """Test DocumentProcessor initialization"""
        processor = DocumentProcessor()
        assert processor is not None
    
    @patch('builtins.open', mock_open(read_data=b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 6\n0000000000 65535 f \ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF'))
    @patch('PyPDF2.PdfReader')
    def test_process_pdf_success(self, mock_pdf_reader):
        """Test successful PDF processing"""
        # Setup mocks
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test content"
        mock_pdf_reader.return_value.pages = [mock_page]
        
        processor = DocumentProcessor()
        result = processor.process_pdf("test.pdf")
        
        assert result['text'] == "Test content"
        assert 'metadata' in result
        assert result['metadata']['source'] == "test.pdf"
    
    def test_process_pdf_failure(self):
        """Test PDF processing failure"""
        processor = DocumentProcessor()
        with pytest.raises(Exception):
            processor.process_pdf("non_existent.pdf")


class TestOllamaEmbedder:
    """Test OllamaEmbedder class"""
    
    @patch('src.rag_pipeline.ollama.Client')
    def test_init(self, mock_client):
        """Test OllamaEmbedder initialization"""
        mock_client.return_value.list.return_value = {'models': []}
        
        embedder = OllamaEmbedder()
        assert embedder.model_name == "nomic-embed-text"
        assert embedder.client is not None
    
    @patch('src.rag_pipeline.ollama.Client')
    def test_embed_text_success(self, mock_client):
        """Test successful text embedding"""
        mock_client.return_value.embeddings.return_value = {
            'embedding': [0.1, 0.2, 0.3]
        }
        
        embedder = OllamaEmbedder()
        result = embedder.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
    
    @patch('src.rag_pipeline.ollama.Client')
    def test_embed_text_failure(self, mock_client):
        """Test text embedding failure"""
        mock_client.return_value.embeddings.side_effect = Exception("Test error")
        
        embedder = OllamaEmbedder()
        with pytest.raises(Exception):
            embedder.embed_text("test text")
    
    @patch('src.rag_pipeline.ollama.Client')
    def test_embed_batch(self, mock_client):
        """Test batch embedding"""
        mock_client.return_value.embeddings.return_value = {
            'embedding': [0.1, 0.2, 0.3]
        }
        
        embedder = OllamaEmbedder()
        result = embedder.embed_batch(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.1, 0.2, 0.3]


class TestSemanticChunker:
    """Test SemanticChunker class"""
    
    @patch('src.rag_pipeline.SentenceTransformer')
    def test_init(self, mock_transformer):
        """Test SemanticChunker initialization"""
        chunker = SemanticChunker()
        assert chunker.model is not None
    
    def test_split_into_sentences(self):
        """Test sentence splitting"""
        chunker = SemanticChunker()
        text = "First sentence. Second sentence! Third sentence?"
        sentences = chunker._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert sentences[0] == "First sentence"
        assert sentences[1] == "Second sentence"
        assert sentences[2] == "Third sentence"
    
    def test_simple_chunk(self):
        """Test simple chunking"""
        chunker = SemanticChunker()
        text = "A" * 1500  # 1500 characters
        chunks = chunker._simple_chunk(text, chunk_size=1000, overlap=200)
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 700  # 1500 - 800 (1000 - 200 overlap)
    
    @patch('src.rag_pipeline.SentenceTransformer')
    def test_chunk_by_similarity_fallback(self, mock_transformer):
        """Test chunking with similarity fallback"""
        mock_transformer.return_value.encode.side_effect = Exception("Test error")
        
        chunker = SemanticChunker()
        text = "A" * 1500
        chunks = chunker.chunk_by_similarity(text)
        
        # Should fallback to simple chunking
        assert len(chunks) >= 1


class TestVectorStore:
    """Test VectorStore class"""
    
    @patch('src.rag_pipeline.chromadb.PersistentClient')
    def test_init(self, mock_client):
        """Test VectorStore initialization"""
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        assert store.collection is not None
    
    @patch('src.rag_pipeline.chromadb.PersistentClient')
    def test_add_documents(self, mock_client):
        """Test adding documents to vector store"""
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        documents = [{
            'id': 'test_1',
            'text': 'test text',
            'embedding': [0.1, 0.2, 0.3],
            'metadata': {'source': 'test.pdf'}
        }]
        
        store.add_documents(documents)
        mock_collection.add.assert_called_once()
    
    @patch('src.rag_pipeline.chromadb.PersistentClient')
    def test_search(self, mock_client):
        """Test searching vector store"""
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['test document']],
            'metadatas': [[{'source': 'test.pdf'}]],
            'distances': [[0.1]]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        results = store.search([0.1, 0.2, 0.3], k=1)
        
        assert len(results) == 1
        assert results[0]['text'] == 'test document'
        assert results[0]['distance'] == 0.1
    
    @patch('src.rag_pipeline.chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_client):
        """Test getting collection statistics"""
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_collection.name = "test_collection"
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore()
        stats = store.get_collection_stats()
        
        assert stats['document_count'] == 5
        assert stats['collection_name'] == "test_collection"


class TestRAGPipeline:
    """Test RAGPipeline class"""
    
    @patch('src.rag_pipeline.OllamaLLM')
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.SemanticChunker')
    @patch('src.rag_pipeline.OllamaEmbedder')
    @patch('src.rag_pipeline.DocumentProcessor')
    def test_init(self, mock_docling, mock_embedder, mock_chunker, mock_vector, mock_llm):
        """Test RAGPipeline initialization"""
        pipeline = RAGPipeline()
        
        assert pipeline.document_processor is not None
        assert pipeline.embedder is not None
        assert pipeline.chunker is not None
        assert pipeline.vector_store is not None
        assert pipeline.llm is not None
    
    @patch('src.rag_pipeline.OllamaLLM')
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.SemanticChunker')
    @patch('src.rag_pipeline.OllamaEmbedder')
    @patch('src.rag_pipeline.DocumentProcessor')
    def test_ingest_document(self, mock_docling, mock_embedder, mock_chunker, mock_vector, mock_llm):
        """Test document ingestion"""
        # Setup mocks
        mock_docling.return_value.process_document.return_value = {
            'text': 'test content',
            'metadata': {'title': 'test.pdf'}
        }
        mock_chunker.return_value.chunk_by_similarity.return_value = ['chunk1', 'chunk2']
        mock_embedder.return_value.embed_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_vector.return_value.add_documents.return_value = None
        
        pipeline = RAGPipeline()
        result = pipeline.ingest_document("test.pdf")
        
        assert 'message' in result
        assert result['chunks_created'] == 2
        assert result['source'] == "test.pdf"
    
    @patch('src.rag_pipeline.OllamaLLM')
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.SemanticChunker')
    @patch('src.rag_pipeline.OllamaEmbedder')
    @patch('src.rag_pipeline.DocumentProcessor')
    def test_query(self, mock_docling, mock_embedder, mock_chunker, mock_vector, mock_llm):
        """Test querying the pipeline"""
        # Setup mocks
        mock_embedder.return_value.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_vector.return_value.search.return_value = [{
            'text': 'relevant text',
            'metadata': {'source': 'test.pdf'},
            'distance': 0.1
        }]
        mock_llm.return_value.generate_response.return_value = "Test answer"
        
        pipeline = RAGPipeline()
        result = pipeline.query("test question")
        
        assert 'answer' in result
        assert result['answer'] == "Test answer"
        assert result['context_used'] == 1
        assert len(result['sources']) == 1
    
    @patch('src.rag_pipeline.OllamaLLM')
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.SemanticChunker')
    @patch('src.rag_pipeline.OllamaEmbedder')
    @patch('src.rag_pipeline.DocumentProcessor')
    def test_query_no_results(self, mock_docling, mock_embedder, mock_chunker, mock_vector, mock_llm):
        """Test querying with no results"""
        # Setup mocks
        mock_embedder.return_value.embed_text.return_value = [0.1, 0.2, 0.3]
        mock_vector.return_value.search.return_value = []
        
        pipeline = RAGPipeline()
        result = pipeline.query("test question")
        
        assert result['answer'] == "I don't have any relevant documents to answer your question."
        assert result['context_used'] == 0
        assert len(result['sources']) == 0
    
    @patch('src.rag_pipeline.OllamaLLM')
    @patch('src.rag_pipeline.VectorStore')
    @patch('src.rag_pipeline.SemanticChunker')
    @patch('src.rag_pipeline.OllamaEmbedder')
    @patch('src.rag_pipeline.DocumentProcessor')
    def test_get_stats(self, mock_docling, mock_embedder, mock_chunker, mock_vector, mock_llm):
        """Test getting pipeline statistics"""
        # Setup mocks
        mock_vector.return_value.get_collection_stats.return_value = {
            'document_count': 5,
            'collection_name': 'test'
        }
        mock_embedder.return_value.model_name = "test_embed_model"
        mock_llm.return_value.model_name = "test_llm_model"
        
        pipeline = RAGPipeline()
        stats = pipeline.get_stats()
        
        assert stats['pipeline_status'] == 'healthy'
        assert stats['embedding_model'] == "test_embed_model"
        assert stats['llm_model'] == "test_llm_model"


class TestBatchProcessor:
    """Test BatchProcessor class"""
    
    def test_init(self):
        """Test BatchProcessor initialization"""
        mock_pipeline = Mock()
        processor = BatchProcessor(mock_pipeline)
        
        assert processor.rag_pipeline == mock_pipeline
        assert processor.max_workers == 4
    
    @pytest.mark.asyncio
    async def test_process_documents_batch(self):
        """Test batch document processing"""
        mock_pipeline = Mock()
        mock_pipeline.ingest_document.return_value = {
            'message': 'Success',
            'chunks_created': 3
        }
        
        processor = BatchProcessor(mock_pipeline)
        results = await processor.process_documents_batch(['test1.pdf', 'test2.pdf'])
        
        assert len(results) == 2
        assert all(r['status'] == 'success' for r in results)
    
    @pytest.mark.asyncio
    async def test_process_documents_batch_with_errors(self):
        """Test batch processing with errors"""
        mock_pipeline = Mock()
        mock_pipeline.ingest_document.side_effect = [
            {'message': 'Success'},  # First succeeds
            Exception('Test error')  # Second fails
        ]
        
        processor = BatchProcessor(mock_pipeline)
        results = await processor.process_documents_batch(['test1.pdf', 'test2.pdf'])
        
        assert len(results) == 2
        assert results[0]['status'] == 'success'
        assert results[1]['status'] == 'error'
        assert 'Test error' in results[1]['error']


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(b'%PDF-1.4\n%Test PDF content\n')
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.integration
    def test_pipeline_integration(self, temp_pdf_file):
        """Test complete pipeline integration"""
        # This would require actual models and services
        # Skip for now, but structure is here for when needed
        pytest.skip("Integration test requires actual services")
    
    @pytest.mark.integration
    def test_api_integration(self):
        """Test API integration"""
        # This would test the actual FastAPI endpoints
        pytest.skip("Integration test requires actual API server")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 