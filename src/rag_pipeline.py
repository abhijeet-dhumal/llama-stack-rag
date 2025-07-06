"""
Local RAG Pipeline Implementation
Combines PyPDF2, Ollama, and Llama-stack for document processing and retrieval
"""

import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio

import ollama
import chromadb
from chromadb.config import Settings
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Simple document processing using PyPDF2 and python-docx"""
    
    def __init__(self):
        pass
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document and extract content"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return self.process_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self.process_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self.process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF and extract text content"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page in pdf_reader.pages:
                    text_content.append(page.extract_text())
                
                full_text = '\n'.join(text_content)
                
                content = {
                    'text': full_text,
                    'metadata': {
                        'source': pdf_path,
                        'num_pages': len(pdf_reader.pages),
                        'title': os.path.basename(pdf_path),
                        'file_type': 'pdf'
                    }
                }
                
                logger.info(f"Processed PDF: {pdf_path}, extracted {len(full_text)} characters from {len(pdf_reader.pages)} pages")
                return content
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def process_docx(self, docx_path: str) -> Dict[str, Any]:
        """Process DOCX file and extract text content"""
        try:
            doc = Document(docx_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)
            
            full_text = '\n'.join(text_content)
            
            content = {
                'text': full_text,
                'metadata': {
                    'source': docx_path,
                    'num_paragraphs': len(doc.paragraphs),
                    'title': os.path.basename(docx_path),
                    'file_type': 'docx'
                }
            }
            
            logger.info(f"Processed DOCX: {docx_path}, extracted {len(full_text)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error processing DOCX {docx_path}: {str(e)}")
            raise
    
    def process_text(self, text_path: str) -> Dict[str, Any]:
        """Process text file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                content_text = file.read()
            
            content = {
                'text': content_text,
                'metadata': {
                    'source': text_path,
                    'title': os.path.basename(text_path),
                    'file_type': 'text'
                }
            }
            
            logger.info(f"Processed text file: {text_path}, extracted {len(content_text)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error processing text file {text_path}: {str(e)}")
            raise


class OllamaEmbedder:
    """Embedding generation using Ollama"""
    
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.client = ollama.Client()
        self._ensure_model()
    
    def _ensure_model(self):
        """Ensure the embedding model is available"""
        try:
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.info(f"Pulling embedding model: {self.model_name}")
                self.client.pull(self.model_name)
        except Exception as e:
            logger.warning(f"Could not verify model availability: {str(e)}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings


class SemanticChunker:
    """Semantic-based text chunking"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def chunk_by_similarity(self, text: str, threshold: float = 0.7, 
                          max_chunk_size: int = 1000) -> List[str]:
        """Chunk text based on semantic similarity"""
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [text]
        
        try:
            embeddings = self.model.encode(sentences)
            chunks = []
            current_chunk = [sentences[0]]
            current_size = len(sentences[0])
            
            for i in range(1, len(sentences)):
                similarity = cosine_similarity(
                    [embeddings[i-1]], 
                    [embeddings[i]]
                )[0][0]
                
                sentence_size = len(sentences[i])
                
                if (similarity > threshold and 
                    current_size + sentence_size < max_chunk_size):
                    current_chunk.append(sentences[i])
                    current_size += sentence_size
                else:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_size = sentence_size
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            # Fallback to simple chunking
            return self._simple_chunk(text, max_chunk_size)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _simple_chunk(self, text: str, chunk_size: int = 1000, 
                     overlap: int = 200) -> List[str]:
        """Simple overlapping text chunking"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks


class VectorStore:
    """Vector storage using ChromaDB"""
    
    def __init__(self, collection_name: str = "documents", 
                 persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if not documents:
            return
        
        ids = [doc['id'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        documents_text = [doc['text'] for doc in documents]
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            return [
                {
                    'text': doc,
                    'metadata': meta,
                    'distance': dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                'document_count': count,
                'collection_name': self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'document_count': 0}


class OllamaLLM:
    """Local LLM using Ollama"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        self.client = ollama.Client()
        self._ensure_model()
    
    def _ensure_model(self):
        """Ensure the LLM model is available"""
        try:
            models = self.client.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.model_name not in model_names:
                logger.info(f"Pulling LLM model: {self.model_name}")
                self.client.pull(self.model_name)
        except Exception as e:
            logger.warning(f"Could not verify model availability: {str(e)}")
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using local LLM"""
        try:
            full_prompt = f"""
Context: {context}

Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, say so clearly.
"""
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                stream=False
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"


class RAGPipeline:
    """Main RAG Pipeline orchestrator"""
    
    def __init__(self, 
                 embedding_model: str = "nomic-embed-text",
                 llm_model: str = "llama3.2:3b",
                 collection_name: str = "documents"):
        
        self.document_processor = DocumentProcessor()
        self.embedder = OllamaEmbedder(embedding_model)
        self.chunker = SemanticChunker()
        self.vector_store = VectorStore(collection_name)
        self.llm = OllamaLLM(llm_model)
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_document(self, pdf_path: str, use_semantic_chunking: bool = True) -> Dict[str, Any]:
        """Ingest a PDF document into the RAG system"""
        try:
            # Process document
            content = self.document_processor.process_document(pdf_path)
            
            # Chunk the content
            if use_semantic_chunking:
                chunks = self.chunker.chunk_by_similarity(content['text'])
            else:
                chunks = self.chunker._simple_chunk(content['text'])
            
            # Generate embeddings
            embeddings = self.embedder.embed_batch(chunks)
            
            # Prepare documents for storage
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc_id = f"{Path(pdf_path).stem}_{i}"
                documents.append({
                    'id': doc_id,
                    'text': chunk,
                    'embedding': embedding,
                    'metadata': {
                        'source': pdf_path,
                        'chunk_index': i,
                        'document_type': 'pdf',
                        'title': content['metadata']['title']
                    }
                })
            
            # Store in vector database
            self.vector_store.add_documents(documents)
            
            result = {
                'message': f"Successfully ingested {len(chunks)} chunks from {pdf_path}",
                'chunks_created': len(chunks),
                'source': pdf_path,
                'metadata': content['metadata']
            }
            
            logger.info(f"Document ingestion completed: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Error ingesting document {pdf_path}: {str(e)}")
            raise
    
    def query(self, question: str, context_limit: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(question)
            
            # Search for relevant documents
            relevant_docs = self.vector_store.search(
                query_embedding, 
                k=context_limit
            )
            
            if not relevant_docs:
                return {
                    'answer': "I don't have any relevant documents to answer your question.",
                    'sources': [],
                    'context_used': 0,
                    'relevance_scores': []
                }
            
            # Build context
            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            
            # Generate response using local LLM
            answer = self.llm.generate_response(question, context)
            
            result = {
                'answer': answer,
                'sources': [doc['metadata'] for doc in relevant_docs],
                'context_used': len(relevant_docs),
                'relevance_scores': [1 - doc['distance'] for doc in relevant_docs]
            }
            
            logger.info(f"Query processed successfully: {question[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            return {
                'pipeline_status': 'healthy',
                'vector_store_stats': vector_stats,
                'embedding_model': self.embedder.model_name,
                'llm_model': self.llm.model_name
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {'pipeline_status': 'error', 'error': str(e)}


class BatchProcessor:
    """Batch processing for multiple documents"""
    
    def __init__(self, rag_pipeline: RAGPipeline, max_workers: int = 4):
        self.rag_pipeline = rag_pipeline
        self.max_workers = max_workers
    
    async def process_documents_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor, 
                    self.rag_pipeline.ingest_document, 
                    file_path
                )
                for file_path in file_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'file': file_paths[i],
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed_results.append({
                    'file': file_paths[i],
                    'status': 'success',
                    'result': result
                })
        
        return processed_results


if __name__ == "__main__":
    # Example usage
    rag = RAGPipeline()
    
    # Test with a sample query
    stats = rag.get_stats()
    print(f"Pipeline stats: {stats}") 