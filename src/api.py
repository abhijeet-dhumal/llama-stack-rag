"""
FastAPI application for the RAG Pipeline
Provides REST API endpoints for document ingestion and querying
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import tempfile
import os
from pathlib import Path
import logging
import asyncio
from contextlib import asynccontextmanager

from .rag_pipeline import RAGPipeline, BatchProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline = None
batch_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global rag_pipeline, batch_processor
    
    # Startup
    logger.info("Starting RAG Pipeline initialization...")
    try:
        # Start initialization in background
        asyncio.create_task(initialize_pipeline())
        yield
    except Exception as e:
        logger.error(f"Failed to start RAG Pipeline: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down RAG Pipeline...")

async def initialize_pipeline():
    """Initialize the RAG pipeline asynchronously"""
    global rag_pipeline, batch_processor
    try:
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline()
        batch_processor = BatchProcessor(rag_pipeline)
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
        rag_pipeline = None
        batch_processor = None

# Create FastAPI app
app = FastAPI(
    title="Local RAG Pipeline",
    description="A local RAG pipeline using Ollama, Docling, and ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    context_limit: int = Field(5, ge=1, le=20, description="Maximum number of context chunks to use")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    context_used: int
    relevance_scores: List[float]

class IngestionResponse(BaseModel):
    message: str
    chunks_created: int
    source: str
    metadata: Dict[str, Any]

class StatsResponse(BaseModel):
    pipeline_status: str
    vector_store_stats: Dict[str, Any]
    embedding_model: str
    llm_model: str

class BatchIngestionRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths to ingest")

class BatchIngestionResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_files: int
    successful: int
    failed: int


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Local RAG Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        return {
            "status": "initializing",
            "message": "RAG pipeline is still initializing (downloading models)"
        }
    
    try:
        stats = rag_pipeline.get_stats()
        return {
            "status": "healthy",
            "pipeline_status": stats.get("pipeline_status", "unknown"),
            "message": "RAG pipeline is running"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"RAG pipeline error: {str(e)}"
        }


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a PDF document into the RAG system"""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is still initializing. Please wait for model download to complete."
        )
    
    if not file.filename.endswith(('.pdf', '.md', '.txt', '.docx')):
        raise HTTPException(
            status_code=400, 
            detail="Supported file types: PDF, Markdown, Text, and Word documents."
        )
    
    # Create temporary file with correct extension
    file_extension = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        try:
            # Read and save uploaded file
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()  # Ensure content is written to disk
            tmp_file_path = tmp_file.name
            
            logger.info(f"Processing uploaded file: {file.filename}")
            
            # Process the document
            result = rag_pipeline.ingest_document(tmp_file_path)
            
            logger.info(f"Successfully ingested: {file.filename}")
            return IngestionResponse(**result)
            
        except Exception as e:
            logger.error(f"Error ingesting document {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to ingest document: {str(e)}"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


@app.post("/ingest/batch", response_model=BatchIngestionResponse)
async def ingest_documents_batch(
    background_tasks: BackgroundTasks,
    request: BatchIngestionRequest
):
    """Ingest multiple documents in batch"""
    if not request.file_paths:
        raise HTTPException(
            status_code=400,
            detail="No file paths provided"
        )
    
    # Validate file paths
    valid_paths = []
    for file_path in request.file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        if not file_path.endswith(('.pdf', '.md', '.txt', '.docx')):
            logger.warning(f"Skipping unsupported file: {file_path}")
            continue
        valid_paths.append(file_path)
    
    if not valid_paths:
        raise HTTPException(
            status_code=400,
            detail="No valid files found (supported: PDF, Markdown, Text, Word)"
        )
    
    try:
        # Process documents in batch
        results = await batch_processor.process_documents_batch(valid_paths)
        
        # Count successful and failed
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        logger.info(f"Batch ingestion completed: {successful} successful, {failed} failed")
        
        return BatchIngestionResponse(
            results=results,
            total_files=len(valid_paths),
            successful=successful,
            failed=failed
        )
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch ingestion failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is still initializing. Please wait for model download to complete."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        result = rag_pipeline.query(
            request.question, 
            context_limit=request.context_limit
        )
        
        logger.info(f"Query processed successfully, found {result['context_used']} relevant chunks")
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse)
async def get_pipeline_stats():
    """Get pipeline statistics"""
    try:
        stats = rag_pipeline.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """List ingested documents"""
    try:
        stats = rag_pipeline.get_stats()
        return {
            "total_documents": stats['vector_store_stats']['document_count'],
            "collection_name": stats['vector_store_stats']['collection_name'],
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    try:
        # This would require implementing a clear method in the vector store
        # For now, return a message
        return {
            "message": "Document clearing not implemented yet",
            "status": "pending"
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List available models"""
    try:
        return {
            "embedding_model": rag_pipeline.embedder.model_name,
            "llm_model": rag_pipeline.llm.model_name,
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "type": "http_error"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred",
            "type": "server_error"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 