"""
FastAPI application for the RAG Pipeline
Provides REST API endpoints for document ingestion and querying
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import tempfile
import os
from pathlib import Path
import logging
import asyncio
from contextlib import asynccontextmanager

from .feast_rag_pipeline import FeastRAGPipeline
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Feast RAG pipeline instance
feast_pipeline = None
batch_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global feast_pipeline, batch_processor
    
    # Startup
    logger.info("Starting Feast RAG Pipeline initialization...")
    try:
        # Initialize pipeline synchronously during startup
        await initialize_pipeline()
        yield
    except Exception as e:
        logger.error(f"Failed to start Feast RAG Pipeline: {str(e)}")
        raise
    
    # Shutdown
    logger.info("Shutting down Feast RAG Pipeline...")

async def initialize_pipeline():
    """Initialize the Feast RAG pipeline asynchronously"""
    global feast_pipeline, batch_processor
    try:
        logger.info("Initializing Feast RAG Pipeline...")
        pipeline = FeastRAGPipeline()
        feast_pipeline = pipeline  # Explicit assignment
        # Note: batch_processor may need updating for Feast integration
        batch_processor = None  # Skip batch processor for now
        logger.info(f"Feast RAG Pipeline initialized successfully: {feast_pipeline is not None}")
        logger.info(f"Pipeline feast_store: {feast_pipeline.feast_store is not None}")
    except Exception as e:
        logger.error(f"Failed to initialize Feast RAG Pipeline: {str(e)}")
        feast_pipeline = None
        batch_processor = None

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description="A RAG pipeline using Feast feature store with file-based milvus-lite for simplified vector storage and retrieval",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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


@app.get("/")
async def root():
    """Serve the main UI"""
    static_dir = Path(__file__).parent.parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        return {
            "message": "Local RAG Pipeline API",
            "version": "1.0.0",
            "status": "running",
            "note": "UI not found - static files may not be available"
        }

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Local RAG Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    if feast_pipeline is None:
        return {
            "status": "initializing",
            "message": "Feast RAG pipeline is still initializing"
        }
    
    try:
        # Use Feast retriever health check
        if hasattr(feast_pipeline, 'feast_retriever') and feast_pipeline.feast_retriever:
            health = feast_pipeline.feast_retriever.health_check()
            return {
                "status": "healthy",
                "feast_store": str(health.get("feast_store", False)),
                "milvus_connection": str(health.get("milvus_connection", False)),
                "embedding_model": str(health.get("embedding_model", False)),
                "message": "Feast RAG pipeline is running with unified Milvus backend"
            }
        else:
            return {
                "status": "healthy",
                "message": "Feast RAG pipeline is running"
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": f"Feast RAG pipeline error: {str(e)}"
        }


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the Feast RAG system"""
    if feast_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Feast RAG pipeline is still initializing. Please wait for initialization to complete."
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
            
            # Process the document with Feast pipeline (async)
            result = await feast_pipeline.process_document(tmp_file_path, file.filename)
            
            # Check if processing failed
            if result.get("status") == "error":
                error_msg = result.get("message", "Unknown error during processing")
                logger.error(f"Failed to ingest document {file.filename}: {error_msg}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to ingest document: {error_msg}"
                )
            
            # Adapt result format for API response (match IngestionResponse model)
            storage_method = result.get('storage_method', 'feast')
            chunks_created = result.get('chunks_created', 0)
            
            # Create appropriate message based on success/failure
            if chunks_created > 0:
                message = f"Successfully ingested {file.filename} with {chunks_created} chunks using {storage_method}"
            else:
                message = f"Processed {file.filename} but no chunks were created using {storage_method}"
            
            adapted_result = {
                "message": message,
                "chunks_created": chunks_created,
                "source": file.filename,
                "metadata": {
                    "storage_method": storage_method,
                    "status": result.get("status", "success"),
                    "file_name": result.get("file_name", file.filename),
                    "document_id": f"feast_{file.filename}_{chunks_created}"
                }
            }
            
            logger.info(f"Processed: {file.filename} with {chunks_created} chunks (status: {result.get('status')})")
            return IngestionResponse(**adapted_result)
            
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
    """Query the Feast RAG system"""
    if feast_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Feast RAG pipeline is still initializing. Please wait for initialization to complete."
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Use Feast pipeline's async query method
        result = await feast_pipeline.query_documents(
            request.question, 
            top_k=request.context_limit
        )
        
        # Adapt result format for API response (match QueryResponse model)
        context_docs = result.get("context_documents", [])
        adapted_result = {
            "answer": result.get("answer", ""),
            "sources": [
                {
                    "content": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": doc.get("similarity_score", 0.0)
                }
                for doc in context_docs
            ],
            "context_used": result.get("retrieved_chunks", 0),
            "relevance_scores": [doc.get("similarity_score", 0.0) for doc in context_docs]
        }
        
        logger.info(f"Query processed successfully, found {adapted_result['context_used']} relevant chunks using feast")
        
        return QueryResponse(**adapted_result)
        
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
        if feast_pipeline is None:
            # Pipeline not initialized yet
            stats = {
                "pipeline_status": "initializing",
                        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
                "vector_store_stats": {
                    "collection_name": "rag_document_embeddings",
                    "document_count": 0,
                    "backend": "feast_milvus_initializing"
                }
            }
        else:
            # Get real database statistics
            db_stats = await feast_pipeline.get_database_stats()
            
            stats = {
                "pipeline_status": "ready",
                        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model, 
                "vector_store_stats": {
                    "collection_name": db_stats.get("collection_name", "rag_document_embeddings"),
                    "document_count": db_stats.get("document_count", 0),
                    "chunk_count": db_stats.get("chunk_count", 0),
                    "backend": db_stats.get("backend", "feast_milvus_lite")
                }
            }
        
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.get("/documents", response_model=Dict[str, Any])
async def list_documents():
    """List ingested documents from Feast"""
    if feast_pipeline is None:
        return {
            "documents": [],
            "total_documents": 0,
            "collection_name": "rag_document_embeddings",
            "backend": "feast_milvus_unavailable",
            "status": "active"
        }
    
    try:
        documents_data = await feast_pipeline.get_documents_list()
        return {
            "documents": documents_data.get("documents", []),
            "total_documents": documents_data.get("total_documents", 0),
            "collection_name": documents_data.get("collection_name", "rag_document_embeddings"),
            "backend": documents_data.get("backend", "feast_milvus_lite"),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error listing documents from Feast: {str(e)}")
        return {
            "documents": [],
            "total_documents": 0,
            "collection_name": "rag_document_embeddings",
            "backend": "feast_milvus_lite_error",
            "status": "active"
        }


@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector store"""
    if feast_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Feast RAG pipeline is still initializing. Please wait for initialization to complete."
        )
    
    try:
        await feast_pipeline.clear_collection()
        return {
            "status": "success",
            "message": "Successfully cleared all documents from Feast Milvus database",
            "backend": "feast_milvus"
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
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "vector_store": "feast_milvus",
            "feature_store": "feast",
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
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info"
    ) 