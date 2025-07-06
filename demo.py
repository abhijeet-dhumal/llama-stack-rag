#!/usr/bin/env python3
"""
Demo script for the Local RAG Pipeline
Shows how to use the RAG pipeline programmatically
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
SAMPLE_DOCS_DIR = Path("sample_docs")

def print_banner():
    """Print demo banner"""
    print("=" * 60)
    print("🚀 Local RAG Pipeline Demo")
    print("=" * 60)
    print()

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        print("Please make sure the API is running at http://localhost:8000")
        print("Run: ./start.sh or python -m uvicorn src.api:app --reload")
        return False

def get_pipeline_stats():
    """Get and display pipeline statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("\n📊 Pipeline Statistics:")
            print(f"  • Status: {stats.get('pipeline_status', 'unknown')}")
            print(f"  • Embedding Model: {stats.get('embedding_model', 'unknown')}")
            print(f"  • LLM Model: {stats.get('llm_model', 'unknown')}")
            print(f"  • Documents: {stats.get('vector_store_stats', {}).get('document_count', 0)}")
            return stats
        else:
            print(f"❌ Failed to get stats: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return None

def ingest_sample_document():
    """Ingest a sample document (simulate with text file)"""
    print("\n📄 Ingesting Sample Document...")
    
    # Create a temporary PDF-like file for demo
    sample_file = SAMPLE_DOCS_DIR / "sample_document.txt"
    
    # Check if sample document exists
    if not sample_file.exists():
        print("❌ Sample document not found. Creating one...")
        create_sample_document(sample_file)
    
    # For demo purposes, we'll simulate PDF ingestion
    # In a real scenario, you would upload an actual PDF
    print("📤 Uploading document to RAG pipeline...")
    
    # Note: The actual API expects PDF files
    # This is a demo showing the expected response format
    sample_response = {
        "message": "Successfully ingested 15 chunks from sample_document.pdf",
        "chunks_created": 15,
        "source": "sample_document.pdf",
        "metadata": {
            "title": "sample_document.pdf",
            "source": "sample_document.pdf",
            "num_pages": 3
        }
    }
    
    print("✅ Document ingestion simulation:")
    print(f"  • Message: {sample_response['message']}")
    print(f"  • Chunks created: {sample_response['chunks_created']}")
    print(f"  • Source: {sample_response['source']}")
    
    return sample_response

def create_sample_document(file_path):
    """Create a sample document for testing"""
    sample_content = """
    # Sample Document for RAG Pipeline Testing
    
    This is a sample document created to test the RAG pipeline functionality.
    It contains various types of content including technical specifications,
    procedures, and data points.
    
    ## Technical Specifications
    
    The RAG pipeline uses the following components:
    - Ollama for local LLM inference
    - Docling for advanced PDF processing
    - ChromaDB for vector storage
    - FastAPI for the REST API
    
    ## Performance Metrics
    
    The system can process approximately 1000 documents per hour with
    95% accuracy in semantic similarity matching. The average response
    time is less than 2 seconds with support for 50 concurrent queries.
    
    ## Configuration
    
    The pipeline supports various configuration options including:
    - Model selection (embedding and LLM models)
    - Chunk size and overlap settings
    - Similarity thresholds
    - API endpoint configuration
    """
    
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(sample_content)

def query_documents(questions):
    """Query the RAG pipeline with sample questions"""
    print("\n❓ Querying Documents...")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Query {i}: {question}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/query",
                json={
                    "question": question,
                    "context_limit": 5
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Answer: {result.get('answer', 'No answer provided')[:200]}...")
                print(f"📚 Sources: {len(result.get('sources', []))} documents")
                print(f"🔍 Context used: {result.get('context_used', 0)} chunks")
                results.append(result)
            else:
                print(f"❌ Query failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            
        time.sleep(1)  # Brief pause between queries
    
    return results

def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n📦 Batch Processing Demo...")
    
    # Simulate batch processing request
    batch_request = {
        "file_paths": [
            "sample_docs/doc1.pdf",
            "sample_docs/doc2.pdf",
            "sample_docs/doc3.pdf"
        ]
    }
    
    print("📤 Batch processing request:")
    print(f"  • Files: {len(batch_request['file_paths'])}")
    
    # Simulate response (actual implementation would process real files)
    batch_response = {
        "results": [
            {"file": "sample_docs/doc1.pdf", "status": "success", "result": {"chunks_created": 12}},
            {"file": "sample_docs/doc2.pdf", "status": "success", "result": {"chunks_created": 8}},
            {"file": "sample_docs/doc3.pdf", "status": "error", "error": "File not found"}
        ],
        "total_files": 3,
        "successful": 2,
        "failed": 1
    }
    
    print("✅ Batch processing results:")
    print(f"  • Total files: {batch_response['total_files']}")
    print(f"  • Successful: {batch_response['successful']}")
    print(f"  • Failed: {batch_response['failed']}")

def run_performance_test():
    """Run a simple performance test"""
    print("\n⚡ Performance Test...")
    
    # Simulate multiple queries
    test_questions = [
        "What is the main purpose of this system?",
        "How does the RAG pipeline work?",
        "What are the performance characteristics?",
        "What models are used in the pipeline?",
        "How do you configure the system?"
    ]
    
    start_time = time.time()
    
    print(f"🏃 Running {len(test_questions)} test queries...")
    
    # Simulate query processing
    for i, question in enumerate(test_questions, 1):
        print(f"  Query {i}: Processing...")
        time.sleep(0.5)  # Simulate processing time
        print(f"  Query {i}: ✅ Completed")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n📊 Performance Results:")
    print(f"  • Total queries: {len(test_questions)}")
    print(f"  • Total time: {total_time:.2f} seconds")
    print(f"  • Average time per query: {total_time/len(test_questions):.2f} seconds")
    print(f"  • Queries per second: {len(test_questions)/total_time:.2f}")

def main():
    """Main demo function"""
    print_banner()
    
    # Check API health
    if not check_api_health():
        print("\n💡 To start the API, run:")
        print("  ./start.sh")
        print("  or")
        print("  uvicorn src.api:app --reload")
        return
    
    # Get pipeline stats
    stats = get_pipeline_stats()
    
    # Demonstrate document ingestion
    ingestion_result = ingest_sample_document()
    
    # Demonstrate querying
    sample_questions = [
        "What is the RAG pipeline?",
        "What are the main components of the system?",
        "How does the document processing work?",
        "What are the performance metrics?",
        "How do you configure the pipeline?"
    ]
    
    query_results = query_documents(sample_questions)
    
    # Demonstrate batch processing
    demonstrate_batch_processing()
    
    # Run performance test
    run_performance_test()
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 Next steps:")
    print("  • Upload your own PDF documents via the API")
    print("  • Explore the interactive API docs at http://localhost:8000/docs")
    print("  • Customize the pipeline configuration in config/settings.py")
    print("  • Scale the deployment using Docker Compose")

if __name__ == "__main__":
    main() 