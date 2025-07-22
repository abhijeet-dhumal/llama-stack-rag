"""
Utility Functions for RAG LlamaStack Application
Common helper functions used across modules
"""

import streamlit as st
import numpy as np
from typing import List, Dict, Any, Union


def format_file_size(size_bytes: int) -> str:
    """Convert file size in bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    # Use 1 decimal place for MB and above, 0 decimals for KB and below
    if i >= 2:  # MB and above
        return f"{size_bytes:.1f}{size_names[i]}"
    else:  # KB and below
        return f"{size_bytes:.0f}{size_names[i]}"


def initialize_session_state() -> None:
    """Initialize all required session state variables"""
    # Debug: Check if we're preserving existing data
    existing_docs = len(getattr(st.session_state, 'uploaded_documents', []))
    if existing_docs > 0:
        print(f"🔄 Rerun detected - preserving {existing_docs} existing documents")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # LlamaStack client
    if 'llamastack_client' not in st.session_state:
        try:
            from .llamastack_client import LlamaStackClient
            st.session_state.llamastack_client = LlamaStackClient()
        except ImportError:
            # Create a minimal fallback client
            st.session_state.llamastack_client = create_fallback_client()
    
    # Available models - will be loaded when needed
    if 'available_models' not in st.session_state:
        st.session_state.available_models = {"embedding": [], "llm": [], "all": []}
    
    # Selected models
    if 'selected_embedding_model' not in st.session_state:
        st.session_state.selected_embedding_model = "all-MiniLM-L6-v2"
    
    if 'selected_llm_model' not in st.session_state:
        st.session_state.selected_llm_model = "llama3.2:1b"
    
    # File processing tracking with interruption handling
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()
    
    # Upload state tracking
    if 'currently_uploading' not in st.session_state:
        st.session_state.currently_uploading = set()
        
    if 'failed_uploads' not in st.session_state:
        st.session_state.failed_uploads = set()
        
    if 'upload_interrupted' not in st.session_state:
        st.session_state.upload_interrupted = False
    
    # Document storage (multiple formats for compatibility) - PRESERVE EXISTING DATA
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        print("📄 Initialized empty documents storage")
    else:
        print(f"📄 Preserved {len(st.session_state.documents)} documents in storage")
    
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
        print("📋 Initialized empty uploaded_documents storage")
    else:
        print(f"📋 Preserved {len(st.session_state.uploaded_documents)} uploaded documents")
    
    # Theme state
    if 'dark_theme' not in st.session_state:
        st.session_state.dark_theme = False  # Light theme default
    
    # Page navigation state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None  # Default to main page


def create_fallback_client():
    """Create a minimal fallback client when backend import fails"""
    class FallbackClient:
        def health_check(self):
            return True  # Always return True for demo
        
        def get_available_models(self):
            return {
                "embedding": [{"identifier": "all-MiniLM-L6-v2", "name": "all-MiniLM-L6-v2"}],
                "llm": [{"identifier": "llama3.2:1b", "name": "llama3.2:1b"}],
                "all": [
                    {"identifier": "all-MiniLM-L6-v2", "name": "all-MiniLM-L6-v2"},
                    {"identifier": "llama3.2:1b", "name": "llama3.2:1b"}
                ]
            }
        
        def get_embeddings(self, text, model="all-MiniLM-L6-v2"):
            # Return varied dummy embeddings
            import random
            dummy_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
            # Add variation based on text content
            text_hash = hash(text) % 1000
            for i in range(min(10, len(dummy_embedding))):
                dummy_embedding[i] += (text_hash / 10000.0)
            return dummy_embedding
        
        def chat_completion(self, user_prompt, system_prompt="", model="llama3.2:1b"):
            return f"Demo response to: {user_prompt[:50]}... (LlamaStack client not available - using model: {model})"
    
    return FallbackClient()


def get_context_type(content: str) -> str:
    """Determine the type of content based on keywords"""
    if not content:
        return "unknown content"
    
    content_lower = content.lower()
    
    # Check for specific patterns
    if any(word in content_lower for word in ['research', 'study', 'findings', 'methodology', 'conclusion']):
        return "research or academic document"
    elif any(word in content_lower for word in ['policy', 'guideline', 'procedure', 'regulation', 'compliance']):
        return "policy or procedural document"
    elif any(word in content_lower for word in ['manual', 'instructions', 'guide', 'tutorial', 'how to']):
        return "instructional or reference material"
    elif any(word in content_lower for word in ['contract', 'agreement', 'legal', 'terms', 'conditions']):
        return "legal or contractual document"
    elif any(word in content_lower for word in ['report', 'analysis', 'summary', 'overview', 'assessment']):
        return "analytical or summary report"
    elif any(word in content_lower for word in ['software', 'code', 'implementation', 'system']):
        return "software development or technical implementation"
    else:
        return "a technical or academic document"


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """Calculate cosine similarity between two vectors and normalize to 0-1 range for relevance"""
    try:
        # Convert numpy arrays to lists to avoid ambiguity issues
        if isinstance(vec1, np.ndarray):
            vec1 = vec1.tolist()
        if isinstance(vec2, np.ndarray):
            vec2 = vec2.tolist()
        
        # Debug logging
        print(f"🔍 DEBUG: cosine_similarity called with vec1 length: {len(vec1)}, vec2 length: {len(vec2)}")
        print(f"🔍 DEBUG: vec1 type: {type(vec1)}, vec2 type: {type(vec2)}")
        
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            print(f"❌ DEBUG: Invalid vectors - vec1: {bool(vec1)}, vec2: {bool(vec2)}, lengths: {len(vec1) if vec1 else 0} vs {len(vec2) if vec2 else 0}")
            return 0.0
        
        # Check if vectors contain numpy arrays or other non-scalar values
        if any(isinstance(x, (list, tuple)) for x in vec1) or any(isinstance(x, (list, tuple)) for x in vec2):
            print(f"❌ DEBUG: Vectors contain non-scalar values - vec1: {[type(x) for x in vec1[:3]]}, vec2: {[type(x) for x in vec2[:3]]}")
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        print(f"🔍 DEBUG: dot_product: {dot_product}, magnitude1: {magnitude1}, magnitude2: {magnitude2}")
        
        if magnitude1 == 0 or magnitude2 == 0:
            print(f"❌ DEBUG: Zero magnitude detected - magnitude1: {magnitude1}, magnitude2: {magnitude2}")
            return 0.0
        
        # Calculate cosine similarity (-1 to 1)
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        # Normalize to 0-1 range for relevance percentage
        # Convert from [-1, 1] to [0, 1] where:
        # -1 (opposite) -> 0% relevance
        # 0 (orthogonal) -> 50% relevance  
        # 1 (identical) -> 100% relevance
        relevance_score = (cosine_sim + 1) / 2
        
        result = max(0.0, min(1.0, relevance_score))
        print(f"🔍 DEBUG: cosine_similarity result: {result:.4f}")
        return result
        
    except Exception as e:
        print(f"❌ DEBUG: Error in cosine_similarity: {e}")
        print(f"❌ DEBUG: vec1 sample: {vec1[:3] if vec1 else 'None'}")
        print(f"❌ DEBUG: vec2 sample: {vec2[:3] if vec2 else 'None'}")
        return 0.0


def validate_llamastack_connection() -> bool:
    """Validate connection to LlamaStack"""
    try:
        return st.session_state.llamastack_client.health_check()
    except Exception:
        return False


def safe_get_session_state(key: str, default: Any = None) -> Any:
    """Safely get a value from session state with a default"""
    return getattr(st.session_state, key, default)


def reset_session_state() -> None:
    """Reset session state to initial values"""
    keys_to_clear = [
        'chat_history',
        'uploaded_documents', 
        'documents',
        'processed_files'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            if isinstance(st.session_state[key], list):
                st.session_state[key] = []
            elif isinstance(st.session_state[key], set):
                st.session_state[key] = set()
            else:
                del st.session_state[key] 


def cleanup_interrupted_uploads():
    """Clean up interrupted upload states and reset if needed"""
    # Check if we had an interruption (e.g., model change during upload)
    if st.session_state.upload_interrupted:
        print("🧹 Cleaning up interrupted upload state")
        
        # Move currently uploading files to failed state for retry
        for file_id in st.session_state.currently_uploading:
            st.session_state.failed_uploads.add(file_id)
            print(f"📝 Moved {file_id} to failed uploads for retry")
        
        # Clear the uploading state
        st.session_state.currently_uploading.clear()
        st.session_state.upload_interrupted = False


def mark_upload_start(file_id: str):
    """Mark a file as starting upload process"""
    st.session_state.currently_uploading.add(file_id)
    # Remove from failed if it was there
    st.session_state.failed_uploads.discard(file_id)


def mark_upload_success(file_id: str):
    """Mark a file as successfully processed"""
    st.session_state.currently_uploading.discard(file_id)
    st.session_state.processed_files.add(file_id)
    st.session_state.failed_uploads.discard(file_id)


def mark_upload_failed(file_id: str):
    """Mark a file as failed for retry"""
    st.session_state.currently_uploading.discard(file_id)
    st.session_state.failed_uploads.add(file_id)


def process_uploaded_files_with_state_tracking(files):
    """Wrapper for process_uploaded_files with proper state tracking"""
    from .document_handler import process_uploaded_files
    
    try:
        # Track which files we're processing
        for file in files:
            file_id = f"{file.name}_{file.size}"
            mark_upload_start(file_id)
        
        # Process the files
        success_count = process_uploaded_files(files)
        
        # Mark successful files
        for file in files:
            file_id = f"{file.name}_{file.size}"
            mark_upload_success(file_id)
            
    except Exception as e:
        # Mark all files as failed for retry
        for file in files:
            file_id = f"{file.name}_{file.size}"
            mark_upload_failed(file_id)
        
        st.error(f"❌ Upload processing failed: {e}")
        st.info("💡 Files have been marked for retry - try uploading again") 