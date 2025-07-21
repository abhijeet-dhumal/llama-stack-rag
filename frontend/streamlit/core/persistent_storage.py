"""
Persistent Storage System for LlamaStack RAG
Handles saving and loading document metadata across sessions
"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st

# Storage file paths
STORAGE_DIR = os.path.join("data", "cache")
DOCUMENTS_FILE = os.path.join(STORAGE_DIR, "persistent_documents.json")
METADATA_FILE = os.path.join(STORAGE_DIR, "document_metadata.json")
CHAT_HISTORY_FILE = os.path.join(STORAGE_DIR, "chat_history.json")

def ensure_storage_dir():
    """Ensure storage directory exists"""
    os.makedirs(STORAGE_DIR, exist_ok=True)

def _convert_to_serializable(obj):
    """Recursively convert objects to JSON-serializable format"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, 'tolist'):  # NumPy arrays
        try:
            return obj.tolist()
        except Exception as e:
            print(f"⚠️ Error converting NumPy array: {type(obj)} - {e}")
            return str(obj)
    elif hasattr(obj, 'item'):  # NumPy scalars
        try:
            return obj.item()
        except Exception as e:
            print(f"⚠️ Error converting NumPy scalar: {type(obj)} - {e}")
            return str(obj)
    elif isinstance(obj, list):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):  # Custom objects
        print(f"⚠️ Converting custom object: {type(obj)}")
        return str(obj)
    else:
        print(f"⚠️ Converting unknown object: {type(obj)}")
        return str(obj)

def save_documents_to_storage(documents: List[Dict[str, Any]]) -> bool:
    """Save documents metadata to persistent storage"""
    try:
        ensure_storage_dir()
        
        # Convert documents to JSON-serializable format using recursive function
        serializable_documents = []
        for doc in documents:
            serializable_doc = _convert_to_serializable(doc)
            serializable_documents.append(serializable_doc)
        
        # Add timestamp for tracking
        storage_data = {
            "timestamp": datetime.now().isoformat(),
            "document_count": len(serializable_documents),
            "documents": serializable_documents
        }
        
        with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(storage_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(serializable_documents)} documents to persistent storage")
        return True
    except Exception as e:
        print(f"❌ Error saving documents to storage: {e}")
        # Print more detailed error info for debugging
        import traceback
        print(f"🔍 Detailed error: {traceback.format_exc()}")
        
        # Fallback: Try to save only essential metadata
        try:
            print("🔄 Attempting fallback save with essential metadata only...")
            
            # Debug the document structure first
            debug_document_structure(documents)
            
            essential_documents = []
            for doc in documents:
                essential_doc = {
                    'name': str(doc.get('name', 'Unknown')),
                    'file_size_mb': float(doc.get('file_size_mb', 0)),
                    'chunk_count': int(doc.get('chunk_count', 0)),
                    'processing_time': float(doc.get('processing_time', 0)),
                    'upload_time': str(doc.get('upload_time', '')),
                    'file_type': str(doc.get('file_type', '')),
                    'embedding_errors': int(doc.get('embedding_errors', 0))
                }
                essential_documents.append(essential_doc)
            
            storage_data = {
                "timestamp": datetime.now().isoformat(),
                "document_count": len(essential_documents),
                "documents": essential_documents,
                "fallback_save": True
            }
            
            with open(DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Fallback save successful: {len(essential_documents)} documents")
            return True
        except Exception as fallback_error:
            print(f"❌ Fallback save also failed: {fallback_error}")
            return False

def load_documents_from_storage() -> List[Dict[str, Any]]:
    """Load documents metadata from persistent storage"""
    try:
        if not os.path.exists(DOCUMENTS_FILE):
            print("📁 No persistent documents found")
            return []
        
        # Check file size to avoid reading empty or corrupted files
        file_size = os.path.getsize(DOCUMENTS_FILE)
        if file_size == 0:
            print("📁 Persistent storage file is empty")
            return []
        
        with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            storage_data = json.load(f)
        
        documents = storage_data.get("documents", [])
        timestamp = storage_data.get("timestamp", "Unknown")
        print(f"📁 Loaded {len(documents)} documents from persistent storage (saved: {timestamp})")
        return documents
    except json.JSONDecodeError as e:
        print(f"❌ Corrupted JSON in persistent storage: {e}")
        # Try to backup and remove corrupted file
        try:
            backup_path = f"{DOCUMENTS_FILE}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(DOCUMENTS_FILE, backup_path)
            print(f"📦 Backed up corrupted file to: {backup_path}")
        except Exception as backup_error:
            print(f"⚠️ Could not backup corrupted file: {backup_error}")
        return []
    except Exception as e:
        print(f"❌ Error loading documents from storage: {e}")
        return []

def save_document_metadata(metadata: Dict[str, Any]) -> bool:
    """Save additional document metadata"""
    try:
        ensure_storage_dir()
        
        # Load existing metadata
        existing_metadata = load_document_metadata()
        
        # Update with new metadata
        existing_metadata.update(metadata)
        
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_metadata, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"❌ Error saving document metadata: {e}")
        return False

def load_document_metadata() -> Dict[str, Any]:
    """Load additional document metadata"""
    try:
        if not os.path.exists(METADATA_FILE):
            return {}
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata
    except Exception as e:
        print(f"❌ Error loading document metadata: {e}")
        return {}

def clear_persistent_storage() -> bool:
    """Clear all persistent storage"""
    try:
        files_to_remove = [DOCUMENTS_FILE, METADATA_FILE]
        removed_count = 0
        
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        print(f"🗑️ Cleared {removed_count} persistent storage files")
        return True
    except Exception as e:
        print(f"❌ Error clearing persistent storage: {e}")
        return False

def cleanup_corrupted_storage() -> bool:
    """Clean up corrupted storage files"""
    try:
        cleaned_count = 0
        
        # Check and clean up corrupted documents file
        if os.path.exists(DOCUMENTS_FILE):
            try:
                with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                    json.load(f)  # Test if JSON is valid
            except (json.JSONDecodeError, Exception):
                # File is corrupted, backup and remove
                backup_path = f"{DOCUMENTS_FILE}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(DOCUMENTS_FILE, backup_path)
                print(f"📦 Cleaned up corrupted documents file: {backup_path}")
                cleaned_count += 1
        
        # Check and clean up corrupted metadata file
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    json.load(f)  # Test if JSON is valid
            except (json.JSONDecodeError, Exception):
                # File is corrupted, backup and remove
                backup_path = f"{METADATA_FILE}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(METADATA_FILE, backup_path)
                print(f"📦 Cleaned up corrupted metadata file: {backup_path}")
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} corrupted storage files")
        else:
            print("✅ No corrupted storage files found")
        
        return True
    except Exception as e:
        print(f"❌ Error cleaning up corrupted storage: {e}")
        return False

def get_storage_stats() -> Dict[str, Any]:
    """Get statistics about persistent storage"""
    try:
        stats = {
            "storage_dir_exists": os.path.exists(STORAGE_DIR),
            "documents_file_exists": os.path.exists(DOCUMENTS_FILE),
            "metadata_file_exists": os.path.exists(METADATA_FILE),
            "document_count": 0,
            "last_saved": None,
            "storage_size_mb": 0
        }
        
        if os.path.exists(DOCUMENTS_FILE):
            file_size = os.path.getsize(DOCUMENTS_FILE) / (1024 * 1024)
            stats["storage_size_mb"] += file_size
            
            try:
                with open(DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stats["document_count"] = data.get("document_count", 0)
                    stats["last_saved"] = data.get("timestamp", None)
            except:
                pass
        
        if os.path.exists(METADATA_FILE):
            file_size = os.path.getsize(METADATA_FILE) / (1024 * 1024)
            stats["storage_size_mb"] += file_size
        
        return stats
    except Exception as e:
        print(f"❌ Error getting storage stats: {e}")
        return {}

def initialize_session_from_storage():
    """Initialize session state from persistent storage"""
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Load documents from persistent storage
    persistent_documents = load_documents_from_storage()
    
    if persistent_documents:
        # Merge with existing session documents (avoid duplicates)
        existing_names = {doc.get('name', '') for doc in st.session_state.uploaded_documents}
        new_documents = [doc for doc in persistent_documents if doc.get('name', '') not in existing_names]
        
        if new_documents:
            st.session_state.uploaded_documents.extend(new_documents)
            print(f"🔄 Restored {len(new_documents)} documents from persistent storage")
        else:
            print("📋 All documents already in session state")
    else:
        print("📋 No persistent documents to restore")

def save_session_to_storage():
    """Save current session documents to persistent storage"""
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        success = save_documents_to_storage(st.session_state.uploaded_documents)
        if success:
            print(f"💾 Session documents saved to persistent storage")
        return success
    return False

def backup_session_documents():
    """Create a backup of current session documents"""
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        backup_file = os.path.join(STORAGE_DIR, f"backup_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            ensure_storage_dir()
            
            # Convert documents to serializable format
            serializable_documents = []
            for doc in st.session_state.uploaded_documents:
                serializable_doc = _convert_to_serializable(doc)
                serializable_documents.append(serializable_doc)
            
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "document_count": len(serializable_documents),
                "documents": serializable_documents
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Backup created: {os.path.basename(backup_file)}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    return False

def debug_document_structure(documents: List[Dict[str, Any]], max_items: int = 3):
    """Debug helper to print document structure"""
    print(f"🔍 Debugging {len(documents)} documents (showing first {max_items}):")
    for i, doc in enumerate(documents[:max_items]):
        print(f"  Document {i+1}:")
        for key, value in doc.items():
            print(f"    {key}: {type(value)} = {value}")
        print()

# Chat History Storage Functions

def save_chat_history_to_storage(chat_history: List[Dict[str, Any]]) -> bool:
    """Save chat history to persistent storage"""
    try:
        ensure_storage_dir()
        
        # Convert chat history to JSON-serializable format
        serializable_history = []
        for message in chat_history:
            serializable_message = _convert_to_serializable(message)
            serializable_history.append(serializable_message)
        
        # Add timestamp and metadata
        storage_data = {
            "timestamp": datetime.now().isoformat(),
            "message_count": len(serializable_history),
            "chat_history": serializable_history
        }
        
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(storage_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(serializable_history)} chat messages to persistent storage")
        return True
    except Exception as e:
        print(f"❌ Error saving chat history to storage: {e}")
        import traceback
        print(f"🔍 Detailed error: {traceback.format_exc()}")
        return False

def load_chat_history_from_storage() -> List[Dict[str, Any]]:
    """Load chat history from persistent storage"""
    try:
        if not os.path.exists(CHAT_HISTORY_FILE):
            print("📝 No chat history file found")
            return []
        
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            storage_data = json.load(f)
        
        chat_history = storage_data.get("chat_history", [])
        message_count = storage_data.get("message_count", 0)
        timestamp = storage_data.get("timestamp", "unknown")
        
        print(f"📝 Loaded {len(chat_history)} chat messages from persistent storage (saved: {timestamp})")
        return chat_history
    except Exception as e:
        print(f"❌ Error loading chat history from storage: {e}")
        return []

def clear_chat_history_storage() -> bool:
    """Clear chat history from persistent storage"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
            print("🗑️ Cleared chat history from persistent storage")
        return True
    except Exception as e:
        print(f"❌ Error clearing chat history storage: {e}")
        return False

def get_chat_history_stats() -> Dict[str, Any]:
    """Get statistics about stored chat history"""
    try:
        if not os.path.exists(CHAT_HISTORY_FILE):
            return {
                "exists": False,
                "message_count": 0,
                "file_size_mb": 0,
                "last_saved": None
            }
        
        # Get file stats
        file_stats = os.stat(CHAT_HISTORY_FILE)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Load and count messages
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            storage_data = json.load(f)
        
        return {
            "exists": True,
            "message_count": storage_data.get("message_count", 0),
            "file_size_mb": round(file_size_mb, 2),
            "last_saved": storage_data.get("timestamp", "unknown")
        }
    except Exception as e:
        print(f"❌ Error getting chat history stats: {e}")
        return {
            "exists": False,
            "message_count": 0,
            "file_size_mb": 0,
            "last_saved": None,
            "error": str(e)
        } 