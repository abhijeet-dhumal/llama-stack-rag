"""
FAISS-SQLite Synchronization Manager
Handles synchronization between FAISS vector database and SQLite metadata storage
"""

import streamlit as st
import hashlib
import json
import time
import os
import pickle
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from .database.models import Document, get_db_manager

class FAISSSyncManager:
    """Manages synchronization between FAISS and SQLite databases"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_dimension = 384  # all-MiniLM-L6-v2 dimension
        self.faiss_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'faiss')
        os.makedirs(self.faiss_data_dir, exist_ok=True)
        
    def get_embedding_model(self):
        """Get or initialize the embedding model"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def get_faiss_file_paths(self, user_id: int = None):
        """Get file paths for FAISS persistent storage"""
        user_suffix = f"_user_{user_id}" if user_id else "_global"
        return {
            'index': os.path.join(self.faiss_data_dir, f"faiss_index{user_suffix}.faiss"),
            'documents': os.path.join(self.faiss_data_dir, f"faiss_documents{user_suffix}.pkl"),
            'chunks': os.path.join(self.faiss_data_dir, f"faiss_chunks{user_suffix}.pkl"),
            'mapping': os.path.join(self.faiss_data_dir, f"faiss_mapping{user_suffix}.pkl")
        }
    
    def save_faiss_data(self, user_id: int = None):
        """Save FAISS data to disk"""
        try:
            if 'faiss_index' not in st.session_state or st.session_state.faiss_index is None:
                return False
                
            paths = self.get_faiss_file_paths(user_id)
            
            # Save FAISS index
            faiss.write_index(st.session_state.faiss_index, paths['index'])
            
            # Save documents metadata
            with open(paths['documents'], 'wb') as f:
                pickle.dump(st.session_state.get('faiss_documents', []), f)
            
            # Save chunks metadata
            with open(paths['chunks'], 'wb') as f:
                pickle.dump(st.session_state.get('faiss_chunks', []), f)
            
            # Save document mapping
            with open(paths['mapping'], 'wb') as f:
                pickle.dump(st.session_state.get('faiss_document_mapping', {}), f)
            
            print(f"✅ FAISS data saved to disk")
            return True
            
        except Exception as e:
            print(f"❌ Error saving FAISS data: {e}")
            return False
    
    def load_faiss_data(self, user_id: int = None):
        """Load FAISS data from disk"""
        try:
            # First try user-specific files
            paths = self.get_faiss_file_paths(user_id)
            
            # Check if all user-specific files exist
            missing_files = [path for path in paths.values() if not os.path.exists(path)]
            
            # If user-specific files don't exist, try global files
            if missing_files and user_id is not None:
                global_paths = self.get_faiss_file_paths(None)  # Global files
                global_missing = [path for path in global_paths.values() if not os.path.exists(path)]
                
                if not global_missing:
                    paths = global_paths
                else:
                    return False
            elif missing_files:
                return False
            
            # Load FAISS index
            st.session_state.faiss_index = faiss.read_index(paths['index'])
            
            # Load documents metadata
            with open(paths['documents'], 'rb') as f:
                st.session_state.faiss_documents = pickle.load(f)
            
            # Load chunks metadata
            with open(paths['chunks'], 'rb') as f:
                st.session_state.faiss_chunks = pickle.load(f)
            
            # Load document mapping
            with open(paths['mapping'], 'rb') as f:
                st.session_state.faiss_document_mapping = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading FAISS data: {e}")
            return False
    
    def initialize_faiss_index(self):
        """Initialize FAISS index if not exists"""
        if 'faiss_index' not in st.session_state or st.session_state.faiss_index is None:
            # Try to load from disk first
            user_id = st.session_state.get('user_id')
            
            if not self.load_faiss_data(user_id):
                # If loading fails, create new index
                st.session_state.faiss_index = faiss.IndexFlatL2(self.vector_dimension)
                st.session_state.faiss_documents = []
                st.session_state.faiss_chunks = []
                st.session_state.faiss_document_mapping = {}
    
    def sync_from_sqlite(self, user_id: int = None):
        """Sync FAISS index with SQLite database"""
        try:
            # Get documents from SQLite
            if user_id:
                documents = Document.get_by_user(user_id)
            else:
                # Get all documents if no user_id specified
                db = get_db_manager()
                results = db.execute_query("SELECT * FROM documents ORDER BY upload_time DESC")
                documents = []
                for row in results:
                    doc = Document(
                        id=row['id'],
                        user_id=row['user_id'],
                        name=row['name'],
                        file_type=row['file_type'],
                        file_size_mb=row['file_size_mb'],
                        content_hash=row['content_hash'],
                        source_url=row['source_url'],
                        domain=row['domain'],
                        processing_status=row['processing_status'],
                        chunk_count=row['chunk_count'],
                        character_count=row['character_count'],
                        upload_time=row['upload_time'],
                        metadata=row['metadata']
                    )
                    documents.append(doc)
            
            # Initialize FAISS
            self.initialize_faiss_index()
            
            # Clear existing FAISS data
            st.session_state.faiss_index.reset()
            st.session_state.faiss_documents = []
            st.session_state.faiss_chunks = []
            st.session_state.faiss_document_mapping = {}
            
            # Process each document
            synced_count = 0
            for doc in documents:
                if doc.processing_status == 'completed':
                    self._add_document_to_faiss(doc)
                    synced_count += 1
            
            # Save to disk after sync
            self.save_faiss_data(user_id)
            
            print(f"✅ Synced {synced_count} documents from SQLite to FAISS")
            return True
            
        except Exception as e:
            print(f"❌ Error syncing from SQLite: {e}")
            return False
    
    def add_document_to_both(self, doc_data: Dict, chunks: List[str], user_id: int = None) -> bool:
        """Add document to both SQLite and FAISS"""
        try:
            # 1. Save to SQLite first
            doc = Document(
                user_id=user_id,
                name=doc_data.get('name'),
                file_type=doc_data.get('file_type'),
                file_size_mb=doc_data.get('file_size_mb', 0),
                content_hash=doc_data.get('content_hash'),
                source_url=doc_data.get('source_url'),
                domain=doc_data.get('domain'),
                processing_status='processing',
                chunk_count=len(chunks),
                character_count=sum(len(chunk) for chunk in chunks),
                metadata=json.dumps(doc_data.get('metadata', {}))
            )
            
            doc_id = doc.save()
            
            # 2. Add to FAISS
            self.initialize_faiss_index()
            self._add_chunks_to_faiss(chunks, doc_data.get('name', ''), doc_id, doc)
            
            # 3. Update SQLite status
            doc.processing_status = 'completed'
            doc.save()
            
            # 4. Save to disk
            self.save_faiss_data(user_id)
            
            print(f"✅ Added document {doc_data.get('name')} to both databases")
            return True
            
        except Exception as e:
            print(f"❌ Error adding document to both databases: {e}")
            return False
    
    def _add_document_to_faiss(self, doc: Document):
        """Add existing document to FAISS index"""
        try:
            # For existing documents, we can't reconstruct the full content
            # So we'll create placeholder chunks based on metadata
            if doc.source_url:
                # Web content - create placeholder
                content = f"Web content from {doc.source_url}\n\nThis document contains {doc.chunk_count} chunks with {doc.character_count} characters."
            else:
                # File content - create placeholder
                content = f"Document: {doc.name}\nType: {doc.file_type}\nSize: {doc.file_size_mb:.2f} MB\n\nThis document contains {doc.chunk_count} chunks with {doc.character_count} characters."
            
            # Create a single chunk with the metadata
            chunks = [content]
            
            # Add to FAISS with full document metadata
            self._add_chunks_to_faiss(chunks, doc.name, doc.id, doc)
            
            print(f"✅ Added document {doc.name} to FAISS (metadata only)")
            
        except Exception as e:
            print(f"❌ Error adding document {doc.name} to FAISS: {e}")
    
    def _add_chunks_to_faiss(self, chunks: List[str], document_name: str, doc_id: int, doc_object: Document = None):
        """Add chunks to FAISS index"""
        try:
            model = self.get_embedding_model()
            
            # Generate embeddings for chunks
            embeddings = model.encode(chunks)
            
            # Add to FAISS index
            st.session_state.faiss_index.add(embeddings.astype('float32'))
            
            # Store chunk metadata
            start_idx = len(st.session_state.faiss_chunks)
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk,
                    'document_name': document_name,
                    'document_id': doc_id,
                    'chunk_index': i,
                    'embedding_index': start_idx + i
                }
                st.session_state.faiss_chunks.append(chunk_data)
            
            # Store document metadata with full information
            if doc_object:
                # Use full document object from SQLite
                doc_data = {
                    'id': doc_object.id,
                    'name': doc_object.name,
                    'file_type': doc_object.file_type,
                    'file_size_mb': doc_object.file_size_mb,
                    'source_url': doc_object.source_url,
                    'domain': doc_object.domain,
                    'chunk_count': doc_object.chunk_count,
                    'character_count': doc_object.character_count,
                    'upload_time': doc_object.upload_time or time.strftime('%Y-%m-%d %H:%M:%S'),  # Use default if None
                    'user_id': doc_object.user_id,
                    'source': 'sync'
                }
            else:
                # Fallback for new documents
                doc_data = {
                    'id': doc_id,
                    'name': document_name,
                    'chunk_count': len(chunks),
                    'upload_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'sync'
                }
            
            st.session_state.faiss_documents.append(doc_data)
            
            # Update document mapping
            st.session_state.faiss_document_mapping[doc_id] = list(range(start_idx, start_idx + len(chunks)))
            
        except Exception as e:
            print(f"❌ Error adding chunks to FAISS: {e}")
    
    def _chunk_content(self, content: str, document_name: str) -> List[str]:
        """Split content into chunks"""
        chunks = []
        
        # Simple chunking by paragraphs
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Only chunks with substantial content
                # Further split long paragraphs
                if len(para) > 1000:
                    sentences = para.split('. ')
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 1000:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                else:
                    chunks.append(para)
        
        return chunks
    
    def remove_document_from_both(self, doc_id: int) -> bool:
        """Remove document from both SQLite and FAISS"""
        try:
            # 1. Remove from SQLite
            db = get_db_manager()
            db.execute_update("DELETE FROM documents WHERE id = ?", (doc_id,))
            
            # 2. Remove from FAISS (rebuild index without this document)
            self._rebuild_faiss_without_document(doc_id)
            
            # 3. Save to disk
            user_id = st.session_state.get('user_id')
            self.save_faiss_data(user_id)
            
            print(f"✅ Removed document {doc_id} from both databases")
            return True
            
        except Exception as e:
            print(f"❌ Error removing document {doc_id}: {e}")
            return False
    
    def _rebuild_faiss_without_document(self, doc_id_to_remove: int):
        """Rebuild FAISS index excluding a specific document"""
        try:
            # Get all documents except the one to remove
            db = get_db_manager()
            results = db.execute_query(
                "SELECT * FROM documents WHERE id != ? ORDER BY upload_time DESC",
                (doc_id_to_remove,)
            )
            
            # Rebuild FAISS index
            st.session_state.faiss_index.reset()
            st.session_state.faiss_documents = []
            st.session_state.faiss_chunks = []
            st.session_state.faiss_document_mapping = {}
            
            # Re-add all documents except the removed one
            for row in results:
                doc = Document(
                    id=row['id'],
                    user_id=row['user_id'],
                    name=row['name'],
                    file_type=row['file_type'],
                    file_size_mb=row['file_size_mb'],
                    content_hash=row['content_hash'],
                    source_url=row['source_url'],
                    domain=row['domain'],
                    processing_status=row['processing_status'],
                    chunk_count=row['chunk_count'],
                    character_count=row['character_count'],
                    upload_time=row['upload_time'],
                    metadata=row['metadata']
                )
                if doc.processing_status == 'completed':
                    self._add_document_to_faiss(doc)
                    
        except Exception as e:
            print(f"❌ Error rebuilding FAISS index: {e}")
    
    def faiss_data_exists(self, user_id: int = None) -> bool:
        """Check if FAISS data exists on disk"""
        try:
            # Check user-specific files first
            paths = self.get_faiss_file_paths(user_id)
            user_files_exist = all(os.path.exists(path) for path in paths.values())
            
            # If user-specific files don't exist and user_id is provided, check global files
            if not user_files_exist and user_id is not None:
                global_paths = self.get_faiss_file_paths(None)
                global_files_exist = all(os.path.exists(path) for path in global_paths.values())
                return global_files_exist
            
            return user_files_exist
        except Exception as e:
            print(f"❌ Error checking FAISS data existence: {e}")
            return False
    
    def get_sync_status(self) -> Dict:
        """Get synchronization status between FAISS and SQLite"""
        try:
            # Get SQLite document count
            db = get_db_manager()
            sqlite_count = db.execute_query("SELECT COUNT(*) as count FROM documents")[0]['count']
            
            # Check if FAISS data exists on disk
            user_id = st.session_state.get('user_id')
            faiss_exists_on_disk = self.faiss_data_exists(user_id)
            
            # Get FAISS document count from session state or disk
            if 'faiss_documents' in st.session_state:
                faiss_count = len(st.session_state.get('faiss_documents', []))
                faiss_chunks = len(st.session_state.get('faiss_chunks', []))
            elif faiss_exists_on_disk:
                # Try to load from disk to get accurate count
                try:
                    # Use the same logic as load_faiss_data to find the correct paths
                    paths = self.get_faiss_file_paths(user_id)
                    if not all(os.path.exists(path) for path in paths.values()) and user_id is not None:
                        # Try global files
                        paths = self.get_faiss_file_paths(None)
                    
                    with open(paths['documents'], 'rb') as f:
                        faiss_docs = pickle.load(f)
                    with open(paths['chunks'], 'rb') as f:
                        faiss_chunks_data = pickle.load(f)
                    faiss_count = len(faiss_docs)
                    faiss_chunks = len(faiss_chunks_data)
                except:
                    faiss_count = 0
                    faiss_chunks = 0
            else:
                faiss_count = 0
                faiss_chunks = 0
            
            # Check for mismatches
            mismatches = []
            if sqlite_count != faiss_count:
                mismatches.append(f"Document count mismatch: SQLite={sqlite_count}, FAISS={faiss_count}")
            
            if not faiss_exists_on_disk and sqlite_count > 0:
                mismatches.append("FAISS data not found on disk")
            
            return {
                'sqlite_documents': sqlite_count,
                'faiss_documents': faiss_count,
                'faiss_chunks': faiss_chunks,
                'faiss_exists_on_disk': faiss_exists_on_disk,
                'in_sync': len(mismatches) == 0,
                'mismatches': mismatches
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'in_sync': False
            }
    
    def delete_all_data(self, user_id: int = None) -> bool:
        """Delete all FAISS data (files and session state)"""
        try:
            # 1. Delete persistent FAISS files
            paths = self.get_faiss_file_paths(user_id)
            for path in paths.values():
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🗑️ Deleted: {path}")
            
            # 2. Also delete global files if user-specific deletion
            if user_id:
                global_paths = self.get_faiss_file_paths(None)
                for path in global_paths.values():
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"🗑️ Deleted global: {path}")
            
            print(f"✅ FAISS data deleted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting FAISS data: {e}")
            return False
    
    def force_sync(self, user_id: int = None) -> bool:
        """Force synchronization between databases"""
        return self.sync_from_sqlite(user_id)
    
    def clear_session_state(self):
        """Clear all document-related session state"""
        # Clear FAISS session state
        if 'faiss_index' in st.session_state:
            st.session_state.faiss_index.reset()
        st.session_state.faiss_documents = []
        st.session_state.faiss_chunks = []
        st.session_state.faiss_document_mapping = {}
        
        # Clear other document-related session state
        if 'uploaded_documents' in st.session_state:
            st.session_state.uploaded_documents = []
        if 'file_upload_success' in st.session_state:
            del st.session_state.file_upload_success
        if 'web_content_success' in st.session_state:
            del st.session_state.web_content_success
        if 'file_upload_warning' in st.session_state:
            del st.session_state.file_upload_warning
        if 'web_content_warning' in st.session_state:
            del st.session_state.web_content_warning
        
        # Reinitialize empty FAISS index
        st.session_state.faiss_index = faiss.IndexFlatL2(self.vector_dimension)
        st.session_state.faiss_documents = []
        st.session_state.faiss_chunks = []
        st.session_state.faiss_document_mapping = {}

# Global instance
faiss_sync_manager = FAISSSyncManager() 