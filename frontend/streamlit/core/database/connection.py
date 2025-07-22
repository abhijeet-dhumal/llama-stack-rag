"""
Database connection manager for RAG LlamaStack
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional

class DatabaseManager:
    """Manages SQLite database connections and operations"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to data directory
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            db_path = data_dir / "rag_llamastack.db"
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Users table
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    role VARCHAR(20) DEFAULT 'user'
                );

                -- User sessions
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    session_token VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                );

                -- Documents table
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    name VARCHAR(255) NOT NULL,
                    file_type VARCHAR(20) NOT NULL,
                    file_size_mb REAL NOT NULL,
                    content_hash VARCHAR(64),
                    source_url TEXT,
                    domain VARCHAR(100),
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status VARCHAR(20) DEFAULT 'pending',
                    chunk_count INTEGER DEFAULT 0,
                    character_count INTEGER DEFAULT 0,
                    metadata TEXT
                );

                -- Document chunks
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER REFERENCES documents(id),
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_vector BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Chat sessions
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    title VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                );

                -- Chat messages
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_session_id INTEGER REFERENCES chat_sessions(id),
                    role VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                );

                -- FAISS indices
                CREATE TABLE IF NOT EXISTS faiss_indices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER REFERENCES users(id),
                    index_name VARCHAR(100) NOT NULL,
                    vector_dimension INTEGER NOT NULL,
                    total_vectors INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    index_file_path TEXT
                );

                -- Vector mappings
                CREATE TABLE IF NOT EXISTS vector_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    faiss_index_id INTEGER REFERENCES faiss_indices(id),
                    document_chunk_id INTEGER REFERENCES document_chunks(id),
                    vector_index INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Migration: Add metadata column to chat_messages if it doesn't exist
            try:
                conn.execute("ALTER TABLE chat_messages ADD COLUMN metadata TEXT")
                print("✅ Added metadata column to chat_messages table")
            except Exception:
                # Column already exists, ignore error
                pass
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        return conn
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an insert query and return the last insert ID"""
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def get_last_insert_id(self) -> int:
        """Get the last inserted row ID"""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT last_insert_rowid()")
            result = cursor.fetchone()
            return result[0] if result else 0
    
    def delete_all_documents(self) -> bool:
        """Safely delete all documents from the database"""
        try:
            with self.get_connection() as conn:
                # Delete from document_chunks first (foreign key constraint)
                conn.execute("DELETE FROM document_chunks")
                
                # Delete from documents
                conn.execute("DELETE FROM documents")
                
                # Reset auto-increment counter
                conn.execute("DELETE FROM sqlite_sequence WHERE name='documents'")
                conn.execute("DELETE FROM sqlite_sequence WHERE name='document_chunks'")
                
                conn.commit()
                return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def delete_documents_by_user(self, user_id: int) -> bool:
        """Safely delete all documents for a specific user"""
        try:
            with self.get_connection() as conn:
                # Get document IDs for this user
                cursor = conn.execute("SELECT id FROM documents WHERE user_id = ?", (user_id,))
                document_ids = [row[0] for row in cursor.fetchall()]
                
                if document_ids:
                    # Delete from document_chunks first
                    placeholders = ','.join('?' * len(document_ids))
                    conn.execute(f"DELETE FROM document_chunks WHERE document_id IN ({placeholders})", document_ids)
                    
                    # Delete from documents
                    conn.execute("DELETE FROM documents WHERE user_id = ?", (user_id,))
                    
                    conn.commit()
                
                return True
        except Exception as e:
            print(f"Error deleting documents for user {user_id}: {e}")
            return False

# Global database instance
_db_manager: Optional[DatabaseManager] = None

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager 