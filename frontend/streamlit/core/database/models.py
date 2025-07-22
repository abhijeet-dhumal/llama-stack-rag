"""
Database models for RAG LlamaStack
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from .connection import get_db_manager

class User:
    """User model with authentication"""
    
    def __init__(self, id: int = None, username: str = None, email: str = None, 
                 password_hash: str = None, role: str = 'user'):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
    
    @staticmethod
    def create(username: str, email: str, password: str) -> 'User':
        """Create a new user"""
        db = get_db_manager()
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        try:
            user_id = db.execute_insert(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username, email, password_hash)
            )
            return User.get_by_id(user_id)
        except Exception as e:
            raise Exception(f"Failed to create user: {e}")
    
    @staticmethod
    def get_by_id(user_id: int) -> Optional['User']:
        """Get user by ID"""
        db = get_db_manager()
        result = db.execute_query(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        if result:
            row = result[0]
            return User(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash'],
                role=row['role']
            )
        return None
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional['User']:
        """Authenticate user"""
        db = get_db_manager()
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        result = db.execute_query(
            "SELECT * FROM users WHERE username = ? AND password_hash = ? AND is_active = 1",
            (username, password_hash)
        )
        
        if result:
            row = result[0]
            # Update last login
            db.execute_update(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (row['id'],)
            )
            return User(
                id=row['id'],
                username=row['username'],
                email=row['email'],
                password_hash=row['password_hash'],
                role=row['role']
            )
        return None

class Document:
    """Document model"""
    
    def __init__(self, id: int = None, user_id: int = None, name: str = None,
                 file_type: str = None, file_size_mb: float = None, 
                 content_hash: str = None, source_url: str = None,
                 domain: str = None, processing_status: str = 'pending',
                 chunk_count: int = 0, character_count: int = 0,
                 upload_time: str = None, metadata: str = None):
        self.id = id
        self.user_id = user_id
        self.name = name
        self.file_type = file_type
        self.file_size_mb = file_size_mb or 0.0
        self.content_hash = content_hash
        self.source_url = source_url
        self.domain = domain
        self.processing_status = processing_status
        self.chunk_count = chunk_count or 0
        self.character_count = character_count or 0
        self.upload_time = upload_time
        self.metadata = metadata
    
    @property
    def file_size(self) -> int:
        """Get file size in bytes for compatibility"""
        return int(self.file_size_mb * 1024 * 1024) if self.file_size_mb else 0
    
    @property
    def created_at(self) -> str:
        """Get upload time for compatibility"""
        return self.upload_time or "Unknown"
    
    def save(self) -> int:
        """Save document to database"""
        db = get_db_manager()
        
        if self.id is None:
            # Insert new document
            self.id = db.execute_insert(
                """INSERT INTO documents 
                   (user_id, name, file_type, file_size_mb, content_hash, source_url, domain, 
                    processing_status, chunk_count, character_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.user_id, self.name, self.file_type, self.file_size_mb, 
                 self.content_hash, self.source_url, self.domain, self.processing_status,
                 self.chunk_count, self.character_count, self.metadata)
            )
        else:
            # Update existing document
            db.execute_update(
                """UPDATE documents SET 
                   name = ?, file_type = ?, file_size_mb = ?, content_hash = ?,
                   source_url = ?, domain = ?, processing_status = ?, 
                   chunk_count = ?, character_count = ?, metadata = ?
                   WHERE id = ?""",
                (self.name, self.file_type, self.file_size_mb, self.content_hash,
                 self.source_url, self.domain, self.processing_status,
                 self.chunk_count, self.character_count, self.metadata, self.id)
            )
        
        return self.id
    
    @staticmethod
    def get_by_user(user_id: int) -> List['Document']:
        """Get all documents for a user"""
        db = get_db_manager()
        results = db.execute_query(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY upload_time DESC",
            (user_id,)
        )
        
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
        
        return documents
    
    @staticmethod
    def get_by_id(doc_id: int) -> Optional['Document']:
        """Get document by ID"""
        db = get_db_manager()
        result = db.execute_query(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        
        if result:
            row = result[0]
            return Document(
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
        return None

class ChatSession:
    """Chat session model"""
    
    def __init__(self, id: int = None, user_id: int = None, title: str = None):
        self.id = id
        self.user_id = user_id
        self.title = title
    
    def save(self) -> int:
        """Save chat session"""
        db = get_db_manager()
        
        if self.id is None:
            self.id = db.execute_insert(
                "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
                (self.user_id, self.title)
            )
        else:
            db.execute_update(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (self.title, self.id)
            )
        
        return self.id
    
    @staticmethod
    def get_by_user(user_id: int) -> List['ChatSession']:
        """Get all chat sessions for a user"""
        db = get_db_manager()
        results = db.execute_query(
            "SELECT * FROM chat_sessions WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC",
            (user_id,)
        )
        
        sessions = []
        for row in results:
            session = ChatSession(
                id=row['id'],
                user_id=row['user_id'],
                title=row['title']
            )
            sessions.append(session)
        
        return sessions

class ChatMessage:
    """Chat message model"""
    
    def __init__(self, id: int = None, chat_session_id: int = None, 
                 role: str = None, content: str = None, metadata: Dict = None):
        self.id = id
        self.chat_session_id = chat_session_id
        self.role = role
        self.content = content
        self.metadata = metadata or {}
    
    def save(self) -> int:
        """Save chat message"""
        db = get_db_manager()
        
        metadata_json = json.dumps(self.metadata) if self.metadata else None
        
        self.id = db.execute_insert(
            "INSERT INTO chat_messages (chat_session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (self.chat_session_id, self.role, self.content, metadata_json)
        )
        
        # Update chat session timestamp
        db.execute_update(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (self.chat_session_id,)
        )
        
        return self.id
    
    @staticmethod
    def get_by_session(chat_session_id: int) -> List['ChatMessage']:
        """Get all messages for a chat session"""
        db = get_db_manager()
        results = db.execute_query(
            "SELECT * FROM chat_messages WHERE chat_session_id = ? ORDER BY timestamp",
            (chat_session_id,)
        )
        
        messages = []
        for row in results:
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
            message = ChatMessage(
                id=row['id'],
                chat_session_id=row['chat_session_id'],
                role=row['role'],
                content=row['content'],
                metadata=metadata
            )
            messages.append(message)
        
        return messages 