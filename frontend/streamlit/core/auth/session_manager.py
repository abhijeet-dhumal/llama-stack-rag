"""
Session manager for RAG LlamaStack
"""

import streamlit as st
from typing import Optional
from ..database.models import ChatSession, ChatMessage
from datetime import datetime

class SessionManager:
    """Manages chat sessions"""
    
    @staticmethod
    def create_chat_session(user_id: int, title: str = None) -> ChatSession:
        """Create a new chat session"""
        if not title:
            title = f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(user_id=user_id, title=title)
        session.save()
        
        # Store in session state
        st.session_state.current_chat_session = session
        return session
    
    @staticmethod
    def get_current_chat_session() -> Optional[ChatSession]:
        """Get current chat session"""
        return st.session_state.get('current_chat_session')
    
    @staticmethod
    def switch_chat_session(session: ChatSession):
        """Switch to a different chat session"""
        st.session_state.current_chat_session = session
    
    @staticmethod
    def add_message(role: str, content: str, metadata: dict = None):
        """Add message to current chat session"""
        session = SessionManager.get_current_chat_session()
        if not session:
            st.error("No active chat session")
            return
        
        message = ChatMessage(
            chat_session_id=session.id,
            role=role,
            content=content,
            metadata=metadata
        )
        message.save()
        
        return message 