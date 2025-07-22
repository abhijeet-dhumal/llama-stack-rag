"""
Authentication manager for RAG LlamaStack
"""

import streamlit as st
import hashlib
import time
import json
import os
from typing import Optional
from ..database.models import User

class AuthManager:
    """Manages user authentication"""
    
    SESSION_FILE = "data/auth_session.json"
    
    @staticmethod
    def _ensure_session_dir():
        """Ensure the session directory exists"""
        os.makedirs(os.path.dirname(AuthManager.SESSION_FILE), exist_ok=True)
    
    @staticmethod
    def _generate_session_token(user_id: int) -> str:
        """Generate a session token for persistent authentication"""
        timestamp = str(int(time.time()))
        token_data = f"{user_id}:{timestamp}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    @staticmethod
    def _save_session_to_file(user_id: int, token: str):
        """Save session data to file for persistence"""
        AuthManager._ensure_session_dir()
        session_data = {
            'user_id': user_id,
            'token': token,
            'timestamp': time.time()
        }
        try:
            with open(AuthManager.SESSION_FILE, 'w') as f:
                json.dump(session_data, f)
        except Exception as e:
            st.error(f"Failed to save session: {e}")
    
    @staticmethod
    def _load_session_from_file() -> Optional[dict]:
        """Load session data from file"""
        try:
            if os.path.exists(AuthManager.SESSION_FILE):
                with open(AuthManager.SESSION_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Failed to load session: {e}")
        return None
    
    @staticmethod
    def _clear_session_file():
        """Clear session data from file"""
        try:
            if os.path.exists(AuthManager.SESSION_FILE):
                os.remove(AuthManager.SESSION_FILE)
        except Exception as e:
            st.error(f"Failed to clear session: {e}")
    
    @staticmethod
    def _get_session_cookie() -> Optional[str]:
        """Get session token from cookie"""
        return st.session_state.get('auth_token')
    
    @staticmethod
    def _set_session_cookie(user_id: int, token: str):
        """Set session token in cookie and file"""
        st.session_state.auth_token = token
        st.session_state.auth_user_id = user_id
        st.session_state.user_id = user_id  # Also set user_id for compatibility
        st.session_state.auth_timestamp = time.time()
        # Save to file for persistence
        AuthManager._save_session_to_file(user_id, token)
    
    @staticmethod
    def _clear_session_cookie():
        """Clear session token from cookie and file"""
        if 'auth_token' in st.session_state:
            del st.session_state.auth_token
        if 'auth_user_id' in st.session_state:
            del st.session_state.auth_user_id
        if 'user_id' in st.session_state:
            del st.session_state.user_id
        if 'auth_timestamp' in st.session_state:
            del st.session_state.auth_timestamp
        # Clear file session
        AuthManager._clear_session_file()
    
    @staticmethod
    def _validate_session() -> Optional[User]:
        """Validate existing session and return user if valid"""
        # First try regular session state
        token = AuthManager._get_session_cookie()
        user_id = st.session_state.get('auth_user_id')
        timestamp = st.session_state.get('auth_timestamp', 0)
        
        # If not in regular session, try file-based session
        if not token:
            session_data = AuthManager._load_session_from_file()
            if session_data:
                token = session_data.get('token')
                user_id = session_data.get('user_id')
                timestamp = session_data.get('timestamp', 0)
                # Set both auth_user_id and user_id for compatibility
                if user_id:
                    st.session_state.auth_user_id = user_id
                    st.session_state.user_id = user_id
        
        if not token or not user_id:
            return None
        
        # Check if session is expired (24 hours)
        if time.time() - timestamp > 86400:  # 24 hours
            AuthManager._clear_session_cookie()
            return None
        
        # Validate token and get user
        try:
            user = User.get_by_id(user_id)
            if user:
                # Regenerate token for security
                new_token = AuthManager._generate_session_token(user_id)
                AuthManager._set_session_cookie(user_id, new_token)
                # Ensure user_id is set for compatibility
                st.session_state.user_id = user_id
                return user
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def login(username: str, password: str) -> Optional[User]:
        """Authenticate user login"""
        try:
            user = User.authenticate(username, password)
            if user:
                # Generate session token
                token = AuthManager._generate_session_token(user.id)
                AuthManager._set_session_cookie(user.id, token)
                
                # Store user in session state
                st.session_state.current_user = user
                st.session_state.is_authenticated = True
                return user
            return None
        except Exception as e:
            st.error(f"Login failed: {e}")
            return None
    
    @staticmethod
    def register(username: str, email: str, password: str) -> Optional[User]:
        """Register new user"""
        try:
            user = User.create(username, email, password)
            if user:
                # Generate session token and auto-login after registration
                token = AuthManager._generate_session_token(user.id)
                AuthManager._set_session_cookie(user.id, token)
                
                # Store user in session state
                st.session_state.current_user = user
                st.session_state.is_authenticated = True
                return user
            return None
        except Exception as e:
            st.error(f"Registration failed: {e}")
            return None
    
    @staticmethod
    def logout():
        """Logout current user"""
        # Clear session cookie
        AuthManager._clear_session_cookie()
        
        # Clear session state
        if 'current_user' in st.session_state:
            del st.session_state.current_user
        if 'is_authenticated' in st.session_state:
            del st.session_state.is_authenticated
        if 'current_chat_session' in st.session_state:
            del st.session_state.current_chat_session
        if 'user_id' in st.session_state:
            del st.session_state.user_id
    
    @staticmethod
    def get_current_user() -> Optional[User]:
        """Get current authenticated user"""
        # First check if we have a user in session state
        user = st.session_state.get('current_user')
        if user:
            return user
        
        # If not, try to validate existing session
        user = AuthManager._validate_session()
        if user:
            st.session_state.current_user = user
            st.session_state.is_authenticated = True
            return user
        
        return None
    
    @staticmethod
    def is_authenticated() -> bool:
        """Check if user is authenticated"""
        # Check session state first
        if st.session_state.get('is_authenticated', False):
            return True
        
        # If not authenticated in session state, try to validate session
        user = AuthManager._validate_session()
        if user:
            st.session_state.current_user = user
            st.session_state.is_authenticated = True
            return True
        
        return False
    
    @staticmethod
    def require_auth():
        """Decorator to require authentication"""
        if not AuthManager.is_authenticated():
            st.error("Please log in to access this feature")
            st.stop()
    
    @staticmethod
    def initialize_session():
        """Initialize authentication session on app startup"""
        # Try to restore session from stored data
        if not AuthManager.is_authenticated():
            user = AuthManager._validate_session()
            if user:
                st.session_state.current_user = user
                st.session_state.is_authenticated = True
                return True
        return False 