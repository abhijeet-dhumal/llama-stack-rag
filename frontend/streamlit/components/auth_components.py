"""
Authentication UI components for RAG LlamaStack
"""

import streamlit as st
from core.auth.authentication import AuthManager

def render_login_form():
    """Render login form"""
    st.markdown("### 🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            login_submitted = st.form_submit_button("🔐 Login", type="primary")
        with col2:
            register_submitted = st.form_submit_button("📝 Register")
        
        if login_submitted and username and password:
            user = AuthManager.login(username, password)
            if user:
                st.success(f"✅ Welcome back, {user.username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        elif register_submitted:
            st.session_state.show_register = True
            st.rerun()

def render_register_form():
    """Render registration form"""
    st.markdown("### 📝 Register")
    
    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Choose a username")
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input("Password", type="password", placeholder="Choose a password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        col1, col2 = st.columns(2)
        with col1:
            register_submitted = st.form_submit_button("📝 Register", type="primary")
        with col2:
            back_submitted = st.form_submit_button("🔙 Back to Login")
        
        if register_submitted:
            if not all([username, email, password, confirm_password]):
                st.error("❌ Please fill in all fields")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            elif len(password) < 6:
                st.error("❌ Password must be at least 6 characters")
            else:
                user = AuthManager.register(username, email, password)
                if user:
                    st.success(f"✅ Account created successfully! Welcome, {user.username}!")
                    st.rerun()
                else:
                    st.error("❌ Registration failed. Username or email may already exist.")
        
        elif back_submitted:
            st.session_state.show_register = False
            st.rerun()

def render_auth_interface():
    """Render authentication interface"""
    if not AuthManager.is_authenticated():
        # Show login or register form
        if st.session_state.get('show_register', False):
            render_register_form()
        else:
            render_login_form()
        return False
    else:
        # User is authenticated - no need to show welcome message here
        # The sidebar already shows user info and logout button
        return True 