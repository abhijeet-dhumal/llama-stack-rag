"""
RAG LlamaStack Streamlit Application
Main application entry point with User Authentication
"""

import streamlit as st

# Import core modules
from core.config import PAGE_CONFIG, LLAMASTACK_BASE, APP_DESCRIPTION
from core.theme import initialize_theme, apply_theme, render_header_theme_toggle
from core.utils import (
    initialize_session_state, validate_llamastack_connection,
    cleanup_interrupted_uploads, process_uploaded_files_with_state_tracking
)
from core.document_handler import (
    initialize_document_storage, render_file_uploader, validate_uploaded_files,
    render_document_library, has_documents
)
from core.model_manager import render_model_dashboard, render_ollama_integration, get_model_info
from core.chat_interface import render_welcome_screen, render_chat_interface


def main():
    """Main application function with Bootstrap styling and Authentication"""
    # Configure page with menu items
    st.set_page_config(
        **PAGE_CONFIG,
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
            # 🦙 RAG LlamaStack
            
            **Retrieval-Augmented Generation with LlamaStack & Ollama**
            
            ### Features:
            - 🔐 User Authentication & Multi-user Support
            - 📄 Document processing (PDF, TXT, MD, DOCX, PPTX)
            - 🔍 Semantic search with embeddings
            - 🤖 Local LLM inference with Ollama
            - 🎨 Dark/Light theme support
            - 📊 Real-time processing metrics
            
            ### Settings & Options:
            - **Theme Toggle:** Use the ☀️/🌙 button in top-right header
            - **Model Selection:** Configure in left sidebar
            - **File Upload:** Max 50MB per file, multiple formats supported
            - **Dashboard & Settings:** Available in this 3-dot menu
            
            ### Tech Stack:
            Built with Streamlit + LlamaStack + Ollama + SQLite for local AI processing
            """
        }
    )
    
    # Load Bootstrap CSS
    load_bootstrap_css()
    
    # Initialize all systems
    initialize_theme()
    initialize_session_state()
    initialize_document_storage()
    
    # Apply theme
    apply_theme()
    
    # Check authentication first
    try:
        from components.auth_components import render_auth_interface
        from core.auth.authentication import AuthManager
        
        # Initialize session on startup
        AuthManager.initialize_session()
        
        is_authenticated = render_auth_interface()
        
        if not is_authenticated:
            # Show authentication interface and stop here
            return
    except ImportError:
        st.warning("⚠️ Authentication system not available - running in demo mode")
        is_authenticated = True
    
    # Initialize FAISS index after authentication (so we have user_id)
    try:
        from core.faiss_sync_manager import faiss_sync_manager
        faiss_sync_manager.initialize_faiss_index()
    except Exception as e:
        print(f"⚠️ FAISS initialization error: {e}")
    
    # Check LlamaStack connection
    if not validate_llamastack_connection():
        st.error("🔴 LlamaStack Offline")
        st.stop()
    
    # Header (no top menu bar)
    render_header()
    
    # Sidebar with authentication-aware navigation
    render_sidebar()
    
    # Main content area
    if st.session_state.current_page == "Document_Management":
        # Import and render Document Management page
        try:
            from pages.Document_Management import main as doc_management_main
            doc_management_main()
        except ImportError as e:
            st.error(f"Failed to load Document Management page: {e}")
    elif st.session_state.current_page == "Database_Dashboard":
        # Import and render Database Dashboard page
        try:
            from pages.Database_Dashboard import main as db_dashboard_main
            db_dashboard_main()
        except ImportError as e:
            st.error(f"Failed to load Database Dashboard page: {e}")
    elif st.session_state.current_page == "FAISS_Dashboard":
        # Import and render FAISS Dashboard page
        try:
            from pages.FAISS_Dashboard import main as faiss_dashboard_main
            faiss_dashboard_main()
        except ImportError as e:
            st.error(f"Failed to load FAISS Dashboard page: {e}")
    else:
        # Default main content
        if not has_documents():
            render_welcome_screen()
        else:
            render_chat_interface()


def load_bootstrap_css():
    """Load Bootstrap CSS and custom styles"""
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        /* CSS Variables for theme colors */
        :root {
            --background-color: #ffffff;
            --text-color: #000000;
            --border-color: #dee2e6;
            --hover-color: #f8f9fa;
        }
        
        [data-theme="dark"] {
            --background-color: #1e1e1e;
            --text-color: #ffffff;
            --border-color: #404040;
            --hover-color: #2d2d2d;
        }
        
        [data-theme="light"] {
            --background-color: #ffffff;
            --text-color: #000000;
            --border-color: #dee2e6;
            --hover-color: #f8f9fa;
        }
        
        /* Custom Bootstrap overrides for Streamlit */
        .stButton > button {
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
            font-weight: 500 !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.15) !important;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            border: 2px dashed #0d6efd !important;
            border-radius: 0.375rem !important;
            padding: 1rem !important;
            text-align: center !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa !important;
        }
        
        /* Dark theme support */
        [data-testid="stSidebar"] {
            background-color: var(--background-color) !important;
        }
        
        /* Authentication form styling */
        .stForm {
            background-color: #f8f9fa !important;
            padding: 1rem !important;
            border-radius: 0.5rem !important;
            border: 1px solid #dee2e6 !important;
        }
        
        /* User info styling */
        .user-info {
            background-color: #e3f2fd !important;
            padding: 0.5rem !important;
            border-radius: 0.25rem !important;
            border-left: 4px solid #2196f3 !important;
        }
        
        /* Comprehensive button styling for all themes */
        
        /* Primary buttons (blue) - Light theme */
        [data-theme="light"] .stButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="light"] .stButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Primary buttons (blue) - Dark theme */
        [data-theme="dark"] .stButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Secondary buttons (gray) - Light theme */
        [data-theme="light"] .stButton > button[data-baseweb="button"] {
            background-color: #6c757d !important;
            border-color: #6c757d !important;
            color: #ffffff !important;
        }
        
        [data-theme="light"] .stButton > button[data-baseweb="button"]:hover {
            background-color: #5a6268 !important;
            border-color: #5a6268 !important;
            color: #ffffff !important;
        }
        
        /* Secondary buttons (gray) - Dark theme */
        [data-theme="dark"] .stButton > button[data-baseweb="button"] {
            background-color: #6c757d !important;
            border-color: #6c757d !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stButton > button[data-baseweb="button"]:hover {
            background-color: #5a6268 !important;
            border-color: #5a6268 !important;
            color: #ffffff !important;
        }
        
        /* Success buttons (green) - Light theme */
        [data-theme="light"] .stButton > button[data-testid="baseButton-secondary"] {
            background-color: #198754 !important;
            border-color: #198754 !important;
            color: #ffffff !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-secondary"]:hover {
            background-color: #157347 !important;
            border-color: #157347 !important;
            color: #ffffff !important;
        }
        
        /* Success buttons (green) - Dark theme */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-secondary"] {
            background-color: #198754 !important;
            border-color: #198754 !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-secondary"]:hover {
            background-color: #157347 !important;
            border-color: #157347 !important;
            color: #ffffff !important;
        }
        
        /* Danger buttons (red) - Light theme */
        [data-theme="light"] .stButton > button[data-testid="baseButton-tertiary"] {
            background-color: #dc3545 !important;
            border-color: #dc3545 !important;
            color: #ffffff !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-tertiary"]:hover {
            background-color: #bb2d3b !important;
            border-color: #bb2d3b !important;
            color: #ffffff !important;
        }
        
        /* Danger buttons (red) - Dark theme */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-tertiary"] {
            background-color: #dc3545 !important;
            border-color: #dc3545 !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-tertiary"]:hover {
            background-color: #bb2d3b !important;
            border-color: #bb2d3b !important;
            color: #ffffff !important;
        }
        
        /* Warning buttons (yellow) - Light theme */
        [data-theme="light"] .stButton > button[data-testid="baseButton-quaternary"] {
            background-color: #ffc107 !important;
            border-color: #ffc107 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-quaternary"]:hover {
            background-color: #e0a800 !important;
            border-color: #e0a800 !important;
            color: #000000 !important;
        }
        
        /* Warning buttons (yellow) - Dark theme */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-quaternary"] {
            background-color: #ffc107 !important;
            border-color: #ffc107 !important;
            color: #000000 !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-quaternary"]:hover {
            background-color: #e0a800 !important;
            border-color: #e0a800 !important;
            color: #000000 !important;
        }
        
        /* Info buttons (cyan) - Light theme */
        [data-theme="light"] .stButton > button[data-testid="baseButton-quinary"] {
            background-color: #0dcaf0 !important;
            border-color: #0dcaf0 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-quinary"]:hover {
            background-color: #0bb5d4 !important;
            border-color: #0bb5d4 !important;
            color: #000000 !important;
        }
        
        /* Info buttons (cyan) - Dark theme */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-quinary"] {
            background-color: #0dcaf0 !important;
            border-color: #0dcaf0 !important;
            color: #000000 !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-quinary"]:hover {
            background-color: #0bb5d4 !important;
            border-color: #0bb5d4 !important;
            color: #000000 !important;
        }
        
        /* Outlined buttons - Light theme */
        [data-theme="light"] .stButton > button[data-testid="baseButton-outlined"] {
            background-color: transparent !important;
            border-color: #0d6efd !important;
            color: #0d6efd !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-outlined"]:hover {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        /* Outlined buttons - Dark theme */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-outlined"] {
            background-color: transparent !important;
            border-color: #0d6efd !important;
            color: #0d6efd !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-outlined"]:hover {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        /* Text buttons - Light theme */
        [data-theme="light"] .stButton > button[data-testid="baseButton-text"] {
            background-color: transparent !important;
            border-color: transparent !important;
            color: #0d6efd !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-text"]:hover {
            background-color: rgba(13, 110, 253, 0.1) !important;
            border-color: transparent !important;
            color: #0d6efd !important;
        }
        
        /* Text buttons - Dark theme */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-text"] {
            background-color: transparent !important;
            border-color: transparent !important;
            color: #0d6efd !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-text"]:hover {
            background-color: rgba(13, 110, 253, 0.2) !important;
            border-color: transparent !important;
            color: #0d6efd !important;
        }
        
        /* Universal button text color fix for all themes */
        .stButton > button,
        .stButton > button span,
        .stButton > button div,
        .stButton > button p {
            text-shadow: none !important;
        }
        
        /* Ensure button text is always visible */
        .stButton > button:not([data-testid="baseButton-text"]):not([data-testid="baseButton-outlined"]) {
            color: #ffffff !important;
        }
        
        /* Special handling for light-colored buttons */
        .stButton > button[data-testid="baseButton-quaternary"],
        .stButton > button[data-testid="baseButton-quinary"] {
            color: #000000 !important;
        }
        
        /* Additional button type handling */
        
        /* Small buttons */
        .stButton > button[data-testid="baseButton-small"] {
            padding: 0.25rem 0.5rem !important;
            font-size: 0.875rem !important;
        }
        
        /* Large buttons */
        .stButton > button[data-testid="baseButton-large"] {
            padding: 0.75rem 1.5rem !important;
            font-size: 1.25rem !important;
        }
        
        /* Disabled buttons */
        .stButton > button:disabled {
            opacity: 0.6 !important;
            cursor: not-allowed !important;
        }
        
        /* Loading buttons */
        .stButton > button[data-testid="baseButton-loading"] {
            position: relative !important;
        }
        
        /* Icon buttons */
        .stButton > button[data-testid="baseButton-icon"] {
            padding: 0.5rem !important;
            border-radius: 50% !important;
            width: 2.5rem !important;
            height: 2.5rem !important;
        }
        
        /* Link-style buttons */
        .stButton > button[data-testid="baseButton-link"] {
            background-color: transparent !important;
            border-color: transparent !important;
            color: #0d6efd !important;
            text-decoration: underline !important;
        }
        
        .stButton > button[data-testid="baseButton-link"]:hover {
            background-color: transparent !important;
            border-color: transparent !important;
            color: #0a58ca !important;
        }
        
        /* Ghost buttons (transparent with border) */
        .stButton > button[data-testid="baseButton-ghost"] {
            background-color: transparent !important;
            border-color: #dee2e6 !important;
            color: #6c757d !important;
        }
        
        .stButton > button[data-testid="baseButton-ghost"]:hover {
            background-color: #f8f9fa !important;
            border-color: #dee2e6 !important;
            color: #495057 !important;
        }
        
        /* Dark theme ghost buttons */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-ghost"] {
            background-color: transparent !important;
            border-color: #495057 !important;
            color: #adb5bd !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-ghost"]:hover {
            background-color: #343a40 !important;
            border-color: #495057 !important;
            color: #ffffff !important;
        }
        
        /* Form submit button styling - Theme-aware */
        .stFormSubmitButton > button {
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.15) !important;
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Dark theme form submit button overrides */
        [data-theme="dark"] .stFormSubmitButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stFormSubmitButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Light theme form submit button overrides */
        [data-theme="light"] .stFormSubmitButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="light"] .stFormSubmitButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Dark theme form submit button overrides */
        
        /* FAISS Dashboard specific styling */
        .faiss-dashboard {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* FAISS dashboard header styling */
        .faiss-dashboard h3 {
            color: var(--text-color) !important;
            font-weight: 600 !important;
        }
        
        .faiss-dashboard h4 {
            color: var(--text-color) !important;
            font-weight: 500 !important;
        }
        
        .faiss-dashboard p {
            color: var(--text-color) !important;
        }
        
        .faiss-dashboard .stMarkdown {
            color: var(--text-color) !important;
        }
        
        /* Dark theme FAISS dashboard text */
        [data-theme="dark"] .faiss-dashboard h3,
        [data-theme="dark"] .faiss-dashboard h4,
        [data-theme="dark"] .faiss-dashboard p,
        [data-theme="dark"] .faiss-dashboard .stMarkdown {
            color: #ffffff !important;
        }
        
        /* Light theme FAISS dashboard text */
        [data-theme="light"] .faiss-dashboard h3,
        [data-theme="light"] .faiss-dashboard h4,
        [data-theme="light"] .faiss-dashboard p,
        [data-theme="light"] .faiss-dashboard .stMarkdown {
            color: #000000 !important;
        }
        
        /* FAISS metrics cards - theme compatible */
        .faiss-metric {
            background-color: var(--background-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 0.5rem !important;
            padding: 1rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* FAISS metric values styling - More specific selectors */
        .faiss-metric .stMetric,
        .faiss-metric .stMetric > div,
        .faiss-metric .stMetric > div > div,
        .faiss-metric .stMetric > div > div > div,
        .faiss-metric .stMetric > div > div > div > div,
        .faiss-metric .stMetric > div > div > div > div > div {
            background-color: transparent !important;
            color: var(--text-color) !important;
        }
        
        /* Target Streamlit metric components directly */
        .faiss-metric [data-testid="metric-container"],
        .faiss-metric [data-testid="metric-container"] > div,
        .faiss-metric [data-testid="metric-container"] > div > div,
        .faiss-metric [data-testid="metric-container"] > div > div > div {
            background-color: transparent !important;
            color: var(--text-color) !important;
        }
        
        /* Target metric labels and values specifically */
        .faiss-metric [data-testid="metric-container"] [data-testid="metric-label"],
        .faiss-metric [data-testid="metric-container"] [data-testid="metric-value"] {
            background-color: transparent !important;
            color: var(--text-color) !important;
        }
        
        /* Dark theme FAISS metrics */
        [data-theme="dark"] .faiss-metric .stMetric,
        [data-theme="dark"] .faiss-metric .stMetric > div,
        [data-theme="dark"] .faiss-metric .stMetric > div > div,
        [data-theme="dark"] .faiss-metric .stMetric > div > div > div {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* Light theme FAISS metrics */
        [data-theme="light"] .faiss-metric .stMetric,
        [data-theme="light"] .faiss-metric .stMetric > div,
        [data-theme="light"] .faiss-metric .stMetric > div > div,
        [data-theme="light"] .faiss-metric .stMetric > div > div > div {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        /* Maximum specificity for metric styling - Override all Streamlit defaults */
        html body [data-theme="light"] .faiss-metric .stMetric,
        html body [data-theme="light"] .faiss-metric .stMetric > div,
        html body [data-theme="light"] .faiss-metric .stMetric > div > div,
        html body [data-theme="light"] .faiss-metric .stMetric > div > div > div,
        html body [data-theme="light"] .faiss-metric .stMetric > div > div > div > div,
        html body [data-theme="light"] .faiss-metric .stMetric > div > div > div > div > div {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        html body [data-theme="dark"] .faiss-metric .stMetric,
        html body [data-theme="dark"] .faiss-metric .stMetric > div,
        html body [data-theme="dark"] .faiss-metric .stMetric > div > div,
        html body [data-theme="dark"] .faiss-metric .stMetric > div > div > div,
        html body [data-theme="dark"] .faiss-metric .stMetric > div > div > div > div,
        html body [data-theme="dark"] .faiss-metric .stMetric > div > div > div > div > div {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* Target all possible metric container variations */
        html body [data-theme="light"] .faiss-metric [data-testid="metric-container"],
        html body [data-theme="light"] .faiss-metric [data-testid="metric-container"] > div,
        html body [data-theme="light"] .faiss-metric [data-testid="metric-container"] > div > div,
        html body [data-theme="light"] .faiss-metric [data-testid="metric-container"] > div > div > div,
        html body [data-theme="light"] .faiss-metric [data-testid="metric-container"] > div > div > div > div {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-container"],
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-container"] > div,
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-container"] > div > div,
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-container"] > div > div > div,
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-container"] > div > div > div > div {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* Target metric labels and values specifically with maximum specificity */
        html body [data-theme="light"] .faiss-metric [data-testid="metric-label"],
        html body [data-theme="light"] .faiss-metric [data-testid="metric-value"],
        html body [data-theme="light"] .faiss-metric [data-testid="metric-delta"] {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-label"],
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-value"],
        html body [data-theme="dark"] .faiss-metric [data-testid="metric-delta"] {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        /* FAISS dataframes - theme compatible */
        .faiss-dataframe {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Target Streamlit dataframe components directly */
        .faiss-dataframe [data-testid="stDataFrame"],
        .faiss-dataframe [data-testid="stDataFrame"] > div,
        .faiss-dataframe [data-testid="stDataFrame"] > div > div,
        .faiss-dataframe [data-testid="stDataFrame"] > div > div > div {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
        }
        
        /* Dataframe table styling */
        .faiss-dataframe table,
        .faiss-dataframe thead,
        .faiss-dataframe tbody,
        .faiss-dataframe tr,
        .faiss-dataframe th,
        .faiss-dataframe td {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }
        
        /* Dark theme dataframe overrides */
        [data-theme="dark"] .faiss-dataframe [data-testid="stDataFrame"],
        [data-theme="dark"] .faiss-dataframe [data-testid="stDataFrame"] > div,
        [data-theme="dark"] .faiss-dataframe [data-testid="stDataFrame"] > div > div,
        [data-theme="dark"] .faiss-dataframe [data-testid="stDataFrame"] > div > div > div,
        [data-theme="dark"] .faiss-dataframe table,
        [data-theme="dark"] .faiss-dataframe thead,
        [data-theme="dark"] .faiss-dataframe tbody,
        [data-theme="dark"] .faiss-dataframe tr,
        [data-theme="dark"] .faiss-dataframe th,
        [data-theme="dark"] .faiss-dataframe td {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border-color: #404040 !important;
        }
        
        /* Light theme dataframe overrides */
        [data-theme="light"] .faiss-dataframe [data-testid="stDataFrame"],
        [data-theme="light"] .faiss-dataframe [data-testid="stDataFrame"] > div,
        [data-theme="light"] .faiss-dataframe [data-testid="stDataFrame"] > div > div,
        [data-theme="light"] .faiss-dataframe [data-testid="stDataFrame"] > div > div > div,
        [data-theme="light"] .faiss-dataframe table,
        [data-theme="light"] .faiss-dataframe thead,
        [data-theme="light"] .faiss-dataframe tbody,
        [data-theme="light"] .faiss-dataframe tr,
        [data-theme="light"] .faiss-dataframe th,
        [data-theme="light"] .faiss-dataframe td {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-color: #dee2e6 !important;
        }
        
        /* FAISS tabs - theme compatible */
        .faiss-tabs {
            background-color: var(--background-color) !important;
        }
        
        /* FAISS status indicators */
        .faiss-status-active {
            color: #28a745 !important;
        }
        
        .faiss-status-empty {
            color: #ffc107 !important;
        }
        
        .faiss-status-error {
            color: #dc3545 !important;
        }
        
        /* Dark theme FAISS overrides */
        [data-theme="dark"] .faiss-dashboard {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .faiss-metric {
            background-color: #2d2d2d !important;
            border-color: #404040 !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .faiss-dataframe {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        /* Light theme FAISS overrides */
        [data-theme="light"] .faiss-dashboard {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .faiss-metric {
            background-color: #f8f9fa !important;
            border-color: #dee2e6 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .faiss-dataframe {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        [data-theme="dark"] .stFormSubmitButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stFormSubmitButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Light theme form submit button overrides */
        [data-theme="light"] .stFormSubmitButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="light"] .stFormSubmitButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
            color: #ffffff !important;
        }
        
        /* Force white text for light theme form submit buttons with higher specificity */
        body:not([data-theme="dark"]) .stFormSubmitButton > button {
            color: #ffffff !important;
        }
        
        html body:not([data-theme="dark"]) .stFormSubmitButton > button {
            color: #ffffff !important;
        }
        
        /* Maximum specificity for light theme form submit button text */
        html body:not([data-theme="dark"]) .stFormSubmitButton > button,
        html body:not([data-theme="dark"]) .stFormSubmitButton > button span,
        html body:not([data-theme="dark"]) .stFormSubmitButton > button div,
        html body:not([data-theme="dark"]) .stFormSubmitButton > button p {
            color: #ffffff !important;
        }
        
        /* Streamlit selectbox styling - Dark theme */
        .stSelectbox [data-baseweb="select"] {
            background-color: #0d1117 !important;
            border-color: #30363d !important;
            color: #ffffff !important;
        }
        
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        
        .stSelectbox [data-baseweb="popover"] {
            background-color: #0d1117 !important;
            border-color: #30363d !important;
        }
        
        .stSelectbox [data-baseweb="popover"] li {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        
        .stSelectbox [data-baseweb="popover"] li:hover {
            background-color: #21262d !important;
        }
        
        /* Light theme selectbox overrides */
        [data-theme="light"] .stSelectbox [data-baseweb="select"] {
            background-color: #ffffff !important;
            border-color: #dee2e6 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stSelectbox [data-baseweb="popover"] {
            background-color: #ffffff !important;
        }
        
        [data-theme="light"] .stSelectbox [data-baseweb="popover"] li {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stSelectbox [data-baseweb="popover"] li:hover {
            background-color: #f8f9fa !important;
        }
        
        /* Force light theme text color for all selectbox elements */
        [data-theme="light"] .stSelectbox * {
            color: #000000 !important;
        }
        
        /* Additional light theme overrides with different selectors */
        .css-1d391kg .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        .css-18e3th9 .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        div[data-testid="block-container"] .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] {
            color: #000000 !important;
        }
        
        /* Force black text for light theme using body class */
        body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] span,
        body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] div,
        body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] p {
            color: #000000 !important;
        }
        
        /* Override any theme.py selectbox styling */
        .stSidebar .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        .stSidebar .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] span,
        .stSidebar .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] div {
            color: inherit !important;
        }
        
        /* Maximum specificity overrides for light theme */
        html body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        html body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] span,
        html body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] div,
        html body:not([data-theme="dark"]) .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] p {
            color: #000000 !important;
        }
        
        /* Override theme.py variables for selectbox */
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] {
            color: var(--text-color, #000000) !important;
        }
        
        /* Force light theme colors */
        :root {
            --selectbox-text-color: #000000;
        }
        
        [data-theme="light"] {
            --selectbox-text-color: #000000;
        }
        
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] {
            color: var(--selectbox-text-color) !important;
        }
        
        /* Streamlit alert compatibility - Dark theme */
        .stAlert {
            background-color: #0d1117 !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        .stAlert[data-baseweb="notification"] {
            background-color: #0d1117 !important;
            border-color: #ffc107 !important;
            color: #ffffff !important;
        }
        
        /* Success box styling - Dark theme */
        .stSuccess {
            background-color: #0d1117 !important;
            border-color: #198754 !important;
            color: #ffffff !important;
        }
        
        /* Error box styling - Dark theme */
        .stError {
            background-color: #0d1117 !important;
            border-color: #dc3545 !important;
            color: #ffffff !important;
        }
        
        /* Warning box styling - Dark theme */
        .stWarning {
            background-color: #0d1117 !important;
            border-color: #ffc107 !important;
            color: #ffffff !important;
        }
        
        /* Light theme alert overrides */
        [data-theme="light"] .stAlert {
            background-color: #f8f9fa !important;
            border-color: #0d6efd !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stSuccess {
            background-color: #d1e7dd !important;
            border-color: #198754 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stError {
            background-color: #f8d7da !important;
            border-color: #dc3545 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stWarning {
            background-color: #fff3cd !important;
            border-color: #ffc107 !important;
            color: #000000 !important;
        }
        
        /* Streamlit expander styling - Dark theme */
        .streamlit-expanderHeader {
            background-color: #0d1117 !important;
            border-color: #30363d !important;
            color: #ffffff !important;
        }
        
        .streamlit-expanderContent {
            background-color: #0d1117 !important;
            border-color: #30363d !important;
            color: #ffffff !important;
        }
        
        /* Light theme expander overrides */
        [data-theme="light"] .streamlit-expanderHeader {
            background-color: #ffffff !important;
            border-color: #dee2e6 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .streamlit-expanderContent {
            background-color: #ffffff !important;
            border-color: #dee2e6 !important;
            color: #000000 !important;
        }
        
        /* Streamlit metric styling - Dark theme */
        .stMetric {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        
        .stMetric [data-testid="metric-container"] {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        
        .stMetric [data-testid="metric-container"] label {
            color: #ffffff !important;
        }
        
        .stMetric [data-testid="metric-container"] div {
            color: #ffffff !important;
        }
        
        /* Light theme metric overrides */
        [data-theme="light"] .stMetric {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stMetric [data-testid="metric-container"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stMetric [data-testid="metric-container"] label {
            color: #000000 !important;
        }
        
        [data-theme="light"] .stMetric [data-testid="metric-container"] div {
            color: #000000 !important;
        }
        
        /* Streamlit dataframe styling - Dark theme */
        .stDataFrame {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        
        .stDataFrame [data-testid="stDataFrame"] {
            background-color: #0d1117 !important;
            color: #ffffff !important;
        }
        
        /* Light theme dataframe overrides */
        [data-theme="light"] .stDataFrame {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stDataFrame [data-testid="stDataFrame"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Streamlit plotly chart styling - Dark theme */
        .stPlotlyChart {
            background-color: #0d1117 !important;
        }
        
        /* Light theme plotly chart overrides */
        [data-theme="light"] .stPlotlyChart {
            background-color: #ffffff !important;
        }
        
        /* Comprehensive Streamlit tabs styling for all themes */
        
        /* Base tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--background-color) !important;
            border-color: var(--border-color) !important;
            border-radius: 0.375rem !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
            font-weight: 500 !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: var(--hover-color) !important;
            color: var(--text-color) !important;
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
            font-weight: 600 !important;
        }
        
        /* Dark theme specific tab styling */
        [data-theme="dark"] .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e !important;
            border-color: #404040 !important;
        }
        
        [data-theme="dark"] .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border-color: #404040 !important;
        }
        
        [data-theme="dark"] .stTabs [data-baseweb="tab"]:hover {
            background-color: #404040 !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* Light theme specific tab styling */
        [data-theme="light"] .stTabs [data-baseweb="tab-list"] {
            background-color: #ffffff !important;
            border-color: #dee2e6 !important;
        }
        
        [data-theme="light"] .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa !important;
            color: #495057 !important;
            border-color: #dee2e6 !important;
        }
        
        [data-theme="light"] .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef !important;
            color: #212529 !important;
        }
        
        [data-theme="light"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* FAISS specific tab styling */
        .faiss-tabs .stTabs [data-baseweb="tab-list"] {
            background-color: var(--background-color) !important;
            border-color: var(--border-color) !important;
        }
        
        .faiss-tabs .stTabs [data-baseweb="tab"] {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
            border-color: var(--border-color) !important;
        }
        
        .faiss-tabs .stTabs [data-baseweb="tab"]:hover {
            background-color: var(--hover-color) !important;
            color: var(--text-color) !important;
        }
        
        .faiss-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* Dark theme FAISS tabs */
        [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e !important;
            border-color: #404040 !important;
        }
        
        [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border-color: #404040 !important;
        }
        
        [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab"]:hover {
            background-color: #404040 !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* Light theme FAISS tabs */
        [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab-list"] {
            background-color: #ffffff !important;
            border-color: #dee2e6 !important;
        }
        
        [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa !important;
            color: #495057 !important;
            border-color: #dee2e6 !important;
        }
        
        [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef !important;
            color: #212529 !important;
        }
        
        [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* Maximum specificity for tab styling - Override Streamlit defaults */
        html body [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab-list"] {
            background-color: #ffffff !important;
            border-color: #dee2e6 !important;
        }
        
        html body [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa !important;
            color: #495057 !important;
            border-color: #dee2e6 !important;
        }
        
        html body [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef !important;
            color: #212529 !important;
        }
        
        html body [data-theme="light"] .faiss-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* Dark theme maximum specificity */
        html body [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab-list"] {
            background-color: #1e1e1e !important;
            border-color: #404040 !important;
        }
        
        html body [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
            border-color: #404040 !important;
        }
        
        html body [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab"]:hover {
            background-color: #404040 !important;
            color: #ffffff !important;
        }
        
        html body [data-theme="dark"] .faiss-tabs .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
            border-color: #0d6efd !important;
        }
        
        /* Universal button visibility fixes - Final overrides */
        
        /* Ensure all button text is visible regardless of theme */
        .stButton > button,
        .stFormSubmitButton > button {
            text-shadow: none !important;
            font-weight: 500 !important;
        }
        
        /* Force proper contrast for all buttons */
        .stButton > button:not([data-testid="baseButton-text"]):not([data-testid="baseButton-outlined"]):not([data-testid="baseButton-link"]):not([data-testid="baseButton-ghost"]):not([data-testid="baseButton-quaternary"]):not([data-testid="baseButton-quinary"]) {
            color: #ffffff !important;
        }
        
        /* Ensure light-colored buttons have dark text */
        .stButton > button[data-testid="baseButton-quaternary"],
        .stButton > button[data-testid="baseButton-quinary"] {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Ensure outlined and text buttons have proper contrast */
        .stButton > button[data-testid="baseButton-outlined"] {
            color: #0d6efd !important;
            font-weight: 500 !important;
        }
        
        .stButton > button[data-testid="baseButton-text"] {
            color: #0d6efd !important;
            font-weight: 500 !important;
        }
        
        .stButton > button[data-testid="baseButton-link"] {
            color: #0d6efd !important;
            font-weight: 500 !important;
        }
        
        .stButton > button[data-testid="baseButton-ghost"] {
            color: #6c757d !important;
            font-weight: 500 !important;
        }
        
        /* Dark theme specific button text colors */
        [data-theme="dark"] .stButton > button[data-testid="baseButton-outlined"] {
            color: #0d6efd !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-text"] {
            color: #0d6efd !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-link"] {
            color: #0d6efd !important;
        }
        
        [data-theme="dark"] .stButton > button[data-testid="baseButton-ghost"] {
            color: #adb5bd !important;
        }
        
        /* Light theme specific button text colors */
        [data-theme="light"] .stButton > button[data-testid="baseButton-outlined"] {
            color: #0d6efd !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-text"] {
            color: #0d6efd !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-link"] {
            color: #0d6efd !important;
        }
        
        [data-theme="light"] .stButton > button[data-testid="baseButton-ghost"] {
            color: #6c757d !important;
        }
        
        /* Final fallback for any remaining button visibility issues */
        .stButton > button * {
            color: inherit !important;
        }
        
        /* Ensure button text is never transparent or invisible */
        .stButton > button {
            opacity: 1 !important;
        }
        
        .stButton > button * {
            opacity: 1 !important;
        }
        
        /* Additional Streamlit component overrides for FAISS dashboard */
        
        /* Override any remaining dark backgrounds in FAISS dashboard */
        .faiss-dashboard * {
            background-color: inherit !important;
        }
        
        /* Force light theme colors for FAISS dashboard in light mode */
        [data-theme="light"] .faiss-dashboard * {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .faiss-dashboard .stMetric *,
        [data-theme="light"] .faiss-dashboard [data-testid="metric-container"] * {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        /* Maximum specificity for light theme metric overrides */
        html body [data-theme="light"] .faiss-dashboard .stMetric,
        html body [data-theme="light"] .faiss-dashboard .stMetric *,
        html body [data-theme="light"] .faiss-dashboard [data-testid="metric-container"],
        html body [data-theme="light"] .faiss-dashboard [data-testid="metric-container"] *,
        html body [data-theme="light"] .faiss-dashboard [data-testid="metric-label"],
        html body [data-theme="light"] .faiss-dashboard [data-testid="metric-value"],
        html body [data-theme="light"] .faiss-dashboard [data-testid="metric-delta"] {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        /* Override any Streamlit default metric styling */
        html body [data-theme="light"] .faiss-dashboard div[data-testid="stMetric"] {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        html body [data-theme="light"] .faiss-dashboard div[data-testid="stMetric"] * {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .faiss-dashboard [data-testid="stDataFrame"] * {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Force dark theme colors for FAISS dashboard in dark mode */
        [data-theme="dark"] .faiss-dashboard * {
            background-color: #1e1e1e !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .faiss-dashboard .stMetric *,
        [data-theme="dark"] .faiss-dashboard [data-testid="metric-container"] * {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .faiss-dashboard [data-testid="stDataFrame"] * {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        /* Maximum specificity for FAISS dashboard theme overrides */
        html body [data-theme="light"] .faiss-dashboard .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa !important;
            color: #495057 !important;
        }
        
        html body [data-theme="light"] .faiss-dashboard .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        html body [data-theme="dark"] .faiss-dashboard .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d !important;
            color: #ffffff !important;
        }
        
        html body [data-theme="dark"] .faiss-dashboard .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        /* Final override for any remaining Streamlit metric styling issues */
        html body [data-theme="light"] .faiss-metric,
        html body [data-theme="light"] .faiss-metric *,
        html body [data-theme="light"] .faiss-dashboard .stMetric,
        html body [data-theme="light"] .faiss-dashboard .stMetric *,
        html body [data-theme="light"] .faiss-dashboard [data-testid="stMetric"],
        html body [data-theme="light"] .faiss-dashboard [data-testid="stMetric"] * {
            background-color: transparent !important;
            color: #000000 !important;
        }
        
        /* Ensure metric text is always visible */
        html body [data-theme="light"] .faiss-metric p,
        html body [data-theme="light"] .faiss-metric span,
        html body [data-theme="light"] .faiss-metric div,
        html body [data-theme="light"] .faiss-dashboard .stMetric p,
        html body [data-theme="light"] .faiss-dashboard .stMetric span,
        html body [data-theme="light"] .faiss-dashboard .stMetric div {
            color: #000000 !important;
            background-color: transparent !important;
        }
        
        /* Dark theme counterparts */
        html body [data-theme="dark"] .faiss-metric,
        html body [data-theme="dark"] .faiss-metric *,
        html body [data-theme="dark"] .faiss-dashboard .stMetric,
        html body [data-theme="dark"] .faiss-dashboard .stMetric *,
        html body [data-theme="dark"] .faiss-dashboard [data-testid="stMetric"],
        html body [data-theme="dark"] .faiss-dashboard [data-testid="stMetric"] * {
            background-color: transparent !important;
            color: #ffffff !important;
        }
        
        html body [data-theme="dark"] .faiss-metric p,
        html body [data-theme="dark"] .faiss-metric span,
        html body [data-theme="dark"] .faiss-metric div,
        html body [data-theme="dark"] .faiss-dashboard .stMetric p,
        html body [data-theme="dark"] .faiss-dashboard .stMetric span,
        html body [data-theme="dark"] .faiss-dashboard .stMetric div {
            color: #ffffff !important;
            background-color: transparent !important;
        }
        
        /* Custom FAISS metric cards styling */
        .faiss-metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .faiss-metric-card {
            background-color: var(--background-color);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            transition: all 0.2s ease-in-out;
        }
        
        .faiss-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .faiss-metric-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .faiss-metric-label {
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
            color: var(--text-color);
        }
        
        .faiss-metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-color);
        }
        
        /* Light theme custom metrics */
        [data-theme="light"] .faiss-metric-card {
            background-color: #ffffff;
            border-color: #dee2e6;
            color: #000000;
        }
        
        [data-theme="light"] .faiss-metric-label,
        [data-theme="light"] .faiss-metric-value {
            color: #000000;
        }
        
        /* Dark theme custom metrics */
        [data-theme="dark"] .faiss-metric-card {
            background-color: #2d2d2d;
            border-color: #404040;
            color: #ffffff;
        }
        
        [data-theme="dark"] .faiss-metric-label,
        [data-theme="dark"] .faiss-metric-value {
            color: #ffffff;
        }
        
        /* Responsive design for smaller screens */
        @media (max-width: 768px) {
            .faiss-metrics-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.5rem;
            }
            
            .faiss-metric-card {
                padding: 0.75rem;
            }
            
            .faiss-metric-icon {
                font-size: 1.5rem;
            }
            
            .faiss-metric-value {
                font-size: 1.25rem;
            }
        }
        
        /* Sidebar theme toggle button styling */
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
            font-weight: 500 !important;
            min-height: 2.5rem !important;
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Theme toggle button specific styling */
        [data-testid="stSidebar"] button[key="sidebar_theme_toggle"] {
            font-size: 1.2rem !important;
            padding: 0.5rem !important;
            min-width: 2.5rem !important;
            border-radius: 0.375rem !important;
        }
        
        /* Light theme sidebar buttons */
        [data-theme="light"] [data-testid="stSidebar"] .stButton > button {
            background-color: #ffffff !important;
            border: 1px solid #dee2e6 !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #f8f9fa !important;
            border-color: #adb5bd !important;
        }
        
        /* Dark theme sidebar buttons */
        [data-theme="dark"] [data-testid="stSidebar"] .stButton > button {
            background-color: #2d2d2d !important;
            border: 1px solid #404040 !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #404040 !important;
            border-color: #6c757d !important;
        }
        
        /* Hide Streamlit's default page navigation */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        
        /* Hide any other page navigation elements */
        .css-1d391kg {
            display: none !important;
        }
        
        /* Additional selectors to hide page navigation */
        .css-1lcbmhc {
            display: none !important;
        }
        
        /* Hide the entire page navigation section */
        section[data-testid="stSidebarNav"] {
            display: none !important;
        }

    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render clean, minimal header with branding"""
    # Simple header with just branding
    st.title("🦙 RAG LlamaStack")
    st.caption("Retrieval-Augmented Generation with LlamaStack & Ollama")


def render_sidebar():
    """Render the sidebar with authentication-aware navigation"""
    with st.sidebar:
        # Authentication section (if available)
        try:
            from core.auth.authentication import AuthManager
            if AuthManager.is_authenticated():
                user = AuthManager.get_current_user()
                st.markdown(f"### 👤 {user.username}")
                st.caption(f"Email: {user.email} | Role: {user.role}")
                
                # User actions in a clean layout
                col1, col2 = st.columns(2)
                with col1:
                    # Theme toggle button
                    theme_icon = "☀️" if st.session_state.dark_theme else "🌙"
                    theme_tooltip = "Switch to Light Theme" if st.session_state.dark_theme else "Switch to Dark Theme"
                    if st.button(theme_icon, key="sidebar_theme_toggle", help=theme_tooltip):
                        st.session_state.dark_theme = not st.session_state.dark_theme
                        st.rerun()
                
                with col2:
                    # Logout button
                    if st.button("🚪 Logout", key="sidebar_logout"):
                        AuthManager.logout()
                        st.success("✅ Logged out successfully")
                        st.rerun()
                
                st.markdown("---")
        except ImportError:
            pass  # Continue without authentication
        
        # System Status Section (Top Priority)
        st.markdown("### 🔌 System Status")
        
        # Connection status with real-time indicators
        col1, col2 = st.columns(2)
        with col1:
            if validate_llamastack_connection():
                st.success("🟢 LlamaStack")
            else:
                st.error("🔴 LlamaStack")
        
        with col2:
            # Check Ollama status
            try:
                import subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
                if result.returncode == 0:
                    st.success("🟢 Ollama")
                else:
                    st.warning("🟡 Ollama")
            except Exception:
                st.error("🔴 Ollama")
        
        # Quick setup info (collapsible)
        with st.expander("⚙️ Configuration & Help", expanded=False):
            st.markdown("""
            **Current Setup:**
            - 🔍 Embeddings: all-MiniLM-L6-v2 (sentence-transformers)
            - 🧠 LLM: See Model Dashboard below
            - 🌐 Web Processing: MCP Server + Fallback
            
            **Connection Issues?**
            - Use diagnostics in Model Dashboard
            - Check `llamastack-config.yaml`
            - Restart with `make restart`
            
            **Performance Tips:**
            - Add API providers for faster responses
            - Use local Ollama models for privacy
            - Mix file uploads with web URLs for comprehensive knowledge base
            """)
        
        st.markdown("---")
        
        # Model Dashboard
        render_model_dashboard()
        
        # Navigation buttons
        st.markdown("### 🧭 Navigation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Document Management", type="primary", use_container_width=True):
                st.session_state.current_page = "Document_Management"
                st.rerun()
        
        with col2:
            if st.button("🗄️ Database Dashboard", type="secondary", use_container_width=True):
                st.session_state.current_page = "Database_Dashboard"
                st.rerun()
        
        # FAISS Dashboard button
        if st.button("🔍 FAISS Dashboard", type="secondary", use_container_width=True):
            st.session_state.current_page = "FAISS_Dashboard"
            st.rerun()
        
        # Ollama Integration
        render_ollama_integration()
        
        # Sync Status
        if st.session_state.get('authenticated'):
            try:
                from core.faiss_sync_manager import faiss_sync_manager
                sync_status = faiss_sync_manager.get_sync_status()
                
                if not sync_status.get('in_sync', False):
                    st.markdown("### 🔄 Database Sync Status")
                    st.warning("⚠️ FAISS and SQLite databases are out of sync!")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**SQLite:** {sync_status.get('sqlite_documents', 0)} documents")
                        st.write(f"**FAISS:** {sync_status.get('faiss_documents', 0)} documents")
                    with col2:
                        if st.button("🔄 Sync Now", type="primary"):
                            with st.spinner("Synchronizing..."):
                                user_id = st.session_state.get('user_id')
                                success = faiss_sync_manager.force_sync(user_id)
                                if success:
                                    st.success("✅ Databases synchronized!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to synchronize")
            except Exception as e:
                st.error(f"❌ Sync status check failed: {e}")
        
        # Connection Diagnostics & Debug Tools (Bottom of navbar)
        st.markdown("---")
        with st.expander("🔍 Connection Diagnostics & Debug", expanded=False):
            st.markdown("**Advanced troubleshooting tools**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🩺 Diagnose LlamaStack", help="Check LlamaStack endpoints and connectivity"):
                    with st.spinner("Diagnosing LlamaStack..."):
                        diagnosis = st.session_state.llamastack_client.diagnose_llamastack()
                        
                        if diagnosis["reachable"]:
                            st.success(f"✅ LlamaStack reachable at {diagnosis['base_url']}")
                            
                            if diagnosis["available_endpoints"]:
                                st.info("📡 **Available Endpoints:**")
                                for endpoint in diagnosis["available_endpoints"]:
                                    st.text(f"• {endpoint}")
                            else:
                                st.warning("⚠️ No API endpoints found")
                        else:
                            st.error(f"❌ Cannot reach LlamaStack at {diagnosis['base_url']}")
                        
                        if diagnosis["recommendations"]:
                            st.warning("💡 **Recommendations:**")
                            for rec in diagnosis["recommendations"]:
                                st.text(f"• {rec}")
            
            with col2:
                if st.button("🦙 Check Ollama", help="Check if Ollama is running and has models"):
                    with st.spinner("Checking Ollama..."):
                        from core.model_manager import check_ollama_status
                        ollama_status = check_ollama_status()
                        
                        if ollama_status["running"]:
                            st.success("✅ Ollama is running")
                            if ollama_status["models"]:
                                st.info(f"📦 **{len(ollama_status['models'])} models available:**")
                                for model in ollama_status["models"][:5]:  # Show first 5
                                    st.text(f"• {model['name']}")
                                if len(ollama_status["models"]) > 5:
                                    st.text(f"... and {len(ollama_status['models']) - 5} more")
                            else:
                                st.warning("⚠️ No models found in Ollama")
                                st.info("💡 Pull a model with: `ollama pull llama3.2:1b`")
                        else:
                            st.error("❌ Ollama is not running")
                            st.info("💡 Start Ollama with: `ollama serve`")
            
            # MCP Server diagnostic check
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                if st.button("🌐 Test MCP Server", help="Check if MCP server is available for web content processing"):
                    with st.spinner("Testing MCP Server..."):
                        try:
                            import subprocess
                            result = subprocess.run(
                                ["npx", "@just-every/mcp-read-website-fast", "--version"],
                                capture_output=True,
                                text=True,
                                timeout=10
                            )
                            
                            if result.returncode == 0:
                                st.success("✅ MCP Server available")
                                st.info("🔧 Web content extraction will use MCP server for optimal quality")
                            else:
                                st.warning("⚠️ MCP Server not available")
                                st.info("🔄 Web content extraction will use fallback method")
                        except Exception as e:
                            st.warning("⚠️ MCP Server check failed")
                            st.info("🔄 Fallback method will be used automatically")
                            st.text(f"Details: {str(e)}")
            
            with col4:
                if st.button("🧪 Test Web Processing", help="Test web content extraction with example URL"):
                    with st.spinner("Testing web content extraction..."):
                        try:
                            # Initialize web processor if not exists
                            if 'web_content_processor' not in st.session_state:
                                from core.web_content_processor import WebContentProcessor
                                st.session_state.web_content_processor = WebContentProcessor()
                            
                            # Test with example.com
                            test_url = "https://example.com"
                            processor = st.session_state.web_content_processor
                            
                            if processor.is_valid_url(test_url):
                                st.success("✅ URL validation works")
                                st.info("🌐 Web content processing is ready!")
                                st.text("Try entering any URL in the 'Web URLs' tab")
                            else:
                                st.error("❌ URL validation failed")
                        except Exception as e:
                            st.error("❌ Web processing test failed")
                            st.text(f"Error: {str(e)}")
            
            # Debug information
            st.markdown("---")
            st.markdown("**🐛 Debug Information**")
            
            if st.button("📋 Show Debug Info", help="Display current configuration and state"):
                debug_info = {
                    "LlamaStack URL": st.session_state.llamastack_client.base_url,
                    "Selected LLM": st.session_state.selected_llm_model,
                    "Selected Embedding": st.session_state.selected_embedding_model,
                    "Documents Loaded": len(st.session_state.uploaded_documents) if 'uploaded_documents' in st.session_state else 0,
                    "Chat History": len(st.session_state.chat_history) if 'chat_history' in st.session_state else 0,
                    "Web Processor Available": 'web_content_processor' in st.session_state
                }
                
                st.json(debug_info)


if __name__ == "__main__":
    main()
