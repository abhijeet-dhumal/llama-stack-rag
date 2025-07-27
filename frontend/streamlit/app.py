"""
RAG LlamaStack Frontend Application
A Streamlit-based interface for RAG (Retrieval-Augmented Generation) with LlamaStack
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Set environment variable to avoid tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import core modules
from core.config import PAGE_CONFIG, LLAMASTACK_BASE, APP_DESCRIPTION
from core.theme import initialize_theme, apply_theme, render_header_theme_toggle, render_sidebar_theme_toggle
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
    """Main application function with Bootstrap styling"""
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
            Built with Streamlit + LlamaStack + Ollama for local AI processing
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
    
    # Check LlamaStack connection
    if not validate_llamastack_connection():
        st.error("🔴 LlamaStack Offline")
        st.stop()
    
    # Header (no top menu bar)
    render_header()
    
    # Sidebar
    render_sidebar()
    
    # Main content area
    if not has_documents():
        render_welcome_screen()
    else:
        render_chat_interface()


def load_bootstrap_css():
    """Load Bootstrap CSS and custom styles"""
    st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        /* Custom Bootstrap overrides for Streamlit */
        .stButton > button {
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
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
            background-color: #f8f9fa !important;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div {
            background: linear-gradient(45deg, #0d6efd, #0b5ed7) !important;
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 0.375rem !important;
            border: none !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa !important;
            border-radius: 0.375rem !important;
            border: 1px solid #dee2e6 !important;
        }
        
        /* Selectbox styling - Dark theme compatible */
        .stSelectbox > div > div > div {
            background-color: #3b3b3b !important;
            border: 1px solid #555555 !important;
            border-radius: 0.375rem !important;
            color: #ffffff !important;
            padding: 0.5rem !important;
            min-height: 2.5rem !important;
        }
        
        .stSelectbox > div > div > div:hover {
            border-color: #0d6efd !important;
            background-color: #4b4b4b !important;
        }
        
        /* Selectbox dropdown options */
        .stSelectbox > div > div > div > div {
            background-color: #3b3b3b !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* Selectbox dropdown arrow */
        .stSelectbox svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }
        
        /* File uploader styling - Dark theme */
        .stFileUploader > div {
            border: 2px dashed #0d6efd !important;
            border-radius: 0.375rem !important;
            padding: 1rem !important;
            text-align: center !important;
            background-color: #2b2b2b !important;
            color: #ffffff !important;
        }
        
        .stFileUploader > div > div {
            color: #ffffff !important;
        }
        
        /* File uploader text visibility */
        .stFileUploader p, .stFileUploader span, .stFileUploader div {
            color: #ffffff !important;
        }
        
        /* Radio button styling - Dark theme */
        .stRadio > div > div > div {
            color: #ffffff !important;
        }
        
        .stRadio > div > div > div > label {
            color: #ffffff !important;
        }
        
        /* Text input styling - Dark theme */
        .stTextInput > div > div > input {
            background-color: #3b3b3b !important;
            border-color: #555555 !important;
            color: #ffffff !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: #aaaaaa !important;
        }
        
        /* Button styling - Theme-aware */
        .stButton > button {
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Dark theme button overrides */
        [data-theme="dark"] .stButton > button {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stButton > button:hover {
            background-color: #0b5ed7 !important;
            border-color: #0b5ed7 !important;
        }
        
        /* Light theme button overrides */
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
        
        /* Form submit button styling - Theme-aware */
        .stFormSubmitButton > button {
            border-radius: 0.375rem !important;
            transition: all 0.2s ease-in-out !important;
        }
        
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 0.25rem 0.5rem rgba(0, 0, 0, 0.15) !important;
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

        [data-theme="light"] .stButton button[kind="primary"] {
            color: white !important;
        }
        
        /* Secondary button styling */
        .stButton > button[kind="secondary"] {
            background-color: #6c757d !important;
            border-color: #6c757d !important;
            color: #ffffff !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background-color: #5a6268 !important;
            border-color: #5a6268 !important;
        }
        
        /* Label styling - Dark theme */
        [data-theme="dark"] .stMarkdown, 
        [data-theme="dark"] .stText, 
        [data-theme="dark"] .stLabel {
            color: #ffffff !important;
        }

        /* Form text styling */
        .form-text, .text-muted {
            color: #aaaaaa !important;
        }
        
        /* Ensure all text in sidebar is visible */
        .css-1d391kg * {
            color: #ffffff !important;
        }
        
        /* Fix any remaining dark text on dark background */
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Selectbox styling - Theme-aware */
        .stSelectbox > div > div > div {
            border: 1px solid #555555 !important;
            border-radius: 0.375rem !important;
        }
        
        .stSelectbox > div > div > div:hover {
            border-color: #0d6efd !important;
        }
        
        /* Selectbox dropdown options */
        .stSelectbox [data-baseweb="select"] > div {
            border: 1px solid #555555 !important;
        }
        
        /* Selectbox dropdown list */
        .stSelectbox [data-baseweb="popover"] {
            border: 1px solid #555555 !important;
        }
        
        /* Additional selectbox fixes for better consistency */
        .stSelectbox label {
            font-weight: 500 !important;
        }
        
        /* Selectbox container */
        .stSelectbox {
            margin-bottom: 1rem !important;
        }
        
        /* Fix for selectbox placeholder text */
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"]::placeholder {
            color: #aaaaaa !important;
        }
        
        /* Light theme overrides for selectbox text */
        [data-theme="light"] .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        [data-theme="light"] .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] span,
        [data-theme="light"] .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] div,
        [data-theme="light"] .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] p {
            color: #000000 !important;
        }
        
        [data-theme="light"] .stSelectbox > div > div > div {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        [data-theme="light"] .stSelectbox [data-testid="stSelectbox"] {
            color: #000000 !important;
        }

        /* This will target the selected value in the selectbox specifically */
        [data-theme="light"] .stSelectbox [data-baseweb="select"] > div > div {
            color: #000000 !important;
        }
        
        /* Light theme dropdown options */
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
        
        /* Nuclear option for selectbox visibility - Force black text in light theme */
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"],
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] span,
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] div,
        .stSelectbox [data-baseweb="select"] [data-testid="stSelectbox"] p,
        .stSelectbox > div > div > div,
        .stSelectbox > div > div > div > div,
        .stSelectbox > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div > div > div,
        .stSelectbox > div > div > div > div > div > div > div > div > div {
            color: #000000 !important;
        }
        
        /* Override any white text colors */
        .stSelectbox [style*="color: white"],
        .stSelectbox [style*="color: #ffffff"],
        .stSelectbox [style*="color: rgb(255, 255, 255)"],
        .stSelectbox [style*="color: rgba(255, 255, 255"] {
            color: #000000 !important;
        }
        
        /* Force all selectbox elements to have black text */
        .stSelectbox * {
            color: #000000 !important;
        }
        
        /* Final nuclear option - Override ANY white text in selectbox */
        .stSelectbox,
        .stSelectbox *,
        .stSelectbox > div,
        .stSelectbox > div *,
        .stSelectbox section,
        .stSelectbox section *,
        .stSelectbox [data-baseweb="select"],
        .stSelectbox [data-baseweb="select"] *,
        .stSelectbox [data-baseweb="popover"],
        .stSelectbox [data-baseweb="popover"] *,
        .stSelectbox [data-testid="stSelectbox"],
        .stSelectbox [data-testid="stSelectbox"] * {
            color: #000000 !important;
        }
        
        /* Override any existing white color rules for selectbox */
        .stSelectbox [style*="color: white"] { color: #000000 !important; }
        .stSelectbox [style*="color: #ffffff"] { color: #000000 !important; }
        .stSelectbox [style*="color: rgb(255, 255, 255)"] { color: #000000 !important; }
        .stSelectbox [style*="color: rgba(255, 255, 255"] { color: #000000 !important; }
        
        /* Streamlit alert compatibility - Dark theme */
        [data-theme="dark"] .stAlert {
            background-color: #0d1117 !important;
            border-color: #0d6efd !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] .stAlert[data-baseweb="notification"] {
            background-color: #0d1117 !important;
            border-color: #ffc107 !important;
            color: #ffffff !important;
        }
        
        /* Success box styling - Dark theme */
        [data-theme="dark"] .stSuccess {
            background-color: #0d1117 !important;
            border-color: #198754 !important;
            color: #ffffff !important;
        }
        
        /* Error box styling - Dark theme */
        [data-theme="dark"] .stError {
            background-color: #0d1117 !important;
            border-color: #dc3545 !important;
            color: #ffffff !important;
        }
        
        /* Warning box styling - Dark theme */
        [data-theme="dark"] .stWarning {
            background-color: #0d1117 !important;
            border-color: #ffc107 !important;
            color: #ffffff !important;
        }
        
        /* Sidebar visibility and styling */
        [data-theme="dark"] section[data-testid="stSidebar"] {
            background-color: #2b2b2b !important;
            color: #ffffff !important;
        }
        
        [data-theme="dark"] section[data-testid="stSidebar"] .css-1d391kg {
            background-color: transparent !important;
        }
        
        /* Ensure all sidebar content is visible */
        [data-theme="dark"] section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Simple text color fixes for dark theme */
        [data-theme="dark"] .text-muted {
            color: #aaaaaa !important;
        }
        
        /* Fix sidebar text color */
        [data-theme="dark"] section[data-testid="stSidebar"] h1,
        [data-theme="dark"] section[data-testid="stSidebar"] h2,
        [data-theme="dark"] section[data-testid="stSidebar"] h3,
        [data-theme="dark"] section[data-testid="stSidebar"] p,
        [data-theme="dark"] section[data-testid="stSidebar"] label {
            color: #ffffff !important;
        }
        
        /* Remove extra horizontal line at top of sidebar */
        section[data-testid="stSidebar"] > div:first-child {
            border-top: none !important;
            border-bottom: none !important;
        }
        
        /* Remove borders from first few elements in sidebar */
        section[data-testid="stSidebar"] > div:nth-child(1),
        section[data-testid="stSidebar"] > div:nth-child(2),
        section[data-testid="stSidebar"] > div:nth-child(3) {
            border-top: none !important;
            border-bottom: none !important;
        }
        
        /* Remove any unwanted borders from sidebar elements */
        section[data-testid="stSidebar"] .stMarkdown:first-child,
        section[data-testid="stSidebar"] .stMarkdown:first-child h3 {
            border-top: none !important;
            border-bottom: none !important;
        }
        
        /* Clean up sidebar section dividers */
        section[data-testid="stSidebar"] .stMarkdown:has(h3) {
            border-top: none !important;
            padding-top: 0.5rem !important;
            margin-top: 0.5rem !important;
        }
        
        /* Fix expander headers in sidebar */
        [data-theme="dark"] section[data-testid="stSidebar"] .streamlit-expanderHeader {
            background-color: #3b3b3b !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
        }
        
        /* Fix expander headers in main content - More specific selectors */
        .streamlit-expanderHeader,
        [data-testid="stExpander"] .streamlit-expanderHeader,
        .stExpander .streamlit-expanderHeader {
            background-color: #3b3b3b !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            border-radius: 0.375rem !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Expander content area */
        .streamlit-expanderContent,
        [data-testid="stExpander"] .streamlit-expanderContent,
        .stExpander .streamlit-expanderContent {
            background-color: #2b2b2b !important;
            border: 1px solid #555555 !important;
            border-top: none !important;
            border-radius: 0 0 0.375rem 0.375rem !important;
            padding: 1rem !important;
        }
        
        /* Expander when collapsed */
        .streamlit-expanderHeader[aria-expanded="false"],
        [data-testid="stExpander"] .streamlit-expanderHeader[aria-expanded="false"] {
            border-radius: 0.375rem !important;
            border-bottom: 1px solid #555555 !important;
        }
        
        /* Expander when expanded */
        .streamlit-expanderHeader[aria-expanded="true"],
        [data-testid="stExpander"] .streamlit-expanderHeader[aria-expanded="true"] {
            border-radius: 0.375rem 0.375rem 0 0 !important;
            border-bottom: none !important;
        }
        
        /* Hover effect for expander headers */
        .streamlit-expanderHeader:hover,
        [data-testid="stExpander"] .streamlit-expanderHeader:hover {
            background-color: #4b4b4b !important;
            border-color: #0d6efd !important;
            box-shadow: 0 2px 4px rgba(13, 110, 253, 0.2) !important;
        }
        
        /* Force expander styling with higher specificity */
        div[data-testid="stExpander"] {
            border: 1px solid #555555 !important;
            border-radius: 0.375rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Override theme.py expander styling with maximum specificity */
        body .streamlit-expanderHeader,
        body [data-testid="stExpander"] .streamlit-expanderHeader,
        body .stExpander .streamlit-expanderHeader,
        html body .streamlit-expanderHeader {
            background-color: #3b3b3b !important;
            color: #ffffff !important;
            border: 1px solid #555555 !important;
            border-radius: 0.375rem !important;
            padding: 0.75rem 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Override theme.py expander content styling */
        body .streamlit-expanderContent,
        body [data-testid="stExpander"] .streamlit-expanderContent,
        body .stExpander .streamlit-expanderContent,
        html body .streamlit-expanderContent {
            background-color: #2b2b2b !important;
            border: 1px solid #555555 !important;
            border-top: none !important;
            border-radius: 0 0 0.375rem 0.375rem !important;
            padding: 1rem !important;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render clean, minimal header with branding"""
    # Render the fixed-position theme toggle
    # render_header_theme_toggle()
    
    # Simple header with just branding and new feature highlight
    # col1, col2 = st.columns([3, 1])
    
    # with col1:
    #     st.title("🦙 RAG LlamaStack")
    #     st.caption("Retrieval-Augmented Generation with LlamaStack & Ollama")


def render_sidebar():
    """Render the sidebar with all components"""
    with st.sidebar:
        # System Status Section (Top Priority)
        st.markdown("### 🔌 System Status")
        
        # Connection status with real-time indicators
        col1, col2, col3 = st.columns(3)
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

        with col3:
            render_sidebar_theme_toggle()
        
        st.markdown("---")
        
        # Model Dashboard
        render_model_dashboard()
        
        # Enhanced Content Sources Section with highlighting
        st.markdown("### 📁 Content Sources")
        
        # Document Upload & Web URL Section
        uploaded_files = render_file_uploader()
        
        if uploaded_files:
            valid_files = validate_uploaded_files(uploaded_files)
            if valid_files:
                # Clean up any previous interrupted state
                cleanup_interrupted_uploads()
                
                # Check for new files that haven't been successfully processed
                new_files = []
                retry_files = []
                
                for file in valid_files:
                    file_id = f"{file.name}_{file.size}"
                    
                    # Check if file is in failed uploads (needs retry)
                    if file_id in st.session_state.failed_uploads:
                        retry_files.append(file)
                        st.session_state.failed_uploads.discard(file_id)
                        continue
                    
                    # Check if file is currently being uploaded
                    if file_id in st.session_state.currently_uploading:
                        st.warning(f"⏳ File `{file.name}` is currently being processed...")
                        continue
                    
                    # Check if file is already successfully processed
                    if file_id not in st.session_state.processed_files:
                        new_files.append(file)
                
                # Show retry message if applicable
                if retry_files:
                    st.info(f"🔄 Retrying {len(retry_files)} previously failed upload(s)")
                
                # Process new and retry files
                files_to_process = new_files + retry_files
                if files_to_process:
                    process_uploaded_files_with_state_tracking(files_to_process)
        
        # Document Library
        render_document_library()
        
        # Ollama Integration
        render_ollama_integration()
        
        # FAISS Database Dashboard
        st.markdown("---")
        try:
            from components.faiss_dashboard import render_faiss_dashboard
            render_faiss_dashboard()
        except ImportError:
            st.markdown("### 🗄️ FAISS Database")
            st.info("FAISS dashboard component not available")
            
        # Connection Diagnostics & Debug Tools (Bottom of navbar)
        st.markdown("---")
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
                if st.button("🌐 Test MCP Servers", help="Check which MCP servers are available for web content processing"):
                    with st.spinner("Testing MCP Servers..."):
                        try:
                            import subprocess
                            
                            # Test all MCP servers
                            mcp_servers = [
                                {
                                    'name': 'Just-Every MCP',
                                    'command': ['npx', '@just-every/mcp-read-website-fast', '--version'],
                                    'description': '📦 Reliable markdown extraction (no API key required)'
                                }
                            ]
                            
                            available_servers = []
                            for server in mcp_servers:
                                try:
                                    # All servers are free (no API keys required)
                                    
                                    result = subprocess.run(
                                        server['command'],
                                        capture_output=True,
                                        text=True,
                                        timeout=5
                                    )
                                    if result.returncode == 0:
                                        available_servers.append(f"✅ {server['name']} - {server['description']}")
                                    else:
                                        available_servers.append(f"❌ {server['name']} - Not available")
                                except Exception:
                                    available_servers.append(f"❌ {server['name']} - Not installed")
                            
                            if any("✅" in server for server in available_servers):
                                st.success("✅ MCP Servers Available")
                                st.info("🔧 Web content extraction will use the best available MCP server")
                                
                                # Show available servers
                                st.markdown("**Available MCP Servers:**")
                                for server in available_servers:
                                    if "✅" in server:
                                        st.success(server)
                                    else:
                                        st.warning(server)
                                
                                # Show priority order
                                st.info("📊 **Priority Order:** Just-Every MCP → BeautifulSoup")
                            else:
                                st.warning("⚠️ No MCP servers available")
                                st.info("🔄 Web content extraction will use BeautifulSoup fallback")
                                st.info("💡 Install MCP servers: `make setup-mcp`")
                                
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
            
            # Quick MCP Server Status
            st.markdown("**🔧 MCP Server Status**")
            try:
                import subprocess
                
                # Quick check for Just-Every MCP (primary)
                result = subprocess.run(
                    ['npx', '@just-every/mcp-read-website-fast', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    st.success("✅ Just-Every MCP (Primary)")
                else:
                    st.warning("⚠️ Just-Every MCP (Not available)")
                    
            except Exception:
                st.warning("⚠️ MCP Server check failed")
            
            st.caption("💡 Click 'Test MCP Servers' for detailed status")


if __name__ == "__main__":
    main()
