"""
RAG LlamaStack Streamlit Application
Main application entry point - now modular and clean!
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
    """Main application function"""
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


def render_header():
    """Render clean, minimal header with branding"""
    # Render the fixed-position theme toggle
    render_header_theme_toggle()
    
    # Simple header with just branding and new feature highlight
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("🦙 RAG LlamaStack")
        st.caption("Retrieval-Augmented Generation with LlamaStack & Ollama")


def render_sidebar():
    """Render the sidebar with all components"""
    with st.sidebar:
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
