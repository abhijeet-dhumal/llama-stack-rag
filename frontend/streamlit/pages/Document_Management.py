"""
Document Management Page for RAG LlamaStack
Provides comprehensive document upload, management, and analytics capabilities
"""

import streamlit as st
import time
import sys
import os

# Add the components directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'components'))

def main():
    """Main function for Document Management page."""
    
    # Page header
    st.title("📄 Document Management")
    st.markdown("Upload, manage, and analyze your documents and web content")
    
    # Back button and debug controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("← Back to Main", type="secondary", key="doc_mgmt_back_btn"):
            st.session_state.current_page = None
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All Documents", type="secondary", key="doc_mgmt_clear_btn"):
            if 'faiss_documents' in st.session_state:
                st.session_state.faiss_documents = []
                st.success("All documents cleared!")
                st.rerun()
    with col3:
        if st.button("🧪 Test Progress", type="secondary", key="doc_mgmt_test_btn"):
            # Test progress indicator
            progress_container = st.container()
            with progress_container:
                st.markdown("### 🧪 Testing Progress Indicator...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(5):
                    progress = (i / 4)  # Use decimal (0.0-1.0) instead of percentage
                    progress_bar.progress(progress)
                    status_text.text(f"Test step {i+1}/5...")
                    time.sleep(0.5)
                
                progress_bar.progress(1.0)  # Use 1.0 instead of 100
                status_text.text("Test complete!")
                st.success("✅ Progress indicator test successful!")
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Documents", 
        "🌐 Web Content", 
        "📚 Document Library", 
        "📊 Analytics"
    ])
    
    # Import and render tab components
    try:
        # Tab 1: File Upload
        with tab1:
            from components.file_upload_tab import render_file_upload_tab
            render_file_upload_tab()
        
        # Tab 2: Web Content
        with tab2:
            from components.web_content_tab import render_web_content_tab
            render_web_content_tab()
        
        # Tab 3: Document Library
        with tab3:
            from components.document_library_tab import render_document_library_tab
            render_document_library_tab()
        
        # Tab 4: Analytics
        with tab4:
            from components.analytics_tab import render_analytics_tab
            render_analytics_tab()
            
    except ImportError as e:
        st.error(f"❌ Error loading tab components: {e}")
        st.info("💡 Make sure all component files are properly created")
    except Exception as e:
        st.error(f"❌ Error rendering tabs: {e}")

if __name__ == "__main__":
    main() 