import streamlit as st
import pandas as pd
from datetime import datetime

def render_document_library_tab():
    """Render the document library tab to display uploaded documents."""
    
    st.markdown("### 📚 Document Library")
    st.markdown("View and manage your uploaded documents and web content.")
    st.markdown("---")
    
    # Check if documents exist - use faiss_documents instead of uploaded_documents
    if 'faiss_documents' not in st.session_state or not st.session_state.faiss_documents:
        st.info("📚 No documents uploaded yet. Go to the 'Upload Documents' or 'Web Content' tabs to add documents.")
        return
    
    documents = st.session_state.faiss_documents
    
    # Document statistics
    st.markdown("#### 📊 Document Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(documents))
    
    with col2:
        file_docs = [d for d in documents if d.get('source') == 'file_upload']
        st.metric("File Documents", len(file_docs))
    
    with col3:
        web_docs = [d for d in documents if d.get('source') in ['bulk_upload', 'manual_entry']]
        st.metric("Web Documents", len(web_docs))
    
    with col4:
        total_size = sum(d.get('file_size_mb', 0) for d in documents)
        st.metric("Total Size (MB)", f"{total_size:.1f}")
    
    st.markdown("---")
    
    # Document filters
    st.markdown("#### 🔍 Document Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Source filter
        sources = list(set(d.get('source', 'unknown') for d in documents))
        selected_source = st.selectbox("Filter by Source", ["All"] + sources, key="library_source_filter")
    
    with col2:
        # File type filter
        file_types = list(set(d.get('file_type', 'unknown') for d in documents))
        selected_type = st.selectbox("Filter by Type", ["All"] + file_types, key="library_type_filter")
    
    with col3:
        # Status filter
        statuses = list(set(d.get('processing_status', 'unknown') for d in documents))
        selected_status = st.selectbox("Filter by Status", ["All"] + statuses, key="library_status_filter")
    
    # Apply filters
    filtered_documents = documents
    if selected_source != "All":
        filtered_documents = [d for d in filtered_documents if d.get('source') == selected_source]
    if selected_type != "All":
        filtered_documents = [d for d in filtered_documents if d.get('file_type') == selected_type]
    if selected_status != "All":
        filtered_documents = [d for d in filtered_documents if d.get('processing_status') == selected_status]
    
    st.markdown(f"**Showing {len(filtered_documents)} of {len(documents)} documents**")
    st.markdown("---")
    
    # Document list
    st.markdown("#### 📋 Document List")
    
    if not filtered_documents:
        st.info("No documents match the selected filters.")
        return
    
    # Create expandable sections for each document
    for i, doc in enumerate(filtered_documents):
        with st.expander(f"📄 {doc.get('name', 'Unknown Document')} ({doc.get('file_type', 'unknown')})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Document details
                st.markdown(f"**Name:** {doc.get('name', 'Unknown')}")
                st.markdown(f"**Type:** {doc.get('file_type', 'Unknown')}")
                st.markdown(f"**Size:** {doc.get('file_size_mb', 0):.2f} MB")
                st.markdown(f"**Chunks:** {doc.get('chunk_count', 0)}")
                st.markdown(f"**Status:** {doc.get('processing_status', 'Unknown')}")
                st.markdown(f"**Created:** {doc.get('created_at', 'Unknown')}")
                
                if doc.get('source_url'):
                    st.markdown(f"**Source URL:** {doc.get('source_url')}")
                
                if doc.get('source'):
                    source_display = {
                        'file_upload': '📤 File Upload',
                        'bulk_upload': '📊 Bulk URL Upload',
                        'manual_entry': '🔗 Manual URL Entry'
                    }.get(doc.get('source'), doc.get('source'))
                    st.markdown(f"**Source:** {source_display}")
            
            with col2:
                # Action buttons
                if st.button("🗑️ Delete", key=f"library_delete_{i}", type="secondary"):
                    delete_document(i)
                    st.rerun()
                
                if st.button("👁️ View Content", key=f"library_view_{i}", type="secondary"):
                    show_document_content(doc)
    
    # Export options
    st.markdown("---")
    st.markdown("#### 📤 Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export as CSV", key="library_export_csv", type="secondary"):
            export_documents_csv(filtered_documents)
    
    with col2:
        if st.button("🗑️ Clear All Documents", key="library_clear_all", type="secondary"):
            if st.button("⚠️ Confirm Clear All", key="library_confirm_clear", type="primary"):
                st.session_state.faiss_documents = []
                st.success("All documents cleared!")
                st.rerun()

def delete_document(index):
    """Delete a document from the session state."""
    if 'faiss_documents' in st.session_state:
        if 0 <= index < len(st.session_state.faiss_documents):
            deleted_doc = st.session_state.faiss_documents.pop(index)
            st.success(f"✅ Deleted: {deleted_doc.get('name', 'Unknown Document')}")

def show_document_content(doc):
    """Show document content in a modal-like display."""
    st.markdown("#### 📄 Document Content")
    st.markdown("---")
    
    # Show content preview (first 500 characters)
    content = doc.get('content', 'No content available')
    preview = content[:500] + "..." if len(content) > 500 else content
    
    st.text_area("Content Preview", preview, height=200, disabled=True)
    
    if len(content) > 500:
        with st.expander("View Full Content"):
            st.text_area("Full Content", content, height=400, disabled=True)

def export_documents_csv(documents):
    """Export documents to CSV format."""
    try:
        # Prepare data for CSV
        data = []
        for doc in documents:
            data.append({
                'Name': doc.get('name', 'Unknown'),
                'Type': doc.get('file_type', 'Unknown'),
                'Size (MB)': doc.get('file_size_mb', 0),
                'Chunks': doc.get('chunk_count', 0),
                'Status': doc.get('processing_status', 'Unknown'),
                'Created': doc.get('created_at', 'Unknown'),
                'Source': doc.get('source', 'Unknown'),
                'Source URL': doc.get('source_url', ''),
                'Content Preview': doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
            })
        
        # Create DataFrame and CSV
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        
        # Download button
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"documents_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ CSV export ready for download!")
        
    except Exception as e:
        st.error(f"❌ Error exporting CSV: {e}") 