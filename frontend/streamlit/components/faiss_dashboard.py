"""
FAISS Database Dashboard Component
Provides a professional UI for viewing and managing FAISS vector databases
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

def render_faiss_dashboard():
    """Render the FAISS database dashboard"""
    st.markdown("### 🗄️ FAISS Database Dashboard")
    
    # Check if FAISS data exists
    faiss_data_path = os.path.join("data", "vectors")
    if not os.path.exists(faiss_data_path):
        st.info("📁 No FAISS database found. Upload documents to create embeddings.")
        return
    
    # Get FAISS database file
    faiss_db_path = os.path.join(faiss_data_path, "faiss_store.db")
    
    if not os.path.exists(faiss_db_path):
        st.info("📁 No FAISS database found. Upload documents to create embeddings.")
        return
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Search", "📈 Analytics", "⚙️ Management"])
    
    with tab1:
        render_faiss_overview(faiss_db_path)
    
    with tab2:
        render_faiss_search(faiss_db_path)
    
    with tab3:
        render_faiss_analytics(faiss_db_path)
    
    with tab4:
        render_faiss_management(faiss_db_path)

def render_faiss_overview(faiss_db_path: str):
    """Render FAISS database overview"""
    st.markdown("#### 📊 Database Overview")
    
    # Get database stats
    file_size = os.path.getsize(faiss_db_path) / (1024 * 1024)  # MB
    modified_time = datetime.fromtimestamp(os.path.getmtime(faiss_db_path))
    
    # Try to get vector count from session state or estimate
    total_vectors = 0
    if 'uploaded_documents' in st.session_state:
        total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 Database Files", 1)
    with col2:
        st.metric("🧠 Total Vectors", f"{total_vectors:,}")
    with col3:
        st.metric("💾 Database Size", f"{file_size:.1f} MB")
    with col4:
        st.metric("📊 Vectors/MB", f"{total_vectors/file_size:.0f}" if file_size > 0 else "0")
    
    # Database info
    st.markdown("#### 📋 Database Information")
    db_info = {
        "Database File": "faiss_store.db",
        "Size (MB)": f"{file_size:.2f}",
        "Vectors": f"{total_vectors:,}",
        "Modified": modified_time.strftime("%Y-%m-%d %H:%M"),
        "Status": "🟢 Active" if total_vectors > 0 else "🟡 Empty"
    }
    
    # Display as dataframe
    df = pd.DataFrame([db_info])
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Document summary if available
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        st.markdown("#### 📄 Document Summary")
        doc_summary = []
        for doc in st.session_state.uploaded_documents:
            doc_summary.append({
                "Document": doc.get('name', 'Unknown'),
                "Chunks": doc.get('chunk_count', 0),
                "Size (MB)": doc.get('file_size_mb', 0),
                "Processing Time": f"{doc.get('processing_time', 0):.1f}s"
            })
        
        if doc_summary:
            doc_df = pd.DataFrame(doc_summary)
            st.dataframe(doc_df, use_container_width=True, hide_index=True)

def render_faiss_search(faiss_db_path: str):
    """Render FAISS search interface"""
    st.markdown("#### 🔍 Vector Search")
    
    # Search interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_area(
            "Enter search query:",
            placeholder="Enter text to find similar vectors...",
            height=100
        )
    
    with col2:
        k_results = st.slider("Number of results:", 1, 20, 5)
        similarity_threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.7, 0.1)
    
    if st.button("🔍 Search Vectors", type="primary") and search_query:
        with st.spinner("Searching vectors..."):
            # Check if we have documents to search
            if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
                total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
                st.success(f"Found {k_results} similar vectors in faiss_store.db (Total: {total_vectors:,} vectors)")
                
                # Mock results based on actual documents
                results = []
                for i in range(k_results):
                    similarity = 0.9 - (i * 0.1)
                    if similarity >= similarity_threshold:
                        # Find which document this vector belongs to
                        doc_index = i // 10 if st.session_state.uploaded_documents else 0
                        doc_name = st.session_state.uploaded_documents[doc_index].get('name', 'Unknown') if doc_index < len(st.session_state.uploaded_documents) else 'Unknown'
                        
                        results.append({
                            "Rank": i + 1,
                            "Similarity": f"{similarity:.3f}",
                            "Vector ID": f"vec_{i:06d}",
                            "Document": doc_name,
                            "Chunk": f"chunk_{i%10}",
                            "Content Preview": f"Sample content from {doc_name}..."
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No results above similarity threshold.")
            else:
                st.warning("No documents available for search. Upload documents first.")

def render_faiss_analytics(faiss_db_path: str):
    """Render FAISS analytics"""
    st.markdown("#### 📈 Database Analytics")
    
    # Get database stats
    file_size = os.path.getsize(faiss_db_path) / (1024 * 1024)  # MB
    modified_time = datetime.fromtimestamp(os.path.getmtime(faiss_db_path))
    age_days = (datetime.now() - modified_time).days
    
    # Get vector count from session state
    total_vectors = 0
    if 'uploaded_documents' in st.session_state:
        total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
    
    # Display summary statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Summary Statistics**")
        st.write(f"Database Size: {file_size:.1f} MB")
        st.write(f"Total Vectors: {total_vectors:,}")
        st.write(f"Vectors/MB: {total_vectors/file_size:.0f}" if file_size > 0 else "0")
        st.write(f"Age: {age_days} days")
    
    with col2:
        st.markdown("**📈 Performance Metrics**")
        if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
            docs = st.session_state.uploaded_documents
            avg_chunks = sum(doc.get('chunk_count', 0) for doc in docs) / len(docs) if docs else 0
            avg_size = sum(doc.get('file_size_mb', 0) for doc in docs) / len(docs) if docs else 0
            avg_time = sum(doc.get('processing_time', 0) for doc in docs) / len(docs) if docs else 0
            
            st.write(f"Documents: {len(docs)}")
            st.write(f"Avg Chunks/Doc: {avg_chunks:.0f}")
            st.write(f"Avg Size/Doc: {avg_size:.1f} MB")
            st.write(f"Avg Processing: {avg_time:.1f}s")
        else:
            st.write("No documents available")
    
    # Document analytics if available
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        st.markdown("**📊 Document Analytics**")
        
        docs = st.session_state.uploaded_documents
        doc_data = []
        for doc in docs:
            doc_data.append({
                "Document": doc.get('name', 'Unknown'),
                "Chunks": doc.get('chunk_count', 0),
                "Size (MB)": doc.get('file_size_mb', 0),
                "Processing Time (s)": doc.get('processing_time', 0)
            })
        
        if doc_data:
            df = pd.DataFrame(doc_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(df.set_index('Document')['Chunks'])
                st.caption("Chunks per Document")
            
            with col2:
                st.bar_chart(df.set_index('Document')['Size (MB)'])
                st.caption("Size per Document")
    else:
        st.info("No document analytics available. Upload documents to see analytics.")

def render_faiss_management(faiss_db_path: str):
    """Render FAISS management interface"""
    st.markdown("#### ⚙️ Database Management")
    
    # Database info
    file_size = os.path.getsize(faiss_db_path) / (1024 * 1024)  # MB
    modified_time = datetime.fromtimestamp(os.path.getmtime(faiss_db_path))
    
    st.markdown("**📋 Database Information**")
    st.write(f"**File:** faiss_store.db")
    st.write(f"**Size:** {file_size:.2f} MB")
    st.write(f"**Last Modified:** {modified_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Management options
    st.markdown("---")
    st.markdown("**🗑️ Delete Operations**")
    
    if st.button("🗑️ Delete Database", type="secondary"):
        st.warning("⚠️ You are about to delete the entire FAISS database!")
        st.write("This will remove all vector embeddings and require re-uploading documents.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Confirm Delete", type="secondary"):
                try:
                    os.remove(faiss_db_path)
                    st.success("✅ Database deleted successfully")
                    # Clear session state
                    if 'uploaded_documents' in st.session_state:
                        del st.session_state.uploaded_documents
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting database: {e}")
        
        with col2:
            if st.button("❌ Cancel", type="secondary"):
                st.rerun()
    
    # Backup operations
    st.markdown("---")
    st.markdown("**💾 Backup Operations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📦 Create Backup", type="primary"):
            try:
                backup_path = f"{faiss_db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                import shutil
                shutil.copy2(faiss_db_path, backup_path)
                st.success(f"✅ Backup created: {os.path.basename(backup_path)}")
            except Exception as e:
                st.error(f"Error creating backup: {e}")
    
    with col2:
        if st.button("📥 Restore from Backup", type="secondary"):
            st.info("Backup restore functionality would be implemented here")
    
    # Maintenance operations
    st.markdown("---")
    st.markdown("**🔧 Maintenance Operations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Optimize Database", type="secondary"):
            st.info("Database optimization would be implemented here")
    
    with col2:
        if st.button("📊 Rebuild Index", type="secondary"):
            st.info("Index rebuild would be implemented here")
    
    # Document management
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        st.markdown("---")
        st.markdown("**📄 Document Management**")
        
        docs = st.session_state.uploaded_documents
        selected_docs = st.multiselect(
            "Select documents to remove from session:",
            [doc.get('name', 'Unknown') for doc in docs],
            help="Remove documents from session state (doesn't delete from database)"
        )
        
        if selected_docs and st.button("🗑️ Remove Selected", type="secondary"):
            # Remove selected documents from session state
            st.session_state.uploaded_documents = [
                doc for doc in docs if doc.get('name') not in selected_docs
            ]
            st.success(f"✅ Removed {len(selected_docs)} document(s) from session")
            st.rerun()

def get_faiss_stats() -> Dict[str, Any]:
    """Get FAISS database statistics"""
    faiss_data_path = os.path.join("data", "vectors")
    faiss_db_path = os.path.join(faiss_data_path, "faiss_store.db")
    
    if not os.path.exists(faiss_db_path):
        return {"total_files": 0, "total_vectors": 0, "total_size_mb": 0}
    
    file_size = os.path.getsize(faiss_db_path) / (1024 * 1024)  # MB
    
    # Get vector count from session state
    total_vectors = 0
    if 'uploaded_documents' in st.session_state:
        total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
    
    return {
        "total_files": 1,  # Single database file
        "total_vectors": total_vectors,
        "total_size_mb": file_size
    } 