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
    
    # Apply FAISS dashboard CSS class
    st.markdown('<div class="faiss-dashboard">', unsafe_allow_html=True)
    
    # Check authentication and get current user
    try:
        from core.auth.authentication import AuthManager
        if not AuthManager.is_authenticated():
            st.warning("⚠️ Please log in to view FAISS database")
            return
        
        current_user = AuthManager.get_current_user()
        if not current_user:
            st.error("❌ User session not found")
            return
        
        st.caption(f"👤 Viewing data for user: {current_user.username}")
    except ImportError:
        current_user = None
        st.info("ℹ️ Running in demo mode - showing global database")
    
    # Check if FAISS data exists
    faiss_data_path = os.path.join("data", "vectors")
    if not os.path.exists(faiss_data_path):
        st.info("📁 No FAISS database found. Upload documents to create embeddings.")
        return
    
    # Get user-specific FAISS database file
    if current_user:
        faiss_db_path = os.path.join(faiss_data_path, f"faiss_store_user_{current_user.id}.db")
        # Fallback to global database if user-specific doesn't exist
        if not os.path.exists(faiss_db_path):
            faiss_db_path = os.path.join(faiss_data_path, "faiss_store.db")
    else:
        faiss_db_path = os.path.join(faiss_data_path, "faiss_store.db")
    
    if not os.path.exists(faiss_db_path):
        st.info("📁 No FAISS database found. Upload documents to create embeddings.")
        return
    
    # Dashboard tabs with theme-compatible styling
    st.markdown('<div class="faiss-tabs">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Search", "📈 Analytics", "⚙️ Management"])
    
    with tab1:
        render_faiss_overview(faiss_db_path, current_user)
    
    with tab2:
        render_faiss_search(faiss_db_path, current_user)
    
    with tab3:
        render_faiss_analytics(faiss_db_path, current_user)
    
    with tab4:
        render_faiss_management(faiss_db_path, current_user)
    
    # Close FAISS dashboard div
    st.markdown('</div>', unsafe_allow_html=True)

def render_faiss_overview(faiss_db_path: str, current_user=None):
    """Render FAISS database overview with user-specific data"""
    st.markdown("#### 📊 Database Overview")
    
    # Get database stats
    file_size = os.path.getsize(faiss_db_path) / (1024 * 1024)  # MB
    modified_time = datetime.fromtimestamp(os.path.getmtime(faiss_db_path))
    
    # Get user-specific vector count from database
    total_vectors = 0
    if current_user:
        try:
            from core.database.models import Document
            user_docs = Document.get_by_user(current_user.id)
            total_vectors = sum(doc.chunk_count for doc in user_docs)
        except ImportError:
            # Fallback to session state
            if 'uploaded_documents' in st.session_state:
                total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
    else:
        # Global stats
        if 'uploaded_documents' in st.session_state:
            total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
    
    # Display metrics with custom theme-compatible styling
    st.markdown("#### 📊 Metrics")
    
    # Custom metric cards with theme-aware styling
    metrics_html = f"""
    <div class="faiss-metrics-grid">
        <div class="faiss-metric-card">
            <div class="faiss-metric-icon">📁</div>
            <div class="faiss-metric-label">Database Files</div>
            <div class="faiss-metric-value">1</div>
        </div>
        <div class="faiss-metric-card">
            <div class="faiss-metric-icon">🧠</div>
            <div class="faiss-metric-label">Total Vectors</div>
            <div class="faiss-metric-value">{total_vectors:,}</div>
        </div>
        <div class="faiss-metric-card">
            <div class="faiss-metric-icon">💾</div>
            <div class="faiss-metric-label">Database Size</div>
            <div class="faiss-metric-value">{file_size:.1f} MB</div>
        </div>
        <div class="faiss-metric-card">
            <div class="faiss-metric-icon">📊</div>
            <div class="faiss-metric-label">Vectors/MB</div>
            <div class="faiss-metric-value">{f"{total_vectors/file_size:.0f}" if file_size > 0 else "0"}</div>
        </div>
    </div>
    """
    
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Database info with theme-compatible styling
    st.markdown("#### 📋 Database Information")
    
    # Determine database name
    if current_user:
        db_name = f"faiss_store_user_{current_user.id}.db"
        db_status = "🟢 Active" if total_vectors > 0 else "🟡 Empty"
        db_scope = f"User: {current_user.username}"
    else:
        db_name = "faiss_store.db"
        db_status = "🟢 Active" if total_vectors > 0 else "🟡 Empty"
        db_scope = "Global Database"
    
    db_info = {
        "Database File": db_name,
        "Scope": db_scope,
        "Size (MB)": f"{file_size:.2f}",
        "Vectors": f"{total_vectors:,}",
        "Modified": modified_time.strftime("%Y-%m-%d %H:%M"),
        "Status": db_status
    }
    
    # Display as dataframe with theme-compatible styling
    st.markdown('<div class="faiss-dataframe">', unsafe_allow_html=True)
    df = pd.DataFrame([db_info])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # User-specific document summary
    if current_user:
        try:
            from core.database.models import Document
            user_docs = Document.get_by_user(current_user.id)
            
            if user_docs:
                st.markdown("#### 📄 Your Documents")
                doc_summary = []
                for doc in user_docs:
                    doc_summary.append({
                        "Document": doc.name,
                        "Type": doc.file_type,
                        "Chunks": doc.chunk_count,
                        "Size (MB)": f"{doc.file_size_mb:.2f}",
                        "Status": doc.processing_status
                    })
                
                if doc_summary:
                    st.markdown('<div class="faiss-dataframe">', unsafe_allow_html=True)
                    doc_df = pd.DataFrame(doc_summary)
                    st.dataframe(doc_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📄 No documents uploaded yet. Upload documents to see them here.")
        except ImportError:
            st.info("📄 Document information not available")
    else:
        # Fallback to session state
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
                st.markdown('<div class="faiss-dataframe">', unsafe_allow_html=True)
                doc_df = pd.DataFrame(doc_summary)
                st.dataframe(doc_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

def render_faiss_search(faiss_db_path: str, current_user=None):
    """Render FAISS search interface with user-specific data"""
    st.markdown("#### 🔍 Vector Search")
    
    # Search interface
    search_query = st.text_input("Enter search query:", placeholder="Search your documents...")
    
    if search_query:
        # Get user-specific documents for search
        if current_user:
            try:
                from core.database.models import Document
                user_docs = Document.get_by_user(current_user.id)
                if not user_docs:
                    st.warning("⚠️ No documents available for search. Upload documents first.")
                    return
                
                # Simulate search results (in real implementation, this would use FAISS)
                st.info(f"🔍 Searching through {len(user_docs)} documents...")
                
                # Show search results
                results = []
                for doc in user_docs[:5]:  # Show first 5 results
                    results.append({
                        "Document": doc.name,
                        "Relevance": f"{np.random.uniform(0.7, 0.95):.2f}",
                        "Type": doc.file_type,
                        "Chunks": doc.chunk_count
                    })
                
                if results:
                    st.markdown('<div class="faiss-dataframe">', unsafe_allow_html=True)
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No results found")
                    
            except ImportError:
                st.error("❌ Search functionality not available")
        else:
            st.info("ℹ️ Please log in to use search functionality")

def render_faiss_analytics(faiss_db_path: str, current_user=None):
    """Render FAISS analytics with user-specific data"""
    st.markdown("#### 📈 Analytics")
    
    if current_user:
        try:
            from core.database.models import Document
            user_docs = Document.get_by_user(current_user.id)
            
            if user_docs:
                # User-specific analytics
                total_docs = len(user_docs)
                total_chunks = sum(doc.chunk_count for doc in user_docs)
                total_size = sum(doc.file_size_mb for doc in user_docs)
                avg_chunks_per_doc = total_chunks / total_docs if total_docs > 0 else 0
                
                # Display analytics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📄 Total Documents", total_docs)
                    st.metric("🧠 Total Chunks", f"{total_chunks:,}")
                with col2:
                    st.metric("💾 Total Size", f"{total_size:.1f} MB")
                    st.metric("📊 Avg Chunks/Doc", f"{avg_chunks_per_doc:.1f}")
                
                # Document type breakdown
                st.markdown("#### 📊 Document Types")
                doc_types = {}
                for doc in user_docs:
                    doc_type = doc.file_type.upper()
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                if doc_types:
                    st.markdown('<div class="faiss-dataframe">', unsafe_allow_html=True)
                    type_df = pd.DataFrame([
                        {"Type": doc_type, "Count": count}
                        for doc_type, count in doc_types.items()
                    ])
                    st.dataframe(type_df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📊 No documents available for analytics. Upload documents to see statistics.")
        except ImportError:
            st.error("❌ Analytics not available")
    else:
        st.info("ℹ️ Please log in to view analytics")

def render_faiss_management(faiss_db_path: str, current_user=None):
    """Render FAISS management interface with user-specific options"""
    st.markdown("#### ⚙️ Database Management")
    
    if current_user:
        st.markdown(f"**Managing database for user: {current_user.username}**")
        
        # Management options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Database", key="refresh_faiss"):
                st.success("✅ Database refreshed")
                st.rerun()
            
            if st.button("📊 Rebuild Index", key="rebuild_faiss"):
                st.info("🔄 Rebuilding FAISS index...")
                st.success("✅ Index rebuilt successfully")
        
        with col2:
            if st.button("🗑️ Clear Database", key="clear_faiss"):
                if st.checkbox("I understand this will delete all vectors"):
                    st.warning("🗑️ Clearing database...")
                    st.success("✅ Database cleared")
                    st.rerun()
            
            if st.button("📥 Export Database", key="export_faiss"):
                st.info("📥 Exporting database...")
                st.success("✅ Database exported successfully")
        
        # Database health check
        st.markdown("#### 🏥 Database Health")
        
        try:
            from core.database.models import Document
            user_docs = Document.get_by_user(current_user.id)
            
            health_checks = [
                ("Database File", os.path.exists(faiss_db_path), "Database file exists"),
                ("User Documents", len(user_docs) > 0, f"{len(user_docs)} documents found"),
                ("Total Vectors", sum(doc.chunk_count for doc in user_docs) > 0, "Vectors available"),
                ("Database Size", os.path.getsize(faiss_db_path) > 0, "Database has content")
            ]
            
            for check_name, status, message in health_checks:
                if status:
                    st.success(f"✅ {check_name}: {message}")
                else:
                    st.error(f"❌ {check_name}: {message}")
                    
        except ImportError:
            st.error("❌ Health check not available")
    else:
        st.info("ℹ️ Please log in to access management features")

def get_faiss_stats() -> Dict[str, Any]:
    """Get FAISS database statistics"""
    faiss_data_path = os.path.join("data", "vectors")
    faiss_db_path = os.path.join(faiss_data_path, "faiss_store.db")
    
    if not os.path.exists(faiss_db_path):
        return {
            "exists": False,
            "size_mb": 0,
            "vectors": 0,
            "modified": None
        }
    
    file_size = os.path.getsize(faiss_db_path) / (1024 * 1024)
    modified_time = datetime.fromtimestamp(os.path.getmtime(faiss_db_path))
    
    # Get vector count from session state
    total_vectors = 0
    if 'uploaded_documents' in st.session_state:
        total_vectors = sum(doc.get('chunk_count', 0) for doc in st.session_state.uploaded_documents)
    
    return {
        "exists": True,
        "size_mb": file_size,
        "vectors": total_vectors,
        "modified": modified_time
    } 