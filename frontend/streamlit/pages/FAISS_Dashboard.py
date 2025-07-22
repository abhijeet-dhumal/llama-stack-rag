import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

def main():
    """Main function for FAISS Database Dashboard page."""
    
    # Page header
    st.title("🗄️ FAISS Vector Database Dashboard")
    st.markdown("Manage and analyze your FAISS vector database for similarity search")
    
    # Back button
    if st.button("← Back to Main", type="secondary", key="faiss_dashboard_back_btn"):
        st.session_state.current_page = None
        st.rerun()
    
    # Check sync status and sync if needed
    from core.faiss_sync_manager import faiss_sync_manager
    
    # Get sync status
    sync_status = faiss_sync_manager.get_sync_status()
    
    # Display sync status
    if not sync_status.get('in_sync', False):
        st.warning("⚠️ FAISS and SQLite databases are out of sync!")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**SQLite Documents:** {sync_status.get('sqlite_documents', 0)}")
            st.write(f"**FAISS Documents:** {sync_status.get('faiss_documents', 0)}")
            if sync_status.get('faiss_exists_on_disk', False):
                st.success("💾 FAISS data found on disk")
            else:
                st.warning("💾 FAISS data not found on disk")
            if 'mismatches' in sync_status:
                for mismatch in sync_status['mismatches']:
                    st.error(f"❌ {mismatch}")
        with col2:
            if st.button("🔄 Sync Now", type="primary"):
                with st.spinner("Synchronizing databases..."):
                    user_id = st.session_state.get('user_id')
                    success = faiss_sync_manager.force_sync(user_id)
                    if success:
                        st.success("✅ Databases synchronized!")
                        # Force a rerun to update the display
                        st.rerun()
                    else:
                        st.error("❌ Failed to synchronize databases")
                        st.info("💡 Check the console for error details")
    else:
        st.success("✅ FAISS and SQLite databases are synchronized!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("SQLite Documents", sync_status.get('sqlite_documents', 0))
        with col2:
            st.metric("FAISS Documents", sync_status.get('faiss_documents', 0))
        with col3:
            st.metric("FAISS Chunks", sync_status.get('faiss_chunks', 0))
        with col4:
            if sync_status.get('faiss_exists_on_disk', False):
                st.success("💾 Persistent")
            else:
                st.warning("💾 Not Saved")
    
    # Check if FAISS index exists
    if 'faiss_index' not in st.session_state or st.session_state.faiss_index is None:
        st.warning("⚠️ No FAISS index found. Please upload some documents first to create embeddings.")
        return
    
    # Get FAISS data
    faiss_index = st.session_state.faiss_index
    faiss_documents = st.session_state.get('faiss_documents', [])
    faiss_chunks = st.session_state.get('faiss_chunks', [])
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Search & Test", 
        "📈 Analytics", 
        "⚙️ Management", 
        "🔧 Advanced"
    ])
    
    with tab1:
        render_overview_tab(faiss_index, faiss_documents, faiss_chunks)
    
    with tab2:
        render_search_tab(faiss_index, faiss_documents, faiss_chunks)
    
    with tab3:
        render_analytics_tab(faiss_documents, faiss_chunks)
    
    with tab4:
        render_management_tab(faiss_index, faiss_documents, faiss_chunks)
    
    with tab5:
        render_advanced_tab(faiss_index, faiss_documents, faiss_chunks)

def render_overview_tab(faiss_index, faiss_documents, faiss_chunks):
    """Render the overview tab with FAISS statistics."""
    
    st.markdown("### 📊 FAISS Database Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Documents",
            value=len(faiss_documents),
            help="Number of documents in FAISS index"
        )
    
    with col2:
        st.metric(
            label="Total Chunks",
            value=len(faiss_chunks),
            help="Number of text chunks in FAISS index"
        )
    
    with col3:
        st.metric(
            label="Vector Dimension",
            value=faiss_index.d if hasattr(faiss_index, 'd') else "Unknown",
            help="Dimension of vectors in FAISS index"
        )
    
    with col4:
        st.metric(
            label="Index Type",
            value=type(faiss_index).__name__,
            help="Type of FAISS index being used"
        )
    
    # Document types breakdown
    if faiss_documents:
        st.markdown("### 📄 Document Types")
        doc_types = {}
        for doc in faiss_documents:
            doc_type = doc.get('file_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        if doc_types:
            fig = px.pie(
                values=list(doc_types.values()),
                names=list(doc_types.keys()),
                title="Document Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent documents
    if faiss_documents:
        st.markdown("### 📋 Recent Documents")
        # Sort documents by upload_time, handling None values and invalid dates safely
        def safe_sort_key(doc):
            upload_time = doc.get('upload_time')
            if not upload_time:
                return ''  # Put documents without upload_time at the end
            return upload_time
        
        recent_docs = sorted(faiss_documents, key=safe_sort_key, reverse=True)[:10]
        
        for doc in recent_docs:
            with st.expander(f"📄 {doc.get('name', 'Unknown')} ({doc.get('file_type', 'Unknown')})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Display file size in appropriate units
                    file_size = doc.get('file_size_mb', 0) or 0
                    if file_size > 0:
                        if file_size < 1:
                            size_display = f"{file_size * 1024:.1f} KB"
                        else:
                            size_display = f"{file_size:.2f} MB"
                    else:
                        size_display = "Unknown"
                    
                    st.write(f"**Size:** {size_display}")
                    st.write(f"**Chunks:** {doc.get('chunk_count', 0) or 0}")
                    st.write(f"**Characters:** {doc.get('character_count', 0) or 0:,}")
                    st.write(f"**Upload Time:** {doc.get('upload_time', 'Unknown')}")
                with col2:
                    st.write(f"**User ID:** {doc.get('user_id', 'Unknown')}")
                    if doc.get('source_url'):
                        st.write(f"**Source:** {doc.get('source_url')}")
                    if doc.get('domain'):
                        st.write(f"**Domain:** {doc.get('domain')}")

def render_search_tab(faiss_index, faiss_documents, faiss_chunks):
    """Render the search and test tab."""
    
    st.markdown("### 🔍 FAISS Similarity Search")
    
    # Search query input
    query = st.text_area(
        "Enter your search query:",
        placeholder="Type your query here to test FAISS similarity search...",
        height=100
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        top_k = st.slider("Number of results (k):", min_value=1, max_value=20, value=5)
    
    with col2:
        similarity_threshold = st.slider(
            "Similarity threshold:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1
        )
    
    if st.button("🔍 Search", type="primary"):
        if query.strip():
            with st.spinner("Searching FAISS index..."):
                results = perform_faiss_search(query, faiss_index, faiss_chunks, top_k, similarity_threshold)
                display_search_results(results, query)
        else:
            st.warning("Please enter a search query.")

def render_analytics_tab(faiss_documents, faiss_chunks):
    """Render the analytics tab with visualizations."""
    
    st.markdown("### 📈 FAISS Analytics")
    
    if not faiss_documents:
        st.info("No documents available for analytics.")
        return
    
    # Debug info
    st.write(f"Debug: {len(faiss_documents)} documents, {len(faiss_chunks)} chunks")
    
    # Chunk length distribution
    if faiss_chunks:
        st.markdown("#### 📏 Chunk Length Distribution")
        try:
            chunk_lengths = [len(chunk.get('text', '')) for chunk in faiss_chunks]
            
            if chunk_lengths and any(length > 0 for length in chunk_lengths):
                fig = px.histogram(
                    x=chunk_lengths,
                    nbins=20,
                    title="Distribution of Chunk Lengths",
                    labels={'x': 'Chunk Length (characters)', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Chunk Length", f"{np.mean(chunk_lengths):.1f} chars")
                with col2:
                    st.metric("Median Chunk Length", f"{np.median(chunk_lengths):.1f} chars")
                with col3:
                    st.metric("Max Chunk Length", f"{np.max(chunk_lengths)} chars")
            else:
                st.info("No valid chunk data available for analysis.")
        except Exception as e:
            st.error(f"Error creating chunk length distribution: {e}")
    
    # Document size vs chunks correlation
    if faiss_documents:
        st.markdown("#### 📊 Document Size vs Chunks")
        
        try:
            # Create DataFrame for the scatter plot
            data = []
            for doc in faiss_documents:
                # Convert file_size_mb to bytes for consistency
                file_size_bytes = doc.get('file_size_mb', 0) * 1024 * 1024
                data.append({
                    'size': file_size_bytes,
                    'chunk_count': doc.get('chunk_count', 0),
                    'name': doc.get('name', 'Unknown')
                })
            
            if data:
                df = pd.DataFrame(data)
                st.write(f"Debug: DataFrame shape {df.shape}")
                
                # Use go.Figure instead of px.scatter to avoid DataFrame issues
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['size'],
                    y=df['chunk_count'],
                    mode='markers',
                    text=df['name'],
                    hovertemplate='<b>%{text}</b><br>Size: %{x}<br>Chunks: %{y}<extra></extra>'
                ))
                fig.update_layout(
                    title="Document Size vs Number of Chunks",
                    xaxis_title="Document Size (bytes)",
                    yaxis_title="Number of Chunks"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No document data available for correlation analysis.")
        except Exception as e:
            st.error(f"Error creating document correlation chart: {e}")
    
    # Upload timeline
    if faiss_documents:
        st.markdown("#### 📅 Upload Timeline")
        try:
            upload_times = []
            for doc in faiss_documents:
                try:
                    upload_time_str = doc.get('upload_time', '')
                    if upload_time_str:
                        # Handle different time formats
                        if 'Z' in upload_time_str:
                            upload_time = datetime.fromisoformat(upload_time_str.replace('Z', '+00:00'))
                        else:
                            upload_time = datetime.fromisoformat(upload_time_str)
                        upload_times.append(upload_time)
                except Exception as e:
                    continue
            
            if upload_times:
                fig = px.histogram(
                    x=upload_times,
                    nbins=20,
                    title="Document Upload Timeline",
                    labels={'x': 'Upload Date', 'y': 'Number of Documents'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid upload time data available.")
        except Exception as e:
            st.error(f"Error creating upload timeline: {e}")

def render_management_tab(faiss_index, faiss_documents, faiss_chunks):
    """Render the management tab."""
    
    st.markdown("### ⚙️ FAISS Management")
    
    # Index information
    st.markdown("#### 📋 Index Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Index Type:** {type(faiss_index).__name__}")
        st.info(f"**Total Vectors:** {faiss_index.ntotal if hasattr(faiss_index, 'ntotal') else 'Unknown'}")
        st.info(f"**Vector Dimension:** {faiss_index.d if hasattr(faiss_index, 'd') else 'Unknown'}")
    
    with col2:
        st.info(f"**Documents:** {len(faiss_documents)}")
        st.info(f"**Chunks:** {len(faiss_chunks)}")
        st.info(f"**Index Size:** {get_index_size_mb(faiss_index):.2f} MB")
    
    # Management actions
    st.markdown("#### 🛠️ Management Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Sync from SQLite", type="secondary"):
            with st.spinner("Synchronizing from SQLite..."):
                user_id = st.session_state.get('user_id')
                success = faiss_sync_manager.force_sync(user_id)
                if success:
                    st.success("✅ FAISS index synchronized from SQLite!")
                    st.rerun()
                else:
                    st.error("❌ Failed to synchronize from SQLite")
    
    with col2:
        if st.button("🗑️ Clear FAISS Index", type="secondary"):
            if st.button("⚠️ Confirm Clear", type="primary"):
                clear_faiss_index()
                st.success("✅ FAISS index cleared successfully!")
                st.rerun()
    
    with col3:
        if st.button("💾 Export Index", type="secondary"):
            export_faiss_index()
    
    # Dangerous operations section
    st.markdown("#### ⚠️ Dangerous Operations")
    st.warning("⚠️ These operations will permanently delete data and cannot be undone!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Delete All Data", type="secondary", help="Delete all documents from both SQLite and FAISS"):
            st.error("🚨 DANGER: This will delete ALL data from both databases!")
            st.write("**This will delete:**")
            st.write("- All documents from SQLite database")
            st.write("- All FAISS embeddings and metadata")
            st.write("- All persistent FAISS files")
            st.write("- All user data")
            
            # Confirmation buttons
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("❌ Cancel", type="primary"):
                    st.rerun()
            with col_confirm2:
                if st.button("💀 DELETE EVERYTHING", type="secondary"):
                    with st.spinner("Deleting all data..."):
                        user_id = st.session_state.get('user_id')
                        success = faiss_sync_manager.delete_all_data(user_id)
                        if success:
                            st.success("✅ All data deleted successfully!")
                            st.info("🔄 Please refresh the page to see the changes.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete all data")
    
    with col2:
        if st.button("🔄 Reset FAISS Only", type="secondary", help="Reset only FAISS data, keep SQLite documents"):
            st.warning("⚠️ This will reset FAISS data but keep SQLite documents!")
            st.write("**This will delete:**")
            st.write("- All FAISS embeddings and metadata")
            st.write("- All persistent FAISS files")
            st.write("- **SQLite documents will be preserved**")
            
            # Confirmation buttons
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("❌ Cancel", type="primary"):
                    st.rerun()
            with col_confirm2:
                if st.button("🔄 Reset FAISS", type="secondary"):
                    with st.spinner("Resetting FAISS data..."):
                        # Clear FAISS session state
                        if 'faiss_index' in st.session_state:
                            st.session_state.faiss_index.reset()
                        st.session_state.faiss_documents = []
                        st.session_state.faiss_chunks = []
                        st.session_state.faiss_document_mapping = {}
                        
                        # Delete persistent FAISS files
                        user_id = st.session_state.get('user_id')
                        paths = faiss_sync_manager.get_faiss_file_paths(user_id)
                        for path in paths.values():
                            if os.path.exists(path):
                                os.remove(path)
                        
                        st.success("✅ FAISS data reset successfully!")
                        st.info("🔄 Please refresh the page to see the changes.")
                        st.rerun()

def render_advanced_tab(faiss_index, faiss_documents, faiss_chunks):
    """Render the advanced tab with technical details."""
    
    st.markdown("### 🔧 Advanced FAISS Operations")
    
    # Index statistics
    st.markdown("#### 📊 Detailed Index Statistics")
    
    if hasattr(faiss_index, 'ntotal'):
        st.code(f"""
FAISS Index Details:
- Total vectors: {faiss_index.ntotal}
- Vector dimension: {faiss_index.d}
- Index type: {type(faiss_index).__name__}
- Is trained: {faiss_index.is_trained if hasattr(faiss_index, 'is_trained') else 'Unknown'}
- Metric type: {faiss_index.metric_type if hasattr(faiss_index, 'metric_type') else 'Unknown'}
        """)
    
    # Performance testing
    st.markdown("#### ⚡ Performance Testing")
    
    if st.button("🏃‍♂️ Run Performance Test", type="primary"):
        with st.spinner("Running performance test..."):
            performance_results = run_performance_test(faiss_index)
            display_performance_results(performance_results)
    
    # Index optimization
    st.markdown("#### 🔧 Index Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Analyze Index", type="secondary"):
            analyze_faiss_index(faiss_index)
    
    with col2:
        if st.button("⚡ Optimize Index", type="secondary"):
            optimize_faiss_index(faiss_index)

# Helper functions

def perform_faiss_search(query, faiss_index, faiss_chunks, top_k, similarity_threshold):
    """Perform FAISS similarity search."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate query embedding
        query_embedding = model.encode([query])[0]
        
        # Search FAISS index
        distances, indices = faiss_index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            top_k
        )
        
        # Filter by similarity threshold
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(faiss_chunks) and distance <= similarity_threshold:
                chunk = faiss_chunks[idx]
                results.append({
                    'chunk': chunk,
                    'distance': float(distance),
                    'similarity': 1.0 - float(distance),
                    'rank': i + 1
                })
        
        return results
    
    except Exception as e:
        st.error(f"Error performing FAISS search: {e}")
        return []

def display_search_results(results, query):
    """Display FAISS search results."""
    if not results:
        st.warning("No similar chunks found for your query.")
        return
    
    st.markdown(f"### 🔍 Search Results for: *{query}*")
    st.markdown(f"Found **{len(results)}** similar chunks:")
    
    for result in results:
        chunk = result['chunk']
        with st.expander(f"📄 Rank {result['rank']}: {chunk.get('document_name', 'Unknown')} (Similarity: {result['similarity']:.3f})"):
            st.write(f"**Similarity Score:** {result['similarity']:.3f}")
            st.write(f"**Distance:** {result['distance']:.3f}")
            st.write(f"**Document:** {chunk.get('document_name', 'Unknown')}")
            st.write(f"**Chunk Index:** {chunk.get('chunk_index', 'Unknown')}")
            st.markdown("**Content:**")
            st.text(chunk.get('text', 'No content available'))

def get_index_size_mb(faiss_index):
    """Get FAISS index size in MB."""
    try:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
            import faiss
            faiss.write_index(faiss_index, tmp.name)
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
            os.unlink(tmp.name)
            return size_mb
    except:
        return 0.0

def rebuild_faiss_index():
    """Rebuild the FAISS index from documents."""
    # This would need to be implemented based on your document processing logic
    st.info("Rebuild functionality would need to be implemented based on your document processing pipeline.")

def clear_faiss_index():
    """Clear the FAISS index."""
    if 'faiss_index' in st.session_state:
        st.session_state.faiss_index = None
    if 'faiss_documents' in st.session_state:
        st.session_state.faiss_documents = []
    if 'faiss_chunks' in st.session_state:
        st.session_state.faiss_chunks = []

def export_faiss_index():
    """Export FAISS index to file."""
    try:
        import faiss
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
            faiss.write_index(st.session_state.faiss_index, tmp.name)
            
            with open(tmp.name, 'rb') as f:
                st.download_button(
                    label="📥 Download FAISS Index",
                    data=f.read(),
                    file_name=f"faiss_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.faiss",
                    mime="application/octet-stream"
                )
            
            os.unlink(tmp.name)
    except Exception as e:
        st.error(f"Error exporting FAISS index: {e}")

def remove_document_from_faiss(document_name):
    """Remove a document from FAISS index."""
    # This would need to be implemented based on your FAISS structure
    st.info(f"Remove functionality for {document_name} would need to be implemented.")

def show_document_chunks(document_name, faiss_chunks):
    """Show chunks for a specific document."""
    doc_chunks = [chunk for chunk in faiss_chunks if chunk.get('document_name') == document_name]
    
    if doc_chunks:
        st.markdown(f"### 📄 Chunks for: {document_name}")
        for i, chunk in enumerate(doc_chunks):
            with st.expander(f"Chunk {i+1}"):
                st.text(chunk.get('text', 'No content'))
    else:
        st.warning(f"No chunks found for document: {document_name}")

def run_performance_test(faiss_index):
    """Run performance test on FAISS index."""
    try:
        import time
        import numpy as np
        
        # Generate random query vectors
        query_vectors = np.random.rand(100, faiss_index.d).astype('float32')
        
        start_time = time.time()
        distances, indices = faiss_index.search(query_vectors, 5)
        end_time = time.time()
        
        return {
            'queries_per_second': 100 / (end_time - start_time),
            'average_query_time': (end_time - start_time) / 100 * 1000,  # in ms
            'total_time': end_time - start_time
        }
    except Exception as e:
        st.error(f"Error running performance test: {e}")
        return {}

def display_performance_results(results):
    """Display performance test results."""
    if results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Queries/sec", f"{results['queries_per_second']:.1f}")
        
        with col2:
            st.metric("Avg Query Time", f"{results['average_query_time']:.2f} ms")
        
        with col3:
            st.metric("Total Test Time", f"{results['total_time']:.3f} s")

def analyze_faiss_index(faiss_index):
    """Analyze FAISS index structure."""
    st.info("Index analysis would provide detailed information about the FAISS index structure and optimization opportunities.")

def optimize_faiss_index(faiss_index):
    """Optimize FAISS index for better performance."""
    st.info("Index optimization would improve search performance and reduce memory usage.")

if __name__ == "__main__":
    main() 