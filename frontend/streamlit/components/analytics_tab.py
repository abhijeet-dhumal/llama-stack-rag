import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_analytics_tab():
    """Render the analytics tab with document statistics and visualizations."""
    
    st.markdown("### 📊 Document Analytics")
    st.markdown("Analytics and insights about your uploaded documents and web content.")
    st.markdown("---")
    
    # Check if documents exist - use faiss_documents instead of uploaded_documents
    if 'faiss_documents' not in st.session_state or not st.session_state.faiss_documents:
        st.info("📊 No documents uploaded yet. Go to the 'Upload Documents' or 'Web Content' tabs to add documents.")
        return
    
    documents = st.session_state.faiss_documents
    
    # Overview metrics
    st.markdown("#### 📈 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(documents))
    
    with col2:
        total_size = sum(d.get('file_size_mb', 0) for d in documents)
        st.metric("Total Size (MB)", f"{total_size:.1f}")
    
    with col3:
        total_chunks = sum(d.get('chunk_count', 0) for d in documents)
        st.metric("Total Chunks", total_chunks)
    
    with col4:
        avg_size = total_size / len(documents) if documents else 0
        st.metric("Avg Size (MB)", f"{avg_size:.2f}")
    
    st.markdown("---")
    
    # Create DataFrame for analysis
    df = create_analytics_dataframe(documents)
    
    # Document type distribution
    st.markdown("#### 📊 Document Type Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for document types
        type_counts = df['Type'].value_counts()
        if len(type_counts) > 0:
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Document Types",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No document type data available")
    
    with col2:
        # Bar chart for document types
        if len(type_counts) > 0:
            fig_bar = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title="Document Types Count",
                labels={'x': 'Document Type', 'y': 'Count'},
                color=type_counts.values,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No document type data available")
    
    # Source distribution
    st.markdown("#### 📤 Source Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for sources
        source_counts = df['Source'].value_counts()
        if len(source_counts) > 0:
            fig_source_pie = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Document Sources",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_source_pie.update_layout(height=400)
            st.plotly_chart(fig_source_pie, use_container_width=True)
        else:
            st.info("No source data available")
    
    with col2:
        # Source statistics table
        st.markdown("**Source Statistics**")
        source_stats = df.groupby('Source').agg({
            'Size (MB)': ['count', 'sum', 'mean'],
            'Chunks': 'sum'
        }).round(2)
        source_stats.columns = ['Count', 'Total Size (MB)', 'Avg Size (MB)', 'Total Chunks']
        st.dataframe(source_stats, use_container_width=True)
    
    # Size analysis
    st.markdown("#### 📏 Size Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of document sizes
        if len(df) > 0:
            fig_hist = px.histogram(
                df,
                x='Size (MB)',
                nbins=20,
                title="Document Size Distribution",
                labels={'Size (MB)': 'Size (MB)', 'count': 'Number of Documents'},
                color_discrete_sequence=['#636EFA']
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No size data available")
    
    with col2:
        # Box plot of document sizes by type
        if len(df) > 0:
            fig_box = px.box(
                df,
                x='Type',
                y='Size (MB)',
                title="Document Size by Type",
                color='Type',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No size data available")
    
    # Timeline analysis
    st.markdown("#### 📅 Timeline Analysis")
    
    # Convert created dates to datetime
    df['Created_Date'] = pd.to_datetime(df['Created'], errors='coerce')
    df_with_dates = df.dropna(subset=['Created_Date'])
    
    if len(df_with_dates) > 0:
        # Group by date and count documents
        daily_counts = df_with_dates.groupby(df_with_dates['Created_Date'].dt.date).size().reset_index(name='Count')
        daily_counts['Date'] = pd.to_datetime(daily_counts['Created_Date'])
        
        # Line chart for document uploads over time
        fig_timeline = px.line(
            daily_counts,
            x='Date',
            y='Count',
            title="Documents Uploaded Over Time",
            labels={'Date': 'Date', 'Count': 'Number of Documents'},
            markers=True
        )
        fig_timeline.update_layout(height=400)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent activity
        st.markdown("**Recent Activity**")
        recent_docs = df_with_dates.nlargest(5, 'Created_Date')[['Name', 'Type', 'Size (MB)', 'Created']]
        st.dataframe(recent_docs, use_container_width=True)
    else:
        st.info("No timeline data available")
    
    # Performance metrics
    st.markdown("#### ⚡ Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Processing success rate
        completed = len([d for d in documents if d.get('processing_status') == 'completed'])
        success_rate = (completed / len(documents)) * 100 if documents else 0
        st.metric("Processing Success Rate", f"{success_rate:.1f}%")
    
    with col2:
        # Average processing time (placeholder)
        st.metric("Avg Processing Time", "~2.5s")
    
    with col3:
        # Storage efficiency
        efficiency = (total_chunks / total_size) if total_size > 0 else 0
        st.metric("Chunks per MB", f"{efficiency:.1f}")
    
    with col4:
        # FAISS vector database status
        if 'faiss_index' in st.session_state and st.session_state.faiss_index is not None:
            faiss_size = st.session_state.faiss_index.ntotal if hasattr(st.session_state.faiss_index, 'ntotal') else 0
            st.metric("FAISS Vectors", f"{faiss_size:,}")
        else:
            st.metric("FAISS Status", "Not Available")
    
    # FAISS Vector Database Status
    st.markdown("#### 🗄️ FAISS Vector Database")
    
    if 'faiss_index' in st.session_state and st.session_state.faiss_index is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Total vectors in FAISS
            faiss_size = st.session_state.faiss_index.ntotal if hasattr(st.session_state.faiss_index, 'ntotal') else 0
            st.metric("Total Vectors", f"{faiss_size:,}")
        
        with col2:
            # Total chunks stored
            total_chunks_stored = len(st.session_state.get('faiss_chunks', []))
            st.metric("Total Chunks", f"{total_chunks_stored:,}")
        
        with col3:
            # Average vectors per document
            avg_vectors_per_doc = faiss_size / len(documents) if documents else 0
            st.metric("Avg Vectors/Doc", f"{avg_vectors_per_doc:.1f}")
        
        # FAISS index information
        with st.expander("🔍 FAISS Index Details"):
            st.markdown(f"**Index Type:** {type(st.session_state.faiss_index).__name__}")
            st.markdown(f"**Dimension:** {st.session_state.faiss_index.d if hasattr(st.session_state.faiss_index, 'd') else 'Unknown'}")
            st.markdown(f"**Total Vectors:** {faiss_size:,}")
            st.markdown(f"**Memory Usage:** ~{faiss_size * 384 * 4 / (1024*1024):.1f} MB")
            
            # Show recent additions
            if st.session_state.get('faiss_documents'):
                st.markdown("**Recent Additions:**")
                recent_docs = st.session_state.faiss_documents[-10:]  # Last 10
                for doc in recent_docs:
                    st.markdown(f"• {doc.get('filename', 'Unknown')} (chunk {doc.get('chunk_index', 0)})")
    
    else:
        st.warning("⚠️ FAISS vector database not initialized")
        st.info("💡 Upload documents to initialize the vector database")
    
    st.markdown("---")
    
    # Export analytics
    st.markdown("---")
    st.markdown("#### 📤 Export Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Analytics Report", key="analytics_export_report", type="secondary"):
            export_analytics_report(df)
    
    with col2:
        if st.button("📈 Export Charts Data", key="analytics_export_charts", type="secondary"):
            export_charts_data(df)

def create_analytics_dataframe(documents):
    """Create a pandas DataFrame for analytics."""
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
            'Content Length': len(doc.get('content', ''))
        })
    
    return pd.DataFrame(data)

def export_analytics_report(df):
    """Export analytics report as CSV."""
    try:
        # Create comprehensive report
        report_data = []
        
        # Summary statistics
        report_data.append({
            'Metric': 'Total Documents',
            'Value': len(df),
            'Category': 'Summary'
        })
        report_data.append({
            'Metric': 'Total Size (MB)',
            'Value': df['Size (MB)'].sum(),
            'Category': 'Summary'
        })
        report_data.append({
            'Metric': 'Total Chunks',
            'Value': df['Chunks'].sum(),
            'Category': 'Summary'
        })
        
        # Type statistics
        for doc_type, count in df['Type'].value_counts().items():
            report_data.append({
                'Metric': f'Documents - {doc_type}',
                'Value': count,
                'Category': 'Type Distribution'
            })
        
        # Source statistics
        for source, count in df['Source'].value_counts().items():
            report_data.append({
                'Metric': f'Documents - {source}',
                'Value': count,
                'Category': 'Source Distribution'
            })
        
        # Create DataFrame and export
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Analytics Report",
            data=csv,
            file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Analytics report ready for download!")
        
    except Exception as e:
        st.error(f"❌ Error exporting analytics report: {e}")

def export_charts_data(df):
    """Export charts data as CSV."""
    try:
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Charts Data",
            data=csv,
            file_name=f"charts_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Charts data ready for download!")
        
    except Exception as e:
        st.error(f"❌ Error exporting charts data: {e}") 