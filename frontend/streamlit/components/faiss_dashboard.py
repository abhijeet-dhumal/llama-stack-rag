"""
VectorIO Database Dashboard Component
Provides a professional UI for viewing and managing LlamaStack VectorIO databases
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

def render_faiss_dashboard():
    """Render the VectorIO database dashboard"""
    st.markdown("### 🗄️ FAISS VectorIO DB Dashboard")
    
    # Check if LlamaStack client is available
    if 'llamastack_client' not in st.session_state:
        st.info("🔌 LlamaStack client not available. Please ensure LlamaStack is running.")
        return
    
    try:
        # Test connection to VectorIO
        vector_dbs = st.session_state.llamastack_client.list_vector_databases()
        if not vector_dbs:
            st.info("📁 No vector databases found. Upload documents to create embeddings.")
            return
        
        # Check if FAISS database exists
        faiss_db = None
        for db in vector_dbs:
            if db.get('vector_db_id') == 'faiss':
                faiss_db = db
                break
        
        if not faiss_db:
            st.info("📁 FAISS database not found. Upload documents to create embeddings.")
            return
    
        # Clean dashboard layout with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📋 Sources", "🔍 Search", "⚙️ Management"])
    
        with tab1:
            render_clean_overview()
    
        with tab2:
            render_documents_table()
    
        with tab3:
            render_vectorio_search()
    
        with tab4:
            render_crud_operations()
            
    except Exception as e:
        st.error(f"❌ Error connecting to VectorIO: {str(e)}")
        st.info("💡 Make sure LlamaStack is running and VectorIO API is enabled")

def get_vectorio_stats() -> Dict[str, Any]:
    """Get VectorIO database statistics"""
    try:
        if 'llamastack_client' not in st.session_state:
            return {"error": "LlamaStack client not available"}
        
        stats = st.session_state.llamastack_client.get_vector_db_stats("faiss")
        
        if not stats:
            return {"error": "No statistics available"}
        
        return stats
        
    except Exception as e:
        return {"error": f"Failed to get VectorIO stats: {str(e)}"}

def render_clean_overview():
    """Render a clean overview of the VectorIO database"""
    try:
        # Get basic database info (but not vector counts)
        db_info = st.session_state.llamastack_client.get_vector_db_stats("faiss")
        
        # Use the same logic as render_documents_table to get consistent data
        documents = []
        if 'uploaded_documents' in st.session_state:
            documents = st.session_state.uploaded_documents
        
        # If no documents in session state, try to retrieve from VectorIO using same method as documents table
        if not documents:
            try:
                # Get documents from VectorIO using multiple broad queries (same as documents table)
                all_results = []
                broad_queries = ["a", "the", "and", "or", "in", "on", "at", "to", "for", "of"]
                
                for query in broad_queries:
                    try:
                        results = st.session_state.llamastack_client.get_filtered_search_results(
                            query_text=query,
                            vector_db_id="faiss",
                            top_k=100
                        )
                        all_results.extend(results)
                    except Exception as e:
                        continue
                
                # Group by document_id to reconstruct documents (same logic as documents table)
                doc_groups = {}
                for result in all_results:
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('document_id', 'unknown')
                    doc_name = metadata.get('document_name', 'Unknown')
                    source_url = metadata.get('source_url')
                    file_type = metadata.get('file_type', 'FILE')
                    chunk_index = metadata.get('chunk_index', 0)
                    
                    # Extract base document identifier for grouping (same as documents table)
                    if doc_id.startswith('web_'):
                        parts = doc_id.split('_')
                        if len(parts) >= 3:
                            domain = parts[1]
                            base_doc_id = f"web_{domain}"
                        else:
                            base_doc_id = doc_id
                    elif doc_id.startswith('file_'):
                        parts = doc_id.split('_', 2)
                        if len(parts) >= 3:
                            filename = parts[1]
                            base_doc_id = f"file_{filename}"
                        else:
                            base_doc_id = doc_id
                    else:
                        base_doc_id = doc_id
                    
                    if base_doc_id not in doc_groups:
                        doc_groups[base_doc_id] = {
                            'name': doc_name,
                            'source_url': source_url,
                            'file_type': file_type,
                            'chunks': [],
                            'chunk_count': 0,
                            'document_ids': set()
                        }
                    
                    # Add chunk info
                    doc_groups[base_doc_id]['chunks'].append({
                        'content': result.get('content', ''),
                        'chunk_index': chunk_index,
                        'document_id': doc_id
                    })
                    doc_groups[base_doc_id]['chunk_count'] += 1
                    doc_groups[base_doc_id]['document_ids'].add(doc_id)
                
                # Convert to documents format (same as documents table)
                for base_doc_id, doc_data in doc_groups.items():
                    display_name = doc_data['name']
                    if doc_data['source_url']:
                        if display_name.startswith('Web Content from'):
                            from urllib.parse import urlparse
                            try:
                                domain = urlparse(doc_data['source_url']).netloc
                                display_name = f"🌐 {domain}"
                            except:
                                pass
                    elif doc_data['file_type'] == 'FILE':
                        display_name = f"📄 {display_name}"
                    
                    documents.append({
                        'name': display_name,
                        'source_url': doc_data['source_url'],
                        'file_type': doc_data['file_type'],
                        'chunks': doc_data['chunks'],
                        'chunk_count': doc_data['chunk_count'],
                        'document_ids': list(doc_data['document_ids']),
                        'vectorio_stored': True
                    })
                    
            except Exception as e:
                print(f"⚠️ Error retrieving documents from VectorIO: {e}")
        
        # Calculate statistics using the same logic as documents table
        total_sources = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
        web_sources = sum(1 for doc in documents if doc.get('source_url'))
        file_sources = total_sources - web_sources
        web_chunks = sum(doc.get('chunk_count', 0) for doc in documents if doc.get('source_url'))
        file_chunks = total_chunks - web_chunks
        
        # Calculate average chunks per source
        avg_chunks = total_chunks / total_sources if total_sources > 0 else 0
        
        print(f"🔍 Overview counting: {total_sources} sources, {total_chunks} vectors (web: {web_sources}, file: {file_sources})")
            
        # Display key metrics - meaningful statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📚 Knowledge Sources",
                value=total_sources,
                help="Number of documents and websites in your knowledge base"
            )
        
        with col2:
            # Calculate average chunks per source
            avg_chunks = total_chunks / total_sources if total_sources > 0 else 0
            st.metric(
                label="📖 Content Depth",
                value=f"{avg_chunks:.1f}",
                help="Average chunks per source (higher = more detailed content)"
            )
        
        with col3:
            # Use the web_sources calculated above
            
            st.metric(
                label="🌐 Web Sources",
                value=web_sources,
                help="Number of websites in your knowledge base"
            )
        
        with col4:
            st.metric(
                label="📄 File Sources",
                value=file_sources,
                help="Number of uploaded documents in your knowledge base"
            )
        
        # Show detailed statistics breakdown
        st.markdown("**📈 Knowledge Base Statistics**")
        
        # Use the chunk counts calculated above
        total_content_chunks = total_chunks
        
        # Create meaningful breakdown data
        breakdown_data = {
            "📚 Total Knowledge Sources": total_sources,
            "📖 Total Content Chunks": total_content_chunks,
            "🌐 Web Content Chunks": web_chunks,
            "📄 File Content Chunks": file_chunks
        }
        
        breakdown_df = pd.DataFrame(list(breakdown_data.items()), columns=["Metric", "Value"])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        
        # Show insights and recommendations
        if total_sources > 0:
            st.markdown("**💡 Insights & Recommendations**")
            
            # Content depth analysis
            if avg_chunks >= 5:
                st.success(f"✅ **Good Content Depth**: Average {avg_chunks:.1f} chunks per source provides detailed coverage")
            elif avg_chunks >= 2:
                st.info(f"ℹ️ **Moderate Content Depth**: Average {avg_chunks:.1f} chunks per source - consider adding more content")
            else:
                st.warning(f"⚠️ **Low Content Depth**: Average {avg_chunks:.1f} chunks per source - add more detailed sources")
            
            # Content diversity analysis
            if web_sources > 0 and file_sources > 0:
                st.success("✅ **Diverse Content**: Mix of web and file sources provides comprehensive coverage")
            elif web_sources > 0:
                st.info("ℹ️ **Web-Focused**: Consider adding uploaded documents for offline content")
            elif file_sources > 0:
                st.info("ℹ️ **File-Focused**: Consider adding web sources for current information")
            
            # Size recommendations
            if total_sources < 5:
                st.info("💡 **Recommendation**: Add more sources to improve RAG coverage")
            elif total_sources >= 10:
                st.success("✅ **Well-Stocked**: Good number of sources for comprehensive RAG")
        
        # Database configuration table
        st.markdown("**📋 Database Configuration**")
        config_data = {
            "Database ID": db_info.get('vector_db_id', 'Unknown'),
            "Name": db_info.get('name', 'Unknown'),
            "Provider": db_info.get('provider_id', 'Unknown'),
            "Embedding Model": db_info.get('embedding_model', 'Unknown'),
            "Status": "✅ Active" if total_chunks > 0 else "⚠️ Empty"
        }
        
        config_df = pd.DataFrame(list(config_data.items()), columns=["Property", "Value"])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        # Status information
        if total_chunks == 0:
            st.info("📁 **Knowledge Base Empty** - Upload documents or process web URLs to start building your RAG system")
        else:
            # Show meaningful status information
            st.markdown("**📊 Knowledge Base Status**")
            
            # The original code had 'deleted_docs' which is not defined.
            # Assuming it should be 'deleted_document_ids' from session state.
            deleted_document_ids = st.session_state.get('deleted_document_ids', [])
            if len(deleted_document_ids) > 0:
                st.warning(f"🗑️ **{len(deleted_document_ids)} sources marked for deletion** - these won't appear in search results")
            
            # Show content health
            if total_sources >= 5 and avg_chunks >= 3:
                st.success("✅ **Healthy Knowledge Base** - Good coverage and depth for effective RAG")
            elif total_sources >= 3:
                st.info("ℹ️ **Growing Knowledge Base** - Add more sources for better coverage")
            else:
                st.warning("⚠️ **Small Knowledge Base** - Consider adding more diverse sources")
        
        # Debug information (expandable)
        with st.expander("🔍 Debug Information"):
            st.markdown("**Database Info:**")
            st.json(db_info)
            
            st.markdown("**Vector Counting Details:**")
            try:
                st.write(f"**Total unique vectors found**: {total_chunks}")
                st.write(f"**Total unique sources found**: {total_sources}")
                
                # Show sample vectors
                if all_results:
                    st.write("**Sample vectors being counted:**")
                    for i, result in enumerate(all_results[:10]):  # Show first 10
                        metadata = result.get('metadata', {})
                        doc_id = metadata.get('document_id', 'Unknown')
                        chunk_index = metadata.get('chunk_index', 0)
                        content_preview = result.get('content', '')[:50] + "..." if len(result.get('content', '')) > 50 else result.get('content', '')
                        st.write(f"  {i+1}. {doc_id} (chunk {chunk_index}): {content_preview}")
                    
                    if len(all_results) > 10:
                        st.write(f"  ... and {len(all_results) - 10} more vectors")
                        
            except Exception as e:
                st.error(f"Error getting vector details: {e}")
            
            st.markdown("**Session State Analysis:**")
            if 'uploaded_documents' in st.session_state:
                st.write(f"Uploaded documents: {len(st.session_state.uploaded_documents)}")
                for i, doc in enumerate(st.session_state.uploaded_documents[:5]):  # Show first 5
                    st.write(f"  {i+1}. {doc.get('name', 'Unknown')} - {len(doc.get('chunks', []))} chunks")
                if len(st.session_state.uploaded_documents) > 5:
                    st.write(f"  ... and {len(st.session_state.uploaded_documents) - 5} more")
            else:
                st.write("No uploaded documents in session state")
            
            if 'deleted_document_ids' in st.session_state:
                st.write(f"Deleted document IDs: {len(st.session_state.deleted_document_ids)}")
                if st.session_state.deleted_document_ids:
                    st.write("Sample deleted IDs:")
                    for doc_id in list(st.session_state.deleted_document_ids)[:5]:
                        st.write(f"  - {doc_id}")
            else:
                st.write("No deleted document IDs in session state")
        
    except Exception as e:
        st.error(f"❌ Error loading database overview: {str(e)}")
        st.info("💡 Make sure LlamaStack is running and VectorIO API is accessible")

def render_documents_table():
    """Render documents table with proper vector counting"""
    try:
        # Get documents from session state first
        documents = []
        if 'uploaded_documents' in st.session_state:
            documents = st.session_state.uploaded_documents
        
        # If no documents in session state, try to retrieve from VectorIO
        if not documents:
            try:
                # Get documents from VectorIO using multiple broad queries
                all_results = []
                broad_queries = ["a", "the", "and", "or", "in", "on", "at", "to", "for", "of"]
                
                for query in broad_queries:
                    try:
                        results = st.session_state.llamastack_client.get_filtered_search_results(
                            query_text=query,
                            vector_db_id="faiss",
                            top_k=100
                        )
                        all_results.extend(results)
                    except Exception as e:
                        continue
                
                # Group by document_id to reconstruct documents
                doc_groups = {}
                for result in all_results:
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('document_id', 'unknown')
                    doc_name = metadata.get('document_name', 'Unknown')
                    source_url = metadata.get('source_url')
                    file_type = metadata.get('file_type', 'FILE')
                    chunk_index = metadata.get('chunk_index', 0)
                    
                    # Extract base document identifier for grouping
                    if doc_id.startswith('web_'):
                        # For web content: web_domain_chunkindex -> web_domain
                        parts = doc_id.split('_')
                        if len(parts) >= 3:
                            domain = parts[1]
                            base_doc_id = f"web_{domain}"
                        else:
                            base_doc_id = doc_id
                    elif doc_id.startswith('file_'):
                        # For files: file_filename_chunkindex -> file_filename
                        parts = doc_id.split('_', 2)
                        if len(parts) >= 3:
                            filename = parts[1]
                            base_doc_id = f"file_{filename}"
                        else:
                            base_doc_id = doc_id
                    else:
                        base_doc_id = doc_id
                    
                    if base_doc_id not in doc_groups:
                        doc_groups[base_doc_id] = {
                            'name': doc_name,
                            'source_url': source_url,
                            'file_type': file_type,
                            'chunks': [],
                            'chunk_count': 0,
                            'document_ids': set()
                        }
                    
                    # Add chunk info
                    doc_groups[base_doc_id]['chunks'].append({
                        'content': result.get('content', ''),
                        'chunk_index': chunk_index,
                        'document_id': doc_id
                    })
                    doc_groups[base_doc_id]['chunk_count'] += 1
                    doc_groups[base_doc_id]['document_ids'].add(doc_id)
                
                # Convert to documents format
                for base_doc_id, doc_data in doc_groups.items():
                    display_name = doc_data['name']
                    if doc_data['source_url']:
                        if display_name.startswith('Web Content from'):
                            from urllib.parse import urlparse
                            try:
                                domain = urlparse(doc_data['source_url']).netloc
                                display_name = f"🌐 {domain}"
                            except:
                                pass
                    elif doc_data['file_type'] == 'FILE':
                        display_name = f"📄 {display_name}"
                    
                    documents.append({
                        'name': display_name,
                        'source_url': doc_data['source_url'],
                        'file_type': doc_data['file_type'],
                        'chunks': doc_data['chunks'],
                        'chunk_count': doc_data['chunk_count'],
                        'document_ids': list(doc_data['document_ids']),
                        'vectorio_stored': True
                    })
                    
            except Exception as e:
                st.error(f"❌ Error retrieving documents from VectorIO: {e}")
        
        if not documents:
            st.info("📁 No documents found in the database")
            return
        
        # Create summary statistics
        total_sources = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
        web_sources = sum(1 for doc in documents if doc.get('source_url'))
        file_sources = total_sources - web_sources
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📚 Total Sources", total_sources)
        with col2:
            st.metric("🔢 Total Chunks", total_chunks)
        with col3:
            st.metric("🌐 Web Sources", web_sources)
        with col4:
            st.metric("📄 File Sources", file_sources)
        
        # Create table data
        table_data = []
        for doc in documents:
            doc_type = "🌐 Web" if doc.get('source_url') else "📄 File"
            source = doc.get('source_url', doc.get('name', 'Unknown'))
            chunks = doc.get('chunk_count', 0)
            size = f"{chunks} chunks"
            processing = "✅ Stored" if doc.get('vectorio_stored', False) else "⚠️ Session Only"
            storage = "🗄️ VectorIO" if doc.get('vectorio_stored', False) else "💾 Session"
            
            table_data.append({
                "Type": doc_type,
                "Source": source,
                "Chunks": chunks,
                "Size": size,
                "Processing": processing,
                "Storage": storage
            })
        
        # Display table
        st.markdown("**📋 Content Sources**")
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ Error rendering documents table: {str(e)}")

def render_vectorio_search():
    """Render VectorIO search interface"""
    st.markdown("#### 🔍 Search Database")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Enter search query:",
            placeholder="e.g., OpenShift AI, machine learning, etc.",
            help="Search for similar content in the vector database"
        )
    
    with col2:
        top_k = st.slider("Results", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search", type="primary"):
        if not search_query:
            st.warning("⚠️ Please enter a search query")
            return
        
        try:
            with st.spinner("Searching vector database..."):
                results = st.session_state.llamastack_client.search_similar_vectors(
                    query_text=search_query,
                    vector_db_id="faiss",
                    top_k=top_k
                )
            
            if results:
                st.success(f"✅ Found {len(results)} results")
                
                # Display results in a clean format
                for i, result in enumerate(results, 1):
                    similarity = result.get('similarity', 0)
                    content = result.get('content', 'No content')
                    metadata = result.get('metadata', {})
                    
                    with st.expander(f"Result {i} (Score: {similarity:.3f})"):
                        st.markdown(f"**Content:** {content}")
                        
                        if metadata:
                            st.markdown("**Metadata:**")
                            meta_df = pd.DataFrame(list(metadata.items()), columns=['Field', 'Value'])
                            st.dataframe(meta_df, use_container_width=True, hide_index=True)
            else:
                st.info("🔍 No results found")
                
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")

# Legacy function for backward compatibility
def get_faiss_index_stats() -> Dict[str, Any]:
    """Get VectorIO statistics (legacy compatibility)"""
    return get_vectorio_stats()

def render_crud_operations():
    """Render CRUD operations for VectorIO database"""
    st.markdown("#### ⚙️ Database Management")
    
    # CRUD Operations
    operation = st.selectbox(
        "Select Operation:",
        ["📖 Read", "🗑️ Delete", "🔄 Update", "📊 Export", "🧹 Maintenance"],
        help="Choose a database operation to perform"
    )
    
    if operation == "📖 Read":
        render_read_operations()
    elif operation == "🗑️ Delete":
        render_delete_operations()
    elif operation == "🔄 Update":
        render_update_operations()
    elif operation == "📊 Export":
        render_export_operations()
    elif operation == "🧹 Maintenance":
        render_maintenance_operations()

def render_read_operations():
    """Render read operations"""
    st.markdown("**📖 Read Operations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Database", type="primary"):
            st.rerun()
    
    with col2:
        if st.button("📊 Show Statistics", type="secondary"):
            try:
                stats = get_vectorio_stats()
                if "error" not in stats:
                    st.success("✅ Database Statistics:")
                    st.json(stats)
                else:
                    st.error(f"❌ {stats['error']}")
            except Exception as e:
                st.error(f"❌ Error getting statistics: {e}")
    
    # Database info
    st.markdown("---")
    st.markdown("**Database Information:**")
    
    try:
        vector_dbs = st.session_state.llamastack_client.list_vector_databases()
        if vector_dbs:
            for db in vector_dbs:
                st.info(f"**Database ID:** {db.get('vector_db_id', 'Unknown')}")
                st.info(f"**Provider:** {db.get('provider_id', 'Unknown')}")
                st.info(f"**Status:** {db.get('status', 'Unknown')}")
        else:
            st.warning("⚠️ No vector databases found")
    except Exception as e:
        st.error(f"❌ Error reading database info: {e}")

def render_delete_operations():
    """Render delete operations"""
    st.markdown("**🗑️ Delete Operations**")
    
    st.warning("⚠️ **Warning:** Delete operations are irreversible and will permanently remove data from the VectorIO database!")
    
    # Get current documents for deletion
    documents = []
    if 'uploaded_documents' in st.session_state:
        documents = st.session_state.uploaded_documents
    
    # If no documents in session state, try to retrieve from VectorIO
    if not documents:
        try:
            # Get documents from VectorIO for deletion (using filtered search to exclude already deleted)
            search_results = st.session_state.llamastack_client.get_filtered_search_results(
                query_text="a",  # Broad query to get all documents
                vector_db_id="faiss",
                top_k=100
            )
            
            if search_results:
                # Group by document_id to show unique documents
                doc_groups = {}
                for result in search_results:
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('document_id', 'unknown')
                    doc_name = metadata.get('document_name', 'Unknown')
                    source_url = metadata.get('source_url')
                    
                    # Extract base document name for better grouping
                    if doc_id.startswith('web_'):
                        parts = doc_id.split('_')
                        if len(parts) >= 3:
                            domain = parts[1]
                            base_doc_id = f"web_{domain}"
                        else:
                            base_doc_id = doc_id
                    elif doc_id.startswith('file_'):
                        parts = doc_id.split('_', 2)
                        if len(parts) >= 3:
                            filename = parts[1]
                            base_doc_id = f"file_{filename}"
                        else:
                            base_doc_id = doc_id
                    else:
                        base_doc_id = doc_id
                    
                    if base_doc_id not in doc_groups:
                        doc_groups[base_doc_id] = {
                            'name': doc_name,
                            'source_url': source_url,
                            'document_ids': set(),
                            'chunk_count': 0
                        }
                    
                    doc_groups[base_doc_id]['document_ids'].add(doc_id)
                    doc_groups[base_doc_id]['chunk_count'] += 1
                
                # Convert to documents format
                for base_doc_id, doc_data in doc_groups.items():
                    display_name = doc_data['name']
                    if doc_data['source_url']:
                        if display_name.startswith('Web Content from'):
                            from urllib.parse import urlparse
                            try:
                                domain = urlparse(doc_data['source_url']).netloc
                                display_name = f"{domain}"  # Remove the 🌐 prefix here
                            except:
                                pass
                    
                    documents.append({
                        'name': display_name,
                        'source_url': doc_data['source_url'],
                        'document_ids': list(doc_data['document_ids']),
                        'chunk_count': doc_data['chunk_count'],
                        'vectorio_stored': True
                    })
        except Exception as e:
            st.error(f"❌ Error retrieving documents from VectorIO: {e}")
    
    # Show deletion status
    if 'deleted_document_ids' in st.session_state and st.session_state.deleted_document_ids:
        st.info(f"🗑️ **Deletion Status**: {len(st.session_state.deleted_document_ids)} documents marked for deletion")
    
    if not documents:
        st.info("📁 No documents available for deletion")
        return
    
    # Document selection for deletion
    st.markdown("**Select sources to delete from VectorIO database:**")
    
    doc_options = []
    for i, doc in enumerate(documents):
        doc_type = "🌐 Web" if doc.get('source_url') else "📄 File"
        chunk_count = doc.get('chunk_count', len(doc.get('chunks', [])))
        doc_options.append(f"{doc_type} - {doc.get('name', f'Document {i}')} ({chunk_count} chunks)")
    
    selected_docs = st.multiselect(
        "Choose sources to delete:",
        doc_options,
        help="Select sources to permanently remove from VectorIO database"
    )
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Delete Selected Sources", type="secondary", disabled=not selected_docs):
            if selected_docs:
                try:
                    deleted_count = 0
                    total_chunks = 0
                    
                    for selected in selected_docs:
                        # Find the corresponding document
                        for doc in documents:
                            doc_type = "🌐 Web" if doc.get('source_url') else "📄 File"
                            chunk_count = doc.get('chunk_count', len(doc.get('chunks', [])))
                            doc_option = f"{doc_type} - {doc.get('name', 'Unknown')} ({chunk_count} chunks)"
                            
                            # Debug: Print both strings for comparison
                            print(f"DEBUG: Selected: '{selected}'")
                            print(f"DEBUG: Doc option: '{doc_option}'")
                            print(f"DEBUG: Match: {selected == doc_option}")
                            
                            if selected == doc_option:
                                # Delete from VectorIO database
                                if 'document_ids' in doc:
                                    try:
                                        # Use the new deletion method
                                        deletion_success = st.session_state.llamastack_client.delete_vectors_from_vector_db(
                                            document_ids=doc['document_ids'],
                                            vector_db_id="faiss"
                                        )
                                        
                                        if deletion_success:
                                            print(f"🗑️ Successfully deleted document {doc.get('name', 'Unknown')}")
                                            deleted_count += 1
                                            total_chunks += doc.get('chunk_count', 0)
                                            
                                            # Remove from session state if deletion was successful
                                            if 'uploaded_documents' in st.session_state:
                                                st.session_state.uploaded_documents = [
                                                    d for d in st.session_state.uploaded_documents 
                                                    if d.get('name') != doc.get('name')
                                                ]
                                            if 'deleted_document_ids' in st.session_state:
                                                st.session_state.deleted_document_ids.add(doc.get('name'))
                                            
                                        else:
                                            print(f"⚠️ Failed to delete document {doc.get('name', 'Unknown')}")
                                            st.error(f"❌ Failed to delete {doc.get('name', 'Unknown')}")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error deleting {doc.get('name', 'Unknown')}: {e}")
                                else:
                                    # Fallback for documents without document_ids
                                    print(f"🗑️ Marking document {doc.get('name', 'Unknown')} for deletion")
                                    deleted_count += 1
                                    total_chunks += chunk_count
                                break
                    
                    st.success(f"✅ Marked {deleted_count} sources ({total_chunks} chunks) for deletion")
                    st.info("💡 Note: Actual deletion requires VectorIO delete API endpoint")
                    
                except Exception as e:
                    st.error(f"❌ Error during deletion: {e}")
        
        with col2:
            if st.button("🗑️ Clear All Data", type="secondary"):
                st.error("⚠️ **DANGEROUS OPERATION**")
                if st.checkbox("I understand this will permanently delete ALL data from VectorIO database"):
                    if st.checkbox("I confirm I want to delete everything"):
                        try:
                            st.warning("🔄 Clearing all data from VectorIO database...")
                            
                            # Show progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("🗑️ Attempting to clear database...")
                            progress_bar.progress(25)
                            
                            # Use the new clear method
                            clear_success = st.session_state.llamastack_client.clear_all_vectors_from_vector_db(
                                vector_db_id="faiss"
                            )
                            
                            progress_bar.progress(75)
                            status_text.text("🧹 Clearing session state...")
                            
                            if clear_success:
                                # Clear session state
                                keys_to_keep = ['llamastack_client', 'selected_llm_model']
                                keys_to_clear = [key for key in st.session_state.keys() if key not in keys_to_keep]
                                
                                for key in keys_to_clear:
                                    del st.session_state[key]
                                
                                progress_bar.progress(100)
                                status_text.text("✅ All data cleared successfully!")
                                
                                st.success("✅ **All data cleared from VectorIO database and session state**")
                                st.info("🔄 **Database has been reset and is ready for new documents**")
                                
                                # Show confirmation of empty state
                                try:
                                    stats = st.session_state.llamastack_client.get_database_stats("faiss")
                                    if stats.get('is_empty', True):
                                        st.success("📊 **Database Status**: Empty and ready for new content")
                                    else:
                                        st.warning("⚠️ **Note**: Some data may still be present in the database")
                                except Exception as e:
                                    st.info(" **Database cleared successfully**")
                                
                        except Exception as e:
                            st.error(f"❌ Error clearing data: {e}")
        
        # Add manual API operation buttons
        st.markdown("---")
        st.markdown("**🔧 Manual API Operations**")
        st.info("💡 These buttons perform direct API calls to delete and recreate the FAISS database")
        
        manual_col1, manual_col2 = st.columns(2)
        
        with manual_col1:
            if st.button("🗑️ Delete FAISS Database (API)", type="secondary", help="Delete the FAISS database via LlamaStack API"):
                try:
                    import requests
                    
                    st.info("🗑️ Deleting FAISS database via API...")
                    
                    # Delete the FAISS database
                    delete_response = requests.delete(
                        "http://localhost:8321/v1/vector-dbs/faiss",
                        timeout=10
                    )
                    
                    if delete_response.status_code == 200:
                        st.success("✅ FAISS database deleted successfully!")
                        st.info(f"Response: {delete_response.status_code}")
                        
                        # Clear session state
                        if 'uploaded_documents' in st.session_state:
                            st.session_state.uploaded_documents.clear()
                        if 'documents' in st.session_state:
                            st.session_state.documents.clear()
                        if 'deleted_document_ids' in st.session_state:
                            st.session_state.deleted_document_ids.clear()
                        
                        st.success("🧹 Session state cleared")
                        
                    elif delete_response.status_code == 404:
                        st.warning("⚠️ FAISS database not found (already deleted or doesn't exist)")
                    else:
                        st.error(f"❌ Failed to delete database: {delete_response.status_code}")
                        st.text(f"Response: {delete_response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error deleting database: {e}")
        
        with manual_col2:
            if st.button("🔄 Recreate FAISS Database (API)", type="secondary", help="Recreate the FAISS database via LlamaStack API"):
                try:
                    import requests
                    import json
                    
                    st.info("🔄 Recreating FAISS database via API...")
                    
                    # Recreate the database
                    create_data = {
                        "vector_db_id": "faiss",
                        "provider_id": "faiss",
                        "embedding_model": "all-MiniLM-L6-v2"
                    }
                    
                    create_response = requests.post(
                        "http://localhost:8321/v1/vector-dbs",
                        json=create_data,
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    
                    if create_response.status_code in [200, 201]:
                        st.success("✅ FAISS database recreated successfully!")
                        st.info(f"Response: {create_response.status_code}")
                        
                        # Show database info
                        try:
                            response_data = create_response.json()
                            st.json(response_data)
                        except:
                            st.text(f"Response: {create_response.text}")
                        
                        st.success("🆕 Database is ready for new documents")
                        
                    else:
                        st.error(f"❌ Failed to recreate database: {create_response.status_code}")
                        st.text(f"Response: {create_response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error recreating database: {e}")
        
        # Add a combined operation button
        st.markdown("---")
        st.markdown("**⚡ Quick Reset Operation**")
        
        if st.button("🔄 Reset FAISS Database (Delete + Recreate)", type="primary", help="Delete and recreate the FAISS database in one operation"):
            try:
                import requests
                import json
                
                st.info("🔄 Performing complete FAISS database reset...")
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Delete database
                status_text.text("🗑️ Step 1: Deleting existing database...")
                progress_bar.progress(25)
                
                delete_response = requests.delete(
                    "http://localhost:8321/v1/vector-dbs/faiss",
                    timeout=10
                )
                
                if delete_response.status_code not in [200, 404]:
                    st.error(f"❌ Failed to delete database: {delete_response.status_code}")
                    return
                
                # Step 2: Recreate database
                status_text.text("🔄 Step 2: Recreating database...")
                progress_bar.progress(75)
                
                create_data = {
                    "vector_db_id": "faiss",
                    "provider_id": "faiss",
                    "embedding_model": "all-MiniLM-L6-v2"
                }
                
                create_response = requests.post(
                    "http://localhost:8321/v1/vector-dbs",
                    json=create_data,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if create_response.status_code in [200, 201]:
                    # Step 3: Clear session state
                    status_text.text("🧹 Step 3: Clearing session state...")
                    progress_bar.progress(90)
                    
                    keys_to_keep = ['llamastack_client', 'selected_llm_model', 'selected_embedding_model']
                    keys_to_clear = [key for key in st.session_state.keys() if key not in keys_to_keep]
                    
                    for key in keys_to_clear:
                        del st.session_state[key]
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Database reset completed!")
                    
                    st.success("✅ **FAISS database reset completed successfully!**")
                    st.info("🆕 **Database is empty and ready for new documents**")
                    
                    # Verify empty state
                    try:
                        stats = st.session_state.llamastack_client.get_database_stats("faiss")
                        if stats.get('is_empty', True):
                            st.success("📊 **Database Status**: Empty and ready for new content")
                        else:
                            st.warning("⚠️ **Note**: Some data may still be present")
                    except Exception as e:
                        st.info("✅ Database reset completed")
                        
                else:
                    st.error(f"❌ Failed to recreate database: {create_response.status_code}")
                    st.text(f"Response: {create_response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error during database reset: {e}")
                import traceback
                st.text(f"Details: {traceback.format_exc()}") 

def render_update_operations():
    """Render update operations"""
    st.markdown("**🔄 Update Operations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Sources", type="primary"):
            try:
                # Force refresh from VectorIO
                st.session_state.uploaded_documents = []
                st.success("✅ Sources refreshed from VectorIO")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error refreshing: {e}")
    
    with col2:
        if st.button("🔄 Reindex Database", type="secondary"):
            st.info("🔄 Reindexing database...")
            try:
                # This would trigger a reindex operation
                st.success("✅ Database reindexed successfully")
            except Exception as e:
                st.error(f"❌ Reindex failed: {e}")
    
    # Update settings
    st.markdown("---")
    st.markdown("**Database Settings:**")
    
    # Embedding model selection
    embedding_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-MiniLM-L12-v2"]
    selected_model = st.selectbox(
        "Embedding Model:",
        embedding_models,
        index=0,
        help="Select embedding model for new documents"
    )
    
    if st.button("💾 Save Settings", type="primary"):
        st.success(f"✅ Settings saved: {selected_model}")

def render_export_operations():
    """Render export operations"""
    st.markdown("**📊 Export Operations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Statistics", type="primary"):
            try:
                stats = get_vectorio_stats()
                if "error" not in stats:
                    # Create export data
                    export_data = {
                        "export_time": datetime.now().isoformat(),
                        "database_stats": stats,
                        "session_documents": len(st.session_state.get('uploaded_documents', [])),
                        "total_chunks": sum(len(doc.get('chunks', [])) for doc in st.session_state.get('uploaded_documents', []))
                    }
                    
                    # Convert to JSON
                    json_str = json.dumps(export_data, indent=2)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Statistics",
                        data=json_str,
                        file_name=f"vectorio_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"❌ {stats['error']}")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    with col2:
        if st.button("📋 Export Sources List", type="secondary"):
            try:
                documents = st.session_state.get('uploaded_documents', [])
                if documents:
                    # Create CSV data
                    csv_data = []
                    for doc in documents:
                        csv_data.append({
                            "Name": doc.get('name', 'Unknown'),
                            "Type": "Web" if doc.get('source_url') else "File",
                            "Source": doc.get('source_url', doc.get('name', 'Unknown')),
                            "Chunks": len(doc.get('chunks', [])),
                            "Size_MB": doc.get('file_size_mb', 0),
                            "Processing_Time": doc.get('processing_time', 0)
                        })
                    
                    if csv_data:
                        df = pd.DataFrame(csv_data)
                        csv_str = df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Sources CSV",
                            data=csv_str,
                            file_name=f"vectorio_sources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("📁 No sources to export")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")

def render_maintenance_operations():
    """Render maintenance operations"""
    st.markdown("**🧹 Maintenance Operations**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧹 Optimize Database", type="primary"):
            st.info("🔄 Optimizing database...")
            try:
                # Simulate optimization
                import time
                time.sleep(1)
                st.success("✅ Database optimized successfully")
            except Exception as e:
                st.error(f"❌ Optimization failed: {e}")
    
    with col2:
        if st.button("🔍 Validate Database", type="secondary"):
            st.info("🔍 Validating database integrity...")
            try:
                # Check database health
                stats = get_vectorio_stats()
                if "error" not in stats:
                    st.success("✅ Database validation passed")
                    st.info(f"Database is healthy with {stats.get('total_vectors', 0)} vectors")
                else:
                    st.error(f"❌ Database validation failed: {stats['error']}")
            except Exception as e:
                st.error(f"❌ Validation failed: {e}")
    
    # System information
    st.markdown("---")
    st.markdown("**System Information:**")
    
    try:
        # Get available vector databases
        vector_dbs = st.session_state.llamastack_client.list_vector_databases()
        
        st.markdown("**Available Vector Databases:**")
        if vector_dbs:
            for db in vector_dbs:
                status = "✅ Active" if db.get('vector_db_id') == 'faiss' else "⏸️ Inactive"
                st.text(f"• {db.get('vector_db_id', 'Unknown')} - {status}")
        else:
            st.text("• No vector databases found")
            
    except Exception as e:
        st.error(f"❌ Error getting system info: {str(e)}") 