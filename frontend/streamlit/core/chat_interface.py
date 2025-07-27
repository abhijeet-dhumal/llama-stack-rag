"""
Chat Interface for RAG LlamaStack Application
Handles chat functionality, welcome screen, and user interactions
"""

import streamlit as st
import streamlit.components.v1 as components
import html
import re
from typing import List, Dict, Any
from .utils import cosine_similarity, get_context_type
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def render_welcome_screen() -> None:
    """Display welcome screen when no documents are uploaded"""
    
    # Center content with padding
    st.markdown("<br>" * 2, unsafe_allow_html=True)
    
    # Main welcome message
    st.markdown("""
    <div style="text-align: center;">
        <h2>Welcome to your intelligent knowledge assistant</h2>
        <p style="font-size: 1.1em; margin: 1.5rem 0;">
            Upload documents or process web URLs to get started. I'll help you find information, 
            answer questions, and explore your knowledge base from multiple sources.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced getting started card with web content highlight
    st.success("""
    **🎯 Get Started - Choose Your Source**
    
    **📄 Upload Files** using the sidebar →  
    *Supported formats: TXT, PDF, MD, DOCX, PPTX (Max 50MB)*
    
    **🌐 Process Web URLs** ✨ →
    *Extract content from articles, docs, Wikipedia in real-time*
    """)
    
    # Features overview with updated capabilities
    st.markdown("### ✨ What you can do:")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **📝 Ask Questions**
        
        Query your documents and web content with natural language
        """)
    
    with col_b:
        st.markdown("""
        **🔍 Hybrid Search**
        
        Find information across files and web sources
        """)
    
    with col_c:
        st.markdown("""
        **💡 Real-time Processing**
        
        Instant embedding and analysis of web content
        """)
    
    # New feature spotlight
    st.markdown("---")
    st.markdown("### 🌟 Feature Spotlight")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.info("""
        **🌐 Web Content Processing**
        
        - Uses MCP Server with Mozilla Readability
        - Clean extraction from articles and documentation  
        - Smart fallback for maximum compatibility
        - Mixed with uploaded files in unified knowledge base
        """)
    
    with col_right:
        st.info("""
        **🚀 Enhanced AI Pipeline**
        
        - LlamaStack orchestration for unified processing
        - Sentence Transformers for high-quality embeddings
        - Local Ollama models for privacy
        - Real-time performance analytics
        """)
    
    st.markdown("<br>" * 2, unsafe_allow_html=True)


def render_chat_interface() -> None:
    """Display the main chat interface with proper chat layout"""
    # Clean chat styling - use st.markdown with unsafe_allow_html=True
    st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: var(--background-color, #ffffff);
        border: 1px solid var(--border-color, #e9ecef);
    }
    
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    
    .message-bubble {
        max-width: 75%;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #007bff, #0056b3);
        background-color: #007bff;
        color: white !important;
        border-bottom-right-radius: 4px;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #0056b3;
    }
    
    .user-bubble * {
        color: white !important;
    }
    
    .user-message .message-bubble.user-bubble {
        color: white !important;
    }
    
    .user-message .message-bubble.user-bubble * {
        color: white !important;
    }
    
    .assistant-bubble {
        background: var(--secondary-background, #f8f9fa);
        border: 1px solid var(--border-color, #e9ecef);
        color: var(--text-color, #262730);
        border-bottom-left-radius: 4px;
    }
    
    .message-metadata {
        margin-top: 0.5rem;
        font-size: 0.75em;
        opacity: 0.7;
        color: var(--secondary-text, #666666);
    }
    
    .source-tag {
        display: inline-block;
        background: var(--primary-color, #ff4b4b);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
        font-size: 0.75em;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
        cursor: pointer;
        border: none;
    }
    
    .source-tag:hover {
        background: #ff6b6b !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .model-info {
        margin-top: 0.4rem;
        padding: 0.3rem 0.5rem;
        background: var(--secondary-background, #f8f9fa);
        border-radius: 6px;
        font-size: 0.7em;
        border-left: 3px solid var(--info-color, #17a2b8);
        color: var(--secondary-text, #666666);
    }
    
    /* Improve input field readability */
    .stTextInput > div > div > input {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #333 !important;
        background-color: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #6c757d !important;
        font-weight: 400 !important;
    }
    
    /* Global override for user bubble text color */
    .user-bubble, .user-bubble *, .user-message .message-bubble {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat history display
    if st.session_state.chat_history:
        # Prepare document content for JavaScript
        document_contents = {}
        source_relevances = {}
        
        for message in st.session_state.chat_history:
            if message['role'] == 'assistant' and 'sources' in message:
                for i, source in enumerate(message['sources']):
                    doc_name = source['document']
                    source_relevances[i] = source['score'] * 100
                    
                    # Get document content from session state
                    all_documents = []
                    if 'documents' in st.session_state:
                        all_documents.extend(st.session_state.documents)
                    if 'uploaded_documents' in st.session_state:
                        all_documents.extend(st.session_state.uploaded_documents)
                    
                    # Find the document and get its content
                    for doc in all_documents:
                        if doc.get('name') == doc_name:
                            content = doc.get('content', 'Content not available')
                            # Truncate for modal display
                            if len(content) > 2000:
                                content = content[:2000] + "...\n\n[Content truncated for display]"
                            document_contents[doc_name] = content
                            break
        
        # Inject document content into JavaScript
        if document_contents:
            st.markdown(f"""
            <script>
            window.documentContents = {document_contents};
            window.sourceRelevances = {source_relevances};
            </script>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                # User message (right side)
                escaped_content = html.escape(message['content']).replace('\n', '<br>')
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-bubble user-bubble" style="color: white !important;">
                        <span style="color: white !important;">{escaped_content}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                # Assistant message (left side)
                escaped_content = html.escape(message['content']).replace('\n', '<br>')
                
                # Build metadata sections
                metadata_html = ""
                
                # Sources with clickable links
                if 'sources' in message and message['sources']:
                    sources_html = ""
                    for i, source in enumerate(message['sources']):
                        relevance_pct = source["score"] * 100
                        escaped_doc_name = html.escape(source["document"])
                        # Create clickable link for document preview with better styling
                        sources_html += f'<button class="source-tag" onclick="showDocumentPreview(\'{escaped_doc_name}\', {i})">📄 {escaped_doc_name} ({relevance_pct:.1f}%)</button>'
                    metadata_html += f'<div class="message-metadata">{sources_html}</div>'
                
                # Model info (simplified)
                if 'models_used' in message:
                    models = message['models_used']
                    embedding_status = "✅ Real" if models.get('embedding_real', False) else "🧪 Demo"
                    model_info = f"🔍 {models.get('embedding', 'unknown')} ({embedding_status}) • 🧠 {models.get('llm', 'unknown')}"
                    metadata_html += f'<div class="model-info">{model_info}</div>'
                
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-bubble assistant-bubble">
                        {escaped_content}
                        {metadata_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add Streamlit-based document preview as separate expandable block after the message
                if 'sources' in message and message['sources']:
                    with st.expander("📄 View Source Documents", expanded=False):
                        for i, source in enumerate(message['sources']):
                            doc_name = source['document']
                            relevance_pct = source['score'] * 100
                            
                            # Find document content
                            all_documents = []
                            if 'documents' in st.session_state:
                                all_documents.extend(st.session_state.documents)
                            if 'uploaded_documents' in st.session_state:
                                all_documents.extend(st.session_state.uploaded_documents)
                            
                            doc_content = "Content not available"
                            for doc in all_documents:
                                if doc.get('name') == doc_name:
                                    doc_content = doc.get('content', 'Content not available')
                                    break
                            
                            with st.expander(f"📄 {doc_name} (Relevance: {relevance_pct:.1f}%)", expanded=False):
                                st.markdown(f"**Source:** {doc_name}")
                                st.markdown(f"**Relevance:** {relevance_pct:.1f}%")
                                
                                # Show document metadata
                                for doc in all_documents:
                                    if doc.get('name') == doc_name:
                                        if doc.get('file_type'):
                                            st.markdown(f"**Type:** {doc.get('file_type')}")
                                        if doc.get('extraction_method'):
                                            method_map = {
                                                'requests_github_raw': '🔄 GitHub Raw',
                                                'requests_beautifulsoup': '🔄 BeautifulSoup',
                                                'requests_plain_text': '🔄 Direct HTTP',
                                                'mcp_just-every': '📦 Just-Every MCP'
                                            }
                                            method_display = method_map.get(doc.get('extraction_method'), doc.get('extraction_method'))
                                            st.markdown(f"**Method:** {method_display}")
                                        if doc.get('source_url'):
                                            st.markdown(f"**URL:** {doc.get('source_url')}")
                                        break
                                
                                st.markdown("---")
                                # Use a unique counter-based key to avoid duplicates
                                if 'doc_preview_counter' not in st.session_state:
                                    st.session_state.doc_preview_counter = 0
                                st.session_state.doc_preview_counter += 1
                                unique_key = f"doc_preview_{st.session_state.doc_preview_counter}"
                                st.text_area("Content", doc_content, height=200, key=unique_key)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add JavaScript for document preview functionality - use components.html to avoid display issues
    
    # Create the JavaScript component
    js_code = """
    <script>
    // Document preview functionality
    function showDocumentPreview(docName, sourceIndex) {
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'doc-modal';
        modal.id = 'docModal';
        
        // Get document content from Streamlit session state
        const docContent = getDocumentContent(docName);
        
        modal.innerHTML = `
            <div class="doc-modal-content">
                <div class="doc-modal-header">
                    <h3>📄 ${docName}</h3>
                    <span class="doc-modal-close" onclick="closeDocumentPreview()">&times;</span>
                </div>
                <div class="doc-modal-body">
                    <p><strong>Source:</strong> ${docName}</p>
                    <p><strong>Relevance:</strong> ${getSourceRelevance(sourceIndex)}%</p>
                    <hr>
                    <div class="doc-content">
                        <pre style="white-space: pre-wrap; font-family: inherit; background: var(--secondary-background, #f8f9fa); padding: 15px; border-radius: 5px; max-height: 400px; overflow-y: auto;">${docContent}</pre>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        modal.style.display = 'block';
    }
    
    function closeDocumentPreview() {
        const modal = document.getElementById('docModal');
        if (modal) {
            modal.remove();
        }
    }
    
    function getDocumentContent(docName) {
        // This will be populated by Streamlit with actual document content
        return window.documentContents ? window.documentContents[docName] || 'Content not available' : 'Content not available';
    }
    
    function getSourceRelevance(sourceIndex) {
        // This will be populated by Streamlit with actual relevance scores
        return window.sourceRelevances ? window.sourceRelevances[sourceIndex] || 'N/A' : 'N/A';
    }
    
    // Close modal when clicking outside
    window.onclick = function(event) {
        const modal = document.getElementById('docModal');
        if (event.target === modal) {
            closeDocumentPreview();
        }
    }
    </script>
    """
    
    # Render the JavaScript component (hidden)
    components.html(js_code, height=0)
    
    # Add modal CSS separately
    st.markdown("""
    <style>
    /* Document preview modal */
    .doc-modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
    }
    
    .doc-modal-content {
        background-color: var(--background-color, #fefefe);
        margin: 5% auto;
        padding: 20px;
        border: 1px solid var(--border-color, #888);
        border-radius: 10px;
        width: 80%;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
    }
    
    .doc-modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border-color, #ddd);
    }
    
    .doc-modal-close {
        color: var(--text-color, #aaa);
        font-size: 28px;
        font-weight: bold;
        cursor: pointer;
    }
    
    .doc-modal-close:hover {
        color: var(--text-color, #000);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Chat input form
    st.markdown("---")
    
    # Form-based input that responds to Enter key
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "💬 Ask a question about your documents...",
                placeholder="Type your question and press Enter...",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.form_submit_button("Send", type="primary", use_container_width=True)
        
        # Process input when form is submitted (Enter key or button click)
        if send_button and user_input.strip():
            process_user_query(user_input.strip())
            st.rerun()


def process_user_query(query: str) -> None:
    """Process user query and generate response with enhanced error handling"""
    print(f"🔍 DEBUG: Processing query: '{query}'")
    
    # Store current query for context filtering
    st.session_state.current_query = query
    
    # Add user message to chat history immediately
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query
    })
    
    # Get current models being used (define at top for all cases)
    embedding_model = "all-MiniLM-L6-v2"  # Fixed embedding model
    llm_model = st.session_state.selected_llm_model
    
    print(f"🔍 DEBUG: Using models - embedding: {embedding_model}, llm: {llm_model}")
    
    # Check if we have documents to search
    total_documents = 0
    if 'documents' in st.session_state:
        total_documents += len(st.session_state.documents)
    if 'uploaded_documents' in st.session_state:
        total_documents += len(st.session_state.uploaded_documents)
    
    print(f"🔍 DEBUG: Total documents available: {total_documents}")
    
    if total_documents == 0:
        print("❌ DEBUG: No documents available for search")
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': "I don't have any documents to search. Please upload some documents or process web URLs first using the sidebar.",
            'models_used': {
                'embedding': embedding_model,
                'llm': llm_model,
                'embedding_real': False
            }
        })
        return
    
    try:
        # Show processing status briefly
        with st.spinner("🔍 Searching documents..."):
            print("🔍 DEBUG: Getting embeddings for query...")
            
            # Get embeddings for the query
            query_embedding = st.session_state.llamastack_client.get_embeddings(query, model=embedding_model)
            
            print(f"🔍 DEBUG: Query embedding received - length: {len(query_embedding) if query_embedding else 0}")
            
            # Check if we got real embeddings (not dummy ones)
            is_real_embedding = (
                query_embedding and 
                len(query_embedding) == 384 and 
                not all(abs(x) < 0.2 for x in query_embedding[:10])  # More reliable check
            )
            
            print(f"🔍 DEBUG: Is real embedding: {is_real_embedding}")
            
            if query_embedding:
                print("🔍 DEBUG: Finding relevant chunks...")
                
                # Find relevant chunks
                relevant_chunks = find_relevant_chunks(query_embedding)
                
                print(f"🔍 DEBUG: Found {len(relevant_chunks)} relevant chunks")
                
                if relevant_chunks:
                    print("🔍 DEBUG: Generating response...")
                    
                    # Generate response using improved method
                    response = generate_improved_response(query, relevant_chunks, llm_model, embedding_model)
                    
                    print(f"🔍 DEBUG: Response generated successfully")
                    
                    # Extract sources with embedding status info (using enhanced count)
                    from .config import TOP_SOURCES_COUNT
                    sources = []
                    for chunk in relevant_chunks[:TOP_SOURCES_COUNT]:  # Configurable source count
                        sources.append({
                            'document': chunk['document'],
                            'score': chunk['similarity']
                        })
                    
                    # Add assistant response to chat history with model info
                    assistant_response = {
                        'role': 'assistant',
                        'content': response,
                        'sources': sources,
                        'models_used': {
                            'embedding': embedding_model,
                            'llm': llm_model,
                            'embedding_real': is_real_embedding
                        }
                    }
                    
                    # Add debug info if using dummy embeddings
                    if not is_real_embedding:
                        assistant_response['debug_info'] = "Using demo embeddings - consider installing docling or configuring real embedding models"
                    
                    st.session_state.chat_history.append(assistant_response)
                else:
                    print("❌ DEBUG: No relevant chunks found")
                    
                    # No relevant chunks found
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"I couldn't find relevant information about '{query}' in your uploaded documents. Try rephrasing your question or upload more relevant documents.",
                        'models_used': {
                            'embedding': embedding_model,
                            'llm': llm_model,
                            'embedding_real': is_real_embedding
                        }
                    })
            else:
                print("❌ DEBUG: Failed to get query embeddings")
                
                # Embedding generation failed
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': "I'm having trouble processing your question right now. Please check if your embedding model is working correctly.",
                    'models_used': {
                        'embedding': embedding_model,
                        'llm': llm_model,
                        'embedding_real': False
                    }
                })
    
    except Exception as e:
        print(f"❌ DEBUG: Error in process_user_query: {e}")
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
        
        # Error handling with model info
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': f"I encountered an error while processing your question: {str(e)}",
            'models_used': {
                'embedding': embedding_model,
                'llm': llm_model,
                'embedding_real': False
            }
        })
    
    # Limit chat history to prevent memory issues
    if len(st.session_state.chat_history) > 50:  # Keep last 50 messages
        st.session_state.chat_history = st.session_state.chat_history[-50:]


def find_relevant_chunks(query_embedding: List[float], top_k: int = None) -> List[Dict]:
    """Find most relevant document chunks using improved retrieval with filtering and reranking"""
    from .config import MAX_RELEVANT_CHUNKS, MIN_SIMILARITY_THRESHOLD, ENABLE_CHUNK_RERANKING
    import streamlit as st  # Import at the top to avoid scope issues
    
    if top_k is None:
        top_k = MAX_RELEVANT_CHUNKS
    
    relevant_chunks = []
    
    # Search through both regular documents and uploaded documents (including web URLs)
    all_documents = []
    
    # Add regular documents
    if 'documents' in st.session_state:
        all_documents.extend(st.session_state.documents)
        print(f"🔍 DEBUG: Found {len(st.session_state.documents)} regular documents")
    
    # Add uploaded documents (including web URLs)
    if 'uploaded_documents' in st.session_state:
        all_documents.extend(st.session_state.uploaded_documents)
        print(f"🔍 DEBUG: Found {len(st.session_state.uploaded_documents)} uploaded documents (including web URLs)")
    
    print(f"🔍 DEBUG: Total documents to search: {len(all_documents)}")
    
    # Enhanced similarity threshold based on query type
    base_threshold = MIN_SIMILARITY_THRESHOLD
    
    if 'current_query' in st.session_state:
        query = st.session_state.current_query.lower()
        # Increase threshold for specific technical queries to avoid irrelevant results
        if any(tech_term in query for tech_term in ['python', 'javascript', 'java', 'c++', 'programming', 'code']):
            base_threshold = max(base_threshold, 0.35)  # Higher threshold for technical queries
            print(f"🔍 DEBUG: Using higher threshold ({base_threshold}) for technical query")
    
    for doc_idx, doc in enumerate(all_documents):
        print(f"🔍 DEBUG: Processing document {doc_idx + 1}/{len(all_documents)}: {doc.get('name', 'Unknown')} (type: {doc.get('file_type', 'FILE')})")
        
        # Skip documents without embeddings
        if 'embeddings' not in doc or not doc['embeddings']:
            print(f"⚠️ DEBUG: Document {doc.get('name', 'Unknown')} has no embeddings, skipping")
            continue
            
        print(f"🔍 DEBUG: Document {doc.get('name', 'Unknown')} has {len(doc['embeddings'])} embeddings")
        
        for i, embedding in enumerate(doc['embeddings']):
            try:
                # Calculate cosine similarity
                similarity = cosine_similarity(query_embedding, embedding)
                print(f"🔍 DEBUG: Chunk {i} similarity: {similarity:.4f}")
                
                # Filter out chunks below similarity threshold
                if similarity >= base_threshold:
                    chunk_data = {
                        'content': doc['chunks'][i],
                        'document': doc['name'],
                        'similarity': similarity,
                        'chunk_index': i,
                        'doc_name': doc['name'],
                        'doc_type': doc.get('file_type', 'FILE'),  # Add document type for debugging
                        'source_url': doc.get('source_url', None)  # Add source URL for web content
                    }
                    relevant_chunks.append(chunk_data)
                    print(f"✅ DEBUG: Added chunk {i} from {doc.get('name', 'Unknown')} (similarity: {similarity:.4f})")
                else:
                    print(f"❌ DEBUG: Chunk {i} below threshold ({similarity:.4f} < {base_threshold})")
                    
            except Exception as e:
                print(f"❌ DEBUG: Error calculating similarity for chunk {i} in {doc.get('name', 'Unknown')}: {e}")
                continue
    
    print(f"🔍 DEBUG: Found {len(relevant_chunks)} relevant chunks total")
    
    # Sort by similarity
    relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Enhanced reranking with better diversity
    if ENABLE_CHUNK_RERANKING and len(relevant_chunks) > top_k:
        reranked_chunks = []
        used_docs = set()
        
        # First pass: get best chunk from each document (prioritize diversity)
        for chunk in relevant_chunks:
            if chunk['doc_name'] not in used_docs and len(reranked_chunks) < top_k:
                reranked_chunks.append(chunk)
                used_docs.add(chunk['doc_name'])
        
        # Second pass: fill remaining slots with highest similarity
        for chunk in relevant_chunks:
            if len(reranked_chunks) >= top_k:
                break
            if chunk not in reranked_chunks:
                reranked_chunks.append(chunk)
        
        print(f"🔍 DEBUG: Reranked to {len(reranked_chunks)} chunks from {len(used_docs)} documents")
        return reranked_chunks[:top_k]
    
    print(f"🔍 DEBUG: Returning top {min(top_k, len(relevant_chunks))} chunks")
    return relevant_chunks[:top_k]


def generate_improved_response(query: str, relevant_chunks: List[Dict], llm_model: str, embedding_model: str) -> str:
    """Generate response using relevant chunks with optimized prompting and context"""
    if not relevant_chunks:
        return "I couldn't find relevant information in your documents to answer that question."
    
    from .config import MAX_CONTEXT_LENGTH, MIN_SIMILARITY_THRESHOLD
    
    # Filter chunks by improved similarity threshold and build focused context
    high_quality_chunks = [
        chunk for chunk in relevant_chunks 
        if chunk['similarity'] >= MIN_SIMILARITY_THRESHOLD
    ]
    
    if not high_quality_chunks:
        return f"I don't have relevant information about that in the provided documents."
    
    # Additional relevance check - ensure content actually relates to the query
    query_terms = query.lower().split()
    relevant_chunks = []
    
    for chunk in high_quality_chunks:
        content_lower = chunk['content'].lower()
        # More strict relevance check - require multiple query terms to match
        matching_terms = sum(1 for term in query_terms if len(term) > 2 and term in content_lower)
        # Require at least 2 matching terms or high similarity (>0.7)
        if matching_terms >= 2 or chunk['similarity'] > 0.7:
            relevant_chunks.append(chunk)
            print(f"✅ DEBUG: Chunk passed relevance check - {matching_terms} terms match, similarity: {chunk['similarity']:.3f}")
        else:
            print(f"❌ DEBUG: Chunk failed relevance check - {matching_terms} terms match, similarity: {chunk['similarity']:.3f}")
    
    if not relevant_chunks:
        return f"I don't have relevant information about '{query}' in the provided documents. Please try rephrasing your question or upload relevant documents."
    
    # Sort by similarity and take only the most relevant chunks
    relevant_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    relevant_chunks = relevant_chunks[:2]  # Using a smaller number of chunks for more focused context

    context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
    
    # Add a debug log to show the final retrieved context
    logger.info(f"Retrieved context for query '{query}':\n{context}")

    # Ensure context does not exceed max length
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]
    
    # Optimized system prompt for concise responses
    system_prompt = f"""You are a helpful AI assistant. Your task is to answer a question based on the provided context.

Follow these rules precisely:
1.  **Be Concise:** Provide a direct and concise answer. Avoid unnecessary details unless the user asks for a detailed explanation.
2.  **Stay on Topic:** Only use information from the provided 'DOCUMENT CONTEXT'.
3.  **No Echoing:** Do not repeat the question or the context in your answer.
4.  **No Prefaces:** Do not start your answer with phrases like "Based on the document..." or "The context says...".
5.  **Handle Irrelevance:** If the context does not contain the answer, simply state: "I don't have relevant information about that in the provided documents."

The user will provide the context and the question. You must provide only the answer."""

    user_prompt = f"""Based on the following context, answer the question.

DOCUMENT CONTEXT:
---
{context}
---

QUESTION: {query}

ANSWER:"""
    
    # Try LlamaStack with optimized parameters, with a fallback to Ollama
    response = ""
    try:
        response = st.session_state.llamastack_client.chat_completion(
            user_prompt,
            system_prompt=system_prompt,
            model=llm_model
        )
    except Exception as e:
        print(f"LlamaStack completion failed: {e}")
        # Try Ollama as a fallback if LlamaStack fails completely
        try:
            response = try_ollama_completion_optimized(query, context, llm_model)
        except Exception as e2:
            print(f"Ollama fallback failed: {e2}")

    # If we got a response from either, clean and return it
    if response and response.strip():
        response = response.strip()

        # Prioritize the text before any conversational turn marker like 'User:' or 'Question:'
        primary_answer = response
        next_turn_markers = [r'user:', r'question:']
        for marker in next_turn_markers:
            match = re.search(marker, primary_answer, flags=re.IGNORECASE)
            if match:
                primary_answer = primary_answer[:match.start()].strip()
        
        # Now, clean up prefixes like 'Assistant:' or 'Answer:' from the extracted primary answer
        primary_answer_lower = primary_answer.lower()
        match = re.search(r'(assistant|answer):', primary_answer_lower)
        if match:
            primary_answer = primary_answer[match.end():].strip()

        response = primary_answer

        # Replace any remaining markers with emojis for display
        response = re.sub(r"user:", "👤", response, flags=re.IGNORECASE)
        response = re.sub(r"question:", "❓", response, flags=re.IGNORECASE)
        response = re.sub(r"assistant:", "🤖", response, flags=re.IGNORECASE)
        response = re.sub(r"answer:", "💡", response, flags=re.IGNORECASE)

        if len(response.strip()) > 0:
            return response.strip()

    # If all attempts failed or returned empty, generate content-based response
    print(f"⚠️ DEBUG: All LLM attempts failed or returned empty. Using content-based fallback.")
    return generate_enhanced_content_response(query, relevant_chunks)


def try_ollama_completion(query: str, context: str, model: str) -> str:
    """Try Ollama completion with clean prompting"""
    import subprocess
    
    try:
        prompt = f"""Based on these document excerpts, answer the question:
        
Question: {query}

Documents:
{context[:2000]}

Answer based only on the provided documents:"""
        
        result = subprocess.run(
            ["ollama", "generate", model, prompt],
            capture_output=True,
            text=True,
            timeout=45
        )
        
        if result.returncode == 0 and result.stdout.strip():
            response = result.stdout.strip()
            # Clean up any prompt echoing
            if "Answer based only on" in response:
                response = response.split("Answer based only on")[-1].strip()
            if len(response) > 50:
                return response
        
    except subprocess.TimeoutExpired:
        print(f"Ollama generation timed out")
    except Exception as e:
        print(f"Ollama error: {e}")
    
    return None


def generate_content_based_response(query: str, chunks: List[Dict]) -> str:
    """Generate a content-based response when LLM fails"""
    if not chunks:
        return "I couldn't find relevant information in your documents to answer that question."
    
    # Simple content extraction
    top_chunk = chunks[0]
    doc_name = top_chunk['document']
    content = top_chunk['content'][:500]  # First 500 chars
    similarity = top_chunk['similarity']
    
    response = f"""Based on your document "{doc_name}" (relevance: {similarity:.1f}):

{content}

This excerpt appears most relevant to your question about: "{query}"

Note: This is a basic content match. For better responses, ensure your LLM models are properly configured."""
    
    return response


def try_ollama_completion_optimized(query: str, context: str, model: str) -> str:
    """Try Ollama completion with optimized prompting and parameters"""
    import subprocess
    
    try:
        from .config import LLM_TEMPERATURE, LLM_MAX_TOKENS
        
        # Focused prompt for Ollama - no repetition
        prompt = (
            f"System: You are an AI assistant for RAG LlamaStack. Answer the user's question based *only* on the provided context. "
            f"If the context doesn't directly answer the question, but has related information, summarize what you can from the context. "
            f"If the context is completely irrelevant, then say you can't answer. Be concise.\n\n"
            f"Context: {context}\n\n"
            f"User: {query}\n\n"
            f"Assistant:"
        )
        
        # Using subprocess to call ollama directly for potentially better performance
        command = [
            "ollama", "generate", model,
            "--temperature", str(LLM_TEMPERATURE),
            "--num-predict", str(min(LLM_MAX_TOKENS, 800)),  # Ollama limit
            prompt
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=45
        )
        
        if result.returncode == 0 and result.stdout.strip():
            response = result.stdout.strip()
            
            # Prioritize the text before any conversational turn marker like 'User:' or 'Question:'
            primary_answer = response
            next_turn_markers = [r'user:', r'question:']
            for marker in next_turn_markers:
                match = re.search(marker, primary_answer, flags=re.IGNORECASE)
                if match:
                    primary_answer = primary_answer[:match.start()].strip()
            
            # Now, clean up prefixes like 'Assistant:' or 'Answer:' from the extracted primary answer
            primary_answer_lower = primary_answer.lower()
            match = re.search(r'(assistant|answer):', primary_answer_lower)
            if match:
                primary_answer = primary_answer[match.end():].strip()

            response = primary_answer

            # Replace any remaining markers with emojis for display
            response = re.sub(r"user:", "👤", response, flags=re.IGNORECASE)
            response = re.sub(r"question:", "❓", response, flags=re.IGNORECASE)
            response = re.sub(r"assistant:", "🤖", response, flags=re.IGNORECASE)
            response = re.sub(r"answer:", "💡", response, flags=re.IGNORECASE)

            if len(response) > 20:
                return response
        
    except subprocess.TimeoutExpired:
        print(f"Ollama generation timed out")
    except Exception as e:
        print(f"Ollama error: {e}")
    
    return None


def generate_enhanced_content_response(query: str, chunks: List[Dict]) -> str:
    """Generate enhanced content-based response when LLM fails"""
    if not chunks:
        return "I couldn't find relevant information in your documents to answer that question."
    
    # Get best matches with higher relevance threshold
    high_relevance_chunks = [chunk for chunk in chunks if chunk['similarity'] > 0.3]
    if not high_relevance_chunks:
        high_relevance_chunks = chunks[:2]  # Fallback to top 2
    
    # Build structured response
    response_parts = []
    
    # Key information extraction
    response_parts.append("### 📋 Key Information Found:")
    response_parts.append("")
    
    for i, chunk in enumerate(high_relevance_chunks[:3], 1):
        doc_name = chunk['document']
        content = chunk['content']
        similarity = chunk['similarity']
        
        # Extract key sentences/points from content
        sentences = content.split('. ')
        key_sentences = []
        
        # Look for sentences containing query terms
        query_words = query.lower().split()
        for sentence in sentences:
            sentence_lower = sentence.lower()
            word_matches = sum(1 for word in query_words if word in sentence_lower)
            if word_matches > 0 and len(sentence.strip()) > 30:
                key_sentences.append(sentence.strip())
        
        # If no matches, take first few sentences
        if not key_sentences:
            key_sentences = sentences[:3]
        
        # Format the response section
        response_parts.append(f"**{i}. From: {doc_name}** (Relevance: {similarity:.0%})")
        response_parts.append("")
        
        # Add key points
        for j, sentence in enumerate(key_sentences[:3]):
            if sentence.strip():
                response_parts.append(f"- {sentence.strip()}")
        
        response_parts.append("")
    
    # Summary section
    avg_similarity = sum(chunk['similarity'] for chunk in high_relevance_chunks[:3]) / len(high_relevance_chunks[:3])
    
    response_parts.append("### 📊 Response Summary:")
    response_parts.append("")
    response_parts.append(f"- **Sources:** {len(high_relevance_chunks[:3])} documents")
    response_parts.append(f"- **Relevance:** {avg_similarity:.0%} average match")
    response_parts.append(f"- **Method:** Content extraction (AI model unavailable)")
    response_parts.append("")
    
    # Helpful suggestions
    if avg_similarity < 0.4:
        response_parts.append("### 💡 Suggestions:")
        response_parts.append("")
        response_parts.append("- Try rephrasing your question with different keywords")
        response_parts.append("- Upload more relevant documents")
        response_parts.append("- Check if your LLM model is properly configured for AI-powered responses")
    else:
        response_parts.append("### ⚡ For Better Results:")
        response_parts.append("")
        response_parts.append("- **Enable AI processing:** Configure your LLM model for intelligent responses")
        response_parts.append("- **Current status:** Using basic content matching")
    
    return "\n".join(response_parts)


# End of functions for chat processing


def clear_chat_history() -> None:
    """Clear the chat history"""
    st.session_state.chat_history = []


def get_chat_history_count() -> int:
    """Get the number of messages in chat history"""
    return len(st.session_state.chat_history)


def export_chat_history() -> str:
    """Export chat history as formatted text"""
    if not st.session_state.chat_history:
        return "No chat history to export."
    
    export_text = "# Chat History Export\n\n"
    
    for i, message in enumerate(st.session_state.chat_history):
        role = "User" if message['role'] == 'user' else "Assistant"
        export_text += f"## {role} (Message {i+1})\n\n"
        export_text += f"{message['content']}\n\n"
        
        if 'sources' in message:
            export_text += "**Sources:**\n"
            for source in message['sources']:
                export_text += f"- {source['document']} (Relevance: {source['score']:.2f})\n"
            export_text += "\n"
        
        export_text += "---\n\n"
    
    return export_text 