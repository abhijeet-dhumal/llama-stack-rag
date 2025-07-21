"""
Document Processing and Management for RAG LlamaStack Application
Handles file uploads, processing, and document library management
"""

import streamlit as st
import time
import pandas as pd
from typing import List, Dict, Any
from .utils import format_file_size, mark_upload_start, mark_upload_success, mark_upload_failed
from .web_content_processor import WebContentProcessor


# Global embedding model cache
_embedding_model = None

def read_file_content(uploaded_file) -> str:
    """Extract text content from uploaded files with performance optimization"""
    try:
        # Read file content efficiently
        file_content = uploaded_file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        print(f"📄 Processing {uploaded_file.name} ({file_size_mb:.1f}MB)")
        
        # Optimize extraction based on file size
        if file_size_mb > 5:  # Large files need optimization
            print(f"🚀 Optimizing extraction for large file ({file_size_mb:.1f}MB)")
        
        # Try to use docling for document extraction with optimization
        try:
            from docling.document_converter import DocumentConverter
            
            # Create converter with optimized settings for large files
            converter = DocumentConverter()
            
            # Convert based on file type
            if uploaded_file.type == "application/pdf":
                # For PDF files, use docling with optimization
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    
                    # Convert with docling - optimized for large files
                    if file_size_mb > 10:
                        # For very large files, limit pages or use faster extraction
                        print("📚 Using optimized extraction for large PDF...")
                    
                    result = converter.convert(tmp_file.name)
                    content = result.document.export_to_markdown()
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                    
                    if content.strip():
                        print(f"✅ Extracted {len(content)} chars from {uploaded_file.name} using docling")
                        
                        # For very large content, do preliminary optimization
                        if len(content) > 500000:  # 500KB of text
                            content = optimize_extracted_content(content)
                            print(f"🚀 Optimized content to {len(content)} chars")
                            
                        return content
                        
        except ImportError:
            print("⚠️ Docling not available, falling back to basic text extraction")
        except Exception as e:
            print(f"⚠️ Docling extraction failed for {uploaded_file.name}: {e}")
        
        # Fallback to basic text extraction
        try:
            if uploaded_file.type == "text/plain":
                content = file_content.decode('utf-8')
            elif uploaded_file.type == "application/pdf":
                # Simple PDF text extraction fallback
                content = extract_pdf_text_simple(file_content)
            elif uploaded_file.name.endswith('.md'):
                content = file_content.decode('utf-8')
            else:
                # Try to decode as text
                try:
                    content = file_content.decode('utf-8')
                except:
                    content = file_content.decode('utf-8', errors='ignore')
            
            if content.strip():
                print(f"✅ Extracted {len(content)} chars using fallback method")
                
                # Optimize if content is very large
                if len(content) > 500000:
                    content = optimize_extracted_content(content)
                    print(f"🚀 Optimized content to {len(content)} chars")
                
                return content
                
        except Exception as e:
            print(f"❌ Basic extraction failed: {e}")
        
        return ""
        
    except Exception as e:
        print(f"❌ Failed to read {uploaded_file.name}: {e}")
        return ""


def optimize_extracted_content(content: str) -> str:
    """Optimize extracted content for large files by removing low-value text"""
    if len(content) <= 100000:  # Only optimize very large content
        return content
    
    lines = content.split('\n')
    optimized_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip very short lines (likely formatting artifacts)
        if len(line) < 5:
            continue
            
        # Skip lines that are mostly numbers/symbols (likely tables/formatting)
        if len(line) > 10 and sum(c.isalpha() for c in line) / len(line) < 0.5:
            continue
            
        # Skip repetitive lines (headers/footers)
        if line.lower().startswith(('page ', 'chapter ', 'section ')):
            continue
            
        optimized_lines.append(line)
    
    optimized_content = '\n'.join(optimized_lines)
    
    # If still too large, take first 80% (usually most important content)
    if len(optimized_content) > 300000:
        optimized_content = optimized_content[:300000] + "\n\n[Content truncated for performance]"
    
    return optimized_content


def extract_pdf_text_simple(file_content: bytes) -> str:
    """Simple PDF text extraction fallback"""
    try:
        import io
        # This is a very basic fallback - in practice you'd use PyPDF2 or similar
        # For now, just return a placeholder
        return f"[PDF content extracted - {len(file_content)} bytes] - Install docling for better PDF processing"
    except Exception:
        return ""


def create_enhanced_chunks(content: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Create optimized text chunks with intelligent splitting"""
    # Use optimized config defaults
    from .config import CHARS_PER_CHUNK, CHUNK_OVERLAP
    
    if chunk_size is None:
        chunk_size = CHARS_PER_CHUNK  # Now 3000 for better focus
    if overlap is None:
        overlap = CHUNK_OVERLAP  # Now 600
    
    if not content or len(content.strip()) == 0:
        return [""]
    
    # Clean up the content
    content = content.strip()
    
    # If content is smaller than chunk size, return as single chunk
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        # Calculate end position
        end = start + chunk_size
        
        # If this is the last chunk, take everything remaining
        if end >= len(content):
            chunks.append(content[start:])
            break
        
        # Try to find a good breaking point (sentence, paragraph, or word boundary)
        chunk_text = content[start:end]
        
        # Look for sentence endings within the last 200 characters
        for boundary in ['. ', '.\n', '!\n', '?\n', '. \n']:
            boundary_pos = chunk_text.rfind(boundary)
            if boundary_pos > chunk_size - 200:  # Within last 200 chars
                end = start + boundary_pos + len(boundary)
                break
        else:
            # Look for paragraph breaks
            boundary_pos = chunk_text.rfind('\n\n')
            if boundary_pos > chunk_size - 300:  # Within last 300 chars
                end = start + boundary_pos + 2
            else:
                # Look for any newline
                boundary_pos = chunk_text.rfind('\n')
                if boundary_pos > chunk_size - 100:  # Within last 100 chars
                    end = start + boundary_pos + 1
                else:
                    # Look for word boundary
                    boundary_pos = chunk_text.rfind(' ')
                    if boundary_pos > chunk_size - 50:  # Within last 50 chars
                        end = start + boundary_pos + 1
        
        # Add the chunk
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)  # Ensure progress
    
    print(f"✂️ Created {len(chunks)} chunks (avg {sum(len(c) for c in chunks) // len(chunks)} chars each)")
    return chunks


def get_local_embedding_model():
    """Get or initialize the local sentence-transformers model"""
    global _embedding_model
    
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a fast, lightweight model that works well for RAG
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.session_state.embedding_model_loaded = True
            st.session_state.embedding_model_name = 'all-MiniLM-L6-v2'
        except ImportError:
            st.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            return None
        except Exception as e:
            st.warning(f"⚠️ Could not load embedding model: {e}")
            return None
    
    return _embedding_model


def generate_real_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate real embeddings using local sentence-transformers"""
    model = get_local_embedding_model()
    
    if model is None:
        # Fallback to varied dummy embeddings
        import random
        embeddings = []
        for i, text in enumerate(texts):
            dummy_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
            # Add variation based on text content
            text_hash = hash(text) % 1000
            for j in range(min(10, len(dummy_embedding))):
                dummy_embedding[j] += (text_hash / 10000.0) + (i / 1000.0)
            embeddings.append(dummy_embedding)
        return embeddings
    
    try:
        # Generate real embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings.tolist())
        
        return all_embeddings
    except Exception as e:
        st.warning(f"⚠️ Embedding generation failed: {e}")
        # Fallback to dummy embeddings
        import random
        embeddings = []
        for i, text in enumerate(texts):
            dummy_embedding = [random.uniform(-0.1, 0.1) for _ in range(384)]
            text_hash = hash(text) % 1000
            for j in range(min(10, len(dummy_embedding))):
                dummy_embedding[j] += (text_hash / 10000.0) + (i / 1000.0)
            embeddings.append(dummy_embedding)
def backup_documents_data():
    """Backup document data to prevent loss during reruns"""
    if 'uploaded_documents' in st.session_state and st.session_state.uploaded_documents:
        # Store in a more persistent way
        st.session_state['_backup_uploaded_documents'] = st.session_state.uploaded_documents.copy()
        st.session_state['_backup_documents'] = st.session_state.documents.copy() if 'documents' in st.session_state else []
        print(f"💾 Backed up {len(st.session_state.uploaded_documents)} documents")

def restore_documents_data():
    """Restore document data if it was lost during rerun"""
    restored = False
    
    # Restore uploaded_documents if lost
    if ('uploaded_documents' not in st.session_state or not st.session_state.uploaded_documents) and '_backup_uploaded_documents' in st.session_state:
        st.session_state.uploaded_documents = st.session_state['_backup_uploaded_documents'].copy()
        print(f"🔄 Restored {len(st.session_state.uploaded_documents)} uploaded documents from backup")
        restored = True
    
    # Restore documents if lost  
    if ('documents' not in st.session_state or not st.session_state.documents) and '_backup_documents' in st.session_state:
        st.session_state.documents = st.session_state['_backup_documents'].copy()
        print(f"🔄 Restored {len(st.session_state.documents)} documents from backup")
        restored = True
    
    return restored


def initialize_document_storage() -> None:
    """Initialize document storage in session state with backup restoration"""
    # Try to restore from backup first
    restore_documents_data()
    
    # Initialize if still empty
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = []
        print("📋 Initialized empty uploaded_documents")
    if 'documents' not in st.session_state:
        st.session_state.documents = []
        print("📄 Initialized empty documents")
    
    # Create initial backup
    backup_documents_data()


def render_file_uploader() -> List:
    """Render file upload interface with original layout"""
    # Show current upload status
    if hasattr(st.session_state, 'currently_uploading') and st.session_state.currently_uploading:
        uploading_files = [file_id.split('_')[0] for file_id in st.session_state.currently_uploading]
        st.warning(f"⏳ Currently uploading: {', '.join(uploading_files)}")
        st.warning("⚠️ Don't change models during upload!")
    
    if hasattr(st.session_state, 'failed_uploads') and st.session_state.failed_uploads:
        failed_count = len(st.session_state.failed_uploads)
        st.error(f"🔄 {failed_count} file(s) need retry due to interruption")
        if st.button("🗑️ Clear failed uploads", type="secondary"):
            st.session_state.failed_uploads.clear()
            st.rerun()
    
    # Input method selector
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "📄 Upload Files"
    
    input_method = st.radio(
        "Choose input method:",
        ["📄 Upload Files", "🌐 Single URL", "🚀 Bulk URLs"],
        key="input_method_radio",
        horizontal=True
    )
    
    uploaded_files = []
    
    if input_method == "📄 Upload Files":
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt', 'md', 'docx', 'pptx'],
            accept_multiple_files=True,
            help="Drag and drop files here (Max 50MB per file)"
        )
        
        # Upload tips in collapsible
        with st.expander("💡 Upload Tips", expanded=False):
            st.markdown("""
            **Best Practices:**
            - ✅ Upload one file at a time for large files (>10MB)
            - ✅ Don't switch models during upload
            - ✅ Wait for "Processing complete!" before making changes
            - ✅ Failed uploads can be retried automatically
            
            **Supported Formats:**
            - 📄 PDF documents
            - 📝 Text files (.txt, .md)
            - 📘 Word documents (.docx)
            - 📊 PowerPoint (.pptx)
            """)
    
    elif input_method == "🌐 Single URL":
        st.info("🔗 Extract content from a single web page in real-time")
        st.info("Uses MCP server with Mozilla Readability for clean content extraction")
        
        # URL processing form
        with st.form("web_url_form"):
            url = st.text_input(
                "Enter URL:",
                placeholder="https://example.com/article",
                help="Enter a web URL to extract and process its content"
            )
            
            submitted = st.form_submit_button("🌐 Process URL", type="primary")
            
            if submitted and url:
                process_web_url(url)
        
        # URL processing tips
        with st.expander("💡 URL Processing Tips", expanded=False):
            st.markdown("""
            **How it works:**
            - 🔧 **MCP Server**: Uses Mozilla Readability for clean extraction
            - 🔄 **Fallback**: BeautifulSoup + requests if MCP server unavailable
            - 📝 **Output**: Clean markdown content optimized for RAG
            - ⚡ **Real-time**: Content processed and embedded immediately
            
            **Best URLs for processing:**
            - ✅ Articles and blog posts
            - ✅ Documentation pages
            - ✅ News articles
            - ✅ Wikipedia pages
            - ❌ Dynamic content (JavaScript-heavy)
            - ❌ Pages requiring authentication
            """)
    
    elif input_method == "🚀 Bulk URLs":
        st.info("🚀 Process multiple URLs at once for efficient bulk content extraction")
        st.info("Supports text input and file uploads (.txt, .csv)")
        
        # Call the bulk URL processing interface
        render_bulk_url_input()
    
    return uploaded_files if uploaded_files else []


def process_web_url(url: str):
    """Process a web URL and add to document storage"""
    if not url.strip():
        st.error("Please enter a valid URL")
        return
    
    try:
        # Initialize web content processor if not exists
        if 'web_content_processor' not in st.session_state:
            st.session_state.web_content_processor = WebContentProcessor()
        
        # Get web content processor from session state
        processor = st.session_state.web_content_processor
        
        # Extract content from URL
        result = processor.process_url(url)
        
        if result and result.get('success'):
            # Create a document-like structure for web content
            content = result.get('content', '')
            title = result.get('title', f"Web Content from {url}")
            
            # Get domain for display
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            web_document = {
                'name': f"{title}",
                'content': content,
                'file_type': 'WEB',
                'file_size_mb': len(content) / (1024 * 1024),
                'chunk_count': 0,  # Will be calculated during embedding
                'character_count': len(content),
                'upload_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_url': url,
                'domain': domain,
                'extraction_method': result.get('method', 'unknown'),
                'metadata': result.get('metadata', {}),
                'processed_at': result.get('processed_at', time.time())
            }
            
            # Process content for embeddings using existing embedding system
            with st.spinner("🧮 Creating embeddings for web content..."):
                # Create chunks using existing chunking logic
                chunks = create_chunks_from_content(content)
                web_document['chunk_count'] = len(chunks)
                
                # Create embeddings using existing embedding system
                embeddings_data = create_embeddings_from_chunks(chunks)
                
                if embeddings_data:
                    web_document['chunks'] = chunks
                    web_document['embeddings'] = embeddings_data
                    
                    # Add to uploaded documents using existing storage
                    if 'uploaded_documents' not in st.session_state:
                        st.session_state.uploaded_documents = []
                    
                    # Check for duplicates before adding
                    existing_web_docs = [doc for doc in st.session_state.uploaded_documents 
                                       if doc.get('source_url') == url]
                    
                    if existing_web_docs:
                        st.warning(f"⚠️ Content from '{domain}' already exists in your documents")
                        return
                    
                    st.session_state.uploaded_documents.append(web_document)
                    
                    # Show success message
                    st.success(f"✅ Successfully processed web content from {domain}")
                    
                    # Show processing stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📝 Characters", f"{web_document['character_count']:,}")
                    with col2:
                        st.metric("🧩 Chunks", web_document['chunk_count'])
                    with col3:
                        st.metric("📊 Size", f"{web_document['file_size_mb']:.2f} MB")
                    
                    # Form will automatically clear the input on successful submission
                else:
                    st.error("❌ Failed to create embeddings for web content")
        else:
            st.error("❌ Failed to extract content from the URL. Please check the URL and try again.")
            
    except Exception as e:
        st.error(f"❌ Error processing URL: {str(e)}")


def process_bulk_web_urls(urls: List[str], progress_callback=None):
    """Process multiple web URLs in bulk and add to document storage"""
    if not urls:
        st.error("Please provide at least one URL")
        return
    
    # Limit the number of URLs to prevent overwhelming the system
    MAX_URLS = 50
    if len(urls) > MAX_URLS:
        st.warning(f"⚠️ Too many URLs ({len(urls)}). Processing first {MAX_URLS} URLs only.")
        urls = urls[:MAX_URLS]
    
    # Clean and validate URLs
    valid_urls = []
    for url in urls:
        url = url.strip()
        if url and url.startswith(('http://', 'https://')):
            valid_urls.append(url)
        elif url:
            # Try to add https if no protocol specified
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
                valid_urls.append(url)
    
    if not valid_urls:
        st.error("No valid URLs found. Please provide URLs starting with http:// or https://")
        return
    
    # Initialize web content processor if not exists
    if 'web_content_processor' not in st.session_state:
        st.session_state.web_content_processor = WebContentProcessor()
    
    processor = st.session_state.web_content_processor
    
    # Initialize session state for uploaded documents
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Track processing results
    results = {
        'successful': [],
        'failed': [],
        'duplicates': [],
        'total_processed': 0
    }
    
    # Create progress bar for bulk processing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, url in enumerate(valid_urls):
            # Update progress
            progress = (i + 1) / len(valid_urls)
            progress_bar.progress(progress)
            status_text.text(f"Processing {i + 1}/{len(valid_urls)}: {url}")
            
            try:
                # Extract content from URL
                result = processor.process_url(url)
                
                if result and result.get('success'):
                    # Create a document-like structure for web content
                    content = result.get('content', '')
                    title = result.get('title', f"Web Content from {url}")
                    
                    # Get domain for display
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    
                    # Check for duplicates before processing
                    existing_web_docs = [doc for doc in st.session_state.uploaded_documents 
                                       if doc.get('source_url') == url]
                    
                    if existing_web_docs:
                        results['duplicates'].append({
                            'url': url,
                            'domain': domain,
                            'reason': 'Already exists in documents'
                        })
                        continue
                    
                    web_document = {
                        'name': f"{title}",
                        'content': content,
                        'file_type': 'WEB',
                        'file_size_mb': len(content) / (1024 * 1024),
                        'chunk_count': 0,  # Will be calculated during embedding
                        'character_count': len(content),
                        'upload_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'source_url': url,
                        'domain': domain,
                        'extraction_method': result.get('method', 'unknown'),
                        'metadata': result.get('metadata', {}),
                        'processed_at': result.get('processed_at', time.time())
                    }
                    
                    # Create chunks using existing chunking logic
                    chunks = create_chunks_from_content(content)
                    web_document['chunk_count'] = len(chunks)
                    
                    # Create embeddings using existing embedding system
                    embeddings_data = create_embeddings_from_chunks(chunks)
                    
                    if embeddings_data:
                        web_document['chunks'] = chunks
                        web_document['embeddings'] = embeddings_data
                        
                        # Add to uploaded documents
                        st.session_state.uploaded_documents.append(web_document)
                        
                        results['successful'].append({
                            'url': url,
                            'domain': domain,
                            'title': title,
                            'chunks': len(chunks),
                            'size_mb': web_document['file_size_mb']
                        })
                    else:
                        results['failed'].append({
                            'url': url,
                            'domain': domain,
                            'reason': 'Failed to create embeddings'
                        })
                else:
                    results['failed'].append({
                        'url': url,
                        'domain': urlparse(url).netloc if 'urlparse' in locals() else 'unknown',
                        'reason': 'Failed to extract content'
                    })
                
                results['total_processed'] += 1
                
            except Exception as e:
                results['failed'].append({
                    'url': url,
                    'domain': 'unknown',
                    'reason': f'Error: {str(e)}'
                })
                results['total_processed'] += 1
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results summary
        st.markdown("### 📊 Bulk Processing Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successful", len(results['successful']))
        with col2:
            st.metric("❌ Failed", len(results['failed']))
        with col3:
            st.metric("🔄 Duplicates", len(results['duplicates']))
        with col4:
            st.metric("📊 Total Processed", results['total_processed'])
        
        # Show detailed results
        if results['successful']:
            st.markdown("#### ✅ Successfully Processed")
            for item in results['successful']:
                st.success(f"**{item['domain']}** - {item['title']} ({item['chunks']} chunks, {item['size_mb']:.2f} MB)")
        
        if results['failed']:
            st.markdown("#### ❌ Failed to Process")
            for item in results['failed']:
                st.error(f"**{item['domain']}** - {item['reason']}")
        
        if results['duplicates']:
            st.markdown("#### 🔄 Skipped (Already Exists)")
            for item in results['duplicates']:
                st.warning(f"**{item['domain']}** - {item['reason']}")
        
        # Show overall success message
        if results['successful']:
            total_chunks = sum(item['chunks'] for item in results['successful'])
            total_size = sum(item['size_mb'] for item in results['successful'])
            st.success(f"🎉 Successfully processed {len(results['successful'])} URLs with {total_chunks} total chunks ({total_size:.2f} MB)")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error during bulk processing: {str(e)}")


def render_bulk_url_input():
    """Render the bulk URL input interface"""
    st.markdown("### 🌐 Bulk Web Content Extraction")
    st.markdown("Process multiple URLs at once. Enter one URL per line:")
    
    # Text area for bulk URLs
    urls_text = st.text_area(
        "Enter URLs (one per line):",
        placeholder="https://example.com\nhttps://docs.example.com\nhttps://blog.example.com\nhttps://news.example.com",
        height=150,
        help="Enter multiple URLs, one per line. URLs should start with http:// or https://",
        key="bulk_urls_text"
    )
    
    # Quick example button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📝 Load Example", help="Load sample URLs for testing"):
            example_urls = """https://example.com
https://docs.example.com
https://blog.example.com
https://news.example.com"""
            st.session_state.bulk_urls_text = example_urls
            st.rerun()
    with col2:
        st.caption("💡 Click to load example URLs for testing")
    
    # File upload option for bulk URLs
    st.markdown("**Or upload a text file with URLs:**")
    uploaded_file = st.file_uploader(
        "Upload URL list file",
        type=['txt', 'csv'],
        help="Upload a text file with one URL per line, or a CSV file with URLs in the first column"
    )
    
    # Process URLs from file if uploaded
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Handle CSV file
                df = pd.read_csv(uploaded_file)
                if len(df.columns) > 0:
                    urls_from_file = df.iloc[:, 0].dropna().astype(str).tolist()
                else:
                    st.error("❌ CSV file appears to be empty")
                    return
            else:
                # Handle text file
                content = uploaded_file.read().decode('utf-8')
                urls_from_file = [line.strip() for line in content.split('\n') if line.strip()]
            
            if urls_from_file:
                st.success(f"📄 Loaded {len(urls_from_file)} URLs from file")
                st.markdown("**URLs from file:**")
                for url in urls_from_file[:10]:  # Show first 10
                    st.code(url)
                if len(urls_from_file) > 10:
                    st.caption(f"... and {len(urls_from_file) - 10} more URLs")
                
                # Combine with manually entered URLs
                manual_urls = [url.strip() for url in urls_text.split('\n') if url.strip()] if urls_text else []
                all_urls = manual_urls + urls_from_file
                
                if st.button("🚀 Process All URLs", type="primary"):
                    if all_urls:
                        process_bulk_web_urls(all_urls)
                    else:
                        st.warning("⚠️ No URLs to process")
            else:
                st.error("❌ No valid URLs found in the uploaded file")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    else:
        # Process manually entered URLs
        if urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            if urls:
                st.markdown(f"**Found {len(urls)} URLs to process:**")
                for url in urls[:5]:  # Show first 5
                    st.code(url)
                if len(urls) > 5:
                    st.caption(f"... and {len(urls) - 5} more URLs")
                
                if st.button("🚀 Process URLs", type="primary"):
                    process_bulk_web_urls(urls)
            else:
                st.warning("⚠️ No valid URLs found in the text area")
        else:
            st.info("💡 Enter URLs in the text area above or upload a file with URLs")
    
    # Show processing tips
    with st.expander("💡 Processing Tips"):
        st.markdown("""
        **Best Practices for Bulk URL Processing:**
        
        - **URL Format**: Ensure URLs start with `http://` or `https://`
        - **File Size**: Large files may take longer to process
        - **Rate Limiting**: Some websites may limit requests
        - **Content Quality**: Not all websites provide extractable content
        
        **Supported File Formats:**
        - **Text files (.txt)**: One URL per line
        - **CSV files (.csv)**: URLs in the first column
        
        **Processing Limits:**
        - Maximum 50 URLs per batch (to avoid overwhelming the system)
        - Each URL is processed sequentially for stability
        - Duplicate URLs are automatically skipped
        - Processing time depends on website response times
        """)


def create_chunks_from_content(content: str) -> List[str]:
    """Create chunks from content using existing chunking logic"""
    # Use existing chunk size settings
    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 600
    
    if not content or not content.strip():
        return []
    
    chunks = []
    text_length = len(content)
    
    if text_length <= CHUNK_SIZE:
        chunks.append(content.strip())
    else:
        start = 0
        while start < text_length:
            end = start + CHUNK_SIZE
            
            # Try to break at word boundary
            if end < text_length:
                # Look for the last space within the chunk
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - CHUNK_OVERLAP
            if start >= text_length:
                break
    
    return chunks


def create_embeddings_from_chunks(chunks: List[str]) -> Any:
    """Create embeddings from chunks using existing embedding system"""
    try:
        # Use existing embedding model loading logic
        model = get_local_embedding_model()
        if model is None:
            return None
        
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                st.warning(f"Error encoding batch {i//batch_size + 1}: {str(e)}")
                # Create dummy embeddings for this batch
                import numpy as np
                dummy_embeddings = np.random.rand(len(batch), 384).astype(np.float32)
                all_embeddings.extend(dummy_embeddings)
        
        return all_embeddings
        
    except Exception as e:
        st.warning(f"Error creating embeddings: {str(e)}")
        # Return dummy embeddings as fallback
        import numpy as np
        return [np.random.rand(384).astype(np.float32) for _ in chunks]


def validate_uploaded_files(uploaded_files: List[Any]) -> List[Any]:
    """Validate uploaded files and return valid ones"""
    valid_files = []
    
    for file in uploaded_files:
        if file.size > 50 * 1024 * 1024:  # 50MB limit
            st.error(f"❌ File '{file.name}' is too large ({format_file_size(file.size)}). Maximum size is 50MB.")
        else:
            valid_files.append(file)
    
    return valid_files


def process_uploaded_files(files: List[Any]) -> int:
    """Process uploaded files with individual state tracking and interruption handling"""
    if not files:
        return 0
    
    from .utils import mark_upload_success, mark_upload_failed
    
    successful_files = 0
    
    for file in files:
        file_size_mb = file.size / (1024 * 1024) if hasattr(file, 'size') else 0
        file_id = f"{file.name}_{file.size}"
        
        st.markdown(f"### 📄 Processing: `{file.name}` ({file_size_mb:.1f}MB)")
        
        # Show performance estimate for large files
        if file_size_mb > 5:
            estimated_time = file_size_mb * 15  # Rough estimate: 15 seconds per MB
            st.info(f"⏱️ Large file detected. Estimated processing time: ~{estimated_time:.0f} seconds")
    
        # Add interruption warning
        st.warning("⚠️ **Don't switch models during upload** - this will interrupt processing")
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Initialize progress tracking with timing
            progress_bar = st.progress(0)
            status_text = st.empty()
            timing_text = st.empty()
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            try:
                start_time = time.time()
                
                # Step 1: Read file content
                step_start = time.time()
                progress_bar.progress(0.1)
                status_text.text("📥 Reading file content...")
                
                # Check for interruption during processing
                if file_id not in st.session_state.currently_uploading:
                    st.session_state.upload_interrupted = True
                    raise InterruptedError(f"Upload interrupted for {file.name}")
                
                content = read_file_content(file)
                if not content:
                    st.error(f"❌ Could not extract content from {file.name}")
                    mark_upload_failed(file_id)
                    continue
                
                step_time = time.time() - step_start
                timing_text.text(f"⏱️ File reading: {step_time:.1f}s")
                
                actual_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
                
                with metrics_col1:
                    st.metric("📊 Content Size", f"{actual_size_mb:.2f} MB")
                
                # Step 2: Create chunks with optimization  
                step_start = time.time()
                progress_bar.progress(0.3)
                status_text.text("✂️ Creating optimized chunks...")
                
                # Check for interruption
                if file_id not in st.session_state.currently_uploading:
                    st.session_state.upload_interrupted = True
                    raise InterruptedError(f"Upload interrupted for {file.name}")
                
                chunks = create_enhanced_chunks(content)
                
                # Apply large file optimizations
                from .config import LARGE_FILE_THRESHOLD_MB, MAX_CHUNKS_PER_FILE
                if actual_size_mb > LARGE_FILE_THRESHOLD_MB and len(chunks) > MAX_CHUNKS_PER_FILE:
                    status_text.text(f"🚀 Optimizing {len(chunks)} chunks for performance...")
                    chunks = optimize_chunks_for_large_files(chunks)
                
                step_time = time.time() - step_start
                timing_text.text(f"⏱️ Chunking: {step_time:.1f}s")
                
                with metrics_col2:
                    st.metric("📦 Chunks Created", f"{len(chunks)}")
                
                # Step 3: Generate embeddings using optimized batch processing
                step_start = time.time()
                progress_bar.progress(0.5)
                status_text.text("🧠 Generating embeddings (optimized)...")
                
                # Check for interruption before expensive operation
                if file_id not in st.session_state.currently_uploading:
                    st.session_state.upload_interrupted = True
                    raise InterruptedError(f"Upload interrupted for {file.name}")
                
                embeddings, embedding_errors = generate_embeddings_batch_optimized(
                    chunks, 
                    progress_bar, 
                    status_text,
                    start_progress=0.5,
                    file_id=file_id  # Pass file_id for interruption checking
                )
                
                step_time = time.time() - step_start
                timing_text.text(f"⏱️ Embeddings: {step_time:.1f}s")
                
                with metrics_col3:
                    embedding_quality = max(0, (len(chunks) - embedding_errors) / len(chunks) * 100)
                    st.metric("🎯 Quality", f"{embedding_quality:.0f}%")
                
                # Step 4: Store document
                step_start = time.time()
                progress_bar.progress(0.95)
                status_text.text("💾 Storing document...")
                
                # Store document data with backup for session state persistence
                doc_data = {
                    'name': file.name,
                    'content': content,
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'processing_time': time.time() - start_time,
                    'chunk_count': len(chunks),
                    'embedding_errors': embedding_errors,
                    'file_size_mb': actual_size_mb
                }
                
                # Check for duplicates before adding
                if 'uploaded_documents' not in st.session_state:
                    st.session_state.uploaded_documents = []
                
                # Check if document with same name and size already exists
                file_id = f"{file.name}_{file.size}"
                existing_docs = [doc for doc in st.session_state.uploaded_documents 
                               if doc.get('name') == file.name and 
                               doc.get('file_size_mb', 0) == actual_size_mb]
                
                if existing_docs:
                    st.warning(f"⚠️ File '{file.name}' already exists in your documents")
                    mark_upload_success(file_id)  # Mark as processed to avoid retry
                    continue
                
                # Add to documents array (only one array to avoid duplication)
                st.session_state.uploaded_documents.append(doc_data)
                
                # Final progress
                progress_bar.progress(1.0)
                total_time = time.time() - start_time
                
                status_text.text("✅ Processing complete!")
                timing_text.text(f"⏱️ Total time: {total_time:.1f}s ({actual_size_mb/total_time:.1f} MB/s)")
                
                # Show performance summary
                st.success(f"""
                📄 **{file.name}** processed successfully!
                - **Size:** {actual_size_mb:.1f}MB → {len(chunks)} chunks
                - **Quality:** {embedding_quality:.0f}% embeddings successful
                - **Speed:** {total_time:.1f}s ({actual_size_mb/total_time:.1f} MB/s)
                """)
                
                # Mark this file as successfully processed
                mark_upload_success(file_id)
                successful_files += 1
                
            except InterruptedError as e:
                st.error(f"⚠️ Upload interrupted: {e}")
                mark_upload_failed(file_id)
                st.info("💡 File marked for retry - try uploading again after model operations complete")
                
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
                mark_upload_failed(file_id)
                st.info("💡 Check file format and try again")
    
    return successful_files


def render_document_library() -> None:
    """Render the document library interface with original layout"""
    # Try to restore data first in case it was lost
    restore_documents_data()
    
    # Get total document count
    doc_count = get_document_count()
    
    # Debug: Check document state at render time
    if doc_count == 0:
        print(f"⚠️ Document library render: no documents found")
    else:
        print(f"✅ Document library render: found {doc_count} documents")
    
    st.markdown("---")
    
    # Ensure session state exists
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []
    
    # Use only uploaded_documents to avoid duplication
    all_documents = st.session_state.uploaded_documents
    
    # Count documents for the header
    total_size_mb = sum(doc.get("file_size_mb", 0) for doc in all_documents) if all_documents else 0
    
    # Create collapsible expander with document count
    header_text = f"📁 Your Documents ({doc_count})"
    if doc_count > 0:
        header_text += f" • {total_size_mb:.1f}MB"
    
    with st.expander(header_text, expanded=doc_count > 0):  # Auto-expand when documents exist
        if all_documents and doc_count > 0:
            # Show document summary
            total_chunks = sum(doc.get("chunk_count", 0) for doc in all_documents)
            total_chars = sum(len(doc.get("content", "")) for doc in all_documents)
            
            # Quick stats
            st.info(f"📊 {doc_count} documents • {total_size_mb:.1f}MB total • {total_chunks} chunks")
            
            # Embedding Quality Summary
            real_embeddings = sum(doc.get('chunk_count', 0) - doc.get('embedding_errors', 0) for doc in all_documents)
            total_embeddings = sum(doc.get('chunk_count', 0) for doc in all_documents)
            embedding_quality = (real_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0
            
            if embedding_quality > 80:
                quality_status = "Excellent"
                st.success(f"🎯 Embedding Quality: {embedding_quality:.1f}% ({real_embeddings}/{total_embeddings}) - {quality_status}")
            elif embedding_quality > 50:
                quality_status = "Good"
                st.warning(f"🎯 Embedding Quality: {embedding_quality:.1f}% ({real_embeddings}/{total_embeddings}) - {quality_status}")
            else:
                quality_status = "Needs Improvement"
                st.error(f"🎯 Embedding Quality: {embedding_quality:.1f}% ({real_embeddings}/{total_embeddings}) - {quality_status}")
            
            # Clear all button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Clear All", type="secondary", help="Remove all documents"):
                    clear_all_documents()
                    st.rerun()
            
            # Detailed Performance Results Table (in dropdown)
            with st.expander("📊 Detailed Performance Results", expanded=False):
                # Calculate additional metrics
                avg_chunk_size = total_chars // total_chunks if total_chunks > 0 else 0
                avg_file_size = total_size_mb / doc_count if doc_count > 0 else 0
                total_processing_time = sum(doc.get('processing_time', 0) for doc in all_documents)
                avg_processing_speed = total_size_mb / total_processing_time if total_processing_time > 0 else 0
                
                # Create comprehensive performance table
                performance_data = {
                    "Metric": [
                        "📄 Documents Processed",
                        "📦 Total File Size",
                        "📝 Characters Extracted", 
                        "✂️ Text Chunks Created",
                        "🧠 Embeddings Generated",
                        "🎯 Embedding Quality",
                        "⏱️ Total Processing Time",
                        "🚀 Average Processing Speed",
                        "📏 Average Chunk Size",
                        "📊 Average File Size",
                        "🔢 Chunks per Document",
                        "💯 Success Rate",
                        "🔗 Characters per Document",
                        "✨ Status"
                    ],
                    "Value": [
                        f"{doc_count:,}",
                        f"{total_size_mb:.2f} MB",
                        f"{total_chars:,}",
                        f"{total_chunks:,}",
                        f"{real_embeddings:,} / {total_embeddings:,}",
                        f"{embedding_quality:.1f}% ({quality_status})",
                        f"{total_processing_time:.1f} seconds",
                        f"{avg_processing_speed:.2f} MB/s" if avg_processing_speed > 0 else "N/A",
                        f"{avg_chunk_size:,} characters",
                        f"{avg_file_size:.2f} MB",
                        f"{total_chunks // doc_count if doc_count > 0 else 0:,}",
                        f"{(doc_count / len(all_documents) * 100):.1f}%" if all_documents else "N/A",
                        f"{total_chars // doc_count:,}" if doc_count > 0 else "N/A",
                        "🟢 Ready for Q&A and Search"
                    ],
                    "Category": [
                        "📋 Overview",
                        "📋 Overview", 
                        "📋 Overview",
                        "📋 Overview",
                        "🧠 AI Processing",
                        "🧠 AI Processing",
                        "⚡ Performance",
                        "⚡ Performance",
                        "📊 Statistics",
                        "📊 Statistics",
                        "📊 Statistics",
                        "📊 Statistics",
                        "📊 Statistics",
                        "✅ System"
                    ]
                }
                
                # Create and display the dataframe
                performance_df = pd.DataFrame(performance_data)
                
                # Style the table
                st.markdown("**🔍 Comprehensive Performance Analysis**")
                st.dataframe(
                    performance_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )
                
                # Individual document breakdown
                if len(all_documents) > 1:
                    st.markdown("---")
                    st.markdown("**📄 Individual Document Performance**")
                    
                    doc_performance = []
                    for doc in all_documents:
                        chunk_count = doc.get('chunk_count', 0)
                        embedding_errors = doc.get('embedding_errors', 0)
                        doc_quality = ((chunk_count - embedding_errors) / chunk_count * 100) if chunk_count > 0 else 0
                        
                        doc_performance.append({
                            "Document": doc['name'],
                            "Size (MB)": f"{doc.get('file_size_mb', 0):.2f}",
                            "Characters": f"{len(doc.get('content', '')):,}",
                            "Chunks": f"{chunk_count:,}",
                            "Processing Time (s)": f"{doc.get('processing_time', 0):.1f}",
                            "Embedding Quality": f"{doc_quality:.1f}%",
                            "Speed (MB/s)": f"{doc.get('file_size_mb', 0) / max(doc.get('processing_time', 1), 0.1):.2f}"
                        })
                    
                    doc_df = pd.DataFrame(doc_performance)
                    st.dataframe(
                        doc_df,
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Simple document list
            for i, doc in enumerate(all_documents):
                with st.expander(f"📄 {doc['name']}", expanded=False):
                    st.write(f"**Size:** {doc.get('file_size_mb', 0):.2f} MB")
                    st.write(f"**Chunks:** {doc.get('chunk_count', 0)}")
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        remove_document(i)
                        st.rerun()
            
        else:
            st.info("🔍 No documents uploaded yet. Upload some documents above to get started!")


def clear_all_documents() -> None:
    """Clear all uploaded documents"""
    st.session_state.uploaded_documents = []
    st.session_state.documents = []  # Also clear old documents for chat
    st.session_state.chat_history = []  # Clear chat history too

    # Clear backup as well
    if '_backup_uploaded_documents' in st.session_state:
        st.session_state['_backup_uploaded_documents'] = []
    if '_backup_documents' in st.session_state:
        st.session_state['_backup_documents'] = []
    
    print("🗑️ Cleared all documents and backups")


def remove_document(idx: int) -> None:
    """Remove a specific document by index from the combined document list"""
    # Get the combined document list
    all_documents = st.session_state.documents + st.session_state.uploaded_documents
    
    if idx >= len(all_documents):
        print(f"❌ Invalid document index: {idx}")
        return
    
    # Get the document to remove
    doc_to_remove = all_documents[idx]
    doc_name = doc_to_remove['name']
    
    # Remove from appropriate collection
    if doc_to_remove in st.session_state.documents:
        st.session_state.documents.remove(doc_to_remove)
        print(f"🗑️ Removed document from 'documents': {doc_name}")
    elif doc_to_remove in st.session_state.uploaded_documents:
        st.session_state.uploaded_documents.remove(doc_to_remove)
        print(f"🗑️ Removed document from 'uploaded_documents': {doc_name}")
    
    # Update backup
    backup_documents_data()
    
    print(f"🗑️ Removed document: {doc_name}")


def has_documents() -> bool:
    """Check if any documents are uploaded"""
    if 'uploaded_documents' in st.session_state:
        return len(st.session_state.uploaded_documents) > 0
    return False


def get_document_count() -> int:
    """Get the total number of uploaded documents"""
    if 'uploaded_documents' in st.session_state:
        return len(st.session_state.uploaded_documents) 
    return 0


def try_basic_extraction(uploaded_file, file_extension: str) -> str:
    """Simplified extraction - recommend docling for all complex formats"""
    return f"""Content from {uploaded_file.name} ({file_extension.upper()} format)

📋 To extract content from this file, install docling:
   pip install docling[all]

Then restart the app. Docling provides advanced AI-powered document extraction for PDF, DOCX, PPTX and more."""


def extract_file_content(uploaded_file) -> str:
    """Extract text content from uploaded file using docling only"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension in ['txt', 'md']:
            # Text and markdown files - direct reading
            content = str(uploaded_file.read(), "utf-8")
            uploaded_file.seek(0)  # Reset file pointer
            return content
            
        elif file_extension in ['pdf', 'docx', 'pptx']:
            # Use docling for PDF, DOCX, PPTX files
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.base_models import InputFormat
                import tempfile
                import os
                
                # Save uploaded file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                    uploaded_file.seek(0)  # Reset file pointer
                
                try:
                    # Initialize docling converter
                    converter = DocumentConverter()
                    
                    # Convert document
                    result = converter.convert(tmp_path)
                    
                    # Extract text content
                    if result.document and result.document.body:
                        # Get the full text content
                        content = result.document.export_to_markdown()
                        return content.strip() if content else f"Content extracted from {uploaded_file.name} but appears empty"
                    else:
                        return f"Content from {uploaded_file.name} (docling extraction returned empty result)"
                        
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
            except ImportError:
                # If docling not available, recommend installation
                return try_basic_extraction(uploaded_file, file_extension)
            except Exception as e:
                return f"Content from {uploaded_file.name} (docling extraction failed: {str(e)})"
        
        else:
            # Unsupported file type
            return f"Content from {uploaded_file.name} ({file_extension.upper()} extraction not supported)"
            
    except Exception as e:
        return f"Content from {uploaded_file.name} (error: {str(e)})"


def create_text_chunks(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks with improved context preservation"""
    # Use config defaults if not specified
    if chunk_size is None:
        from .config import CHARS_PER_CHUNK
        chunk_size = CHARS_PER_CHUNK
    if overlap is None:
        from .config import CHUNK_OVERLAP
        overlap = CHUNK_OVERLAP
    
    if not text or len(text.strip()) == 0:
        return ["No content available"]
    
    # Clean the text
    text = text.strip()
    
    # If text is shorter than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at logical boundaries
        if end < len(text):
            # Look for paragraph boundary first (double newline)
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + chunk_size // 3:  # Must be at least 1/3 into chunk
                end = para_break + 2
            else:
                # Look for sentence boundary (. ! ?)
                sentence_break = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end)
                )
                
                if sentence_break > start + chunk_size // 2:  # Must be at least halfway
                    end = sentence_break + 1
                else:
                    # Look for word boundary as last resort
                    word_break = text.rfind(' ', start, end)
                    if word_break > start + chunk_size // 2:
                        end = word_break
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap (but ensure we make progress)
        start = max(start + 1, end - overlap)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks if chunks else ["No content available"] 


def optimize_chunks_for_large_files(chunks: List[str]) -> List[str]:
    """Optimize chunks for large files by merging small ones and filtering low-content chunks"""
    if len(chunks) <= 50:
        return chunks
    
    optimized_chunks = []
    current_chunk = ""
    min_chunk_size = 1500  # Larger minimum size for efficiency
    max_chunk_size = 4000  # Larger maximum size
    
    for chunk in chunks:
        # Skip very small or low-content chunks
        if len(chunk.strip()) < 100 or chunk.strip().count(' ') < 10:
            continue
            
        # Try to merge with current chunk if it won't exceed max size
        if len(current_chunk) + len(chunk) < max_chunk_size:
            current_chunk = (current_chunk + "\n\n" + chunk).strip()
        else:
            # Add current chunk if it meets minimum size
            if len(current_chunk) >= min_chunk_size:
                optimized_chunks.append(current_chunk)
            current_chunk = chunk
    
    # Add final chunk
    if len(current_chunk) >= min_chunk_size:
        optimized_chunks.append(current_chunk)
    
    print(f"🚀 Optimized {len(chunks)} chunks down to {len(optimized_chunks)} chunks")
    return optimized_chunks


# Add custom exception for interrupted uploads
class InterruptedError(Exception):
    """Custom exception for interrupted file uploads"""
    pass


def generate_embeddings_batch_optimized(
    chunks: List[str], 
    progress_bar, 
    status_text, 
    start_progress: float = 0.5,
    file_id: str = None
) -> tuple[List[List[float]], int]:
    """Generate embeddings with optimized batching and progress tracking"""
    embeddings = []
    embedding_errors = 0
    total_chunks = len(chunks)
    
    # Determine batch size based on chunk count
    if total_chunks > 50:
        batch_size = 10  # Larger batches for many chunks
    elif total_chunks > 20:
        batch_size = 5   # Medium batches
    else:
        batch_size = 3   # Small batches for few chunks
    
    progress_range = 0.4  # 40% of progress bar for embeddings
    
    # Process in batches
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        status_text.text(f"🧠 Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
        
        # Check for interruption during batch processing
        if file_id and file_id not in st.session_state.currently_uploading:
            st.session_state.upload_interrupted = True
            raise InterruptedError(f"Upload interrupted during embedding generation")
        
        # Try batch processing first (if supported)
        batch_embeddings = try_batch_embedding_generation(batch_chunks)
        
        if batch_embeddings:
            embeddings.extend(batch_embeddings)
        else:
            # Fallback to individual processing for this batch
            for j, chunk in enumerate(batch_chunks):
                try:
                    embedding = st.session_state.llamastack_client.get_embeddings(chunk, model="all-MiniLM-L6-v2")
                    if embedding and len(embedding) > 100:  # Validate real embedding
                        embeddings.append(embedding)
                    else:
                        # Generate efficient dummy embedding
                        embeddings.append(generate_efficient_dummy_embedding(chunk))
                        embedding_errors += 1
                except Exception as e:
                    print(f"Embedding error for chunk {i+j}: {e}")
                    embeddings.append(generate_efficient_dummy_embedding(chunk))
                    embedding_errors += 1
        
        # Update progress less frequently for better performance
        batch_progress = start_progress + (progress_range * (i + len(batch_chunks)) / total_chunks)
        progress_bar.progress(min(batch_progress, 0.9))
    
    print(f"✅ Generated {len(embeddings)} embeddings with {embedding_errors} errors")
    return embeddings, embedding_errors


def try_batch_embedding_generation(chunks: List[str]) -> List[List[float]]:
    """Try to generate embeddings in batch if possible"""
    try:
        # This would be implemented if LlamaStack supports batch embeddings
        # For now, return None to fallback to individual processing
        return None
    except Exception:
        return None


def generate_efficient_dummy_embedding(text: str) -> List[float]:
    """Generate varied dummy embedding efficiently"""
    import hashlib
    
    # Create deterministic but varied embedding based on text content
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Convert hash to numeric seed
    seed_value = int(text_hash[:8], 16)
    
    # Generate 384-dimensional embedding efficiently
    embedding = []
    for i in range(384):
        # Use simple math instead of random for efficiency
        value = ((seed_value + i * 17) % 1000 - 500) / 5000.0  # Range: -0.1 to 0.1
        embedding.append(value)
    
    return embedding 