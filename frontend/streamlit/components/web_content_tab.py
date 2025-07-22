import streamlit as st
import time
import re
import pandas as pd
from pathlib import Path

def render_web_content_tab():
    """Render the web content tab with proper processing and progress indicators."""
    
    # Display persistent success/warning messages
    if 'web_content_success' in st.session_state:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success(st.session_state.web_content_success)
        with col2:
            if st.button("🗑️", key="clear_web_success"):
                del st.session_state.web_content_success
                st.rerun()
    
    if 'web_content_warning' in st.session_state:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.warning(st.session_state.web_content_warning)
        with col2:
            if st.button("🗑️", key="clear_web_warning"):
                del st.session_state.web_content_warning
                st.rerun()
    
    st.markdown("### 🌐 Web Content Processing")
    st.markdown("Process web content from URLs to build your knowledge base.")
    st.markdown("---")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📊 Bulk URL Upload", "🔗 Manual URL Entry"])
    
    with tab1:
        render_bulk_url_upload()
    
    with tab2:
        render_manual_url_entry()

def render_bulk_url_upload():
    """Render bulk URL upload section."""
    
    st.markdown("#### 📊 Bulk URL Upload")
    st.markdown("Upload a CSV file with URLs to process multiple web pages at once.")
    
    # File uploader for CSV
    uploaded_csv = st.file_uploader(
        "Upload CSV file with URLs",
        type=['csv'],
        help="CSV file should have a column named 'url' or 'URL' containing the web addresses"
    )
    
    if uploaded_csv:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_csv)
            
            # Find URL column
            url_column = None
            for col in df.columns:
                if col.lower() in ['url', 'urls', 'link', 'links']:
                    url_column = col
                    break
            
            if url_column is None:
                st.error("❌ No URL column found. Please ensure your CSV has a column named 'url' or 'URL'")
                return
            
            # Extract URLs
            urls = df[url_column].dropna().tolist()
            
            if not urls:
                st.warning("⚠️ No URLs found in the CSV file")
                return
            
            # Display URLs
            st.markdown(f"#### 📋 Found {len(urls)} URLs")
            with st.expander("View URLs"):
                for i, url in enumerate(urls, 1):
                    st.write(f"{i}. {url}")
            
            # Process button
            if st.button("🚀 Process All URLs", type="primary", key="bulk_url_process_btn"):
                process_urls_with_progress(urls, "bulk_upload")
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            st.info("💡 Make sure your CSV file is properly formatted")

def render_manual_url_entry():
    """Render manual URL entry section."""
    
    st.markdown("#### 🔗 Manual URL Entry")
    st.markdown("Enter URLs manually (one per line or comma-separated)")
    
    # Text area for manual URL entry
    url_input = st.text_area(
        "Enter URLs",
        height=150,
        placeholder="Enter URLs here...\nOne per line or comma-separated\n\nExample:\nhttps://example.com\nhttps://another-example.com",
        help="Enter URLs one per line or separated by commas"
    )
    
    if url_input.strip():
        # Parse URLs
        urls = parse_urls_from_text(url_input)
        
        if not urls:
            st.warning("⚠️ No valid URLs found in the input")
            return
        
        # Display parsed URLs
        st.markdown(f"#### 📋 Found {len(urls)} URLs")
        with st.expander("View URLs"):
            for i, url in enumerate(urls, 1):
                st.write(f"{i}. {url}")
        
        # Process button
        if st.button("🚀 Process URL(s)", type="primary", key="manual_url_process_btn"):
            process_urls_with_progress(urls, "manual_entry")

def parse_urls_from_text(text):
    """Parse URLs from text input."""
    urls = []
    
    # Split by lines first
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split by commas if present
        if ',' in line:
            parts = [part.strip() for part in line.split(',')]
        else:
            parts = [line]
        
        for part in parts:
            if part:
                # Basic URL validation
                if is_valid_url(part):
                    urls.append(part)
    
    return list(set(urls))  # Remove duplicates

def is_valid_url(url):
    """Basic URL validation."""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def process_urls_with_progress(urls, source_type):
    """Process URLs with detailed progress indicators and FAISS integration."""
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🌐 Processing URLs...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        try:
            # Initialize FAISS if not exists
            if 'faiss_index' not in st.session_state:
                initialize_faiss_index()
            
            # Check for duplicate URLs
            existing_urls = set()
            if 'faiss_documents' in st.session_state:
                existing_urls = {doc.get('source_url', '') for doc in st.session_state.faiss_documents if doc.get('source_url')}
            
            # Filter out duplicates
            new_urls = [url for url in urls if url not in existing_urls]
            duplicate_urls = [url for url in urls if url in existing_urls]
            
            if duplicate_urls:
                st.warning(f"⚠️ Skipping {len(duplicate_urls)} duplicate URLs")
            
            if not new_urls:
                st.warning("⚠️ All URLs have already been processed")
                return
            
            total_urls = len(new_urls)
            processed_count = 0
            failed_count = 0
            
            with results_container:
                for i, url in enumerate(new_urls):
                    try:
                        # Update progress
                        progress = (i / total_urls)  # Use decimal (0.0-1.0) instead of percentage
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {url[:50]}... ({i+1}/{total_urls})")
                        
                        # Process URL and get content
                        content = fetch_web_content(url)
                        chunks = chunk_content(content, url)
                        
                        # Create document data for sync manager
                        doc_data = {
                            'name': f"Web Content - {url[:50]}...",
                            'file_type': '.web',
                            'file_size_mb': len(content) / (1024*1024),  # Approximate size
                            'content': content,
                            'source_url': url,
                            'domain': url.split('/')[2] if len(url.split('/')) > 2 else url,
                            'source': source_type,
                            'metadata': {
                                'original_url': url,
                                'content_length': len(content),
                                'upload_time': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        }
                        
                        # Use sync manager to add to both SQLite and FAISS
                        from core.faiss_sync_manager import faiss_sync_manager
                        user_id = st.session_state.get('user_id')
                        success = faiss_sync_manager.add_document_to_both(doc_data, chunks, user_id)
                        
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                        
                        # Show success for each URL
                        st.success(f"✅ Processed: {url} ({len(chunks)} chunks)")
                        
                    except Exception as e:
                        failed_count += 1
                        st.error(f"❌ Error processing {url}: {e}")
                
                # Final progress update
                progress_bar.progress(1.0)  # Use 1.0 instead of 100
                status_text.text("Processing complete!")
                
                # Summary
                if processed_count > 0:
                    # Store success message in session state for persistence
                    st.session_state.web_content_success = f"✅ Successfully processed {processed_count} URL{'s' if processed_count > 1 else ''}!"
                    if failed_count > 0:
                        st.session_state.web_content_warning = f"⚠️ {failed_count} URL{'s' if failed_count > 1 else ''} failed to process"
                    
                    # Show success message
                    st.success(st.session_state.web_content_success)
                    if failed_count > 0:
                        st.warning(st.session_state.web_content_warning)
                    
                    # Don't clear messages automatically - let them persist
                    # They will be cleared when the user navigates away or performs another action
                else:
                    st.error("❌ No URLs were processed successfully")
                    
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            st.info("💡 Check the URLs and try again")

def fetch_web_content(url):
    """Fetch content from web URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text if text else f"Content from {url}\n\nThis is a placeholder for web content processing."
        
    except ImportError:
        return f"Content from {url}\n\nWeb content processing requires requests and beautifulsoup4. Install with: pip install requests beautifulsoup4"
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"

def chunk_content(content, url):
    """Split web content into chunks for processing."""
    # Simple chunking by sentences/paragraphs
    chunks = []
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if len(para) > 50:  # Only chunks with substantial content
            # Further split long paragraphs
            if len(para) > 1000:
                sentences = para.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < 1000:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para)
    
    # If no chunks created, create at least one
    if not chunks:
        chunks = [content[:1000] + "..." if len(content) > 1000 else content]
    
    return chunks

def initialize_faiss_index():
    """Initialize FAISS index for vector storage."""
    try:
        import faiss
        import numpy as np
        
        # Create a simple FAISS index (you can customize this based on your needs)
        dimension = 384  # Default dimension for sentence transformers
        index = faiss.IndexFlatL2(dimension)
        
        st.session_state.faiss_index = index
        st.session_state.faiss_documents = []  # Store document metadata
        st.session_state.faiss_chunks = []     # Store chunk texts
        
        st.info("🔧 FAISS vector database initialized")
        
    except ImportError:
        st.error("❌ FAISS not available. Install with: pip install faiss-cpu")
        st.session_state.faiss_index = None
    except Exception as e:
        st.error(f"❌ Error initializing FAISS: {e}")
        st.session_state.faiss_index = None

def add_chunks_to_faiss(chunks, url):
    """Add web content chunks to FAISS vector database."""
    if st.session_state.faiss_index is None:
        st.warning("⚠️ FAISS not available - skipping vector storage")
        return
    
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for chunks
        embeddings = model.encode(chunks)
        
        # Add to FAISS index
        st.session_state.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata
        start_idx = len(st.session_state.faiss_chunks)
        for i, chunk in enumerate(chunks):
            st.session_state.faiss_chunks.append(chunk)
            st.session_state.faiss_documents.append({
                'filename': url,
                'chunk_index': start_idx + i,
                'embedding_index': start_idx + i
            })
        
        st.info(f"📊 Added {len(chunks)} chunks to FAISS vector database")
        
    except ImportError:
        st.warning("⚠️ Sentence transformers not available - install with: pip install sentence-transformers")
    except Exception as e:
        st.error(f"❌ Error adding to FAISS: {e}") 