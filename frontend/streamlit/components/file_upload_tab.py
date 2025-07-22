import streamlit as st
import time
from pathlib import Path
import io
import hashlib

def render_file_upload_tab():
    """Render the file upload tab with proper processing and progress indicators."""
    
    # Display persistent success/warning messages
    if 'file_upload_success' in st.session_state:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.success(st.session_state.file_upload_success)
        with col2:
            if st.button("🗑️", key="clear_file_success"):
                del st.session_state.file_upload_success
                st.rerun()
    
    if 'file_upload_warning' in st.session_state:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.warning(st.session_state.file_upload_warning)
        with col2:
            if st.button("🗑️", key="clear_file_warning"):
                del st.session_state.file_upload_warning
                st.rerun()
    
    st.markdown("### 📤 Upload Documents")
    st.markdown("Upload documents to build your knowledge base.")
    st.markdown("---")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'md', 'docx', 'pptx', 'csv', 'json'],
        accept_multiple_files=True,
        help="Drag and drop files here or click to browse. Supported formats: PDF, TXT, MD, DOCX, PPTX, CSV, JSON"
    )
    
    if uploaded_files:
        # Validation section
        st.markdown("#### 📋 File Validation")
        valid_files = []
        invalid_files = []
        duplicate_files = []
        
        # Get existing document names for duplicate detection
        existing_docs = set()
        if 'faiss_documents' in st.session_state:
            existing_docs = {doc.get('name', '') for doc in st.session_state.faiss_documents}
        
        for file in uploaded_files:
            if file is not None:
                # Check for duplicates
                if file.name in existing_docs:
                    duplicate_files.append(file.name)
                    st.warning(f"⚠️ {file.name} - Already uploaded")
                    continue
                
                # Check file size (max 50MB)
                file_size = getattr(file, 'size', 0) or 0
                
                if file_size > 50 * 1024 * 1024:
                    invalid_files.append((file.name, f"File too large (max 50MB)"))
                    continue
                
                # Check file type
                allowed_extensions = ['.txt', '.pdf', '.docx', '.md', '.csv', '.json', '.pptx']
                file_extension = Path(file.name).suffix.lower()
                if file_extension not in allowed_extensions:
                    invalid_files.append((file.name, f"Unsupported format"))
                    continue
                
                valid_files.append(file)
                st.success(f"✅ {file.name} - {file_size / (1024*1024):.1f}MB")
        
        # Show invalid files
        if invalid_files:
            st.markdown("#### ❌ Invalid Files")
            for file_name, reason in invalid_files:
                st.error(f"❌ {file_name}: {reason}")
        
        # Show duplicate files
        if duplicate_files:
            st.markdown("#### ⚠️ Duplicate Files")
            for file_name in duplicate_files:
                st.warning(f"⚠️ {file_name}: Already uploaded")
        
        # Processing section
        if valid_files:
            st.markdown("#### 🚀 Process Files")
            st.success(f"✅ {len(valid_files)} valid files ready for processing")
            
            if st.button("🚀 Process Documents", type="primary", key="file_upload_process_btn"):
                process_files_with_progress(valid_files)
        else:
            if duplicate_files or invalid_files:
                st.warning("⚠️ No new valid files to process")
            else:
                st.warning("⚠️ No valid files to process")
    
    # Supported file types info
    st.markdown("---")
    st.markdown("#### 📋 Supported File Types")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("• **PDF** - Documents and reports")
        st.markdown("• **DOCX** - Word documents")
        st.markdown("• **TXT** - Plain text files")
    with col2:
        st.markdown("• **MD** - Markdown files")
        st.markdown("• **CSV** - Data files")
        st.markdown("• **JSON** - Structured data")

def process_files_with_progress(files):
    """Process files with detailed progress indicators and FAISS integration."""
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 📄 Processing Documents...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        try:
            # Initialize FAISS if not exists
            if 'faiss_index' not in st.session_state:
                initialize_faiss_index()
            
            total_files = len(files)
            processed_count = 0
            failed_count = 0
            
            with results_container:
                for i, file in enumerate(files):
                    try:
                        # Update progress
                        progress = (i / total_files)  # Use decimal (0.0-1.0) instead of percentage
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {file.name}... ({i+1}/{total_files})")
                        
                        # Get file size safely
                        file_size = getattr(file, 'size', 0) or 0
                        
                        # Read file content based on type
                        content = read_file_content(file)
                        
                        # Generate file hash for deduplication
                        file_hash = generate_file_hash(file)
                        
                        # Create document data for sync manager
                        doc_data = {
                            'name': file.name,
                            'file_type': Path(file.name).suffix.lower(),
                            'file_size_mb': file_size / (1024*1024),
                            'content': content,
                            'content_hash': file_hash,
                            'source': 'file_upload',
                            'metadata': {
                                'original_filename': file.name,
                                'file_size_bytes': file_size,
                                'upload_time': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        }
                        
                        # Process content and add to both databases
                        chunks = chunk_content(content, file.name)
                        
                        # Use sync manager to add to both SQLite and FAISS
                        from core.faiss_sync_manager import faiss_sync_manager
                        user_id = st.session_state.get('user_id')
                        success = faiss_sync_manager.add_document_to_both(doc_data, chunks, user_id)
                        
                        if success:
                            processed_count += 1
                        else:
                            failed_count += 1
                        
                        # Show success for each file
                        st.success(f"✅ Processed: {file.name} ({len(chunks)} chunks)")
                        
                    except Exception as e:
                        failed_count += 1
                        st.error(f"❌ Error processing {file.name}: {e}")
                
                # Final progress update
                progress_bar.progress(1.0)  # Use 1.0 instead of 100
                status_text.text("Processing complete!")
                
                # Summary
                if processed_count > 0:
                    # Store success message in session state for persistence
                    st.session_state.file_upload_success = f"✅ Successfully processed {processed_count} document{'s' if processed_count > 1 else ''}!"
                    if failed_count > 0:
                        st.session_state.file_upload_warning = f"⚠️ {failed_count} file{'s' if failed_count > 1 else ''} failed to process"
                    
                    # Show success message
                    st.success(st.session_state.file_upload_success)
                    if failed_count > 0:
                        st.warning(st.session_state.file_upload_warning)
                    
                    # Don't clear messages automatically - let them persist
                    # They will be cleared when the user navigates away or performs another action
                else:
                    st.error("❌ No documents were processed successfully")
                    
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            st.info("💡 Check that all files are valid and try again")

def generate_file_hash(file):
    """Generate a hash for file deduplication."""
    try:
        # Read file content and generate hash
        content = file.read()
        file.seek(0)  # Reset file pointer
        return hashlib.md5(content).hexdigest()
    except Exception:
        # Fallback to filename hash if content reading fails
        return hashlib.md5(file.name.encode()).hexdigest()

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

def chunk_content(content, filename):
    """Split content into chunks for processing."""
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

def add_chunks_to_faiss(chunks, filename):
    """Add document chunks to FAISS vector database."""
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
                'filename': filename,
                'chunk_index': start_idx + i,
                'embedding_index': start_idx + i
            })
        
        st.info(f"📊 Added {len(chunks)} chunks to FAISS vector database")
        
    except ImportError:
        st.warning("⚠️ Sentence transformers not available - install with: pip install sentence-transformers")
    except Exception as e:
        st.error(f"❌ Error adding to FAISS: {e}")

def read_file_content(file):
    """Read file content based on file type."""
    try:
        if file.name.endswith(('.txt', '.md')):
            return file.read().decode('utf-8')
        elif file.name.endswith('.csv'):
            # For CSV, read as text for now
            return file.read().decode('utf-8')
        elif file.name.endswith('.json'):
            # For JSON, read as text for now
            return file.read().decode('utf-8')
        else:
            # For other file types (PDF, DOCX, PPTX), return placeholder
            return f"Content from {file.name}\n\nThis file type requires specialized processing."
    except UnicodeDecodeError:
        return f"Binary content from {file.name}\n\nThis file contains binary data that requires specialized processing."
    except Exception as e:
        return f"Error reading {file.name}: {str(e)}" 