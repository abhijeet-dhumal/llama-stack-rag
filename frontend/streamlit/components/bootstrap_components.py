"""
Bootstrap-based UI Components for RAG LlamaStack
Simplified components using Bootstrap classes and minimal custom CSS
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd


def render_bootstrap_header(title: str, subtitle: str = None):
    """Render a Bootstrap-styled header"""
    st.markdown(f"""
    <div class="container-fluid p-4 bg-primary text-white rounded mb-4">
        <h1 class="display-6 fw-bold mb-2">{title}</h1>
        {f'<p class="lead mb-0">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_model_status_card(model_name: str, status: str, details: Dict[str, Any] = None):
    """Render a Bootstrap model status card"""
    status_colors = {
        'active': 'success',
        'inactive': 'danger', 
        'loading': 'warning',
        'error': 'danger'
    }
    
    color = status_colors.get(status, 'secondary')
    icon = {
        'active': '✅',
        'inactive': '❌',
        'loading': '⏳',
        'error': '⚠️'
    }.get(status, '❓')
    
    details_html = ""
    if details:
        details_html = f"""
        <div class="mt-2">
            <small class="text-muted">
                {f'<div>Version: {details.get("version", "N/A")}</div>' if details.get("version") else ''}
                {f'<div>Size: {details.get("size", "N/A")}</div>' if details.get("size") else ''}
            </small>
        </div>
        """
    
    st.markdown(f"""
    <div class="card border-{color} border-3 mb-3 shadow-sm">
        <div class="card-body text-center p-3">
            <div class="h5 mb-2">{icon} {model_name}</div>
            <span class="badge bg-{color} rounded-pill px-3 py-1">{status.title()}</span>
            {details_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_upload_zone():
    """Render a Bootstrap-styled upload zone"""
    st.markdown("""
    <div class="border border-2 border-dashed border-primary rounded p-4 text-center bg-light mb-3">
        <div class="h4 text-muted mb-2">📁</div>
        <h5 class="text-primary">Drag & Drop Files Here</h5>
        <p class="text-muted mb-0">Max 50MB per file • PDF, TXT, MD, DOCX, PPTX</p>
    </div>
    """, unsafe_allow_html=True)


def render_document_card(doc: Dict[str, Any], index: int):
    """Render a Bootstrap document card"""
    doc_type_icon = "🌐" if doc.get('file_type') == 'WEB' else "📄"
    
    # Simple card for sidebar
    st.markdown(f"""
    <div class="mb-2 p-2 border rounded">
        <div class="d-flex justify-content-between align-items-center mb-1">
            <span class="fw-bold">{doc_type_icon} {doc['name']}</span>
            <small class="text-muted">{doc.get('file_size_mb', 0):.1f}MB</small>
        </div>
        <div class="small text-muted">
            {doc.get('chunk_count', 0)} chunks • {doc.get('processing_time', 0):.1f}s
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add remove button using Streamlit
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️", key=f"remove_doc_{index}", help="Remove document"):
            # This will be handled by the calling function
            st.session_state.remove_document_index = index


def get_embedding_quality_badge(doc: Dict[str, Any]) -> str:
    """Get embedding quality badge HTML"""
    chunk_count = doc.get('chunk_count', 0)
    embedding_errors = doc.get('embedding_errors', 0)
    
    if chunk_count == 0:
        return '<span class="badge bg-secondary">N/A</span>'
    
    quality = ((chunk_count - embedding_errors) / chunk_count) * 100
    real_count = chunk_count - embedding_errors
    
    if quality > 90:
        return f'<span class="badge bg-success">{real_count}/{chunk_count}</span>'
    elif quality > 70:
        return f'<span class="badge bg-warning text-dark">{real_count}/{chunk_count}</span>'
    else:
        return f'<span class="badge bg-danger">{real_count}/{chunk_count}</span>'


def render_metrics_grid(metrics: List[Dict[str, Any]]):
    """Render a Bootstrap metrics grid"""
    html = '<div class="row g-3 mb-3">'
    
    for metric in metrics:
        html += f"""
        <div class="col-md-3 col-sm-6">
            <div class="card text-center p-2 shadow-sm">
                <div class="h3 mb-0 text-primary fw-bold">{metric['value']}</div>
                <div class="text-muted small text-uppercase fw-semibold">{metric['label']}</div>
            </div>
        </div>
        """
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_chat_message(message: str, is_user: bool = True):
    """Render a Bootstrap chat message"""
    if is_user:
        st.markdown(f"""
        <div class="d-flex justify-content-end mb-2">
            <div class="bg-primary text-white rounded p-2" style="max-width: 75%; border-bottom-right-radius: 0.25rem;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="d-flex justify-content-start mb-2">
            <div class="bg-white border rounded p-2" style="max-width: 75%; border-bottom-left-radius: 0.25rem;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_progress_bar(progress: float, label: str = "Processing..."):
    """Render a Bootstrap progress bar"""
    st.markdown(f"""
    <div class="mb-3">
        <div class="d-flex justify-content-between align-items-center mb-1">
            <small class="text-muted">{label}</small>
            <small class="text-muted">{progress:.0%}</small>
        </div>
        <div class="progress" style="height: 8px;">
            <div class="progress-bar bg-primary" style="width: {progress:.0%}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_alert(message: str, alert_type: str = "info"):
    """Render a Bootstrap alert with dark theme support"""
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "danger": "❌"
    }
    
    icon = icons.get(alert_type, "ℹ️")
    
    # Dark theme colors
    bg_colors = {
        "info": "#0d1117",
        "success": "#0d1117", 
        "warning": "#0d1117",
        "danger": "#0d1117"
    }
    
    border_colors = {
        "info": "#0d6efd",
        "success": "#198754", 
        "warning": "#ffc107",
        "danger": "#dc3545"
    }
    
    bg_color = bg_colors.get(alert_type, "#0d1117")
    border_color = border_colors.get(alert_type, "#0d6efd")
    
    st.markdown(f"""
    <div class="alert d-flex align-items-center gap-2" role="alert" 
         style="background-color: {bg_color}; border: 1px solid {border_color}; color: #ffffff; border-radius: 0.375rem; padding: 0.75rem; margin-bottom: 0.5rem;">
        <span>{icon}</span>
        <div>{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_performance_table(data: List[Dict[str, Any]]):
    """Render a Bootstrap performance table"""
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    st.markdown("""
    <div class="table-responsive">
        <table class="table table-sm table-striped table-hover">
    """, unsafe_allow_html=True)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("</table></div>", unsafe_allow_html=True)


def render_file_uploader_with_bootstrap():
    """Render file uploader with Bootstrap styling"""
    st.markdown("""
    <div class="mb-3">
        <label class="form-label fw-bold text-white">📄 Upload Files</label>
        <div class="border border-2 border-dashed border-primary rounded p-4 text-center" style="background-color: #2b2b2b; color: #ffffff;">
            <div class="h4 mb-2" style="color: #ffffff;">📁</div>
            <h6 style="color: #0d6efd;">Choose files to upload</h6>
            <p class="small mb-0" style="color: #ffffff;">Drag and drop files here (Max 50MB per file)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'md', 'docx', 'pptx'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )


def render_url_input_with_bootstrap():
    """Render URL input with Bootstrap styling"""
    st.markdown("""
    <div class="mb-3">
        <label class="form-label fw-bold text-white">🌐 Web URLs</label>
        <div class="input-group">
            <input type="url" class="form-control" placeholder="https://example.com/article" id="url-input" style="background-color: #3b3b3b; border-color: #555555; color: #ffffff;">
            <button class="btn btn-primary" type="button" onclick="processUrl()">
                🌐 Process URL
            </button>
        </div>
        <div class="form-text text-white">Extract content from web pages in real-time</div>
    </div>
    """, unsafe_allow_html=True)


def render_document_library_with_bootstrap(documents: List[Dict[str, Any]]):
    """Render document library with Bootstrap styling"""
    if not documents:
        st.markdown("""
        <div class="text-center py-5">
            <div class="h4 text-muted mb-3">📚</div>
            <h5 class="text-muted">No documents uploaded yet</h5>
            <p class="text-muted">Upload some documents above to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Simple summary instead of metrics grid for sidebar
    total_size = sum(doc.get('file_size_mb', 0) for doc in documents)
    total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
    
    st.markdown(f"""
    <div class="mb-3 p-2 bg-light rounded">
        <div class="row text-center">
            <div class="col-3">
                <div class="fw-bold text-primary">{len(documents)}</div>
                <small class="text-muted">Docs</small>
            </div>
            <div class="col-3">
                <div class="fw-bold text-primary">{total_size:.1f}</div>
                <small class="text-muted">MB</small>
            </div>
            <div class="col-3">
                <div class="fw-bold text-primary">{total_chunks}</div>
                <small class="text-muted">Chunks</small>
            </div>
            <div class="col-3">
                <div class="fw-bold text-success">🟢</div>
                <small class="text-muted">Ready</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Document cards
    for i, doc in enumerate(documents):
        render_document_card(doc, i)
    
    # Handle remove button clicks
    if hasattr(st.session_state, 'remove_document_index'):
        index = st.session_state.remove_document_index
        if 0 <= index < len(documents):
            # Import the remove function
            try:
                from core.document_handler import remove_document
                remove_document(index)
                st.rerun()
            except ImportError:
                st.error("Could not remove document - import error")
        del st.session_state.remove_document_index


def render_loading_spinner(message: str = "Loading..."):
    """Render a Bootstrap loading spinner"""
    st.markdown(f"""
    <div class="d-flex justify-content-center align-items-center p-4">
        <div class="spinner-border text-primary me-2" role="status">
            <span class="visually-hidden">{message}</span>
        </div>
        <span class="text-muted">{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_tooltip(text: str, tooltip: str):
    """Render text with Bootstrap tooltip"""
    st.markdown(f"""
    <span class="text-muted small ms-1" data-bs-toggle="tooltip" data-bs-placement="top" title="{tooltip}">
        {text}
    </span>
    """, unsafe_allow_html=True) 

def render_bootstrap_sidebar_header(title: str, icon: str = "📋"):
    """Render a Bootstrap-styled sidebar header"""
    st.markdown(f"""
    <div class="d-flex align-items-center mb-3 p-2 bg-primary text-white rounded">
        <span class="me-2">{icon}</span>
        <h6 class="mb-0 fw-bold">{title}</h6>
    </div>
    """, unsafe_allow_html=True)

def render_bootstrap_status_card(title: str, status: str, icon: str = "🔌"):
    """Render a Bootstrap status card for sidebar"""
    status_colors = {
        "success": "success",
        "warning": "warning", 
        "danger": "danger",
        "info": "info"
    }
    
    color = status_colors.get(status, "secondary")
    
    st.markdown(f"""
    <div class="card mb-2 border-{color}">
        <div class="card-body p-2">
            <div class="d-flex align-items-center">
                <span class="me-2">{icon}</span>
                <div class="flex-grow-1">
                    <small class="text-muted">{title}</small>
                    <div class="fw-bold text-{color}">{status}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_bootstrap_system_status():
    """Render system status with Bootstrap styling"""
    render_bootstrap_sidebar_header("System Status", "🔌")
    
    # Check LlamaStack connection
    try:
        from core.utils import validate_llamastack_connection
        llamastack_status = "🟢 Online" if validate_llamastack_connection() else "🔴 Offline"
        llamastack_color = "success" if validate_llamastack_connection() else "danger"
    except:
        llamastack_status = "🔴 Error"
        llamastack_color = "danger"
    
    # Check Ollama status
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
        if result.returncode == 0:
            ollama_status = "🟢 Online"
            ollama_color = "success"
        else:
            ollama_status = "🟡 Warning"
            ollama_color = "warning"
    except Exception:
        ollama_status = "🔴 Offline"
        ollama_color = "danger"
    
    render_bootstrap_status_card("LlamaStack", llamastack_status, "🦙")
    render_bootstrap_status_card("Ollama", ollama_status, "🤖")

def render_bootstrap_content_sources_header():
    """Render content sources header with Bootstrap styling"""
    render_bootstrap_sidebar_header("Content Sources", "📁")

def render_bootstrap_file_upload_section():
    """Render file upload section with Bootstrap styling"""
    st.markdown("""
    <div class="card mb-3">
        <div class="card-header bg-light">
            <h6 class="mb-0">📄 Upload Files</h6>
        </div>
        <div class="card-body p-2">
            <div class="border border-2 border-dashed border-primary rounded p-3 text-center">
                <div class="h4 mb-2">📁</div>
                <h6 class="text-primary">Choose files to upload</h6>
                <p class="small text-muted mb-0">Drag and drop files here (Max 50MB per file)</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'txt', 'md', 'docx', 'pptx'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

def render_bootstrap_url_section():
    """Render URL input section with Bootstrap styling"""
    st.markdown("""
    <div class="card mb-3">
        <div class="card-header bg-light">
            <h6 class="mb-0">🌐 Web URLs</h6>
        </div>
        <div class="card-body p-2">
            <div class="alert alert-info d-flex align-items-center gap-2 mb-2">
                <span>ℹ️</span>
                <div>Extract content from web pages in real-time</div>
            </div>
            <div class="alert alert-info d-flex align-items-center gap-2 mb-3">
                <span>ℹ️</span>
                <div>Uses MCP server with Mozilla Readability for clean content extraction</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_bootstrap_document_summary(documents: List[Dict[str, Any]]):
    """Render document summary with Bootstrap styling"""
    if not documents:
        st.markdown("""
        <div class="card">
            <div class="card-body text-center py-4">
                <div class="h4 text-muted mb-2">📚</div>
                <h6 class="text-muted">No documents uploaded yet</h6>
                <p class="text-muted small">Upload some documents above to get started!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    total_size = sum(doc.get('file_size_mb', 0) for doc in documents)
    total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
    
    st.markdown(f"""
    <div class="card mb-3">
        <div class="card-header bg-light">
            <h6 class="mb-0">📁 Your Documents ({len(documents)})</h6>
        </div>
        <div class="card-body p-2">
            <div class="row g-2 text-center">
                <div class="col-3">
                    <div class="fw-bold text-primary">{len(documents)}</div>
                    <small class="text-muted">Docs</small>
                </div>
                <div class="col-3">
                    <div class="fw-bold text-primary">{total_size:.1f}</div>
                    <small class="text-muted">MB</small>
                </div>
                <div class="col-3">
                    <div class="fw-bold text-primary">{total_chunks}</div>
                    <small class="text-muted">Chunks</small>
                </div>
                <div class="col-3">
                    <div class="fw-bold text-success">🟢</div>
                    <small class="text-muted">Ready</small>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_bootstrap_document_list(documents: List[Dict[str, Any]]):
    """Render document list with Bootstrap styling"""
    for i, doc in enumerate(documents):
        doc_type_icon = "🌐" if doc.get('file_type') == 'WEB' else "📄"
        
        st.markdown(f"""
        <div class="card mb-2">
            <div class="card-body p-2">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <div class="fw-bold">{doc_type_icon} {doc['name']}</div>
                        <div class="small text-muted">
                            {doc.get('file_size_mb', 0):.1f}MB • {doc.get('chunk_count', 0)} chunks • {doc.get('processing_time', 0):.1f}s
                        </div>
                    </div>
                    <button class="btn btn-sm btn-outline-danger" onclick="removeDocument({i})">
                        🗑️
                    </button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_bootstrap_model_dashboard():
    """Render model dashboard with Bootstrap styling"""
    render_bootstrap_sidebar_header("Model Dashboard", "🤖")
    
    st.markdown("""
    <div class="card mb-3">
        <div class="card-body p-2">
            <div class="alert alert-info d-flex align-items-center gap-2 mb-2">
                <span>ℹ️</span>
                <div>Select your preferred LLM model</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_bootstrap_ollama_integration():
    """Render Ollama integration with Bootstrap styling"""
    render_bootstrap_sidebar_header("Ollama Integration", "🦙")
    
    st.markdown("""
    <div class="card mb-3">
        <div class="card-body p-2">
            <div class="alert alert-success d-flex align-items-center gap-2">
                <span>✅</span>
                <div>5 models available</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True) 