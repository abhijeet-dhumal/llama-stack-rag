"""
UI Components for RAG LlamaStack
"""

from .bootstrap_components import (
    render_alert,
    render_file_uploader_with_bootstrap,
    render_url_input_with_bootstrap,
    render_document_library_with_bootstrap,
    render_bootstrap_header,
    render_model_status_card,
    render_upload_zone,
    render_document_card,
    render_metrics_grid,
    render_chat_message,
    render_progress_bar,
    render_performance_table,
    render_loading_spinner,
    render_tooltip,
    # New Bootstrap sidebar components
    render_bootstrap_sidebar_header,
    render_bootstrap_status_card,
    render_bootstrap_system_status,
    render_bootstrap_content_sources_header,
    render_bootstrap_file_upload_section,
    render_bootstrap_url_section,
    render_bootstrap_document_summary,
    render_bootstrap_document_list,
    render_bootstrap_model_dashboard,
    render_bootstrap_ollama_integration
)

from .faiss_dashboard import (
    render_faiss_dashboard,
    get_vectorio_stats,
    get_faiss_index_stats
)

__all__ = [
    'render_alert',
    'render_file_uploader_with_bootstrap', 
    'render_url_input_with_bootstrap',
    'render_document_library_with_bootstrap',
    'render_bootstrap_header',
    'render_model_status_card',
    'render_upload_zone',
    'render_document_card',
    'render_metrics_grid',
    'render_chat_message',
    'render_progress_bar',
    'render_performance_table',
    'render_loading_spinner',
    'render_tooltip',
    # New Bootstrap sidebar components
    'render_bootstrap_sidebar_header',
    'render_bootstrap_status_card',
    'render_bootstrap_system_status',
    'render_bootstrap_content_sources_header',
    'render_bootstrap_file_upload_section',
    'render_bootstrap_url_section',
    'render_bootstrap_document_summary',
    'render_bootstrap_document_list',
    'render_bootstrap_model_dashboard',
    'render_bootstrap_ollama_integration',
    # FAISS Dashboard components
    'render_faiss_dashboard',
    'get_vectorio_stats',
    'get_faiss_index_stats'
]
