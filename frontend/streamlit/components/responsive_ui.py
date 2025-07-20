"""
Responsive UI Components using Tailwind CSS
Modern, mobile-first design components for RAG LlamaStack
"""

import streamlit as st
from typing import Dict, List, Optional, Any


class ResponsiveUI:
    """Responsive UI components using Tailwind CSS classes"""
    
    @staticmethod
    def responsive_header(
        title: str,
        subtitle: str,
        status_text: str = "Connected",
        status_type: str = "success"
    ) -> None:
        """Create a responsive header with gradient background"""
        
        status_classes = {
            "success": "bg-green-100 text-green-800",
            "warning": "bg-yellow-100 text-yellow-800", 
            "error": "bg-red-100 text-red-800",
            "info": "bg-blue-100 text-blue-800"
        }
        
        header_html = f"""
        <header class="bg-gradient-to-r from-purple-500 to-purple-700 text-white shadow-lg rounded-xl mb-8">
            <div class="container mx-auto px-4 py-6">
                <div class="flex flex-col md:flex-row md:items-center md:justify-between">
                    <div class="mb-4 md:mb-0">
                        <h1 class="text-2xl md:text-3xl font-bold">{title}</h1>
                        <p class="text-purple-100 mt-1">{subtitle}</p>
                    </div>
                    <div class="flex items-center space-x-4">
                        <div class="px-3 py-1 rounded-full text-xs font-semibold {status_classes.get(status_type, status_classes['info'])}">
                            🟢 {status_text}
                        </div>
                    </div>
                </div>
            </div>
        </header>
        """
        
        st.markdown(header_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_grid(
        items: List[Dict[str, Any]], 
        columns: Dict[str, int] = None
    ) -> None:
        """Create a responsive grid layout"""
        
        if columns is None:
            columns = {"sm": 1, "md": 2, "lg": 3, "xl": 4}
        
        grid_classes = f"grid grid-cols-{columns['sm']} md:grid-cols-{columns['md']} lg:grid-cols-{columns['lg']} xl:grid-cols-{columns['xl']} gap-6 mb-8"
        
        grid_html = f'<div class="{grid_classes}">'
        
        for item in items:
            card_html = f"""
            <div class="card hover-lift">
                <div class="text-center">
                    <div class="text-3xl mb-3">{item.get('icon', '📄')}</div>
                    <h3 class="font-semibold text-gray-800 mb-2">{item.get('title', 'Title')}</h3>
                    <p class="text-sm text-gray-600">{item.get('description', 'Description')}</p>
                    {f'<div class="mt-4"><span class="metric-value text-2xl">{item["value"]}</span></div>' if item.get('value') else ''}
                </div>
            </div>
            """
            grid_html += card_html
        
        grid_html += '</div>'
        
        st.markdown(grid_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_sidebar_card(
        title: str,
        content: Dict[str, Any],
        card_type: str = "default"
    ) -> None:
        """Create responsive sidebar cards"""
        
        card_classes = {
            "default": "card",
            "gradient": "card-gradient",
            "stats": "card bg-gradient-to-br from-purple-50 to-purple-100 border-l-4 border-purple-500"
        }
        
        card_html = f"""
        <div class="{card_classes.get(card_type, card_classes['default'])} mb-6">
            <h3 class="text-lg font-semibold mb-4 {'text-white' if card_type == 'gradient' else 'text-gray-800'}">{title}</h3>
            <div class="space-y-3">
        """
        
        for key, value in content.items():
            if isinstance(value, dict):
                # Handle status items
                status_class = f"status-{value.get('status', 'info')}"
                card_html += f"""
                <div class="flex justify-between items-center">
                    <span class="text-sm {'text-purple-100' if card_type == 'gradient' else 'text-gray-600'}">{key}</span>
                    <div class="status-badge {status_class}">{value.get('text', 'Unknown')}</div>
                </div>
                """
            else:
                # Handle simple key-value pairs
                card_html += f"""
                <div class="flex justify-between">
                    <span class="text-sm {'text-purple-100' if card_type == 'gradient' else 'text-gray-600'}">{key}</span>
                    <span class="font-semibold {'text-white' if card_type == 'gradient' else 'text-purple-600'}">{value}</span>
                </div>
                """
        
        card_html += """
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_chat_interface(
        messages: List[Dict[str, str]] = None,
        height: str = "h-96"
    ) -> None:
        """Create a responsive chat interface"""
        
        if messages is None:
            messages = [
                {"role": "assistant", "content": "Hello! I'm your AI assistant. Upload some documents and I'll help you analyze them."},
                {"role": "user", "content": "What can you help me with?"}
            ]
        
        chat_html = f"""
        <div class="chat-interface mb-8">
            <div class="chat-header">
                <h2 class="text-xl font-semibold">💬 AI Assistant</h2>
                <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span class="text-sm">Online</span>
                </div>
            </div>
            
            <div class="chat-messages {height} scrollbar-thin">
        """
        
        for message in messages:
            if message["role"] == "assistant":
                chat_html += f"""
                <div class="flex items-start space-x-3 mb-4">
                    <div class="message-avatar message-avatar-bot">AI</div>
                    <div class="message-bot">
                        <p>{message["content"]}</p>
                    </div>
                </div>
                """
            else:
                chat_html += f"""
                <div class="flex items-start space-x-3 mb-4 justify-end">
                    <div class="message-user">
                        <p>{message["content"]}</p>
                    </div>
                    <div class="message-avatar message-avatar-user">You</div>
                </div>
                """
        
        chat_html += """
            </div>
            
            <div class="chat-input">
                <input type="text" placeholder="Type your message..." class="focus-purple" id="chat-input">
                <button class="btn-primary" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <script>
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (message) {
                // This would integrate with Streamlit's chat functionality
                console.log('Sending message:', message);
                input.value = '';
            }
        }
        
        document.getElementById('chat-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        </script>
        """
        
        st.markdown(chat_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_upload_zone(
        accepted_types: List[str] = None,
        max_size: str = "50MB"
    ) -> None:
        """Create a responsive file upload zone"""
        
        if accepted_types is None:
            accepted_types = ["PDF", "TXT", "MD", "DOCX", "PPTX"]
        
        upload_html = f"""
        <div class="card mb-6">
            <h3 class="text-lg font-semibold mb-4 text-gray-800">📄 Upload Documents</h3>
            <div class="upload-zone" id="upload-zone">
                <div class="file-icon">📁</div>
                <p class="text-gray-600 mb-2 font-semibold">Drop files here or click to browse</p>
                <p class="text-xs text-gray-500 mb-2">{", ".join(accepted_types)} files supported</p>
                <p class="text-xs text-purple-600 font-medium">Maximum size: {max_size} per file</p>
                <input type="file" id="file-input" class="hidden" multiple accept=".pdf,.txt,.md,.docx,.pptx">
            </div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const uploadZone = document.getElementById('upload-zone');
            const fileInput = document.getElementById('file-input');
            
            uploadZone.addEventListener('click', () => fileInput.click());
            
            uploadZone.addEventListener('dragover', (e) => {{
                e.preventDefault();
                uploadZone.classList.add('dragover');
            }});
            
            uploadZone.addEventListener('dragleave', (e) => {{
                e.preventDefault();
                uploadZone.classList.remove('dragover');
            }});
            
            uploadZone.addEventListener('drop', (e) => {{
                e.preventDefault();
                uploadZone.classList.remove('dragover');
                const files = e.dataTransfer.files;
                handleFiles(files);
            }});
            
            fileInput.addEventListener('change', (e) => {{
                handleFiles(e.target.files);
            }});
            
            function handleFiles(files) {{
                console.log('Files selected:', files.length);
                // This would integrate with Streamlit's file handling
            }}
        }});
        </script>
        """
        
        st.markdown(upload_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_progress_timeline(
        steps: List[Dict[str, Any]],
        current_step: int = 0
    ) -> None:
        """Create a responsive progress timeline"""
        
        timeline_html = """
        <div class="card mb-8">
            <h3 class="text-lg font-semibold mb-6 text-gray-800">📈 Processing Pipeline</h3>
            <div class="space-y-4">
        """
        
        for i, step in enumerate(steps):
            if i < current_step:
                status_class = "timeline-completed"
                status_text = "✓"
            elif i == current_step:
                status_class = "timeline-current"
                status_text = str(i + 1)
            else:
                status_class = "timeline-pending"
                status_text = str(i + 1)
            
            timeline_html += f"""
            <div class="timeline-item">
                <div class="timeline-icon {status_class}">{status_text}</div>
                <div class="timeline-content">
                    <h4 class="font-semibold text-gray-800">{step.get('title', f'Step {i+1}')}</h4>
                    <p class="text-sm text-gray-600">{step.get('description', 'Processing...')}</p>
                    {f'<p class="text-xs text-purple-600 mt-1">{step["timing"]}</p>' if step.get('timing') else ''}
                </div>
            </div>
            """
        
        timeline_html += """
            </div>
        </div>
        """
        
        st.markdown(timeline_html, unsafe_allow_html=True)
    
    @staticmethod
    def responsive_metrics_dashboard(
        metrics: Dict[str, Any],
        layout: str = "grid"
    ) -> None:
        """Create a responsive metrics dashboard"""
        
        if layout == "grid":
            layout_class = "performance-grid"
        else:
            layout_class = "flex flex-wrap gap-4"
        
        metrics_html = f"""
        <div class="mb-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-6">⚡ Performance Metrics</h2>
            <div class="{layout_class}">
        """
        
        for label, data in metrics.items():
            if isinstance(data, dict):
                value = data.get('value', '0')
                unit = data.get('unit', '')
                change = data.get('change', '')
                change_type = data.get('change_type', 'neutral')
            else:
                value = data
                unit = ''
                change = ''
                change_type = 'neutral'
            
            change_class = {
                'positive': 'text-green-600',
                'negative': 'text-red-600',
                'neutral': 'text-gray-600'
            }.get(change_type, 'text-gray-600')
            
            metrics_html += f"""
            <div class="metric-card hover-lift">
                <div class="metric-value">{value}<span class="text-lg">{unit}</span></div>
                <div class="metric-label">{label.replace('_', ' ').title()}</div>
                {f'<div class="text-sm {change_class} mt-2">{change}</div>' if change else ''}
            </div>
            """
        
        metrics_html += """
            </div>
        </div>
        """
        
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    @staticmethod
    def mobile_optimized_layout() -> str:
        """Return CSS for mobile-optimized layout"""
        
        return """
        <style>
        @media (max-width: 768px) {
            .container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            
            .chat-interface {
                margin-left: -1rem;
                margin-right: -1rem;
                border-radius: 0;
            }
            
            .chat-messages {
                height: 20rem;
            }
            
            .performance-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.75rem;
            }
            
            .metric-card {
                padding: 1rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
            
            .upload-zone {
                padding: 2rem 1rem;
            }
            
            .upload-zone::after {
                display: none;
            }
            
            .card {
                margin-bottom: 1rem;
                padding: 1rem;
            }
            
            .btn-primary, .btn-secondary, .btn-success {
                width: 100%;
                margin-bottom: 0.5rem;
            }
        }
        
        @media (max-width: 480px) {
            .performance-grid {
                grid-template-columns: 1fr;
            }
            
            .timeline-content {
                margin-left: 0.5rem;
                padding: 0.75rem;
            }
            
            .message-user, .message-bot {
                max-width: 90%;
            }
        }
        </style>
        """ 