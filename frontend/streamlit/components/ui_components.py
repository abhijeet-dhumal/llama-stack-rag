"""
Reusable UI Components for RAG LlamaStack Application
"""

import streamlit as st
from typing import Dict, List, Optional, Any
import time


class UIComponents:
    """Collection of reusable UI components"""
    
    @staticmethod
    def load_css(css_file: str = "frontend/streamlit/assets/styles.css") -> None:
        """Load custom CSS styles"""
        try:
            with open(css_file, 'r') as f:
                css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"CSS file not found: {css_file}")
    
    @staticmethod
    def load_javascript(js_file: str = "frontend/streamlit/assets/main.js") -> None:
        """Load custom JavaScript"""
        try:
            with open(js_file, 'r') as f:
                js_content = f.read()
            st.markdown(f'<script>{js_content}</script>', unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"JavaScript file not found: {js_file}")
    
    @staticmethod
    def model_card(
        title: str, 
        model_name: str, 
        status: str = "active",
        description: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Display a model card with status and metrics"""
        
        status_colors = {
            "active": "#4CAF50",
            "inactive": "#F44336", 
            "loading": "#FF9800"
        }
        
        status_icons = {
            "active": "✅",
            "inactive": "❌",
            "loading": "🔄"
        }
        
        card_html = f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3>{title}</h3>
                <span style="background: {status_colors.get(status, '#757575')}; 
                           color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem;">
                    {status_icons.get(status, "❓")} {status.upper()}
                </span>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <strong>Model:</strong> <code>{model_name}</code>
            </div>
        """
        
        if description:
            card_html += f'<div style="margin-bottom: 0.5rem; opacity: 0.9;">{description}</div>'
        
        if metrics:
            card_html += '<div style="display: flex; gap: 1rem; margin-top: 1rem;">'
            for key, value in metrics.items():
                card_html += f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{value}</div>
                    <div style="font-size: 0.8rem; opacity: 0.8;">{key}</div>
                </div>
                """
            card_html += '</div>'
        
        card_html += '</div>'
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def progress_with_steps(
        steps: List[str], 
        current_step: int, 
        step_timings: Optional[Dict[int, float]] = None
    ) -> None:
        """Display a multi-step progress indicator"""
        
        st.markdown("### 📊 Progress")
        
        # Overall progress
        overall_progress = current_step / len(steps) if steps else 0
        st.progress(overall_progress)
        
        # Step-by-step breakdown
        for i, step in enumerate(steps):
            if i < current_step:
                # Completed step
                timing_text = ""
                if step_timings and i in step_timings:
                    timing_text = f" ({step_timings[i]:.2f}s)"
                st.markdown(f"✅ **{step}**{timing_text}")
            elif i == current_step:
                # Current step
                st.markdown(f"🔄 **{step}** _(in progress)_")
            else:
                # Pending step
                st.markdown(f"⏳ {step}")
    
    @staticmethod
    def metric_cards(metrics: Dict[str, Any], columns: int = 3) -> None:
        """Display metrics in card format"""
        
        cols = st.columns(columns)
        
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                card_html = f"""
                <div class="metric-container">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def document_preview(
        content: str, 
        title: str = "Document Preview",
        max_height: str = "400px",
        show_download: bool = True,
        filename: Optional[str] = None
    ) -> None:
        """Display document content preview with optional download"""
        
        with st.expander(f"📄 {title}", expanded=False):
            
            # Document stats
            lines = content.split('\n')
            words = len(content.split())
            chars = len(content)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Lines", len(lines))
            with col2:
                st.metric("Words", words)
            with col3:
                st.metric("Characters", chars)
            with col4:
                if show_download and filename:
                    st.download_button(
                        "📥 Download",
                        content,
                        file_name=filename,
                        mime="text/plain"
                    )
            
            # Content preview
            preview_html = f"""
            <div class="document-preview" style="max-height: {max_height};">
                {content.replace('\n', '<br>').replace(' ', '&nbsp;')}
            </div>
            """
            st.markdown(preview_html, unsafe_allow_html=True)
    
    @staticmethod
    def performance_breakdown(
        timings: Dict[str, float], 
        title: str = "Performance Breakdown"
    ) -> None:
        """Display performance metrics in a clean breakdown"""
        
        with st.expander(f"⚡ {title}", expanded=False):
            total_time = sum(timings.values())
            
            st.metric("Total Time", f"{total_time:.3f}s")
            
            st.markdown("**Step Breakdown:**")
            
            for step, timing in timings.items():
                percentage = (timing / total_time * 100) if total_time > 0 else 0
                
                # Create a visual progress bar for each step
                progress_html = f"""
                <div class="performance-item">
                    <span class="performance-label">{step}</span>
                    <span class="performance-value">{timing:.3f}s ({percentage:.1f}%)</span>
                </div>
                <div style="background: #e9ecef; border-radius: 10px; height: 4px; margin: 0.25rem 0;">
                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); 
                                height: 100%; border-radius: 10px; width: {percentage}%;"></div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
    
    @staticmethod
    def alert_message(
        message: str, 
        alert_type: str = "info", 
        dismissible: bool = False,
        icon: Optional[str] = None
    ) -> None:
        """Display styled alert messages"""
        
        icons = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️", 
            "info": "ℹ️"
        }
        
        display_icon = icon or icons.get(alert_type, "ℹ️")
        dismiss_button = '<button onclick="this.parentElement.style.display=\'none\'" style="background:none;border:none;color:white;float:right;font-size:1.2rem;cursor:pointer;">×</button>' if dismissible else ""
        
        alert_html = f"""
        <div class="st{alert_type.title()}" style="margin: 1rem 0;">
            {display_icon} {message}
            {dismiss_button}
        </div>
        """
        
        st.markdown(alert_html, unsafe_allow_html=True)
    
    @staticmethod
    def loading_spinner(message: str = "Loading...") -> None:
        """Display a loading spinner with message"""
        
        spinner_html = f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: 2rem;">
            <div style="border: 3px solid #f3f3f3; border-top: 3px solid #667eea; 
                        border-radius: 50%; width: 30px; height: 30px; 
                        animation: spin 1s linear infinite; margin-right: 1rem;"></div>
            <span style="color: #667eea; font-weight: 600;">{message}</span>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        
        st.markdown(spinner_html, unsafe_allow_html=True)
    
    @staticmethod
    def status_indicator(
        service: str, 
        status: bool, 
        details: Optional[str] = None
    ) -> None:
        """Display service status indicator"""
        
        status_color = "#4CAF50" if status else "#F44336"
        status_icon = "✅" if status else "❌"
        status_text = "Online" if status else "Offline"
        
        status_html = f"""
        <div style="display: flex; align-items: center; padding: 0.5rem; 
                    border-left: 4px solid {status_color}; background: rgba(0,0,0,0.05); 
                    border-radius: 0 8px 8px 0; margin: 0.25rem 0;">
            <span style="margin-right: 0.5rem;">{status_icon}</span>
            <strong>{service}:</strong>&nbsp;
            <span style="color: {status_color};">{status_text}</span>
            {f'<span style="margin-left: 0.5rem; opacity: 0.7;">({details})</span>' if details else ''}
        </div>
        """
        
        st.markdown(status_html, unsafe_allow_html=True)


class AnimatedComponents:
    """Components with built-in animations"""
    
    @staticmethod
    def typewriter_text(text: str, speed: float = 0.05) -> None:
        """Display text with typewriter effect"""
        placeholder = st.empty()
        displayed_text = ""
        
        for char in text:
            displayed_text += char
            placeholder.markdown(f"**{displayed_text}**")
            time.sleep(speed)
    
    @staticmethod
    def progress_animation(
        total_steps: int, 
        step_duration: float = 0.1,
        messages: Optional[List[str]] = None
    ) -> None:
        """Animated progress bar with optional step messages"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(total_steps + 1):
            progress = i / total_steps
            progress_bar.progress(progress)
            
            if messages and i < len(messages):
                status_text.text(messages[i])
            else:
                status_text.text(f"Step {i}/{total_steps}")
                
            time.sleep(step_duration)
        
        status_text.text("✅ Complete!")
    
    @staticmethod
    def fade_in_content(content_func, delay: float = 0.5) -> None:
        """Display content with fade-in effect"""
        placeholder = st.empty()
        
        # Show loading state
        with placeholder.container():
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            content_func()
            st.markdown('</div>', unsafe_allow_html=True)
            
        time.sleep(delay) 

    @staticmethod
    def file_upload_validator(
        uploaded_files: List[Any],
        max_size_mb: int = 50,
        allowed_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate uploaded files for size and type restrictions"""
        
        def format_file_size(size_bytes: int) -> str:
            """Convert bytes to human readable format"""
            if size_bytes == 0:
                return "0B"
            
            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            
            return f"{size_bytes:.1f}{size_names[i]}"
        
        max_size_bytes = max_size_mb * 1024 * 1024
        valid_files = []
        oversized_files = []
        invalid_type_files = []
        total_size = 0
        
        if allowed_types is None:
            allowed_types = ['txt', 'pdf', 'md', 'docx', 'pptx']
        
        for uploaded_file in uploaded_files:
            file_size = uploaded_file.size
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Check file type
            if file_extension not in allowed_types:
                invalid_type_files.append((uploaded_file.name, file_extension))
                continue
            
            # Check file size
            if file_size > max_size_bytes:
                oversized_files.append((uploaded_file.name, file_size))
                continue
            
            valid_files.append(uploaded_file)
            total_size += file_size
        
        return {
            "valid_files": valid_files,
            "oversized_files": oversized_files,
            "invalid_type_files": invalid_type_files,
            "total_size": total_size,
            "total_size_formatted": format_file_size(total_size),
            "max_size_mb": max_size_mb,
            "allowed_types": allowed_types
        }
    
    @staticmethod
    def display_upload_validation(validation_result: Dict[str, Any]) -> bool:
        """Display upload validation results and return True if files are valid"""
        
        oversized = validation_result["oversized_files"]
        invalid_types = validation_result["invalid_type_files"]
        valid_files = validation_result["valid_files"]
        max_size_mb = validation_result["max_size_mb"]
        allowed_types = validation_result["allowed_types"]
        
        # Show file type errors
        if invalid_types:
            st.error("🚫 **Invalid File Types:**")
            for filename, file_type in invalid_types:
                st.error(f"• `{filename}` (.{file_type}) - **Only {', '.join(allowed_types)} files allowed**")
            st.info(f"💡 **Allowed types**: {', '.join(allowed_types).upper()}")
        
        # Show file size errors
        if oversized:
            st.error(f"🚫 **Files Too Large (Max {max_size_mb}MB):**")
            for filename, size in oversized:
                size_formatted = validation_result["total_size_formatted"] if "total_size_formatted" in validation_result else f"{size/(1024*1024):.1f}MB"
                st.error(f"• `{filename}` ({UIComponents.format_file_size(size)}) - **Exceeds {max_size_mb}MB limit**")
            st.info("💡 **Tip**: Split large documents or compress before uploading")
        
        # Show valid files summary
        if valid_files:
            st.success(f"✅ **{len(valid_files)} valid file(s)** - Total size: {validation_result['total_size_formatted']}")
            
            # Show individual file info
            with st.expander("📋 File Details", expanded=False):
                for uploaded_file in valid_files:
                    file_size = UIComponents.format_file_size(uploaded_file.size)
                    st.markdown(f"• `{uploaded_file.name}` ({file_size})")
        
        # Return whether we have valid files to process
        return len(valid_files) > 0
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Convert file size in bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}" 