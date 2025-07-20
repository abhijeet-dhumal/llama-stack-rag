"""
Asset Manager for RAG LlamaStack Streamlit Application
Handles loading and managing static assets (CSS, JS, HTML templates)
"""

import streamlit as st
import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class AssetManager:
    """Manages static assets for the Streamlit application"""
    
    def __init__(self, base_path: str = "frontend/streamlit"):
        self.base_path = Path(base_path)
        self.assets_path = self.base_path / "assets"
        self.components_path = self.base_path / "components"
        
        # Cache for loaded assets
        self._css_cache = {}
        self._js_cache = {}
        self._html_cache = {}
    
    def load_css(self, filename: str = "styles.css") -> bool:
        """Load CSS file and inject into Streamlit"""
        css_path = self.assets_path / filename
        
        try:
            if filename not in self._css_cache:
                with open(css_path, 'r', encoding='utf-8') as f:
                    self._css_cache[filename] = f.read()
            
            st.markdown(
                f'<style>{self._css_cache[filename]}</style>', 
                unsafe_allow_html=True
            )
            logger.info(f"Loaded CSS: {filename}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"CSS file not found: {css_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading CSS {filename}: {e}")
            return False
    
    def load_javascript(self, filename: str = "main.js") -> bool:
        """Load JavaScript file and inject into Streamlit"""
        js_path = self.assets_path / filename
        
        try:
            if filename not in self._js_cache:
                with open(js_path, 'r', encoding='utf-8') as f:
                    self._js_cache[filename] = f.read()
            
            st.markdown(
                f'<script>{self._js_cache[filename]}</script>', 
                unsafe_allow_html=True
            )
            logger.info(f"Loaded JavaScript: {filename}")
            return True
            
        except FileNotFoundError:
            logger.warning(f"JavaScript file not found: {js_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading JavaScript {filename}: {e}")
            return False
    
    def load_html_template(self, filename: str = "template.html") -> Optional[str]:
        """Load HTML template file"""
        html_path = self.assets_path / filename
        
        try:
            if filename not in self._html_cache:
                with open(html_path, 'r', encoding='utf-8') as f:
                    self._html_cache[filename] = f.read()
            
            logger.info(f"Loaded HTML template: {filename}")
            return self._html_cache[filename]
            
        except FileNotFoundError:
            logger.warning(f"HTML template not found: {html_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading HTML template {filename}: {e}")
            return None
    
    def render_template(
        self, 
        template_name: str, 
        **kwargs
    ) -> Optional[str]:
        """Render HTML template with provided variables"""
        template = self.load_html_template(template_name)
        
        if template is None:
            return None
        
        try:
            # Simple template variable replacement
            for key, value in kwargs.items():
                placeholder = f"{{{{{key}}}}}"
                template = template.replace(placeholder, str(value))
            
            # Handle any remaining placeholders with empty strings
            import re
            template = re.sub(r'\{\{[^}]+\}\}', '', template)
            
            return template
            
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return None
    
    def inject_custom_html(
        self, 
        html_content: str, 
        container_id: Optional[str] = None
    ) -> None:
        """Inject custom HTML into Streamlit"""
        if container_id:
            html_content = f'<div id="{container_id}">{html_content}</div>'
        
        st.markdown(html_content, unsafe_allow_html=True)
    
    def load_all_assets(self) -> Dict[str, bool]:
        """Load essential static assets for performance"""
        results = {}
        
        # Load only main CSS for better performance
        results['css'] = self.load_css()
        
        # Skip Tailwind and template components for now to improve performance
        # results['tailwind'] = self.load_css("tailwind.css")
        # results['template_components'] = self.load_css("template-components.css")
        
        # Skip JavaScript for better initial load performance  
        # results['js'] = self.load_javascript()
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the asset cache"""
        self._css_cache.clear()
        self._js_cache.clear()
        self._html_cache.clear()
        logger.info("Asset cache cleared")
    
    def get_asset_info(self) -> Dict[str, Any]:
        """Get information about loaded assets"""
        return {
            "base_path": str(self.base_path),
            "assets_path": str(self.assets_path),
            "components_path": str(self.components_path),
            "cached_css": list(self._css_cache.keys()),
            "cached_js": list(self._js_cache.keys()),
            "cached_html": list(self._html_cache.keys())
        }


class ThemeManager:
    """Manages UI themes and color schemes"""
    
    THEMES = {
        "default": {
            "primary": "#667eea",
            "secondary": "#764ba2",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "info": "#2196F3"
        },
        "dark": {
            "primary": "#bb86fc",
            "secondary": "#3700b3",
            "success": "#03dac6",
            "warning": "#ff9800",
            "error": "#cf6679",
            "info": "#03dac6"
        },
        "ocean": {
            "primary": "#006994",
            "secondary": "#0582ca",
            "success": "#007f5f",
            "warning": "#f77f00",
            "error": "#d62828",
            "info": "#4cc9f0"
        }
    }
    
    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager
    
    def apply_theme(self, theme_name: str = "default") -> bool:
        """Apply a color theme to the application"""
        if theme_name not in self.THEMES:
            logger.warning(f"Theme '{theme_name}' not found")
            return False
        
        theme = self.THEMES[theme_name]
        
        css_vars = []
        for key, value in theme.items():
            css_vars.append(f"--{key}-color: {value};")
        
        theme_css = f"""
        <style>
        :root {{
            {' '.join(css_vars)}
        }}
        </style>
        """
        
        st.markdown(theme_css, unsafe_allow_html=True)
        logger.info(f"Applied theme: {theme_name}")
        return True
    
    def get_theme_selector(self) -> str:
        """Create a theme selector widget"""
        return st.selectbox(
            "🎨 Choose Theme",
            options=list(self.THEMES.keys()),
            format_func=lambda x: x.title()
        )


class ComponentLoader:
    """Loads and manages reusable UI components"""
    
    def __init__(self, asset_manager: AssetManager):
        self.asset_manager = asset_manager
    
    def load_components(self) -> None:
        """Import and initialize UI components"""
        try:
            # Import UI components (this will be relative to the current working directory)
            import sys
            import os
            
            # Add the components directory to path
            components_path = str(self.asset_manager.components_path)
            if components_path not in sys.path:
                sys.path.insert(0, str(self.asset_manager.base_path))
            
            # Import UI components
            from components.ui_components import UIComponents, AnimatedComponents
            
            # Make components available in session state
            st.session_state.ui_components = UIComponents
            st.session_state.animated_components = AnimatedComponents
            
            logger.info("UI components loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import UI components: {e}")
        except Exception as e:
            logger.error(f"Error loading components: {e}")


# Global asset manager instance
_asset_manager = None

def get_asset_manager() -> AssetManager:
    """Get or create the global asset manager instance"""
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = AssetManager()
    return _asset_manager


def initialize_ui() -> Dict[str, bool]:
    """Initialize the UI with all assets and components"""
    asset_manager = get_asset_manager()
    theme_manager = ThemeManager(asset_manager)
    component_loader = ComponentLoader(asset_manager)
    
    # Load all assets
    results = asset_manager.load_all_assets()
    
    # Apply default theme
    theme_manager.apply_theme()
    
    # Load components
    component_loader.load_components()
    
    # Store managers in session state
    st.session_state.asset_manager = asset_manager
    st.session_state.theme_manager = theme_manager
    
    logger.info("UI initialization complete")
    return results


def inject_page_config() -> None:
    """Set Streamlit page configuration with custom styling"""
    st.set_page_config(
        page_title="RAG LlamaStack",
        page_icon="🦙",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/rag-llama-stack',
            'Report a bug': 'https://github.com/your-repo/rag-llama-stack/issues',
            'About': """
            # RAG LlamaStack Application
            
            A modern RAG (Retrieval-Augmented Generation) application built with LlamaStack and Streamlit.
            
            **Features:**
            - 📄 Document processing with OCR support (Max 50MB per file)
            - 🔍 Semantic search with FAISS
            - 🤖 Local LLM inference with Ollama
            - 📊 Performance monitoring
            - 🎨 Modern UI with custom components
            """
        }
    )
    
    # Set maximum upload size to 50MB
    # Note: This should also be configured in .streamlit/config.toml for production
    st._config.set_option("server.maxUploadSize", 50) 