"""
Theme Management for RAG LlamaStack Application
Handles dark/light theme switching and styling
"""

import streamlit as st


def initialize_theme():
    """Initialize theme state"""
    if "dark_theme" not in st.session_state:
        st.session_state.dark_theme = False  # Default to light theme


def apply_theme():
    """Apply the selected theme with comprehensive styling"""
    if st.session_state.dark_theme:
        inject_dark_theme_css()
    else:
        inject_light_theme_css()


def render_header_theme_toggle() -> None:
    """Render a small theme toggle in the top-right area"""
    theme_icon = "☀️" if st.session_state.dark_theme else "🌙"
    theme_tooltip = "Switch to Light Theme" if st.session_state.dark_theme else "Switch to Dark Theme"
    
    # Create a small theme toggle button in a container
    st.markdown("""
    <style>
    /* Style for small theme toggle - aligned with Streamlit UI */
    .theme-toggle-small .stButton button {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 50% !important;
        width: 28px !important;
        height: 28px !important;
        padding: 0 !important;
        font-size: 13px !important;
        margin: 0 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        min-width: 28px !important;
        min-height: 28px !important;
    }
    
    .theme-toggle-small .stButton button:hover {
        background: rgba(255, 255, 255, 1) !important;
        transform: scale(1.02) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
    }
    
    .theme-toggle-small .stButton button:focus {
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
        outline: none !important;
    }
    
    /* Position aligned with Streamlit's UI elements */
    .theme-toggle-container {
        position: fixed;
        top: 0.65rem;
        right: 4.5rem;
        z-index: 999;
        width: 28px;
        height: 28px;
    }
    
    /* Remove button container padding */
    .theme-toggle-small .stButton {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Dark theme adjustments */
    .dark-theme .theme-toggle-small .stButton button {
        background: rgba(0, 0, 0, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3) !important;
    }
    
    .dark-theme .theme-toggle-small .stButton button:hover {
        background: rgba(0, 0, 0, 0.95) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Ensure it works well with Streamlit's responsive design */
    @media (max-width: 768px) {
        .theme-toggle-container {
            right: 4rem;
            top: 0.5rem;
        }
        
        .theme-toggle-small .stButton button {
            width: 26px !important;
            height: 26px !important;
            font-size: 12px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create the theme toggle in a fixed position container
    st.markdown('<div class="theme-toggle-container">', unsafe_allow_html=True)
    st.markdown('<div class="theme-toggle-small">', unsafe_allow_html=True)
    
    if st.button(theme_icon, key="header_theme_toggle", help=theme_tooltip):
        st.session_state.dark_theme = not st.session_state.dark_theme
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_sidebar_theme_toggle():
    """Renders a theme toggle button, suitable for the sidebar."""
    theme_icon = "☀️" if st.session_state.dark_theme else "🌙"
    theme_tooltip = "Switch to Light Theme" if st.session_state.dark_theme else "Switch to Dark Theme"

    if st.button(theme_icon, key="sidebar_theme_toggle", help=theme_tooltip):
        st.session_state.dark_theme = not st.session_state.dark_theme
        st.rerun()


def toggle_theme():
    """Toggle between dark and light themes"""
    st.session_state.dark_theme = not st.session_state.dark_theme


def get_theme_name() -> str:
    """Get the current theme name"""
    return "dark" if st.session_state.dark_theme else "light" 


def inject_dark_theme_css():
    """Inject comprehensive dark theme CSS for entire app"""
    st.markdown("""
    <style>
    /* Dark Theme Variables */
    :root {
        --background-color: #0e1117;
        --secondary-background: #262730;
        --sidebar-background: #1e1e2e;
        --text-color: #fafafa;
        --secondary-text: #b3b3b3;
        --border-color: #4a4a4a;
        --user-message-bg: #007bff;
        --assistant-message-bg: #3a3a3a;
        --assistant-message-text: #fafafa;
        --section-border: #555555;
        --separator-color: #444444;
    }
    
    /* Main App Background - Full Page */
    .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Header/Toolbar Area - Dark Theme */
    header[data-testid="stHeader"],
    div[data-testid="stHeader"],
    .stAppHeader,
    section[data-testid="stHeader"] {
        background-color: var(--background-color) !important;
        border-bottom: none !important;
    }
    
    /* Header elements styling */
    header[data-testid="stHeader"] *,
    div[data-testid="stHeader"] * {
        color: var(--text-color) !important;
    }
    
    /* Streamlit's built-in header buttons */
    div[data-testid="stHeader"] button,
    header[data-testid="stHeader"] button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    div[data-testid="stHeader"] button:hover,
    header[data-testid="stHeader"] button:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Comprehensive header area coverage */
    .stApp > header,
    .stApp header,
    [data-testid="stHeader"],
    .css-18ni7ap,
    .css-1dp5vir {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Header toolbar buttons and icons */
    [data-testid="stHeader"] svg,
    header svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Sidebar - Comprehensive Targeting */
    .stSidebar, 
    .css-1d391kg, 
    .css-1cypcdb,
    .css-17eq0hr,
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        background-color: var(--sidebar-background) !important;
        color: var(--text-color) !important;
        border-right: 2px solid var(--border-color) !important;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Sidebar Content */
    .stSidebar .stMarkdown,
    .stSidebar .stSelectbox,
    .stSidebar .stButton,
    .stSidebar .stTextInput,
    .stSidebar .stNumberInput,
    .stSidebar .stSlider,
    .stSidebar .stExpander,
    .stSidebar .stRadio,
    .stSidebar .stCheckbox {
        color: var(--text-color) !important;
    }
    
    /* Sidebar Form Elements with Visible Borders */
    .stSidebar .stTextInput > div > div > input,
    .stSidebar .stSelectbox > div > div > div,
    .stSidebar .stNumberInput > div > div > input {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
    }
    
    .stSidebar .stTextInput > div > div > input:focus,
    .stSidebar .stSelectbox > div > div > div:focus,
    .stSidebar .stNumberInput > div > div > input:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25) !important;
    }
    
    /* Specific styling for dropdown/select elements */
    .stSidebar .stSelectbox > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        min-height: 38px !important;
    }
    
    /* Dropdown arrow and container */
    .stSidebar .stSelectbox > div > div > div > div {
        border: none !important;
        background: transparent !important;
    }
    
    /* Dropdown options container */
    .stSidebar .stSelectbox > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Individual dropdown options */
    .stSidebar .stSelectbox > div > div > div > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-bottom: 1px solid var(--separator-color) !important;
        padding: 8px 12px !important;
    }
    
    .stSidebar .stSelectbox > div > div > div > div > div > div:hover {
        background-color: var(--border-color) !important;
    }
    
    .stSidebar .stSelectbox > div > div > div > div > div > div:last-child {
        border-bottom: none !important;
    }
    
    /* Dropdown arrow styling */
    .stSidebar .stSelectbox > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
    }
    
    /* Selected value in dropdown */
    .stSidebar .stSelectbox > div > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown arrow icon */
    .stSidebar .stSelectbox svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Global select box styling for consistency */
    .stSelectbox > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div > div:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25) !important;
    }
    
    /* Global select box options */
    .stSelectbox > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stSelectbox > div > div > div > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-bottom: 1px solid var(--separator-color) !important;
    }
    
    .stSelectbox > div > div > div > div > div > div:hover {
        background-color: var(--border-color) !important;
    }
    
    /* Sidebar Section Separators */
    .stSidebar hr {
        border-color: var(--section-border) !important;
        border-width: 1px !important;
        margin: 1rem 0 !important;
        opacity: 1 !important;
    }
    
    /* Sidebar Section Headers */
    .stSidebar h1,
    .stSidebar h2, 
    .stSidebar h3,
    .stSidebar h4 {
        color: var(--text-color) !important;
        border-bottom: 2px solid var(--section-border) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Expander Headers in Sidebar */
    .stSidebar .streamlit-expanderHeader {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 2px solid var(--section-border) !important;
        border-radius: 4px !important;
    }
    
    /* File Upload Area */
    .stSidebar .stFileUploader {
        border: 2px dashed var(--section-border) !important;
        border-radius: 8px !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        padding: 1rem !important;
    }
    
    /* Section Containers */
    .stSidebar > div {
        border-bottom: 2px solid var(--separator-color) !important;
        padding-bottom: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    .stSidebar > div:last-child {
        border-bottom: none !important;
    }
    
    /* Main Content Area */
    .main .block-container,
    .css-18e3th9,
    .css-1d391kg,
    div[data-testid="block-container"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Header and Title Area */
    h1, h2, h3, h4, h5, h6,
    .stTitle,
    .stHeader,
    .stSubheader {
        color: var(--text-color) !important;
    }
    
    /* Text Elements */
    .stMarkdown,
    .stText,
    .stCaption,
    p, span, div {
        color: var(--text-color) !important;
    }
    
    /* High contrast text for important elements */
    .stSidebar .stMarkdown strong,
    .stSidebar .stMarkdown b,
    .stSidebar label,
    .stSidebar .stSelectbox label,
    .stSidebar .stTextInput label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Status indicators with proper contrast */
    .stSidebar .stSuccess {
        background-color: rgba(34, 197, 94, 0.2) !important;
        border: 1px solid rgba(34, 197, 94, 0.5) !important;
        color: #ffffff !important;
    }
    
    .stSidebar .stError {
        background-color: rgba(239, 68, 68, 0.2) !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Better visibility for model info */
    .stSidebar .stSelectbox > div {
        background-color: var(--secondary-background) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
    }
    
    /* Improved section dividers */
    .stSidebar .stMarkdown:has(h3) {
        border-top: 2px solid var(--section-border) !important;
        padding-top: 1rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Better contrast for embedded text */
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown li,
    .stSidebar .stMarkdown span {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Text Elements - More Specific */
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4,
    .stSidebar .stMarkdown h5,
    .stSidebar .stMarkdown h6,
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown span,
    .stSidebar .stMarkdown div,
    .stSidebar .stMarkdown ul,
    .stSidebar .stMarkdown ol,
    .stSidebar .stMarkdown li,
    .stSidebar .stMarkdown strong,
    .stSidebar .stMarkdown em,
    .stSidebar .stMarkdown code {
        color: var(--text-color) !important;
    }
    
    /* Caption Text */
    .stCaption,
    .caption {
        color: var(--secondary-text) !important;
    }
    
    /* Success/Info/Warning/Error Messages */
    .stSuccess,
    .stInfo,
    .stWarning,
    .stError {
        color: var(--text-color) !important;
    }
    
    .stSuccess .stMarkdown,
    .stInfo .stMarkdown,
    .stWarning .stMarkdown,
    .stError .stMarkdown {
        color: var(--text-color) !important;
    }
    
    /* Comprehensive alert block styling for dark theme */
    .stAlert,
    .stSuccess,
    .stInfo,
    .stWarning,
    .stError {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    /* Alert content styling */
    .stAlert > div,
    .stSuccess > div,
    .stInfo > div,
    .stWarning > div,
    .stError > div {
        background-color: transparent !important;
        color: var(--text-color) !important;
    }
    
    /* Alert text and markdown content */
    .stAlert *,
    .stSuccess *,
    .stInfo *,
    .stWarning *,
    .stError * {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    
    /* Specific alert types with themed backgrounds */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.2) !important;
        border-color: rgba(34, 197, 94, 0.5) !important;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-color: rgba(59, 130, 246, 0.5) !important;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.2) !important;
        border-color: rgba(245, 158, 11, 0.5) !important;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.2) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
    }
    
    /* Force override any inline styles in alerts */
    .stApp .stAlert[style*="background"],
    .stApp .stSuccess[style*="background"],
    .stApp .stInfo[style*="background"],
    .stApp .stWarning[style*="background"],
    .stApp .stError[style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Alert icons and symbols */
    .stAlert svg,
    .stSuccess svg,
    .stInfo svg,
    .stWarning svg,
    .stError svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Markdown Lists and Content */
    .stMarkdown ul li,
    .stMarkdown ol li,
    .stMarkdown strong,
    .stMarkdown em,
    .stMarkdown code {
        color: var(--text-color) !important;
    }
    
    /* Code blocks */
    .stMarkdown pre,
    .stMarkdown code {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Progress Bar - Single Blue Bar */
    .stProgress > div > div > div > div {
        background-color: #007bff !important;
    }
    .stProgress > div > div > div {
        background-color: rgba(255,255,255,0.1) !important;
    }
    
    /* Form and Input Elements */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Dropdown/Select Elements */
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Model Selection Dropdown - Dark Theme */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    /* Dropdown menu container */
    .stSelectbox [data-baseweb="popover"] {
        background-color: var(--secondary-background) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
    }
    
    /* Dropdown options */
    .stSelectbox [data-baseweb="menu"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Individual dropdown items */
    .stSelectbox [role="option"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Dropdown item hover */
    .stSelectbox [role="option"]:hover {
        background-color: var(--sidebar-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Selected dropdown item */
    .stSelectbox [aria-selected="true"] {
        background-color: rgba(0, 123, 255, 0.2) !important;
        color: var(--text-color) !important;
    }
    
    /* Dropdown arrow/icon */
    .stSelectbox svg {
        fill: var(--text-color) !important;
    }
    
    /* Model list specific styling */
    .stSelectbox [data-testid="stSelectbox"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Additional dropdown coverage for any missed elements */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Ensure dropdown text is visible */
    .stSelectbox * {
        color: var(--text-color) !important;
    }
    
    /* BaseWeb select component */
    [data-baseweb="select"] [data-baseweb="select-control"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--section-border) !important;
    }
    
    /* BaseWeb dropdown list */
    [data-baseweb="menu"] li {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: var(--sidebar-background) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        border: none !important;
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader,
    .stExpander > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Tabs */
    .stTabs > div > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Columns and Containers */
    .stColumn,
    .stContainer {
        background-color: transparent !important;
    }
    
    /* Chat Message Styling */
    .user-bubble {
        background-color: var(--user-message-bg) !important;
        color: white !important;
    }
    
    .assistant-bubble {
        background-color: var(--assistant-message-bg) !important;
        color: var(--assistant-message-text) !important;
    }
    
    .chat-container {
        background-color: var(--background-color) !important;
        border: 1px solid var(--border-color);
    }
    
    .source-tag {
        background-color: rgba(255,255,255,0.1) !important;
        color: var(--assistant-message-text) !important;
    }
    
    .model-info {
        color: var(--secondary-text) !important;
        border-top: 1px solid var(--border-color) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: var(--secondary-background) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background-color: var(--secondary-background) !important;
        border-color: var(--border-color) !important;
    }
    
    /* File Upload Drag & Drop Area - Dark Theme */
    .stFileUploader > div > div {
        background-color: var(--secondary-background) !important;
        border: 2px dashed var(--section-border) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
    }
    
    /* File upload text */
    .stFileUploader label,
    .stFileUploader > div > div > div {
        color: var(--text-color) !important;
    }
    
    /* Drag and drop text */
    .stFileUploader > div > div > div > div {
        color: var(--text-color) !important;
    }
    
    /* Upload button */
    .stFileUploader button {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    .stFileUploader button:hover {
        background-color: var(--sidebar-background) !important;
    }
    
    /* File upload area when dragging */
    .stFileUploader > div > div:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-color: #007bff !important;
    }
    
    /* Uploaded file indicators */
    .stFileUploader .uploadedFile {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    /* Additional file uploader text elements */
    .stFileUploader * {
        color: var(--text-color) !important;
    }
    
    /* File uploader specific text nodes */
    .stFileUploader p,
    .stFileUploader span,
    .stFileUploader div[data-testid="stFileDropzoneInstructions"] {
        color: var(--text-color) !important;
    }
    
    /* Browse files button text */
    .stFileUploader button[data-testid="stFileUploaderDropzoneButton"] {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
    }
    
    /* More specific file upload area targeting */
    .stFileUploader,
    .stFileUploader > div,
    .stFileUploader > div > div,
    .stFileUploader > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* File upload dropzone */
    [data-testid="stFileDropzone"] {
        background-color: var(--secondary-background) !important;
        border: 2px dashed var(--section-border) !important;
        color: var(--text-color) !important;
    }
    
    /* Browse files button specific */
    button[data-testid="stFileUploaderDropzoneButton"],
    .stFileUploader button {
        background-color: var(--sidebar-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
    }
    
    button[data-testid="stFileUploaderDropzoneButton"]:hover,
    .stFileUploader button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* File uploader instructions text */
    [data-testid="stFileDropzoneInstructions"] {
        color: var(--text-color) !important;
    }
    
    /* Force all file uploader descendants to use theme colors */
    .stFileUploader, .stFileUploader * {
        color: var(--text-color) !important;
    }
    
    .stFileUploader div[style] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Additional aggressive file upload targeting */
    .stFileUploader section,
    .stFileUploader section *,
    .stFileUploader [data-testid],
    .stFileUploader [data-testid] * {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Force file upload area styling */
    .stFileUploader > section > div,
    .stFileUploader > section > div > div,
    .stFileUploader > section > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--section-border) !important;
    }
    
    /* Model dropdown aggressive targeting */
    .stSelectbox,
    .stSelectbox *,
    .stSelectbox > div,
    .stSelectbox > div *,
    .stSelectbox section,
    .stSelectbox section * {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* BaseWeb dropdown override - more aggressive */
    [data-baseweb="select"],
    [data-baseweb="select"] *,
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [data-baseweb="menu"],
    [data-baseweb="menu"] * {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--section-border) !important;
    }
    
    /* Force override any inline styles */
    .stSelectbox [style*="background"],
    .stFileUploader [style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    .stSelectbox [style*="color"],
    .stFileUploader [style*="color"] {
        color: var(--text-color) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #007bff !important;
    }
    
    /* Hide only Streamlit Footer (keep menu and header visible) */
    footer {visibility: hidden;}
    
    /* Menu Items and Labels */
    .stSelectbox label,
    .stTextInput label,
    .stFileUploader label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label,
    .stCheckbox label,
    .stTextArea label {
        color: var(--text-color) !important;
    }
    
    /* Streamlit Native Menu (3-dot menu) */
    div[data-testid="stAppViewContainer"] *,
    div[role="menuitem"] *,
    div[role="menu"] * {
        color: var(--text-color) !important;
    }
    
    /* Any remaining text elements */
    * {
        color: inherit !important;
    }
    
    /* Force text color for all text nodes */
    body, html {
        color: var(--text-color) !important;
    }
    
    /* Critical override for problematic text colors */
    .stSidebar *:not(svg):not(path) {
        color: var(--text-color) !important;
    }
    
    /* Ensure no invisible text in dark theme */
    .stApp *[style*="color: rgb(49, 51, 63)"],
    .stApp *[style*="color: #31333f"],
    .stApp *[style*="color: rgb(38, 39, 48)"] {
        color: var(--text-color) !important;
    }
    
    /* Nuclear option - force all white backgrounds to theme background */
    .stApp *[style*="background-color: rgb(255, 255, 255)"],
    .stApp *[style*="background-color: #ffffff"],
    .stApp *[style*="background-color: white"],
    .stApp *[style*="background: rgb(255, 255, 255)"],
    .stApp *[style*="background: #ffffff"],
    .stApp *[style*="background: white"] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Force specific Streamlit component backgrounds */
    div[data-baseweb="select"] div[style*="background"],
    div[data-baseweb="popover"] div[style*="background"],
    .stFileUploader div[style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Catch any remaining BaseWeb white backgrounds */
    [class*="BaseButton"][style*="background"],
    [class*="Select"][style*="background"],
    [class*="Popover"][style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Override for interactive elements */
    .stSidebar button,
    .stSidebar input,
    .stSidebar select,
    .stSidebar textarea {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-color: var(--section-border) !important;
    }
    
    /* Ensure dropdown text is visible */
    .stSidebar .stSelectbox div[data-baseweb="select"] {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
    }
    
    /* Force visibility for any remaining elements */
    .stSidebar .stMarkdown *:not(code):not(pre) {
        color: var(--text-color) !important;
    }
    
    /* Custom Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: var(--secondary-background);
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #007bff;
        border-radius: 3px;
    }
    
    /* Theme Toggle Dark Mode Support */
    .dark-theme .theme-toggle-small .stButton button {
        background: rgba(0, 0, 0, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    .dark-theme .theme-toggle-small .stButton button:hover {
        background: rgba(0, 0, 0, 0.9) !important;
    }
    
    /* Enhanced dropdown styling to ensure visible borders */
    .stSelectbox > div > div > div {
        border: 3px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        outline: none !important;
        box-shadow: 0 0 0 1px var(--section-border) !important;
    }
    
    /* Force visible borders on dropdown containers */
    .stSelectbox > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        min-height: 38px !important;
    }
    
    /* Dropdown trigger element */
    .stSelectbox > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        min-height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 8px 12px !important;
    }
    
    /* Dropdown options container */
    .stSelectbox > div > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Individual dropdown options */
    .stSelectbox > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-bottom: 1px solid var(--separator-color) !important;
        padding: 8px 12px !important;
    }
    
    .stSelectbox > div > div > div > div > div > div > div:hover {
        background-color: var(--border-color) !important;
    }
    
    /* Dropdown arrow */
    .stSelectbox svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* COMPREHENSIVE FIXES FOR HORIZONTAL LINES AND DROPDOWN BORDERS */
    
    /* Remove ALL horizontal lines from header and sidebar */
    header,
    div[data-testid="stHeader"],
    .stAppHeader,
    section[data-testid="stHeader"],
    .stSidebar,
    .css-1d391kg,
    .css-1cypcdb,
    .css-17eq0hr,
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"],
    .stApp > header,
    .stApp header,
    [data-testid="stHeader"],
    .css-18ni7ap,
    .css-1dp5vir {
        border-top: none !important;
        border-bottom: none !important;
        border-left: none !important;
        border-right: none !important;
    }
    
    /* Force remove any Streamlit default borders */
    .stApp,
    .main,
    .block-container {
        border-top: none !important;
        border-bottom: none !important;
    }
    
    /* AGGRESSIVE DROPDOWN BORDER FIXING */
    
    /* Target ALL possible selectbox selectors */
    .stSelectbox,
    .stSelectbox *,
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    .stSelectbox > div,
    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    .stSelectbox > div > div > div > div,
    .stSelectbox > div > div > div > div > div,
    .stSelectbox > div > div > div > div > div > div {
        border: 3px solid var(--section-border) !important;
        border-radius: 6px !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        outline: 2px solid var(--section-border) !important;
        box-shadow: 0 0 0 2px var(--section-border) !important;
    }
    
    /* Specific styling for the main selectbox container */
    .stSelectbox > div > div > div {
        border: 4px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 8px !important;
        outline: 3px solid var(--section-border) !important;
        box-shadow: 
            0 0 0 3px var(--section-border),
            0 2px 8px rgba(0, 0, 0, 0.3) !important;
        min-height: 40px !important;
        padding: 4px !important;
    }
    
    /* Force visible borders on the dropdown trigger */
    .stSelectbox > div > div > div > div > div {
        border: 3px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        min-height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 8px 12px !important;
        outline: 2px solid var(--section-border) !important;
        box-shadow: 0 0 0 2px var(--section-border) !important;
    }
    
    /* Dropdown options container with strong borders */
    .stSelectbox > div > div > div > div > div > div {
        border: 3px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        box-shadow: 
            0 4px 12px rgba(0, 0, 0, 0.4),
            0 0 0 2px var(--section-border) !important;
        outline: 2px solid var(--section-border) !important;
    }
    
    /* Individual dropdown options */
    .stSelectbox > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-bottom: 2px solid var(--separator-color) !important;
        padding: 10px 12px !important;
        border-left: 2px solid var(--section-border) !important;
        border-right: 2px solid var(--section-border) !important;
    }
    
    .stSelectbox > div > div > div > div > div > div > div:hover {
        background-color: var(--border-color) !important;
        border-left: 3px solid #007bff !important;
        border-right: 3px solid #007bff !important;
    }
    
    /* Remove borders from last option */
    .stSelectbox > div > div > div > div > div > div > div:last-child {
        border-bottom: none !important;
    }
    
    /* Ensure dropdown arrow is visible */
    .stSelectbox svg,
    .stSelectbox * svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
        stroke: var(--text-color) !important;
    }
    
    /* Force focus states */
    .stSelectbox > div > div > div:focus,
    .stSelectbox > div > div > div > div:focus,
    .stSelectbox > div > div > div > div > div:focus {
        border-color: #007bff !important;
        outline-color: #007bff !important;
        box-shadow: 
            0 0 0 3px rgba(0, 123, 255, 0.5),
            0 0 0 2px #007bff !important;
    }
    
    /* Override Streamlit's default styling */
    .stSelectbox > div > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
        font-weight: 500 !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Force remove any Streamlit default borders that might be interfering */
    .stSelectbox > div > div > div > div > div > div > div > div > div {
        border: none !important;
        background-color: transparent !important;
    }
    
    /* Additional specificity for dropdown borders */
    .stSelectbox > div > div > div > div > div > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
    }
    
    /* Ensure the dropdown is visible even if Streamlit tries to hide it */
    .stSelectbox > div > div > div > div > div > div {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 9999 !important;
        position: relative !important;
    }
    
    /* Force the dropdown to be above other elements */
    .stSelectbox > div > div > div > div > div > div > div {
        position: relative !important;
        z-index: 10000 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def inject_light_theme_css():
    """Inject comprehensive light theme CSS for entire app"""
    st.markdown("""
    <style>
    /* Light Theme Variables */
    :root {
        --background-color: #ffffff;
        --secondary-background: #f8f9fa;
        --sidebar-background: #f1f3f4;
        --text-color: #262730;
        --secondary-text: #666666;
        --border-color: #dee2e6;
        --user-message-bg: #ff4b4b;
        --assistant-message-bg: #f8f9fa;
        --assistant-message-text: #262730;
        --section-border: #d1d5db;
        --separator-color: #e5e7eb;
        --primary-color: #ff4b4b;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --info-color: #17a2b8;
    }
    
    /* Main App Background - Full Page */
    .stApp {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Header/Toolbar Area - Light Theme */
    header[data-testid="stHeader"],
    div[data-testid="stHeader"],
    .stAppHeader,
    section[data-testid="stHeader"] {
        background-color: var(--background-color) !important;
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    /* Header elements styling */
    header[data-testid="stHeader"] *,
    div[data-testid="stHeader"] * {
        color: var(--text-color) !important;
    }
    
    /* Streamlit's built-in header buttons */
    div[data-testid="stHeader"] button,
    header[data-testid="stHeader"] button {
        background-color: rgba(0, 0, 0, 0.05) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    div[data-testid="stHeader"] button:hover,
    header[data-testid="stHeader"] button:hover {
        background-color: rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Comprehensive header area coverage */
    .stApp > header,
    .stApp header,
    [data-testid="stHeader"],
    .css-18ni7ap,
    .css-1dp5vir {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Header toolbar buttons and icons */
    [data-testid="stHeader"] svg,
    header svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Sidebar - Comprehensive Targeting */
    .stSidebar, 
    .css-1d391kg, 
    .css-1cypcdb,
    .css-17eq0hr,
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        background-color: var(--sidebar-background) !important;
        color: var(--text-color) !important;
        border-right: 2px solid var(--border-color) !important;
        box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Sidebar Content */
    .stSidebar .stMarkdown,
    .stSidebar .stSelectbox,
    .stSidebar .stButton,
    .stSidebar .stTextInput,
    .stSidebar .stNumberInput,
    .stSidebar .stSlider,
    .stSidebar .stExpander,
    .stSidebar .stRadio,
    .stSidebar .stCheckbox {
        color: var(--text-color) !important;
    }
    
    /* Sidebar Form Elements with Visible Borders */
    .stSidebar .stTextInput > div > div > input,
    .stSidebar .stSelectbox > div > div > div,
    .stSidebar .stNumberInput > div > div > input {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
    }
    
    .stSidebar .stTextInput > div > div > input:focus,
    .stSidebar .stSelectbox > div > div > div:focus,
    .stSidebar .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.25) !important;
    }
    
    /* Specific styling for dropdown/select elements */
    .stSidebar .stSelectbox > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        min-height: 38px !important;
    }
    
    /* Dropdown arrow and container */
    .stSidebar .stSelectbox > div > div > div > div {
        border: none !important;
        background: transparent !important;
    }
    
    /* Dropdown options container */
    .stSidebar .stSelectbox > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Individual dropdown options */
    .stSidebar .stSelectbox > div > div > div > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-bottom: 1px solid var(--separator-color) !important;
        padding: 8px 12px !important;
    }
    
    .stSidebar .stSelectbox > div > div > div > div > div > div:hover {
        background-color: var(--border-color) !important;
    }
    
    .stSidebar .stSelectbox > div > div > div > div > div > div:last-child {
        border-bottom: none !important;
    }
    
    /* Dropdown arrow styling */
    .stSidebar .stSelectbox > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
    }
    
    /* Selected value in dropdown */
    .stSidebar .stSelectbox > div > div > div > div > div > div > div > div {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    
    /* Dropdown arrow icon */
    .stSidebar .stSelectbox svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Global select box styling for consistency */
    .stSelectbox > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox > div > div > div:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.25) !important;
    }
    
    /* Global select box options */
    .stSelectbox > div > div > div > div > div {
        border: 2px solid var(--section-border) !important;
        background-color: var(--secondary-background) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stSelectbox > div > div > div > div > div > div {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-bottom: 1px solid var(--separator-color) !important;
    }
    
    .stSelectbox > div > div > div > div > div > div:hover {
        background-color: var(--border-color) !important;
    }
    
    /* Sidebar Section Separators */
    .stSidebar hr {
        border-color: var(--section-border) !important;
        border-width: 1px !important;
        margin: 1rem 0 !important;
    }
    
    /* Sidebar Section Headers */
    .stSidebar h1,
    .stSidebar h2, 
    .stSidebar h3,
    .stSidebar h4 {
        color: var(--text-color) !important;
        border-bottom: 1px solid var(--section-border) !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Expander Headers in Sidebar */
    .stSidebar .streamlit-expanderHeader {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
    }
    
    /* File Upload Area */
    .stSidebar .stFileUploader {
        border: 2px dashed var(--section-border) !important;
        border-radius: 8px !important;
        background-color: rgba(255, 255, 255, 0.5) !important;
        padding: 1rem !important;
    }
    
    /* Section Containers */
    .stSidebar > div {
        border-bottom: 1px solid var(--separator-color) !important;
        padding-bottom: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    .stSidebar > div:last-child {
        border-bottom: none !important;
    }
    
    /* Main Content Area */
    .main .block-container,
    .css-18e3th9,
    .css-1d391kg,
    div[data-testid="block-container"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Header and Title Area */
    h1, h2, h3, h4, h5, h6,
    .stTitle,
    .stHeader,
    .stSubheader {
        color: var(--text-color) !important;
    }
    
    /* Text Elements */
    .stMarkdown,
    .stText,
    .stCaption,
    p, span, div {
        color: var(--text-color) !important;
    }
    
    /* High contrast text for important elements in light theme */
    .stSidebar .stMarkdown strong,
    .stSidebar .stMarkdown b,
    .stSidebar label,
    .stSidebar .stSelectbox label,
    .stSidebar .stTextInput label {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    /* Status indicators with proper contrast for light theme */
    .stSidebar .stSuccess {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border: 1px solid rgba(40, 167, 69, 0.3) !important;
        color: #1f2937 !important;
    }
    
    .stSidebar .stError {
        background-color: rgba(220, 53, 69, 0.1) !important;
        border: 1px solid rgba(220, 53, 69, 0.3) !important;
        color: #1f2937 !important;
    }
    
    /* Better visibility for model info in light theme */
    .stSidebar .stSelectbox > div {
        background-color: var(--background-color) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
    }
    
    /* Better contrast for embedded text in light theme */
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown li,
    .stSidebar .stMarkdown span {
        color: #4b5563 !important;
    }
    
    /* Sidebar Text Elements - More Specific */
    .stSidebar .stMarkdown h1,
    .stSidebar .stMarkdown h2,
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4,
    .stSidebar .stMarkdown h5,
    .stSidebar .stMarkdown h6,
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown span,
    .stSidebar .stMarkdown div,
    .stSidebar .stMarkdown ul,
    .stSidebar .stMarkdown ol,
    .stSidebar .stMarkdown li,
    .stSidebar .stMarkdown strong,
    .stSidebar .stMarkdown em,
    .stSidebar .stMarkdown code {
        color: var(--text-color) !important;
    }
    
    /* Caption Text */
    .stCaption,
    .caption {
        color: var(--secondary-text) !important;
    }
    
    /* Success/Error/Info/Warning Messages */
    .stSuccess,
    .stInfo,
    .stWarning,
    .stError {
        color: var(--text-color) !important;
    }
    
    .stSuccess .stMarkdown,
    .stInfo .stMarkdown,
    .stWarning .stMarkdown,
    .stError .stMarkdown {
        color: var(--text-color) !important;
    }
    
    /* Comprehensive alert block styling for light theme */
    .stAlert,
    .stSuccess,
    .stInfo,
    .stWarning,
    .stError {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    /* Alert content styling */
    .stAlert > div,
    .stSuccess > div,
    .stInfo > div,
    .stWarning > div,
    .stError > div {
        background-color: transparent !important;
        color: var(--text-color) !important;
    }
    
    /* Alert text and markdown content */
    .stAlert *,
    .stSuccess *,
    .stInfo *,
    .stWarning *,
    .stError * {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    
    /* Specific alert types with light theme colors */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border-color: rgba(40, 167, 69, 0.3) !important;
    }
    
    .stInfo {
        background-color: rgba(23, 162, 184, 0.1) !important;
        border-color: rgba(23, 162, 184, 0.3) !important;
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1) !important;
        border-color: rgba(255, 193, 7, 0.3) !important;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1) !important;
        border-color: rgba(220, 53, 69, 0.3) !important;
    }
    
    /* Force override any inline styles in alerts for light theme */
    .stApp .stAlert[style*="background"],
    .stApp .stSuccess[style*="background"],
    .stApp .stInfo[style*="background"],
    .stApp .stWarning[style*="background"],
    .stApp .stError[style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Alert icons and symbols for light theme */
    .stAlert svg,
    .stSuccess svg,
    .stInfo svg,
    .stWarning svg,
    .stError svg {
        fill: var(--text-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Markdown Lists and Content */
    .stMarkdown ul li,
    .stMarkdown ol li,
    .stMarkdown strong,
    .stMarkdown em,
    .stMarkdown code {
        color: var(--text-color) !important;
    }
    
    /* Code blocks */
    .stMarkdown pre,
    .stMarkdown code {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Progress Bar - Single Blue Bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color) !important;
    }
    .stProgress > div > div > div {
        background-color: rgba(0,0,0,0.1) !important;
    }
    
    /* Form and Input Elements */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Dropdown/Select Elements */
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Model Selection Dropdown - Light Theme */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    /* Dropdown menu container */
    .stSelectbox [data-baseweb="popover"] {
        background-color: var(--background-color) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Dropdown options */
    .stSelectbox [data-baseweb="menu"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Individual dropdown items */
    .stSelectbox [role="option"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Dropdown item hover */
    .stSelectbox [role="option"]:hover {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Selected dropdown item */
    .stSelectbox [aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.1) !important;
        color: var(--text-color) !important;
    }
    
    /* Dropdown arrow/icon */
    .stSelectbox svg {
        fill: var(--text-color) !important;
    }
    
    /* Model list specific styling */
    .stSelectbox [data-testid="stSelectbox"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Additional dropdown coverage for any missed elements */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Ensure dropdown text is visible */
    .stSelectbox * {
        color: var(--text-color) !important;
    }
    
    /* BaseWeb select component */
    [data-baseweb="select"] [data-baseweb="select-control"] {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border-color: var(--section-border) !important;
    }
    
    /* BaseWeb dropdown list */
    [data-baseweb="menu"] li {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: var(--secondary-background) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader,
    .stExpander > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--border-color) !important;
    }
    
    /* Tabs */
    .stTabs > div > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Columns and Containers */
    .stColumn,
    .stContainer {
        background-color: transparent !important;
    }
    
    /* Chat Message Styling */
    .user-bubble {
        background-color: var(--user-message-bg) !important;
        color: white !important;
    }
    
    .assistant-bubble {
        background-color: var(--assistant-message-bg) !important;
        color: var(--assistant-message-text) !important;
    }
    
    .chat-container {
        background-color: var(--background-color) !important;
        border: 1px solid var(--border-color);
    }
    
    .source-tag {
        background-color: rgba(0,0,0,0.1) !important;
        color: var(--assistant-message-text) !important;
    }
    
    .model-info {
        color: var(--secondary-text) !important;
        border-top: 1px solid var(--border-color) !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: var(--secondary-background) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-color) !important;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        background-color: var(--secondary-background) !important;
        border-color: var(--border-color) !important;
    }
    
    /* File Upload Drag & Drop Area - Light Theme */
    .stFileUploader > div > div {
        background-color: var(--secondary-background) !important;
        border: 2px dashed var(--section-border) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
    }
    
    /* File upload text */
    .stFileUploader label,
    .stFileUploader > div > div > div {
        color: var(--text-color) !important;
    }
    
    /* Drag and drop text */
    .stFileUploader > div > div > div > div {
        color: var(--text-color) !important;
    }
    
    /* Upload button */
    .stFileUploader button {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    .stFileUploader button:hover {
        background-color: var(--secondary-background) !important;
    }
    
    /* File upload area when dragging */
    .stFileUploader > div > div:hover {
        background-color: rgba(0, 0, 0, 0.02) !important;
        border-color: var(--primary-color) !important;
    }
    
    /* Uploaded file indicators */
    .stFileUploader .uploadedFile {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
    }
    
    /* Additional file uploader text elements */
    .stFileUploader * {
        color: var(--text-color) !important;
    }
    
    /* File uploader specific text nodes */
    .stFileUploader p,
    .stFileUploader span,
    .stFileUploader div[data-testid="stFileDropzoneInstructions"] {
        color: var(--text-color) !important;
    }
    
    /* Browse files button text */
    .stFileUploader button[data-testid="stFileUploaderDropzoneButton"] {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
    }
    
    /* More specific file upload area targeting for light theme */
    .stFileUploader,
    .stFileUploader > div,
    .stFileUploader > div > div,
    .stFileUploader > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* File upload dropzone for light theme */
    [data-testid="stFileDropzone"] {
        background-color: var(--secondary-background) !important;
        border: 2px dashed var(--section-border) !important;
        color: var(--text-color) !important;
    }
    
    /* Browse files button specific for light theme */
    button[data-testid="stFileUploaderDropzoneButton"],
    .stFileUploader button {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--section-border) !important;
        border-radius: 4px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    button[data-testid="stFileUploaderDropzoneButton"]:hover,
    .stFileUploader button:hover {
        background-color: rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Force all file uploader descendants to use theme colors */
    .stFileUploader div[style] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Additional aggressive file upload targeting for light theme */
    .stFileUploader section,
    .stFileUploader section *,
    .stFileUploader [data-testid],
    .stFileUploader [data-testid] * {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
    }
    
    /* Force file upload area styling for light theme */
    .stFileUploader > section > div,
    .stFileUploader > section > div > div,
    .stFileUploader > section > div > div > div {
        background-color: var(--secondary-background) !important;
        color: var(--text-color) !important;
        border-color: var(--section-border) !important;
    }
    
    /* Model dropdown aggressive targeting for light theme */
    .stSelectbox,
    .stSelectbox *,
    .stSelectbox > div,
    .stSelectbox > div *,
    .stSelectbox section,
    .stSelectbox section * {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* BaseWeb dropdown override - more aggressive for light theme */
    [data-baseweb="select"],
    [data-baseweb="select"] *,
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [data-baseweb="menu"],
    [data-baseweb="menu"] * {
        background-color: var(--background-color) !important;
        color: var(--text-color) !important;
        border-color: var(--section-border) !important;
    }
    
    /* Force override any inline styles for light theme */
    .stSelectbox [style*="background"] {
        background-color: var(--background-color) !important;
    }
    
    .stFileUploader [style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    .stSelectbox [style*="color"],
    .stFileUploader [style*="color"] {
        color: var(--text-color) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Hide only Streamlit Footer (keep menu and header visible) */
    footer {visibility: hidden;}
    
    /* Menu Items and Labels */
    .stSelectbox label,
    .stTextInput label,
    .stFileUploader label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label,
    .stCheckbox label,
    .stTextArea label {
        color: var(--text-color) !important;
    }
    
    /* Streamlit Native Menu (3-dot menu) */
    div[data-testid="stAppViewContainer"] *,
    div[role="menuitem"] *,
    div[role="menu"] * {
        color: var(--text-color) !important;
    }
    
    /* Any remaining text elements */
    * {
        color: inherit !important;
    }
    
    /* Force text color for all text nodes */
    body, html {
        color: var(--text-color) !important;
    }
    
    /* Critical override for problematic text colors */
    .stSidebar *:not(svg):not(path) {
        color: var(--text-color) !important;
    }
    
    /* Ensure no invisible text in light theme */
    .stApp *[style*="color: rgb(49, 51, 63)"],
    .stApp *[style*="color: #31333f"],
    .stApp *[style*="color: rgb(38, 39, 48)"] {
        color: var(--text-color) !important;
    }
    
    /* Override for interactive elements */
    .stSidebar button,
    .stSidebar input,
    .stSidebar select,
    .stSidebar textarea {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
        border-color: var(--section-border) !important;
    }
    
    /* Ensure dropdown text is visible */
    .stSidebar .stSelectbox div[data-baseweb="select"] {
        color: var(--text-color) !important;
        background-color: var(--secondary-background) !important;
    }
    
    /* Force visibility for any remaining elements */
    .stSidebar .stMarkdown *:not(code):not(pre) {
        color: var(--text-color) !important;
    }
    
    /* Custom Scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: var(--secondary-background);
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 3px;
    }
    
    /* Nuclear option for light theme - force any problematic backgrounds */
    .stApp div[data-baseweb="select"] div[style*="background"],
    .stApp div[data-baseweb="popover"] div[style*="background"],
    .stApp .stFileUploader div[style*="background"] {
        background-color: var(--secondary-background) !important;
    }
    
    /* Light theme BaseWeb overrides */
    .stApp [class*="BaseButton"][style*="background"],
    .stApp [class*="Select"][style*="background"],
    .stApp [class*="Popover"][style*="background"] {
        background-color: var(--background-color) !important;
    }
    
    /* Theme Toggle Light Mode Support */
    .theme-toggle-small .stButton button {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    .theme-toggle-small .stButton button:hover {
        background: rgba(255, 255, 255, 1) !important;
    }
    </style>
    """, unsafe_allow_html=True) 