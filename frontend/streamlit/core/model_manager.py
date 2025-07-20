"""
Model Management for RAG LlamaStack Application
Handles model selection, dashboard, and Ollama integration
"""

import streamlit as st
import subprocess
import json
from typing import Dict, List, Optional, Any


def render_model_dashboard() -> None:
    """Display model selection and provider dashboard"""
    # Header with aligned refresh button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("🤖 Model Dashboard")
    with col2:
        st.markdown("")  # Add space for alignment
        if st.button("🔄 Refresh", help="Refresh available models", type="secondary"):
            st.session_state.available_models = get_all_available_models()
            st.rerun()
    
    # Load models if not already loaded
    if not st.session_state.available_models["all"]:
        with st.spinner("Loading available models..."):
            st.session_state.available_models = get_all_available_models()
    
    # Current model status
    with st.expander("📊 Current Models", expanded=True):
        # Determine model status  
        embedding_status = "🧠 LlamaStack (sentence-transformers)"  # Fixed embedding model
        if st.session_state.selected_llm_model in ["llama3.2:1b", "llama3.2:3b"]:
            llm_status = "🏠 Local (Ollama)"
        else:
            llm_status = "🧪 Demo"  # Fallback for unknown models
        
        st.markdown(f"""
        **🔤 Embedding**: `all-MiniLM-L6-v2` {embedding_status}  
        **🧠 LLM**: `{st.session_state.selected_llm_model}` {llm_status}
        """)
        
        if llm_status == "🧪 Demo":
            st.info("💡 Install Ollama models below for real AI responses!")
    
    # Model Selection
    st.subheader("🎛️ Model Selection")
    
    # Fixed embedding model info (no dropdown)
    st.markdown("**🔤 Embedding Model (Fixed)**")
    st.info("📌 Using `all-MiniLM-L6-v2` via LlamaStack sentence-transformers for consistent, high-quality embeddings")
    
    # LLM model selection (keep this dropdown)
    llm_models = st.session_state.available_models["llm"]
    if llm_models:
        # Remove duplicates while preserving order
        unique_llm_models = []
        seen = set()
        for model in llm_models:
            model_id = model.get('identifier', model.get('model_id', model.get('name', str(model))))
            if model_id not in seen:
                seen.add(model_id)
                unique_llm_models.append(model_id)
        
        selected_llm = st.selectbox(
            "🧠 LLM Model",
            options=unique_llm_models,
            index=0 if st.session_state.selected_llm_model not in unique_llm_models 
                  else unique_llm_models.index(st.session_state.selected_llm_model),
            help="Model used for generating responses and chat",
            key="llm_model_selector"
        )
        
        if selected_llm != st.session_state.selected_llm_model:
            st.session_state.selected_llm_model = selected_llm
            st.rerun()
    else:
        st.warning("⚠️ No LLM models available")


def render_ollama_integration() -> None:
    """Render Ollama integration section as collapsed expander"""
    st.markdown("---")
    
    # Check Ollama status first for header
    ollama_status = check_ollama_status()
    
    # Create header with status
    status_text = "🦙 Ollama Integration"
    if ollama_status["running"]:
        status_text += f" (✅ {len(ollama_status['models'])} models)"
    else:
        status_text += " (❌ offline)"
    
    with st.expander(status_text, expanded=False):
        if ollama_status["running"]:
            st.success(f"✅ **Ollama Running** - {len(ollama_status['models'])} models available")
            
            # Show available models
            if ollama_status["models"]:
                with st.expander("📋 Available Ollama Models", expanded=False):
                    for model in ollama_status["models"]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"• **{model['name']}** ({model['size']})")
                        with col2:
                            if st.button("🗑️", key=f"remove_ollama_{model['name']}", help=f"Remove {model['name']}"):
                                remove_ollama_model(model['name'])
                                st.rerun()
            
            # Auto-pull missing default models
            default_models = ["llama3.2:1b", "nomic-embed-text"]
            missing_models = [m for m in default_models if not any(m in model['name'] for model in ollama_status["models"])]
            
            if missing_models:
                st.warning(f"⚠️ **Missing default models**: {', '.join(missing_models)}")
                if st.button("📥 Auto-pull Default Models", help="Pull llama3.2:1b and nomic-embed-text"):
                    pull_default_ollama_models(missing_models)
                    st.rerun()
            
            # Manual model installation
            st.markdown("**📥 Install New Model:**")
            col1, col2 = st.columns([3, 1])
            with col1:
                model_name = st.text_input(
                    "Model name",
                    placeholder="e.g., llama3.2:3b, mistral:7b",
                    help="Enter any model available in Ollama registry",
                    key="ollama_model_input"
                )
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Align button
                if st.button("📥 Pull", disabled=not model_name):
                    if model_name:
                        pull_ollama_model(model_name)
                        st.rerun()
        
        else:
            st.error("❌ **Ollama Not Running**")
            st.markdown("""
            **To use Ollama models:**
            1. Install Ollama: `brew install ollama` (macOS) or visit [ollama.ai](https://ollama.ai)
            2. Start Ollama: `ollama serve`
            3. Refresh this page
            """)


def check_ollama_status() -> Dict[str, any]:
    """Check if Ollama is running and get available models"""
    try:
        # Check if Ollama is running by listing models
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            models = parse_ollama_models(result.stdout)
            return {"running": True, "models": models}
        else:
            return {"running": False, "models": []}
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return {"running": False, "models": []}


def parse_ollama_models(output: str) -> List[Dict[str, str]]:
    """Parse Ollama list output into structured model data"""
    models = []
    lines = output.strip().split('\n')[1:]  # Skip header
    
    for line in lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                models.append({
                    "name": parts[0],
                    "id": parts[1] if len(parts) > 1 else "",
                    "size": parts[2] if len(parts) > 2 else "",
                    "modified": " ".join(parts[3:]) if len(parts) > 3 else ""
                })
    
    return models


def pull_ollama_model(model_name: str) -> None:
    """Pull a specific Ollama model"""
    with st.spinner(f"Pulling {model_name}... This may take several minutes."):
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Show progress
            progress_placeholder = st.empty()
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    progress_placeholder.text(f"📥 {output.strip()}")
            
            process.wait()
            
            if process.returncode == 0:
                st.success(f"✅ Successfully pulled {model_name}")
                # Refresh available models
                st.session_state.available_models = get_all_available_models()
            else:
                error_output = process.stderr.read()
                st.error(f"❌ Failed to pull {model_name}: {error_output}")
                
        except Exception as e:
            st.error(f"❌ Error pulling model: {str(e)}")


def pull_default_ollama_models(missing_models: List[str]) -> None:
    """Pull multiple default Ollama models"""
    for model in missing_models:
        st.info(f"📥 Pulling {model}...")
        pull_ollama_model(model)


def remove_ollama_model(model_name: str) -> None:
    """Remove an Ollama model"""
    try:
        result = subprocess.run(
            ["ollama", "rm", model_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            st.success(f"✅ Removed {model_name}")
            # Refresh available models
            st.session_state.available_models = get_all_available_models()
        else:
            st.error(f"❌ Failed to remove {model_name}: {result.stderr}")
            
    except Exception as e:
        st.error(f"❌ Error removing model: {str(e)}")


def get_all_available_models() -> Dict[str, List[Any]]:
    """Get models from both Ollama and LlamaStack - embedding model is fixed to all-MiniLM-L6-v2"""
    embedding_models = []
    llm_models = []
    all_models = []
    
    # Debug info
    debug_info = []
    
    # Fix embedding model to all-MiniLM-L6-v2 (no user selection)
    fixed_embedding = {
        "identifier": "all-MiniLM-L6-v2",
        "name": "all-MiniLM-L6-v2", 
        "source": "LlamaStack (sentence-transformers)"
    }
    embedding_models.append(fixed_embedding)
    all_models.append(fixed_embedding)
    debug_info.append("Using fixed embedding model: all-MiniLM-L6-v2")
    
    # Ensure session state uses the fixed embedding model
    st.session_state.selected_embedding_model = "all-MiniLM-L6-v2"
    
    # 1. Get LLM models from Ollama directly  
    ollama_status = check_ollama_status()
    debug_info.append(f"Ollama Status: {'Running' if ollama_status['running'] else 'Not Running'}")
    
    if ollama_status["running"] and ollama_status["models"]:
        debug_info.append(f"Ollama Models Found: {len(ollama_status['models'])}")
        for model in ollama_status["models"]:
            model_name = model["name"]
            # Only add LLM models, skip embedding models
            if "embed" not in model_name.lower():
                model_info = {
                    "identifier": model_name,
                    "name": model_name,
                    "source": "Ollama"
                }
                llm_models.append(model_info)
                all_models.append(model_info)
                debug_info.append(f"Added LLM model: {model_name}")
    else:
        debug_info.append("No Ollama models found")
    
    # 2. Get LLM models from LlamaStack (if available)
    try:
        llamastack_models = st.session_state.llamastack_client.get_available_models()
        debug_info.append(f"LlamaStack Models: {len(llamastack_models.get('all', []))}")
        
        # Add LlamaStack LLM models that aren't already from Ollama
        for model in llamastack_models.get("llm", []):
            model_name = model.get("identifier", model.get("name", "unknown"))
            
            # Check if we already have this model from Ollama
            if not any(m["identifier"] == model_name for m in llm_models):
                model_info = {
                    "identifier": model_name,
                    "name": model_name,
                    "source": "LlamaStack"
                }
                llm_models.append(model_info)
                all_models.append(model_info)
                debug_info.append(f"Added LLM model from LlamaStack: {model_name}")
    
    except Exception as e:
        debug_info.append(f"LlamaStack Error: {str(e)}")
        print(f"Error getting LlamaStack models: {e}")
    
    # 3. Add default LLM model if none found
    if not llm_models:
        default_llm = {"identifier": "llama3.2:1b", "name": "llama3.2:1b", "source": "Default"}
        llm_models.append(default_llm)
        all_models.append(default_llm)
        debug_info.append("Added default LLM model")
    
    # Show debug info in expander
    with st.expander("🔍 Model Detection Debug", expanded=False):
        for info in debug_info:
            st.text(info)
        st.json({
            "total_models": len(all_models),
            "embedding_models": len(embedding_models), 
            "llm_models": len(llm_models)
        })
    
    return {
        "embedding": embedding_models,
        "llm": llm_models,
        "all": all_models
    }


def get_model_info() -> str:
    """Get formatted model information string"""
    return f"📊 **Models**: all-MiniLM-L6-v2 (embedding) • {st.session_state.selected_llm_model} (LLM)" 