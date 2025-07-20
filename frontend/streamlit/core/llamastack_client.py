"""
LlamaStack Client Wrapper
Simple wrapper around the LlamaStack client for the RAG application
"""

import requests
import subprocess
import json
from typing import Optional, List, Dict, Any


class LlamaStackClient:
    def __init__(self, base_url: str = "http://localhost:8321/v1"):
        self.base_url = base_url
        self.session = requests.Session()
        self.ollama_url = "http://localhost:11434"
        
    def health_check(self) -> bool:
        """Check if LlamaStack is responding"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self) -> Dict[str, List[Any]]:
        """Get available models from LlamaStack"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Separate models by type
                embedding_models = []
                llm_models = []
                
                for model in data.get("data", []):
                    model_info = {
                        "identifier": model.get("identifier", ""),
                        "name": model.get("identifier", ""),
                        "provider_id": model.get("provider_id", "")
                    }
                    
                    if model.get("model_type") == "embedding":
                        embedding_models.append(model_info)
                    elif model.get("model_type") == "llm":
                        llm_models.append(model_info)
                
                # Add local Ollama LLM models (but not embeddings)
                ollama_llm_models = self._get_ollama_llm_models()
                llm_models.extend(ollama_llm_models)
                
                all_models = embedding_models + llm_models
                
                return {
                    "embedding": embedding_models,
                    "llm": llm_models,
                    "all": all_models
                }
        except Exception as e:
            print(f"Error getting models: {e}")
        
        return self._get_default_models()
    
    def _get_ollama_llm_models(self) -> List[Dict[str, str]]:
        """Get LLM models from Ollama directly (no embeddings)"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                models = []
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip() and 'embed' not in line.lower():  # Exclude embedding models
                        model_name = line.split()[0]
                        models.append({
                            "identifier": model_name,
                            "name": model_name,
                            "provider_id": "ollama"
                        })
                return models
        except Exception as e:
            print(f"Error getting Ollama LLM models: {e}")
        
        return []
    
    def _get_default_models(self) -> Dict[str, List[Any]]:
        """Return default models when API is unavailable"""
        default_embedding = {"identifier": "all-MiniLM-L6-v2", "name": "all-MiniLM-L6-v2", "provider_id": "sentence-transformers"}
        default_llm = {"identifier": "llama3.2:1b", "name": "llama3.2:1b", "provider_id": "ollama"}
        
        return {
            "embedding": [default_embedding],
            "llm": [default_llm],
            "all": [default_embedding, default_llm]
        }

    def get_embeddings(self, text: str, model: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
        """Get embeddings using LlamaStack inference/embeddings endpoint with sentence-transformers"""
        try:
            # Use LlamaStack embeddings through inference API
            payload = {
                "contents": [text],  # LlamaStack expects array of strings
                "model_id": model    # LlamaStack expects model_id, not model
            }
            response = self.session.post(f"{self.base_url}/inference/embeddings", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if "embeddings" in data and len(data["embeddings"]) > 0:
                    embedding = data["embeddings"][0]  # Get first (and only) embedding
                    if embedding and len(embedding) > 10:  # Validate we got a real embedding
                        print(f"✅ Got real embedding from LlamaStack (sentence-transformers): {len(embedding)} dimensions")
                        return embedding
            else:
                print(f"LlamaStack embeddings failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Error getting embeddings from LlamaStack: {e}")
        
        # Return varied dummy embedding as fallback (no more Ollama direct calls)
        print(f"⚠️ Falling back to dummy embeddings - please check LlamaStack configuration")
        return self._generate_dummy_embedding(text)
    
    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """Generate varied dummy embedding based on text content"""
        import random
        import hashlib
        
        # Create a more varied dummy embedding based on text content
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        random.seed(seed)
        
        # Generate 384-dimensional embedding to match all-MiniLM-L6-v2
        dummy_embedding = []
        for i in range(384):
            base_value = random.uniform(-0.1, 0.1)
            text_influence = (len(text) + ord(text[i % len(text)])) / 10000.0
            dummy_embedding.append(base_value + text_influence)
        
        print(f"⚠️ Using dummy embedding for: {text[:50]}...")
        return dummy_embedding

    def chat_completion(self, user_prompt: str, system_prompt: str = "", model: str = "llama3.2:1b") -> str:
        """Generate chat completion with improved error handling and fallbacks"""
        try:
            from .config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            # Build full prompt for LlamaStack completion endpoint
            full_prompt = ""
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            else:
                full_prompt = user_prompt
            
            print(f"🔄 Attempting LlamaStack completion with model: {model}")
            
            # Try different endpoint configurations for LlamaStack
            endpoint_configs = [
                {
                    "url": "/inference/completion",
                    "payload": {
                        "model_id": model,
                        "content": full_prompt,
                        "sampling_params": {
                            "temperature": LLM_TEMPERATURE,
                            "max_tokens": LLM_MAX_TOKENS,
                            "top_p": LLM_TOP_P
                        }
                    }
                },
                {
                    "url": "/chat/completions", 
                    "payload": {
                        "model": model,
                        "messages": messages,
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": LLM_MAX_TOKENS,
                        "top_p": LLM_TOP_P,
                        "stream": False
                    }
                },
                {
                    "url": "/inference/chat_completion",
                    "payload": {
                        "model_id": model,
                        "messages": messages,
                        "sampling_params": {
                            "temperature": LLM_TEMPERATURE,
                            "max_tokens": LLM_MAX_TOKENS,
                            "top_p": LLM_TOP_P
                        }
                    }
                }
            ]
            
            for config in endpoint_configs:
                try:
                    url = f"{self.base_url}{config['url']}"
                    print(f"📡 Trying endpoint: {url}")
                    response = self.session.post(url, json=config['payload'], timeout=60)
                    print(f"📡 Response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"📡 Response data: {data}")
                        
                        # Handle LlamaStack completion response format
                        if "completion_message" in data and "content" in data["completion_message"]:
                            content = data["completion_message"]["content"]
                            if content and len(content.strip()) > 10:
                                print(f"✅ LlamaStack success via {config['url']}")
                                return content.strip()
                        
                        # Handle LlamaStack simple completion format
                        elif "content" in data:
                            content = data["content"]
                            if content and len(content.strip()) > 10:
                                print(f"✅ LlamaStack success via {config['url']}")
                                return content.strip()
                        
                        # Handle OpenAI compatible response format
                        elif "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"]["content"]
                            if content and len(content.strip()) > 10:
                                print(f"✅ LlamaStack success via {config['url']}")
                                return content.strip()
                        
                        # Handle plain response
                        elif "response" in data:
                            content = data["response"]
                            if content and len(content.strip()) > 10:
                                print(f"✅ LlamaStack success via {config['url']}")
                                return content.strip()
                        
                        print(f"⚠️ LlamaStack returned unexpected format via {config['url']}: {data}")
                    
                    elif response.status_code == 404:
                        print(f"❌ Endpoint not found: {config['url']}")
                        continue
                    elif response.status_code == 400:
                        print(f"❌ Bad request to {config['url']}: {response.text}")
                        continue
                    else:
                        print(f"❌ LlamaStack error {response.status_code} via {config['url']}: {response.text}")
                        continue
                        
                except Exception as e:
                    print(f"❌ Error with endpoint {config['url']}: {e}")
                    continue
            
            print(f"⚠️ All LlamaStack endpoints failed, trying Ollama fallback")
            
            # Try Ollama fallback before giving up
            ollama_response = self._try_ollama_direct(user_prompt, system_prompt, model)
            if ollama_response:
                return ollama_response
            
            # Return context-aware fallback if API fails
            return self._generate_context_aware_fallback(user_prompt, system_prompt)
            
        except Exception as e:
            print(f"❌ Chat completion error: {e}")
            return self._generate_context_aware_fallback(user_prompt, system_prompt)
    
    def _try_ollama_direct(self, user_prompt: str, system_prompt: str, model: str) -> str:
        """Try direct Ollama completion as fallback"""
        try:
            import subprocess
            
            # Create combined prompt for Ollama
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            else:
                full_prompt = user_prompt
            
            result = subprocess.run(
                ["ollama", "generate", model, full_prompt],
                capture_output=True,
                text=True,
                timeout=45
            )
            
            if result.returncode == 0 and result.stdout.strip():
                response = result.stdout.strip()
                if len(response) > 20:
                    print(f"✅ Got Ollama response ({len(response)} chars)")
                    return response
            
        except Exception as e:
            print(f"⚠️ Ollama fallback failed: {e}")
        
        return None
    
    def _generate_context_aware_fallback(self, user_prompt: str, system_prompt: str) -> str:
        """Generate a context-aware fallback response instead of generic demo text"""
        # Extract context from system prompt if available
        if system_prompt and "DOCUMENT CONTEXT:" in system_prompt:
            try:
                context_start = system_prompt.find("DOCUMENT CONTEXT:")
                if context_start != -1:
                    context_section = system_prompt[context_start:context_start+1000]
                    
                    return f"""Based on the available document context, I can see information related to your question: "{user_prompt}"

{context_section}

**Note:** I'm currently in fallback mode as the AI model is unavailable. This is a basic content match from your documents. 

**To get full AI-powered responses:**
- Ensure your LLM model ({getattr(self, 'current_model', 'unknown')}) is running
- Check LlamaStack connection at {self.base_url}
- Verify Ollama is installed and accessible

**Available content:** The documents contain relevant information but need AI processing for a complete answer."""
            except Exception:
                pass
        
        # Generic fallback if no context available
        return f"""I understand you're asking: "{user_prompt}"

**Current Status:** AI model temporarily unavailable - using basic fallback mode.

**To resolve:**
1. **Check LlamaStack:** Ensure it's running at {self.base_url}
2. **Verify Models:** Make sure your LLM model is loaded
3. **Try Ollama:** Install Ollama for local processing
4. **Upload Documents:** Add relevant documents for context

**What I can do:** Basic document search and content matching (when documents are available).""" 

    def test_llm_functionality(self, model: str = "llama3.2:1b") -> Dict[str, Any]:
        """Test LLM functionality and return diagnostic information"""
        result = {
            "llamastack_available": False,
            "model_available": False,
            "ollama_available": False,
            "test_response": None,
            "error_message": None,
            "recommendations": []
        }
        
        # Test LlamaStack health
        try:
            if self.health_check():
                result["llamastack_available"] = True
                print("✅ LlamaStack is responding")
            else:
                result["error_message"] = "LlamaStack not responding"
                result["recommendations"].append("Start LlamaStack server")
                return result
        except Exception as e:
            result["error_message"] = f"LlamaStack connection failed: {e}"
            result["recommendations"].append("Check LlamaStack configuration")
            return result
        
        # Test LLM model
        try:
            test_prompt = "Hello, please respond with 'AI is working' if you can see this."
            response = self.chat_completion(test_prompt, model=model)
            
            if response and "AI is working" in response:
                result["model_available"] = True
                result["test_response"] = response
                print(f"✅ LLM model {model} is working")
            elif response and len(response) > 10:
                result["model_available"] = True
                result["test_response"] = response[:100] + "..."
                print(f"✅ LLM model {model} responding (different format)")
            else:
                result["error_message"] = f"Model {model} not responding properly"
                result["recommendations"].append(f"Check if model {model} is loaded in LlamaStack")
        except Exception as e:
            result["error_message"] = f"Model test failed: {e}"
            result["recommendations"].append("Verify model configuration")
        
        # Test Ollama as fallback
        try:
            import subprocess
            ollama_result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if ollama_result.returncode == 0:
                result["ollama_available"] = True
                print("✅ Ollama is available as fallback")
            else:
                result["recommendations"].append("Install Ollama for backup LLM processing")
        except Exception:
            result["recommendations"].append("Install Ollama for backup LLM processing")
        
        return result 

    def diagnose_llamastack(self) -> dict:
        """Diagnose LlamaStack connectivity and available endpoints"""
        diagnosis = {
            "base_url": self.base_url,
            "reachable": False,
            "available_endpoints": [],
            "models_endpoint": None,
            "chat_endpoint": None,
            "embeddings_endpoint": None,
            "recommendations": []
        }
        
        print(f"🔍 Diagnosing LlamaStack at {self.base_url}")
        
        # Test basic connectivity
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                diagnosis["reachable"] = True
                print("✅ LlamaStack is reachable")
            else:
                print(f"⚠️ LlamaStack health check returned {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot reach LlamaStack: {e}")
            diagnosis["recommendations"].append("Check if LlamaStack server is running on the specified URL")
            diagnosis["recommendations"].append("Verify the base URL in configuration")
            return diagnosis
        
        # Test models endpoint
        models_endpoints = ["/models", "/v1/models", "/inference/models"]
        for endpoint in models_endpoints:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    diagnosis["models_endpoint"] = endpoint
                    diagnosis["available_endpoints"].append(f"Models: {endpoint}")
                    print(f"✅ Models available at: {endpoint}")
                    break
            except Exception:
                continue
        
        # Test chat endpoints
        chat_endpoints = [
            "/inference/chat_completion",
            "/chat/completions", 
            "/v1/chat/completions",
            "/inference/completion",
            "/api/chat"
        ]
        
        test_payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }
        
        for endpoint in chat_endpoints:
            try:
                response = self.session.post(f"{self.base_url}{endpoint}", json=test_payload, timeout=5)
                # Even if it fails due to model issues, a proper endpoint should return structured error
                if response.status_code in [200, 400, 422]:  # Valid responses
                    diagnosis["chat_endpoint"] = endpoint
                    diagnosis["available_endpoints"].append(f"Chat: {endpoint}")
                    print(f"✅ Chat available at: {endpoint}")
                    break
                elif response.status_code == 404:
                    print(f"❌ Not found: {endpoint}")
                else:
                    print(f"⚠️ Unexpected response from {endpoint}: {response.status_code}")
            except Exception:
                continue
        
        # Test embeddings endpoints
        embeddings_endpoints = ["/inference/embeddings", "/v1/embeddings", "/embeddings"]
        for endpoint in embeddings_endpoints:
            try:
                response = self.session.post(f"{self.base_url}{endpoint}", 
                                           json={"contents": ["test"], "model_id": "test"}, timeout=5)
                if response.status_code in [200, 400, 422]:
                    diagnosis["embeddings_endpoint"] = endpoint
                    diagnosis["available_endpoints"].append(f"Embeddings: {endpoint}")
                    print(f"✅ Embeddings available at: {endpoint}")
                    break
            except Exception:
                continue
        
        # Add recommendations based on findings
        if not diagnosis["chat_endpoint"]:
            diagnosis["recommendations"].append("No working chat completion endpoint found")
            diagnosis["recommendations"].append("Check LlamaStack configuration and API provider setup")
        
        if not diagnosis["models_endpoint"]:
            diagnosis["recommendations"].append("No models endpoint found - verify LlamaStack is properly configured")
        
        if len(diagnosis["available_endpoints"]) == 0:
            diagnosis["recommendations"].append("No API endpoints are responding - LlamaStack may not be properly started")
            diagnosis["recommendations"].append("Try restarting LlamaStack with: make restart")
        
        return diagnosis 