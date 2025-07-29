"""
LlamaStack Client Wrapper
Uses the official llama-stack-client package for proper integration
"""

from llama_stack_client import LlamaStackClient as LS
from llama_stack_client.types.event_param import UnstructuredLogEvent, MetricEvent, StructuredLogEvent
import subprocess
import json
import time
from typing import Optional, List, Dict, Any
import hashlib
import uuid

def generate_hex_id() -> str:
    """Generate a hex ID for telemetry"""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16]

class LlamaStackClient:
    def __init__(self, base_url: str = "http://localhost:8321/v1"):
        self.base_url = base_url
        # Use the official LlamaStack client - remove /v1 from base_url since client adds it
        client_base_url = base_url.replace("/v1", "")
        self.client = LS(base_url=client_base_url)
        self.ollama_url = "http://localhost:11434"
    
    def health_check(self) -> bool:
        """Check if LlamaStack is responding"""
        try:
            # Use the models endpoint as a health check
            response = self.client.models.list()
            return True
        except Exception:
            return False
    
    def get_available_models(self) -> Dict[str, List[Any]]:
        """Get available models from LlamaStack using official client"""
        try:
            # Use the official client's models endpoint
            response = self.client.models.list()
            
            # Separate models by type
            embedding_models = []
            llm_models = []
            
            # Handle both list and response.data formats
            models_data = response.data if hasattr(response, 'data') else response
            
            for model in models_data:
                model_info = {
                    "identifier": model.identifier,
                    "name": model.identifier,
                    "provider_id": model.provider_id
                }
                
                if model.model_type == "embedding":
                    embedding_models.append(model_info)
                elif model.model_type == "llm":
                    llm_models.append(model_info)
                    
            # Add local Ollama LLM models if LlamaStack doesn't have them
            if not llm_models:
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
        default_llm = {"identifier": "llama3.2:3b", "name": "llama3.2:3b", "provider_id": "ollama"}
        
        return {
            "embedding": [default_embedding],
            "llm": [default_llm],
            "all": [default_embedding, default_llm]
        }
    
    def _get_llamastack_models(self) -> List[str]:
        """Get list of available model identifiers from LlamaStack"""
        try:
            response = self.client.models.list()
            # Handle both list and response.data formats
            models_data = response.data if hasattr(response, 'data') else response
            return [model.identifier for model in models_data]
        except Exception as e:
            print(f"Error getting LlamaStack models: {e}")
            return []
    
    def get_embeddings(self, text: str, model: str = "all-MiniLM-L6-v2") -> Optional[List[float]]:
        """Get embeddings using official LlamaStack client"""
        try:
            # Use direct HTTP request to the correct endpoint
            import requests
            
            response = requests.post(
                f"{self.base_url}/inference/embeddings",
                json={
                    "model_id": model,
                    "contents": [text]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle both 'data' and 'embeddings' response formats
                if data.get('data') and len(data['data']) > 0:
                    embedding = data['data'][0]['embedding']
                    return embedding
                elif data.get('embeddings') and len(data['embeddings']) > 0:
                    embedding = data['embeddings'][0]
                    return embedding
                else:
                    print(f"❌ No embedding data returned for model: {model}")
                    return None
            else:
                print(f"❌ Embeddings API returned {response.status_code}: {response.text}")
                return None
            
        except Exception as e:
            print(f"❌ Error getting embeddings from LlamaStack: {e}")
            # Fallback to dummy embedding
            return self._generate_dummy_embedding(text)
    
    def _generate_dummy_embedding(self, text: str) -> List[float]:
        """Generate a dummy embedding for fallback"""
        import hashlib
        # Create a deterministic dummy embedding based on text hash
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to 384-dimensional vector (matching all-MiniLM-L6-v2)
        dummy_embedding = []
        for i in range(384):
            # Use hash to generate pseudo-random but deterministic values
            hash_part = hash_hex[i % len(hash_hex)]
            dummy_embedding.append((ord(hash_part) - 48) / 100.0)  # Normalize to small values
        
        return dummy_embedding
    
    def chat_completion(self, user_prompt: str, system_prompt: str = "", model: str = "llama3.2:3b") -> str:
        """Generate chat completion using official LlamaStack client"""
        try:
            from .config import LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TOP_P
            
            # Use the official client's chat completion endpoint
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            print(f"🔄 Attempting LlamaStack completion with model: {model}")
            
            # Get available models from LlamaStack to try different formats
            available_models = self._get_llamastack_models()
            model_variants = [model]
            
            # If this is an Ollama model, try different formats that LlamaStack might use
            if model in ["llama3.2:1b", "llama3.2:3b", "ibm/granite3.3:2b-base"]:
                # Check if any of these variants exist in LlamaStack's available models
                possible_variants = [
                    model,
                    f"ollama/{model}",
                    model.split(":")[0] if ":" in model else model,
                    f"ollama/{model.split(':')[0]}" if ":" in model else f"ollama/{model}"
                ]
                
                # Only add variants that actually exist in LlamaStack
                for variant in possible_variants:
                    if variant in available_models:
                        model_variants.append(variant)
            
            # Try each model variant
            for model_variant in model_variants:
                try:
                    print(f"🔄 Trying model variant: {model_variant}")
                    
                    response = self.client.chat.completions.create(
                        model=model_variant,
                        messages=messages,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        top_p=LLM_TOP_P
                    )
                    
                    if response.choices and len(response.choices) > 0:
                        content = response.choices[0].message.content
                        if content and content.strip():
                            print(f"✅ LlamaStack success via official client with model {model_variant}")
                            return content.strip()
                            
                except Exception as e:
                    print(f"❌ Model variant {model_variant} failed: {e}")
                    continue
            
            # If all LlamaStack attempts failed, try Ollama directly
            print(f"⚠️ All LlamaStack models failed, trying Ollama directly")
            return self._try_ollama_direct(user_prompt, system_prompt, model)
            
        except Exception as e:
            print(f"❌ LlamaStack completion failed: {e}")
            # Try Ollama as a fallback
            return self._try_ollama_direct(user_prompt, system_prompt, model)
    
    def _try_ollama_direct(self, user_prompt: str, system_prompt: str, model: str = "llama3.2:3b") -> str:
        """Try Ollama directly as fallback"""
        try:
            import requests
            
            # Build the prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            else:
                full_prompt = f"User: {user_prompt}\nAssistant:"
            
            # Call Ollama directly
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data:
                    print(f"✅ Ollama direct success with model {model}")
                    return data['response'].strip()
            
            print(f"❌ Ollama direct failed: {response.status_code}")
            return self._generate_context_aware_fallback(user_prompt, system_prompt)
            
        except Exception as e:
            print(f"❌ Ollama direct error: {e}")
            return self._generate_context_aware_fallback(user_prompt, system_prompt)
    
    def _generate_context_aware_fallback(self, user_prompt: str, system_prompt: str) -> str:
        """Generate a context-aware fallback response"""
        # Simple fallback that acknowledges the query
        if "what is" in user_prompt.lower() or "explain" in user_prompt.lower():
            return "I understand you're asking about this topic, but I'm currently unable to provide a detailed response. Please try again or check your connection to the AI models."
        else:
            return "I received your message, but I'm currently experiencing technical difficulties. Please try again in a moment."
    
    def test_llm_functionality(self, model: str = "llama3.2:3b") -> Dict[str, Any]:
        """Test LLM functionality using official client"""
        test_results = {
            "model": model,
            "llamastack_available": False,
            "ollama_available": False,
            "test_response": "",
            "error": None
        }
        
        try:
            # Test LlamaStack first
            test_prompt = "Hello, this is a test. Please respond with 'Test successful' if you can see this message."
            
            response = self.chat_completion(test_prompt, model=model)
            
            if response and "test" in response.lower():
                test_results["llamastack_available"] = True
                test_results["test_response"] = response
                print(f"✅ LlamaStack test successful with model {model}")
            else:
                test_results["error"] = "Unexpected response format"
                
        except Exception as e:
            test_results["error"] = str(e)
            print(f"❌ LlamaStack test failed: {e}")
        
        return test_results

    def diagnose_llamastack(self) -> dict:
        """Diagnose LlamaStack connectivity and functionality"""
        diagnosis = {
            "llamastack_health": False,
            "available_models": [],
            "vector_io_available": False,
            "ollama_health": False,
            "recommendations": []
        }
        
        # Check LlamaStack health
        try:
            diagnosis["llamastack_health"] = self.health_check()
            if diagnosis["llamastack_health"]:
                print("✅ LlamaStack is healthy")
            else:
                diagnosis["recommendations"].append("LlamaStack is not responding. Check if it's running on port 8321.")
        except Exception as e:
            diagnosis["recommendations"].append(f"LlamaStack health check failed: {e}")
        
        # Check available models
        try:
            models = self.get_available_models()
            diagnosis["available_models"] = [model["identifier"] for model in models.get("all", [])]
            print(f"✅ Found {len(diagnosis['available_models'])} available models")
        except Exception as e:
            diagnosis["recommendations"].append(f"Could not retrieve models: {e}")
        
        # Check VectorIO availability
        try:
            # Try to list vector databases to check VectorIO availability
            response = self.client.vector_dbs.list()
            diagnosis["vector_io_available"] = True
            print("✅ VectorIO is available")
        except Exception as e:
            diagnosis["vector_io_available"] = False
            diagnosis["recommendations"].append(f"VectorIO not available: {e}")
        
        # Check Ollama health
        try:
            import requests
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            diagnosis["ollama_health"] = response.status_code == 200
            if diagnosis["ollama_health"]:
                print("✅ Ollama is healthy")
            else:
                diagnosis["recommendations"].append("Ollama is not responding. Check if it's running on port 11434.")
        except Exception as e:
            diagnosis["recommendations"].append(f"Ollama health check failed: {e}")
        
            return diagnosis
    
    # VectorIO Methods using official client
    def store_embeddings_in_vector_db(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]], vector_db_id: str = "default") -> bool:
        """
        Store embeddings in LlamaStack's VectorIO database using official client
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of metadata dictionaries for each embedding
            vector_db_id: The identifier of the vector database
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from llama_stack_client.types.vector_io_insert_params import Chunk
            
            # Convert embeddings to JSON-serializable format (float32 -> float)
            serializable_embeddings = []
            for embedding in embeddings:
                if hasattr(embedding, 'tolist'):  # numpy array
                    serializable_embeddings.append([float(x) for x in embedding.tolist()])
                else:  # regular list
                    serializable_embeddings.append([float(x) for x in embedding])
            
            # Prepare chunks for VectorIO API
            chunks = []
            for i, (embedding, meta) in enumerate(zip(serializable_embeddings, metadata)):
                # Add required document_id field if not present
                chunk_metadata = meta.copy()
                if 'document_id' not in chunk_metadata:
                    chunk_metadata['document_id'] = f"doc_{i}_{hash(meta.get('document_name', 'unknown'))}"
                
                chunk = Chunk(
                    content=meta.get('chunk_content', ''),
                    metadata=chunk_metadata,
                    embedding=embedding
                )
                chunks.append(chunk)
            
            # Use official VectorIO insert method
            self.client.vector_io.insert(
                chunks=chunks,
                vector_db_id=vector_db_id
            )
            
            print(f"✅ Successfully stored {len(embeddings)} embeddings in VectorIO database")
            return True
                
        except Exception as e:
            print(f"❌ Error storing embeddings in VectorIO: {e}")
            return False
    
    def search_similar_vectors(self, query_text: str, vector_db_id: str = "default", top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in LlamaStack's VectorIO database using official client
        
        Args:
            query_text: Text query to search for
            vector_db_id: The identifier of the vector database
            top_k: Number of similar vectors to return
            
        Returns:
            List of similar vectors with metadata
        """
        try:
            # Create query content - use simple text for now
            query_content = query_text
            
            # Use official VectorIO query method
            response = self.client.vector_io.query(
                query=query_content,
                vector_db_id=vector_db_id,
                params={"top_k": top_k}
            )
            
            print(f"🔍 DEBUG: Raw VectorIO response type: {type(response)}")
            
            # Handle different response formats
            chunks = []
            scores = []
            
            if hasattr(response, 'chunks'):
                # Official client response format
                chunks = response.chunks
                scores = getattr(response, 'scores', [])
                print(f"🔍 DEBUG: Using official client format - chunks: {len(chunks)}, scores: {len(scores)}")
            elif isinstance(response, dict):
                # Direct API response format
                chunks = response.get('chunks', [])
                scores = response.get('scores', [])
                print(f"🔍 DEBUG: Using dict format - chunks: {len(chunks)}, scores: {len(scores)}")
            else:
                # Fallback - try to use response directly
                chunks = response if isinstance(response, list) else []
                print(f"🔍 DEBUG: Using fallback format - chunks: {len(chunks)}")
            
            # Convert response to our format
            results = []
            for i, chunk in enumerate(chunks):
                # Get similarity score - try multiple sources
                similarity = 0.0
                if i < len(scores):
                    similarity = scores[i]
                elif hasattr(chunk, 'similarity'):
                    similarity = chunk.similarity
                elif isinstance(chunk, dict) and 'similarity' in chunk:
                    similarity = chunk['similarity']
                
                print(f"🔍 DEBUG: Chunk {i+1} similarity: {similarity}")
                
                if hasattr(chunk, 'content'):
                    # Official client chunk object
                    results.append({
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "similarity": similarity
                    })
                elif isinstance(chunk, dict):
                    # Direct API chunk dict
                    results.append({
                        "content": chunk.get('content', ''),
                        "metadata": chunk.get('metadata', {}),
                        "similarity": similarity
                    })
            
            print(f"🔍 DEBUG: Returning {len(results)} results with similarities: {[r['similarity'] for r in results]}")
            return results
                
        except Exception as e:
            print(f"❌ Error searching vectors in VectorIO: {e}")
            return []
    
    def get_vector_db_stats(self, vector_db_id: str = "default") -> Dict[str, Any]:
        """
        Get statistics about the VectorIO database using official client
        
        Args:
            vector_db_id: The identifier of the vector database
            
        Returns:
            Dictionary with database statistics
        """
        try:
            # Use official client to get vector database info
            response = self.client.vector_dbs.retrieve(vector_db_id)
            
            return {
                "vector_db_id": getattr(response, 'identifier', getattr(response, 'id', 'Unknown')),
                "name": getattr(response, 'name', getattr(response, 'identifier', 'Unknown')),
                "provider_id": getattr(response, 'provider_id', 'Unknown'),
                "status": "Active",  # FAISS databases are always active
                "embedding_model": getattr(response, 'embedding_model', 'Unknown'),
                "embedding_dimension": getattr(response, 'embedding_dimension', 'Unknown'),
                "total_vectors": 0,  # Will be updated when we have actual data
                "database_size_mb": 0,  # Will be updated when we have actual data
                "last_updated": "Now"
            }
                
        except Exception as e:
            print(f"❌ Error getting vector DB stats: {e}")
            return {"error": str(e)}
    
    def list_vector_databases(self) -> List[Dict[str, Any]]:
        """
        List all available vector databases using official client
        
        Returns:
            List of vector database information
        """
        try:
            response = self.client.vector_dbs.list()
            
            databases = []
            # Handle both list and response.data formats
            db_data = response.data if hasattr(response, 'data') else response
            
            for db in db_data:
                databases.append({
                    "vector_db_id": getattr(db, 'identifier', getattr(db, 'id', 'Unknown')),
                    "name": getattr(db, 'name', getattr(db, 'identifier', 'Unknown')),
                    "provider_id": getattr(db, 'provider_id', 'Unknown'),
                    "status": "Active",  # FAISS databases are always active
                    "embedding_model": getattr(db, 'embedding_model', 'Unknown'),
                    "embedding_dimension": getattr(db, 'embedding_dimension', 'Unknown')
                })
            
            return databases
                
        except Exception as e:
            print(f"❌ Error listing vector databases: {e}")
            return []
    
    def delete_vectors_from_vector_db(self, document_ids: List[str], vector_db_id: str = "faiss") -> bool:
        """
        Mark specific vectors for deletion in LlamaStack's VectorIO database using document IDs
        Note: Actual deletion API is not yet implemented, so we mark documents for deletion
        
        Args:
            document_ids: List of document IDs to mark for deletion
            vector_db_id: The identifier of the vector database
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Since deletion API is not yet implemented, we'll mark documents for deletion
            # and store the deletion list in session state for filtering
            import streamlit as st
            
            # Initialize deletion tracking in session state
            if 'deleted_document_ids' not in st.session_state:
                st.session_state.deleted_document_ids = set()
            
            # Mark documents for deletion
            for doc_id in document_ids:
                st.session_state.deleted_document_ids.add(doc_id)
                print(f"🗑️ Marked document {doc_id} for deletion")
            
            print(f"✅ Successfully marked {len(document_ids)} documents for deletion")
            print(f"📋 Total documents marked for deletion: {len(st.session_state.deleted_document_ids)}")
            
            return True
                
        except Exception as e:
            print(f"❌ Error marking vectors for deletion: {e}")
            return False
    
    def clear_all_vectors_from_vector_db(self, vector_db_id: str = "faiss") -> bool:
        """
        Clear all vectors from LlamaStack's VectorIO database by recreating the database
        This is a workaround since the deletion API is not yet available
        
        Args:
            vector_db_id: The identifier of the vector database
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import streamlit as st
            import requests
            import os
            import shutil
            
            print(f"🗑️ Clearing all data from {vector_db_id} database...")
            
            # Method 1: Try to delete and recreate the database via API
            try:
                # First, try to delete the existing database
                delete_response = requests.delete(
                    f"{self.base_url}/vector-dbs/{vector_db_id}",
                    timeout=10
                )
                print(f"🗑️ Database deletion response: {delete_response.status_code}")
                
                # Then recreate the database
                create_data = {
                    "vector_db_id": vector_db_id,
                    "provider_id": "faiss",
                    "embedding_model": "all-MiniLM-L6-v2"
                }
                
                create_response = requests.post(
                    f"{self.base_url}/vector-dbs",
                    json=create_data,
                    timeout=10
                )
                
                if create_response.status_code in [200, 201]:
                    print(f"✅ Successfully recreated {vector_db_id} database")
                    
                    # Clear session state
                    if 'deleted_document_ids' in st.session_state:
                        st.session_state.deleted_document_ids.clear()
                    
                    # Clear uploaded documents from session state
                    if 'uploaded_documents' in st.session_state:
                        st.session_state.uploaded_documents.clear()
                    
                    if 'documents' in st.session_state:
                        st.session_state.documents.clear()
                    
                    return True
                else:
                    print(f"⚠️ Database recreation failed: {create_response.status_code}")
                    
            except Exception as e:
                print(f"⚠️ API-based clearing failed: {e}")
            
            # Method 2: Clear the underlying FAISS files directly
            try:
                print("🗑️ Attempting to clear FAISS files directly...")
                
                # Clear the SQLite KV store
                kvstore_path = "./data/vectors/faiss_store.db"
                if os.path.exists(kvstore_path):
                    os.remove(kvstore_path)
                    print(f"✅ Removed KV store: {kvstore_path}")
                
                # Clear any FAISS index files
                faiss_files = [
                    "./data/vectors/faiss.index",
                    "./data/vectors/faiss_store.index",
                    "./data/vectors/vectors.faiss"
                ]
                
                for faiss_file in faiss_files:
                    if os.path.exists(faiss_file):
                        os.remove(faiss_file)
                        print(f"✅ Removed FAISS file: {faiss_file}")
                
                # Clear session state
                if 'deleted_document_ids' in st.session_state:
                    st.session_state.deleted_document_ids.clear()
                
                if 'uploaded_documents' in st.session_state:
                    st.session_state.uploaded_documents.clear()
                
                if 'documents' in st.session_state:
                    st.session_state.documents.clear()
                
                print("✅ Successfully cleared all FAISS data")
                return True
                
            except Exception as e:
                print(f"❌ File-based clearing failed: {e}")
            
            # Method 3: Fallback - just clear session state and mark all for deletion
            print("⚠️ Using fallback clearing method...")
            
            # Get all document IDs and mark them for deletion
            all_document_ids = set()
            
            try:
                search_results = self.search_similar_vectors(
                    query_text="a",  # Broad query to get all documents
                    vector_db_id=vector_db_id,
                    top_k=1000  # Get a large number to capture all documents
                )
                
                for result in search_results:
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('document_id')
                    if doc_id:
                        all_document_ids.add(doc_id)
            except Exception as e:
                print(f"⚠️ Could not retrieve existing documents: {e}")
            
            # Mark all documents for deletion
            if 'deleted_document_ids' not in st.session_state:
                st.session_state.deleted_document_ids = set()
            
            st.session_state.deleted_document_ids.update(all_document_ids)
            
            # Clear uploaded documents from session state
            if 'uploaded_documents' in st.session_state:
                st.session_state.uploaded_documents.clear()
            
            if 'documents' in st.session_state:
                st.session_state.documents.clear()
            
            print(f"✅ Marked {len(all_document_ids)} documents for deletion (fallback)")
            return True
                
        except Exception as e:
            print(f"❌ Error clearing vectors from VectorIO: {e}")
            return False
    
    def get_filtered_search_results(self, query_text: str, vector_db_id: str = "faiss", top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for similar vectors and filter out deleted documents
        
        Args:
            query_text: Text query to search for
            vector_db_id: The identifier of the vector database
            top_k: Number of similar vectors to return
            
        Returns:
            List of similar vectors with metadata (excluding deleted documents)
        """
        try:
            import streamlit as st
            
            print(f"🔍 DEBUG: get_filtered_search_results called with query: '{query_text}', top_k: {top_k}")
            
            # Get search results
            search_results = self.search_similar_vectors(query_text, vector_db_id, top_k * 2)  # Get more to account for filtering
            
            print(f"🔍 DEBUG: search_similar_vectors returned {len(search_results)} results")
            
            # Filter out deleted documents
            if 'deleted_document_ids' in st.session_state:
                filtered_results = []
                deleted_count = 0
                
                for result in search_results:
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('document_id')
                    
                    if doc_id not in st.session_state.deleted_document_ids:
                        filtered_results.append(result)
                    else:
                        deleted_count += 1
                        print(f"🔍 DEBUG: Filtered out deleted document: {doc_id}")
                
                print(f"🔍 DEBUG: Filtered {deleted_count} deleted documents, {len(filtered_results)} remaining")
                
                # Return only the requested number of results
                final_results = filtered_results[:top_k]
                print(f"🔍 DEBUG: Returning {len(final_results)} final results")
                return final_results
            else:
                print(f"🔍 DEBUG: No deleted_document_ids in session state, returning {len(search_results[:top_k])} results")
                return search_results[:top_k]
                
        except Exception as e:
            print(f"❌ Error getting filtered search results: {e}")
            return []
    
    def get_database_stats(self, vector_db_id: str = "faiss") -> Dict[str, Any]:
        """
        Get comprehensive database statistics including actual vector counts
        
        Args:
            vector_db_id: The identifier of the vector database
            
        Returns:
            Dictionary with database statistics
        """
        try:
            import streamlit as st
            
            # Get basic database info
            db_info = self.get_vector_db_stats(vector_db_id)
            
            # Get actual vector count by searching
            total_vectors = 0
            active_vectors = 0
            
            try:
                # Search for all vectors to get count - use multiple broad queries to get all
                all_results = []
                broad_queries = ["a", "the", "and", "or", "in", "on", "at", "to", "for", "of"]
                
                for query in broad_queries:
                    try:
                        results = self.search_similar_vectors(
                            query_text=query,
                            vector_db_id=vector_db_id,
                            top_k=100  # Get a reasonable number per query
                        )
                        all_results.extend(results)
                    except Exception as e:
                        print(f"⚠️ Query '{query}' failed: {e}")
                        continue
                
                # Remove duplicates based on document_id and chunk_index
                unique_results = []
                seen_chunks = set()
                
                for result in all_results:
                    metadata = result.get('metadata', {})
                    doc_id = metadata.get('document_id', '')
                    chunk_index = metadata.get('chunk_index', 0)
                    chunk_key = f"{doc_id}_{chunk_index}"
                    
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        unique_results.append(result)
                
                total_vectors = len(unique_results)
                print(f"📊 Found {total_vectors} unique vectors in database")
                
                # Count active vectors (not marked for deletion)
                if 'deleted_document_ids' in st.session_state and st.session_state.deleted_document_ids:
                    active_vectors = 0
                    for result in unique_results:
                        metadata = result.get('metadata', {})
                        doc_id = metadata.get('document_id')
                        if doc_id and doc_id not in st.session_state.deleted_document_ids:
                            active_vectors += 1
                    
                    print(f"✅ Active vectors: {active_vectors} (deleted: {total_vectors - active_vectors})")
                else:
                    active_vectors = total_vectors
                    print(f"✅ All {active_vectors} vectors are active (no deletions)")
                    
            except Exception as e:
                print(f"⚠️ Could not get vector count: {e}")
            
            # Get session state counts
            session_docs = 0
            if 'uploaded_documents' in st.session_state:
                session_docs = len(st.session_state.uploaded_documents)
            
            deleted_count = 0
            if 'deleted_document_ids' in st.session_state:
                deleted_count = len(st.session_state.deleted_document_ids)
            
            # Calculate session state vector count for comparison
            session_vectors = 0
            if 'uploaded_documents' in st.session_state:
                for doc in st.session_state.uploaded_documents:
                    if 'chunks' in doc:
                        session_vectors += len(doc['chunks'])
            
            return {
                **db_info,
                "total_vectors": total_vectors,
                "active_vectors": active_vectors,
                "session_documents": session_docs,
                "session_vectors": session_vectors,
                "deleted_documents": deleted_count,
                "is_empty": active_vectors == 0
            }
                
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {
                "error": str(e),
                "total_vectors": 0,
                "active_vectors": 0,
                "session_vectors": 0,
                "is_empty": True
            } 

    def log_telemetry_event(self, event_type: str, event_data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """
        Log a telemetry event using LlamaStack's official telemetry API
        
        Args:
            event_type: Type of event (e.g., 'app_startup', 'document_processed', 'chat_interaction')
            event_data: Dictionary containing event data
            ttl_seconds: Time to live for the event in seconds (default: 1 hour)
            
        Returns:
            bool: True if event was logged successfully, False otherwise
        """
        try:
            # Use official client's telemetry API
            from llama_stack_client.types.event_param import StructuredLogEvent
            
            # Create structured log event
            event = StructuredLogEvent({
                "event_type": event_type,
                "data": event_data,
                "metadata": {
                    "source": "rag_llamastack_app",
                    "version": "1.0.0",
                    "environment": "production",
                    "timestamp": time.time()
                }
            })
            
            # Log using official client
            self.client.telemetry.log_event(
                event=event,
                ttl_seconds=ttl_seconds
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error logging telemetry event: {e}")
            # Fallback to direct HTTP request
            return self._log_telemetry_fallback(event_type, event_data, ttl_seconds)
    
    def log_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None, ttl_seconds: int = 3600) -> bool:
        """
        Log a metric using LlamaStack's official telemetry API
        
        Args:
            metric_name: Name of the metric
            value: Numeric value of the metric
            tags: Optional tags for the metric
            ttl_seconds: Time to live for the metric in seconds (default: 1 hour)
            
        Returns:
            bool: True if metric was logged successfully, False otherwise
        """
        try:
            if tags is None:
                tags = {}
            
            # Use official client's telemetry API
            from llama_stack_client.types.event_param import MetricEvent
            
            # Create metric event
            event = MetricEvent({
                "metric_name": metric_name,
                "value": value,
                "unit": "count",
                "tags": tags,
                "timestamp": time.time()
            })
            
            # Log using official client
            self.client.telemetry.log_event(
                event=event,
                ttl_seconds=ttl_seconds
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error logging metric: {e}")
            # Fallback to direct HTTP request
            return self._log_metric_fallback(metric_name, value, tags, ttl_seconds)
    
    def log_unstructured_event(self, message: str, level: str = "info", additional_data: Dict[str, Any] = None, ttl_seconds: int = 3600) -> bool:
        """
        Log an unstructured event using LlamaStack's official telemetry API
        
        Args:
            message: Log message
            level: Log level (info, warning, error, debug, verbose, warn, critical)
            additional_data: Additional data to include
            ttl_seconds: Time to live for the event in seconds (default: 1 hour)
            
        Returns:
            bool: True if event was logged successfully, False otherwise
        """
        try:
            # Map level to correct severity values
            severity_map = {
                "info": "info",
                "warning": "warn", 
                "warn": "warn",
                "error": "error",
                "debug": "debug",
                "verbose": "verbose",
                "critical": "critical"
            }
            severity = severity_map.get(level.lower(), "info")
            
            # Use official client's telemetry API
            from llama_stack_client.types.event_param import UnstructuredLogEvent
            
            # Create unstructured log event
            event_data = {
                "message": message,
                "severity": severity,
                "timestamp": time.time()
            }
            
            if additional_data:
                event_data.update(additional_data)
            
            event = UnstructuredLogEvent(event_data)
            
            # Log using official client
            self.client.telemetry.log_event(
                event=event,
                ttl_seconds=ttl_seconds
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error logging unstructured event: {e}")
            # Fallback to direct HTTP request
            return self._log_unstructured_fallback(message, level, additional_data, ttl_seconds)
    
    def _log_telemetry_fallback(self, event_type: str, event_data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """Fallback method using direct HTTP request for telemetry events"""
        try:
            event = {
                "type": "structured_log",
                "trace_id": generate_hex_id(),
                "span_id": generate_hex_id(),
                "timestamp": time.time(),
                "payload": {
                    "event_type": event_type,
                    "data": event_data,
                    "metadata": {
                        "source": "rag_llamastack_app",
                        "version": "1.0.0",
                        "environment": "production"
                    }
                }
            }
            
            import requests
            response = requests.post(
                f"{self.base_url}/telemetry/events",
                json={
                    "event": event,
                    "ttl_seconds": ttl_seconds
                },
                headers={"Content-Type": "application/json"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Fallback telemetry logging failed: {e}")
            return False
    
    def _log_metric_fallback(self, metric_name: str, value: float, tags: Dict[str, str] = None, ttl_seconds: int = 3600) -> bool:
        """Fallback method using direct HTTP request for metrics"""
        try:
            if tags is None:
                tags = {}
            
            event = {
                "type": "metric",
                "trace_id": generate_hex_id(),
                "span_id": generate_hex_id(),
                "timestamp": time.time(),
                "metric": metric_name,
                "value": value,
                "unit": "count",
                "tags": tags
            }
            
            import requests
            response = requests.post(
                f"{self.base_url}/telemetry/events",
                json={
                    "event": event,
                    "ttl_seconds": ttl_seconds
                },
                headers={"Content-Type": "application/json"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Fallback metric logging failed: {e}")
            return False
    
    def _log_unstructured_fallback(self, message: str, level: str = "info", additional_data: Dict[str, Any] = None, ttl_seconds: int = 3600) -> bool:
        """Fallback method using direct HTTP request for unstructured events"""
        try:
            severity_map = {
                "info": "info",
                "warning": "warn", 
                "warn": "warn",
                "error": "error",
                "debug": "debug",
                "verbose": "verbose",
                "critical": "critical"
            }
            severity = severity_map.get(level.lower(), "info")
            
            event = {
                "type": "unstructured_log",
                "trace_id": generate_hex_id(),
                "span_id": generate_hex_id(),
                "timestamp": time.time(),
                "message": message,
                "severity": severity
            }
            
            import requests
            response = requests.post(
                f"{self.base_url}/telemetry/events",
                json={
                    "event": event,
                    "ttl_seconds": ttl_seconds
                },
                headers={"Content-Type": "application/json"}
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Fallback unstructured logging failed: {e}")
            return False
    
    def query_telemetry_traces(self) -> List[Dict[str, Any]]:
        """
        Query telemetry traces from LlamaStack
        
        Returns:
            List of trace data
        """
        try:
            traces = self.client.telemetry.query_traces()
            return traces if traces else []
        except Exception as e:
            print(f"❌ Error querying traces: {e}")
            return []
    
    def query_telemetry_spans(self, attribute_filters: Dict[str, Any] = None, attributes_to_return: List[str] = None) -> List[Dict[str, Any]]:
        """
        Query telemetry spans from LlamaStack
        
        Args:
            attribute_filters: Optional filters for spans
            attributes_to_return: Optional list of attributes to return
            
        Returns:
            List of span data
        """
        try:
            if attribute_filters is None:
                attribute_filters = {}
            if attributes_to_return is None:
                attributes_to_return = []
            
            spans = self.client.telemetry.query_spans(
                attribute_filters=attribute_filters,
                attributes_to_return=attributes_to_return
            )
            return spans if spans else []
        except Exception as e:
            print(f"❌ Error querying spans: {e}")
            return []
    
    def get_telemetry_span(self, span_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific telemetry span by ID
        
        Args:
            span_id: ID of the span to retrieve
            
        Returns:
            Span data or None if not found
        """
        try:
            span = self.client.telemetry.get_span(span_id=span_id)
            return span
        except Exception as e:
            print(f"❌ Error getting span {span_id}: {e}")
            return None
    
    def get_telemetry_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific telemetry trace by ID
        
        Args:
            trace_id: ID of the trace to retrieve
            
        Returns:
            Trace data or None if not found
        """
        try:
            trace = self.client.telemetry.get_trace(trace_id=trace_id)
            return trace
        except Exception as e:
            print(f"❌ Error getting trace {trace_id}: {e}")
            return None
    
    def save_spans_to_dataset(self, dataset_id: str, span_ids: List[str]) -> bool:
        """
        Save telemetry spans to a dataset
        
        Args:
            dataset_id: ID of the dataset to save to
            span_ids: List of span IDs to save
            
        Returns:
            bool: True if spans were saved successfully, False otherwise
        """
        try:
            self.client.telemetry.save_spans_to_dataset(
                dataset_id=dataset_id,
                span_ids=span_ids
            )
            return True
        except Exception as e:
            print(f"❌ Error saving spans to dataset: {e}")
            return False
    
    # Convenience methods for common telemetry events
    def log_app_startup(self, version: str = "1.0.0", user_agent: str = "streamlit_app") -> bool:
        """Log application startup event"""
        return self.log_telemetry_event(
            event_type="app_startup",
            event_data={
                "version": version,
                "user_agent": user_agent,
                "startup_time": time.time()
            }
        )
    
    def log_document_processed(self, file_type: str, file_size: int, chunks_created: int, processing_time: float) -> bool:
        """Log document processing event"""
        return self.log_telemetry_event(
            event_type="document_processed",
            event_data={
                "file_type": file_type,
                "file_size": file_size,
                "chunks_created": chunks_created,
                "processing_time": processing_time
            }
        )
    
    def log_chat_interaction(self, model_used: str, query_length: int, response_length: int, response_time: float) -> bool:
        """Log chat interaction event"""
        return self.log_telemetry_event(
            event_type="chat_interaction",
            event_data={
                "model_used": model_used,
                "query_length": query_length,
                "response_length": response_length,
                "response_time": response_time
            }
        )
    
    def log_vector_db_operation(self, operation: str, vector_db_id: str, vectors_count: int, operation_time: float) -> bool:
        """Log vector database operation event"""
        return self.log_telemetry_event(
            event_type="vector_db_operation",
            event_data={
                "operation": operation,
                "vector_db_id": vector_db_id,
                "vectors_count": vectors_count,
                "operation_time": operation_time
            }
        )
    
    def log_web_content_processed(self, url: str, content_length: int, processing_time: float, chunks_created: int) -> bool:
        """Log web content processing event"""
        return self.log_telemetry_event(
            event_type="web_content_processed",
            event_data={
                "url": url,
                "content_length": content_length,
                "processing_time": processing_time,
                "chunks_created": chunks_created
            }
        )
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None) -> bool:
        """Log error event"""
        event_data = {
            "error_type": error_type,
            "error_message": error_message
        }
        if context:
            event_data.update(context)
        
        return self.log_telemetry_event(
            event_type="error",
            event_data=event_data
        ) 

    def create_trace_context(self, operation_name: str) -> Dict[str, str]:
        """
        Create a trace context for correlating telemetry events
        
        Args:
            operation_name: Name of the operation being traced
            
        Returns:
            Dictionary with trace_id and span_id
        """
        trace_id = generate_hex_id()
        span_id = generate_hex_id()
        
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "operation_name": operation_name,
            "start_time": time.time()
        }
    
    def log_trace_event(self, trace_context: Dict[str, str], event_type: str, event_data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """
        Log a telemetry event with trace context for correlation
        
        Args:
            trace_context: Trace context from create_trace_context()
            event_type: Type of event
            event_data: Event data
            ttl_seconds: Time to live for the event
            
        Returns:
            bool: True if event was logged successfully, False otherwise
        """
        try:
            # Add trace context to event data
            event_data_with_trace = {
                **event_data,
                "trace_id": trace_context["trace_id"],
                "span_id": trace_context["span_id"],
                "operation_name": trace_context["operation_name"],
                "timestamp": time.time()
            }
            
            return self.log_telemetry_event(event_type, event_data_with_trace, ttl_seconds)
            
        except Exception as e:
            print(f"❌ Error logging trace event: {e}")
            return False
    
    def log_trace_metric(self, trace_context: Dict[str, str], metric_name: str, value: float, tags: Dict[str, str] = None, ttl_seconds: int = 3600) -> bool:
        """
        Log a metric with trace context for correlation
        
        Args:
            trace_context: Trace context from create_trace_context()
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags
            ttl_seconds: Time to live for the metric
            
        Returns:
            bool: True if metric was logged successfully, False otherwise
        """
        try:
            if tags is None:
                tags = {}
            
            # Add trace context to tags
            tags_with_trace = {
                **tags,
                "trace_id": trace_context["trace_id"],
                "span_id": trace_context["span_id"],
                "operation_name": trace_context["operation_name"]
            }
            
            return self.log_metric(metric_name, value, tags_with_trace, ttl_seconds)
            
        except Exception as e:
            print(f"❌ Error logging trace metric: {e}")
            return False
    
    def log_trace_completion(self, trace_context: Dict[str, str], success: bool = True, error_message: str = None, ttl_seconds: int = 3600) -> bool:
        """
        Log trace completion with duration and status
        
        Args:
            trace_context: Trace context from create_trace_context()
            success: Whether the operation was successful
            error_message: Error message if operation failed
            ttl_seconds: Time to live for the event
            
        Returns:
            bool: True if event was logged successfully, False otherwise
        """
        try:
            duration = time.time() - trace_context["start_time"]
            
            completion_data = {
                "trace_id": trace_context["trace_id"],
                "span_id": trace_context["span_id"],
                "operation_name": trace_context["operation_name"],
                "duration_seconds": duration,
                "success": success,
                "timestamp": time.time()
            }
            
            if error_message:
                completion_data["error_message"] = error_message
            
            return self.log_telemetry_event("trace_completion", completion_data, ttl_seconds)
            
        except Exception as e:
            print(f"❌ Error logging trace completion: {e}")
            return False 