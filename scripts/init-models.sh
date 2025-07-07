#!/bin/bash

# Initialize Ollama models for RAG Pipeline
echo "🚀 Initializing Ollama models for RAG Pipeline..."

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama service to be ready..."
until curl -f http://ollama:11434/api/tags >/dev/null 2>&1; do
    echo "   Ollama not ready yet, waiting 5 seconds..."
    sleep 5
done

echo "✅ Ollama service is ready!"

# Function to pull model with retry logic
pull_model() {
    local model=$1
    local max_retries=3
    local retry=0
    
    echo "📥 Pulling model: $model"
    
    while [ $retry -lt $max_retries ]; do
        if curl -X POST http://ollama:11434/api/pull \
           -H "Content-Type: application/json" \
           -d "{\"name\":\"$model\"}" \
           --max-time 1800; then  # 30 minute timeout
            echo "✅ Successfully pulled $model"
            return 0
        else
            retry=$((retry + 1))
            echo "❌ Failed to pull $model (attempt $retry/$max_retries)"
            if [ $retry -lt $max_retries ]; then
                echo "   Retrying in 10 seconds..."
                sleep 10
            fi
        fi
    done
    
    echo "💥 Failed to pull $model after $max_retries attempts"
    return 1
}

# Pull required models
echo "📦 Pulling required models..."

# Embedding model (smaller, pull first)
pull_model "nomic-embed-text"

# LLM model (smaller, pull second)
pull_model "llama3.2:3b"

echo ""
echo "🎉 Model initialization complete!"
echo "📋 Available models:"
curl -s http://ollama:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "   Unable to list models (jq not available)"

echo ""
echo "🚀 RAG Pipeline is ready to use!"
echo "   Web UI: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs" 