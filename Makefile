# RAG LlamaStack - Streamlit Edition
# Makefile for easy development and deployment

.PHONY: help setup setup-mcp install start stop clean test health status logs

# Default target
help:
	@echo "🦙 RAG LlamaStack - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup        - Complete setup (Python + MCP server)"
	@echo "  setup-mcp    - Setup MCP server for web content processing"
	@echo "  install      - Install Python dependencies only"
	@echo ""
	@echo "Run Commands:"
	@echo "  start        - Start the Streamlit application"
	@echo "  start-dev    - Start with debug logging"
	@echo ""
	@echo "LlamaStack Commands:"
	@echo "  llamastack   - Start LlamaStack server"
	@echo "  ollama       - Start Ollama server"
	@echo ""
	@echo "Test Commands:"
	@echo "  test         - Run Python tests"
	@echo "  test-web     - Test web content processing integration"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  health       - Check system health"
	@echo "  status       - Show service status"

# Complete setup including MCP server
setup: install setup-mcp
	@echo "✅ Complete setup finished!"
	@echo "   You can now run: make start"

# Setup MCP server for web content processing
setup-mcp:
	@echo "🔧 Setting up MCP server for web content processing..."
	@./setup_mcp.sh

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt

# Start the Streamlit application
start:
	@echo "🚀 Starting RAG LlamaStack application..."
	@echo "🌐 Open http://localhost:8501 in your browser"
	streamlit run frontend/streamlit/app.py --server.port 8501

# Start with debug logging
start-dev:
	@echo "🐛 Starting in development mode with debug logging..."
	STREAMLIT_LOGGER_LEVEL=debug streamlit run frontend/streamlit/app.py --server.port 8501

# Start LlamaStack server
llamastack:
	@echo "🦙 Starting LlamaStack server..."
	llamastack run ./llamastack/config/llamastack-config.yaml

# Start Ollama server
ollama:
	@echo "🏠 Starting Ollama server..."
	ollama serve

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

# Test web content processing integration
test-web:
	@echo "🌐 Testing web content processing integration..."
	python test_web_integration.py

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .streamlit/cache 2>/dev/null || true
	rm -rf node_modules/.cache 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Check system health
health:
	@echo "🩺 Checking system health..."
	@echo ""
	@echo "Python version:"
	@python --version || echo "❌ Python not found"
	@echo ""
	@echo "Node.js version:"
	@node --version || echo "❌ Node.js not found"
	@echo ""
	@echo "MCP server status:"
	@npx @just-every/mcp-read-website-fast --version 2>/dev/null || echo "❌ MCP server not available"
	@echo ""
	@echo "Streamlit status:"
	@streamlit --version || echo "❌ Streamlit not found"
	@echo ""
	@echo "LlamaStack status:"
	@curl -s http://localhost:8321/v1/health > /dev/null && echo "✅ LlamaStack running" || echo "❌ LlamaStack not running"
	@echo ""
	@echo "Ollama status:"
	@curl -s http://localhost:11434/api/version > /dev/null && echo "✅ Ollama running" || echo "❌ Ollama not running"

# Show service status
status: health

# Show logs (if any)
logs:
	@echo "📋 Recent logs:"
	@ls -la logs/ 2>/dev/null || echo "No logs directory found"
