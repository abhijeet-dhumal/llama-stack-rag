# RAG LlamaStack - Streamlit Edition
# Makefile for easy development and deployment

.PHONY: help setup setup-mcp install start start-frontend start-dev stop stop-streamlit stop-llamastack stop-ollama restart llamastack ollama test test-web clean clean-all health status logs

# Default target
help:
	@echo "🦙 RAG LlamaStack - Available Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup        - Create virtual environment (step 1)"
	@echo "  venv         - Create Python virtual environment"
	@echo "  install      - Install Python dependencies (step 2)"
	@echo "  setup-mcp    - Setup MCP server for web content processing (step 3)"
	@echo ""
	@echo "Run Commands:"
	@echo "  start        - Start all services + Streamlit (step 4)"
	@echo "  start-frontend - Start just Streamlit (services must be running)"
	@echo "  start-dev    - Start with debug logging"
	@echo "  stop         - Stop all running services"
	@echo "  stop-streamlit - Stop only Streamlit"
	@echo "  stop-llamastack - Stop only LlamaStack"
	@echo "  stop-ollama  - Stop only Ollama"
	@echo "  restart      - Restart all services"
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
	@echo "  clean-all    - Full cleanup (cache, venv, node_modules)"
	@echo "  health       - Check system health"
	@echo "  status       - Show service status"
	@echo ""
	@echo "📋 Quick Start:"
	@echo "  1. make setup"
	@echo "  2. source venv/bin/activate"
	@echo "  3. make install"
	@echo "  4. make setup-mcp"
	@echo "  5. make start (starts all services + frontend)"
	@echo ""
	@echo "🔧 Alternative Flow (manual service control):"
	@echo "  1. make setup"
	@echo "  2. source venv/bin/activate"
	@echo "  3. make install"
	@echo "  4. make setup-mcp"
	@echo "  5. make ollama (in one terminal)"
	@echo "  6. make llamastack (in another terminal)"
	@echo "  7. make start-frontend (in third terminal)"

# Complete setup including MCP server
setup: venv
	@echo "✅ Virtual environment created!"
	@echo "   Please activate it with: source venv/bin/activate"
	@echo "   Then run: make install setup-mcp"
	@echo ""
	@echo "📋 Complete setup steps:"
	@echo "   1. source venv/bin/activate"
	@echo "   2. make install"
	@echo "   3. make setup-mcp"
	@echo "   4. make start"

# Setup MCP server for web content processing
setup-mcp:
	@echo "🚀 Setting up MCP server for web content processing..."
	@echo ""
	@echo "🔍 Checking Node.js installation..."
	@if ! command -v node &> /dev/null; then \
		echo "❌ Node.js is not installed. Please install Node.js (>=16.0.0) first:"; \
		echo "   Visit: https://nodejs.org/"; \
		exit 1; \
	fi
	@NODE_VERSION=$$(node --version | cut -d'v' -f2); \
	echo "✅ Node.js version: $$NODE_VERSION"
	@echo ""
	@echo "📦 Installing MCP server dependencies..."
	@npm install
	@echo ""
	@echo "🧪 Testing MCP server installation..."
	@echo "   Testing Just-Every MCP (reliable markdown extraction)..."
	@timeout 15s npx @just-every/mcp-read-website-fast fetch "https://example.com" --output markdown > /dev/null 2>&1; \
	if [ $$? -eq 0 ]; then \
		echo "✅ Just-Every MCP web content extraction test passed!"; \
		echo ""; \
		echo "🎉 MCP server setup complete!"; \
		echo "   🥇 Primary: Just-Every MCP (reliable markdown extraction)"; \
		echo "   🥈 Fallback: BeautifulSoup (Python fallback)"; \
		echo ""; \
		echo "📖 Usage:"; \
		echo "   1. Start the application: make start"; \
		echo "   2. Go to 'Web URLs' tab in the sidebar"; \
		echo "   3. Enter any web URL to extract and process content"; \
		echo "   4. The system will use Just-Every MCP with BeautifulSoup fallback"; \
	else \
		echo "⚠️  Just-Every MCP test failed, will use BeautifulSoup fallback"; \
		echo ""; \
		echo "🎉 MCP server setup complete!"; \
		echo "   Primary: Just-Every MCP (no API key required)"; \
		echo "   Backup: BeautifulSoup (Python fallback)"; \
	fi
	@echo ""
	@echo "🔧 Troubleshooting:"
	@echo "   - Just-Every MCP works without API keys (100% free!)"
	@echo "   - Just-Every MCP is reliable markdown extraction"
	@echo "   - If MCP server fails, the app will use BeautifulSoup fallback"
	@echo "   - Check that your firewall allows outbound HTTP/HTTPS connections"
	@echo ""
	@echo "📚 Supported by this setup:"
	@echo "   ✅ GitHub README files (raw.githubusercontent.com)"
	@echo "   ✅ Complex web applications"
	@echo "   ✅ News articles and blog posts"
	@echo "   ✅ Documentation pages"
	@echo "   ✅ Wikipedia articles"
	@echo "   ✅ Most static and dynamic content websites"

# Install Python dependencies
install:
	@echo "🐍 Installing Python dependencies..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Virtual environment not detected!"; \
		echo "   Creating virtual environment..."; \
		python -m venv venv; \
		echo "   Please activate it with: source venv/bin/activate"; \
		echo "   Then run: make install"; \
		exit 1; \
	fi
	@echo "✅ Virtual environment detected: $$VIRTUAL_ENV"
	pip install -r requirements.txt

# Create virtual environment
venv:
	@echo "🐍 Creating Python virtual environment..."
	python -m venv venv
	@echo "✅ Virtual environment created!"
	@echo "   Activate it with: source venv/bin/activate"
	@echo "   Then run: make install"

# Start the Streamlit application
start:
	@echo "🚀 Starting RAG LlamaStack application..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Virtual environment not activated!"; \
		echo "   Please activate it first: source venv/bin/activate"; \
		exit 1; \
	fi
	@if ! command -v streamlit &> /dev/null; then \
		echo "❌ Streamlit not found!"; \
		echo "   Please install dependencies: make install"; \
		exit 1; \
	fi
	@echo "✅ Virtual environment: $$VIRTUAL_ENV"
	@echo "✅ Streamlit available"
	@echo ""
	@echo "🔧 Starting required services..."
	@echo "   Starting Ollama..."
	@ollama serve > /dev/null 2>&1 &
	@sleep 3
	@echo "   Starting LlamaStack..."
	@source venv/bin/activate && llama stack run ./llamastack/config/llamastack-config.yaml > /dev/null 2>&1 &
	@sleep 3
	@echo "✅ Services started!"
	@echo "🌐 Open http://localhost:8501 in your browser"
	streamlit run frontend/streamlit/app.py --server.port 8501

# Start with debug logging
start-dev:
	@echo "🐛 Starting in development mode with debug logging..."
	STREAMLIT_LOGGER_LEVEL=debug streamlit run frontend/streamlit/app.py --server.port 8501

# Start just the Streamlit frontend (services must be running)
start-frontend:
	@echo "🚀 Starting Streamlit frontend only..."
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Virtual environment not activated!"; \
		echo "   Please activate it first: source venv/bin/activate"; \
		exit 1; \
	fi
	@if ! command -v streamlit &> /dev/null; then \
		echo "❌ Streamlit not found!"; \
		echo "   Please install dependencies: make install"; \
		exit 1; \
	fi
	@echo "✅ Virtual environment: $$VIRTUAL_ENV"
	@echo "✅ Streamlit available"
	@echo "⚠️  Make sure Ollama and LlamaStack are running first!"
	@echo "🌐 Open http://localhost:8501 in your browser"
	streamlit run frontend/streamlit/app.py --server.port 8501

# Stop all running services
stop:
	@echo "🛑 Stopping all running services..."
	@echo "   Stopping Streamlit processes..."
	@pkill -f "streamlit run frontend/streamlit/app.py" 2>/dev/null || echo "   No Streamlit processes found"
	@echo "   Stopping LlamaStack processes..."
	@pkill -f "llama stack run" 2>/dev/null || echo "   No LlamaStack processes found"
	@echo "   Stopping Ollama processes..."
	@pkill -f "ollama serve" 2>/dev/null || echo "   No Ollama processes found"
	@echo "   Stopping any remaining Python processes..."
	@pkill -f "python.*streamlit" 2>/dev/null || echo "   No Python Streamlit processes found"
	@echo "✅ All services stopped"

# Restart all services
restart: stop
	@echo "🔄 Restarting services..."
	@echo "   Starting Ollama..."
	@ollama serve > /dev/null 2>&1 &
	@sleep 3
	@echo "   Starting LlamaStack..."
	@source venv/bin/activate && llama stack run ./llamastack/config/llamastack-config.yaml > /dev/null 2>&1 &
	@sleep 3
	@echo "   Starting Streamlit..."
	@echo "✅ Services restarted! Run 'make start' to open the application"

# Stop individual services
stop-streamlit:
	@echo "🛑 Stopping Streamlit..."
	@pkill -f "streamlit run frontend/streamlit/app.py" 2>/dev/null || echo "   No Streamlit processes found"
	@echo "✅ Streamlit stopped"

stop-llamastack:
	@echo "🛑 Stopping LlamaStack..."
	@pkill -f "llama stack run" 2>/dev/null || echo "   No LlamaStack processes found"
	@echo "✅ LlamaStack stopped"

stop-ollama:
	@echo "🛑 Stopping Ollama..."
	@pkill -f "ollama serve" 2>/dev/null || echo "   No Ollama processes found"
	@echo "✅ Ollama stopped"

# Start LlamaStack server
llamastack:
	@echo "🦙 Starting LlamaStack server..."
	@echo "   Activating virtual environment..."
	@source venv/bin/activate && llama stack run ./llamastack/config/llamastack-config.yaml

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
	@echo ""
	@echo "🔍 Testing MCP server availability..."
	@if npx @just-every/mcp-read-website-fast --version &> /dev/null; then \
		echo "✅ MCP server is available"; \
		echo ""; \
		echo "🧪 Testing web content extraction..."; \
		echo "   Test URL: https://example.com"; \
		timeout 15s npx @just-every/mcp-read-website-fast fetch "https://example.com" --output markdown > /dev/null 2>&1; \
		if [ $$? -eq 0 ]; then \
			echo "✅ Web content extraction test passed!"; \
			echo "   MCP server is working correctly"; \
		else \
			echo "⚠️  Web content extraction test failed"; \
			echo "   This might be due to network issues or timeouts"; \
		fi; \
	else \
		echo "❌ MCP server not available"; \
		echo "   Run 'make setup-mcp' to install it"; \
	fi
	@echo ""
	@echo "🔍 Testing Python dependencies..."
	@python -c "import requests, bs4, markdownify; print('✅ All Python dependencies available')" 2>/dev/null || echo "❌ Missing Python dependencies - run 'pip install beautifulsoup4 markdownify requests'"
	@echo ""
	@echo "🌐 Web integration test complete!"

# Test improved telemetry integration
test-telemetry:
	@echo "📊 Testing improved telemetry integration..."
	@echo ""
	@echo "🧪 Running comprehensive telemetry tests..."
	@python test_improved_telemetry.py
	@echo ""
	@echo "📊 Telemetry integration test complete!"

# Clean cache and temporary files
clean:
	@echo "🧹 Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .streamlit/cache 2>/dev/null || true
	rm -rf node_modules/.cache 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Full cleanup (cache, venv, node_modules)
clean-all: stop
	@echo "🧹🧹 Full cleanup - removing all generated files and dependencies..."
	@echo "   Removing Python cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "   Removing Streamlit cache..."
	rm -rf .streamlit/cache 2>/dev/null || true
	@echo "   Removing Node.js cache..."
	rm -rf node_modules/.cache 2>/dev/null || true
	@echo "   Removing virtual environment..."
	rm -rf venv 2>/dev/null || true
	@echo "   Removing Node.js dependencies..."
	rm -rf node_modules 2>/dev/null || true
	@echo "   Removing package-lock.json..."
	rm -f package-lock.json 2>/dev/null || true
	@echo "   Removing vector database files..."
	rm -f data/vectors/*.db 2>/dev/null || true
	@echo "   Removing log files..."
	rm -rf logs/* 2>/dev/null || true
	@echo "✅ Full cleanup complete!"
	@echo "   To start fresh, run: make setup"

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
