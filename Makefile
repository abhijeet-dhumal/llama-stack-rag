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
	@echo "Service Commands (for separate terminals):"
	@echo "  ollama       - Start Ollama server (Terminal 1)"
	@echo "  llamastack   - Start LlamaStack server (Terminal 2)"
	@echo "  mcp          - Check MCP server status (Terminal 3, optional)"
	@echo "  start-frontend - Start Streamlit frontend (Terminal 4)"
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
	@echo "  4. make start (starts all services + frontend)"
	@echo ""
	@echo "🌐 Optional: make setup-mcp (for web URL processing)"
	@echo ""
	@echo "🔧 Multi-Terminal Development Flow:"
	@echo "  1. make setup"
	@echo "  2. source venv/bin/activate"
	@echo "  3. make install"
	@echo "  4. Terminal 1: make ollama"
	@echo "  5. Terminal 2: make llamastack"
	@echo "  6. Terminal 3: make start-frontend"
	@echo ""
	@echo "🌐 Optional: Web Content Processing"
	@echo "  make setup-mcp (for web URL processing)"
	@echo "  Terminal 4: make mcp (check MCP status)"
	@echo ""
	@echo "📋 Benefits of separate terminals:"
	@echo "  • Better debugging and log visibility"
	@echo "  • Independent service control"
	@echo "  • Easier troubleshooting"
	@echo "  • Service-specific monitoring"

# Complete setup
setup: venv
	@echo "✅ Virtual environment created!"
	@echo "   Please activate it with: source venv/bin/activate"
	@echo "   Then run: make install"
	@echo ""
	@echo "📋 Complete setup steps:"
	@echo "   1. source venv/bin/activate"
	@echo "   2. make install"
	@echo "   3. make start (or use multi-terminal flow)"
	@echo ""
	@echo "🌐 Optional: Web Content Processing"
	@echo "   make setup-mcp (for web URL processing)"

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
	@if npx @just-every/mcp-read-website-fast --help &> /dev/null; then \
		echo "✅ MCP server installed successfully!"; \
		echo ""; \
		echo "🌐 Testing web content extraction..."; \
		echo "   Test URL: https://example.com"; \
		timeout 10s npx @just-every/mcp-read-website-fast fetch "https://example.com" --output markdown > /dev/null 2>&1; \
		if [ $$? -eq 0 ]; then \
			echo "✅ Web content extraction test passed!"; \
			echo ""; \
			echo "🎉 MCP server setup complete!"; \
			echo "   You can now use web URLs in the RAG application"; \
			echo ""; \
			echo "📖 Usage:"; \
			echo "   1. Start the application: make start"; \
			echo "   2. Go to 'Web URLs' tab in the sidebar"; \
			echo "   3. Enter any web URL to extract and process content"; \
		else \
			echo "⚠️  MCP server installed but test failed (this might be normal)"; \
			echo "   Try using the application - fallback method will work if needed"; \
		fi; \
	else \
		echo "❌ MCP server installation failed"; \
		echo "   The application will use fallback method (BeautifulSoup)"; \
		echo "   Make sure you have the required Python packages:"; \
		echo "   pip install beautifulsoup4 markdownify requests"; \
	fi
	@echo ""
	@echo "🔧 Troubleshooting:"
	@echo "   - If MCP server fails, the app will automatically use fallback"
	@echo "   - Check that your firewall allows outbound HTTP/HTTPS connections"
	@echo "   - Some websites may block automated access"
	@echo ""
	@echo "📚 Supported by this setup:"
	@echo "   ✅ News articles and blog posts"
	@echo "   ✅ Documentation pages"
	@echo "   ✅ Wikipedia articles"
	@echo "   ✅ Most static content websites"

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

# Start LlamaStack server (for separate terminal)
llamastack:
	@echo "🦙 Starting LlamaStack server..."
	@echo "   Port: 8321"
	@echo "   Config: ./llamastack/config/llamastack-config.yaml"
	@echo "   Health check: http://localhost:8321/v1/health"
	@echo ""
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Virtual environment not detected!"; \
		echo "   Activating virtual environment..."; \
		source venv/bin/activate && llama stack run ./llamastack/config/llamastack-config.yaml; \
	else \
		echo "✅ Virtual environment detected: $$VIRTUAL_ENV"; \
		llama stack run ./llamastack/config/llamastack-config.yaml; \
	fi

# Start Ollama server (for separate terminal)
ollama:
	@echo "🏠 Starting Ollama server..."
	@echo "   Port: 11434"
	@echo "   Health check: http://localhost:11434/api/version"
	@echo "   Models: http://localhost:11434/api/tags"
	@echo ""
	ollama serve

# Start MCP server (for separate terminal)
mcp:
	@echo "🔧 Starting MCP server for web content processing..."
	@echo "   Testing MCP server availability..."
	@if npx @just-every/mcp-read-website-fast --version &> /dev/null; then \
		echo "✅ MCP server is available"; \
		echo "   Usage: npx @just-every/mcp-read-website-fast fetch <URL> --output markdown"; \
		echo ""; \
		echo "🌐 MCP server is ready for web content extraction!"; \
		echo "   The Streamlit app will automatically use this service."; \
		echo ""; \
		echo "📖 Test command:"; \
		echo "   npx @just-every/mcp-read-website-fast fetch https://example.com --output markdown"; \
	else \
		echo "❌ MCP server not available"; \
		echo "   Run 'make setup-mcp' to install it"; \
		echo "   The app will use fallback method (BeautifulSoup)"; \
	fi

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
