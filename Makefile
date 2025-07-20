.PHONY: help install dev test lint format clean setup llamastack-start llamastack-stop llamastack-restart frontend-setup frontend start stop restart status health podman-build podman-up podman-down

# Default target
help:
	@echo "🦙 RAG NotebookLM with LlamaStack"
	@echo "================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make setup    - Install dependencies and configure everything"
	@echo "  make start    - Start LlamaStack + Frontend (complete app)"
	@echo "  make status   - Check service health and show URLs"
	@echo "  make stop     - Stop all services"
	@echo ""
	@echo "🔧 Individual Services:"
	@echo "  make llamastack-start  - Start LlamaStack server only"
	@echo "  make llamastack-stop   - Stop LlamaStack server"
	@echo "  make frontend         - Start Streamlit frontend only"
	@echo "  make frontend-stop    - Stop Streamlit frontend"
	@echo ""
	@echo "🛠️  Development:"
	@echo "  make test     - Run tests"
	@echo "  make health   - Check service health"
	@echo "  make restart  - Restart everything"
	@echo ""
	@echo "🐳 Container Commands:"
	@echo "  make podman-up    - Start with Podman containers"
	@echo "  make podman-down  - Stop containers"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt

# Complete setup (run this first)
setup: install frontend-setup
	mkdir -p data/{documents,vectors,models,cache} logs llamastack/data/vectors
	@echo "✅ Setup complete! You can now run: make start"

# Setup Streamlit configuration (prevents first-time setup issues)
frontend-setup:
	@mkdir -p ~/.streamlit
	@echo "[general]" > ~/.streamlit/config.toml
	@echo "email = \"\"" >> ~/.streamlit/config.toml
	@echo "" >> ~/.streamlit/config.toml
	@echo "[server]" >> ~/.streamlit/config.toml
	@echo "headless = true" >> ~/.streamlit/config.toml
	@echo "port = 8501" >> ~/.streamlit/config.toml
	@echo "" >> ~/.streamlit/config.toml
	@echo "[browser]" >> ~/.streamlit/config.toml
	@echo "gatherUsageStats = false" >> ~/.streamlit/config.toml
	@echo "✅ Streamlit configuration created"

# Run development server
dev:
	source venv/bin/activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Stop any existing Streamlit processes
frontend-stop:
	@echo "🔄 Stopping Streamlit..."
	@pkill -f "streamlit run" 2>/dev/null || echo "No Streamlit process found"

# Run Streamlit frontend (NotebookLM-style RAG interface)
frontend: frontend-stop
	@echo "🎨 Starting RAG NotebookLM frontend..."
	source venv/bin/activate && streamlit run frontend/streamlit/app.py --server.port 8501

# Start complete RAG application (LlamaStack + Frontend)
start: llamastack-start
	@echo "⏳ Waiting for LlamaStack to be ready..."
	@sleep 5
	@make frontend

# Stop everything
stop: llamastack-stop frontend-stop
	@echo "🛑 All services stopped"

# Restart everything
restart: stop start

# Run tests
test:
	pytest backend/tests/ -v

# Code formatting
format:
	black backend/ frontend/
	isort backend/ frontend/

# Development helpers
dev-setup: install
	mkdir -p data/{documents,vectors,models,cache} logs
	cp .env.example .env
	echo "Setup complete! Edit .env file with your configuration."

# Health check for all services
health:
	@echo "🔍 Checking service health..."
	@echo "LlamaStack (port 8321):"
	@curl -s http://localhost:8321/v1/providers >/dev/null 2>&1 && echo "  ✅ Running" || echo "  ❌ Not responding"
	@echo "Streamlit Frontend (port 8501):"
	@curl -s http://localhost:8501 >/dev/null 2>&1 && echo "  ✅ Running" || echo "  ❌ Not responding"

# Show application status and URLs
status: health
	@echo ""
	@echo "📱 Application URLs:"
	@echo "  • RAG NotebookLM UI: http://localhost:8501"
	@echo "  • LlamaStack API: http://localhost:8321"
	@echo "  • LlamaStack Docs: http://localhost:8321/docs"
	@echo ""
	@echo "🏗️  Architecture: Frontend → LlamaStack (no intermediate backend needed)"
	@echo ""

# LlamaStack server management
llamastack-status:
	@curl -s http://localhost:8321/v1/providers >/dev/null 2>&1 && echo "✅ LlamaStack is running" || echo "❌ LlamaStack is not running"

llamastack-stop:
	@echo "🔄 Stopping LlamaStack..."
	@pkill -f "llama stack run" 2>/dev/null || echo "No LlamaStack process found"
	@sleep 2

llamastack-start: llamastack-stop
	@echo "🚀 Starting LlamaStack..."
	cd llamastack && source ../venv/bin/activate && llama stack run config/llamastack-config.yaml &
	@echo "⏳ Waiting for LlamaStack to start..."
	@sleep 10
	@curl -s http://localhost:8321/v1/providers >/dev/null 2>&1 && echo "✅ LlamaStack started successfully" || echo "⚠️  LlamaStack may still be starting..."

llamastack-restart: llamastack-stop llamastack-start

# Podman commands (using files in podman/ directory)
podman-build:
	cd podman && podman-compose build

podman-up:
	cd podman && podman-compose up -d

podman-down:
	cd podman && podman-compose down

podman-logs:
	cd podman && podman-compose logs -f

# Alternative: Use podman directly
podman-build-direct:
	podman build -f podman/Containerfile.api -t rag-notebooklm:latest .

podman-run-direct:
	podman run -d -p 8000:8000 -p 8501:8501 -p 5001:5001 --name rag-app rag-notebooklm:latest

podman-stop-direct:
	podman stop rag-app && podman rm rag-app

# Clean Podman resources
podman-clean:
	podman system prune -f
