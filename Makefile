.PHONY: install dev test lint format clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run development server
dev:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit frontend
frontend:
	streamlit run frontend/streamlit/app.py --server.port 8501

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

# Health check
health:
	curl -s http://localhost:8000/health
	curl -s http://localhost:5001/health

# LlamaStack commands
llamastack-start:
	cd llamastack && llama stack run config/m4_config.yaml
