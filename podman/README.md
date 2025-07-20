# 🐙 Podman Container Configuration

This directory contains all container-related files for the RAG NotebookLM application.

## Files

- **`podman-compose.yml`** - Multi-service orchestration configuration
- **`Containerfile.api`** - FastAPI backend container (M4 optimized)
- **`Containerfile.llamastack`** - LlamaStack service container (M4 optimized)  
- **`Containerfile.frontend`** - Streamlit frontend container
- **`.containerignore`** - Build context optimization

## Usage

```bash
# From project root directory:

# Build all services
make podman-build

# Start services
make podman-up

# View logs
make podman-logs

# Stop services
make podman-down
```

## Direct Podman Commands

```bash
# Build specific service
podman build -f podman/Containerfile.api -t rag-api .
podman build -f podman/Containerfile.llamastack -t rag-llamastack .
podman build -f podman/Containerfile.frontend -t rag-frontend .

# Run multi-service stack
cd podman && podman-compose up -d
```

All containers are optimized for M4 MacBooks with ARM64 architecture and include health checks. 