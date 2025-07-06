#!/bin/bash

# RAG Pipeline Startup Script
# Provides easy commands to start, stop, and manage the RAG pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_PORT=8000
DEFAULT_HOST="0.0.0.0"
OLLAMA_MODELS=("llama3.2:1b" "nomic-embed-text")

# Global variables for container runtime
CONTAINER_CMD=""
COMPOSE_CMD=""

# Functions
print_banner() {
    echo -e "${GREEN}"
    echo "=================================================="
    echo "🚀 Local RAG Pipeline with Ollama & Docling"
    echo "=================================================="
    echo -e "${NC}"
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found${NC}"
        exit 1
    fi
    
    # Check Ollama
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}Error: Ollama not found. Please install from https://ollama.com${NC}"
        exit 1
    fi
    
    # Check container runtime (Podman preferred, Docker fallback)
    if command -v podman &> /dev/null; then
        export CONTAINER_CMD="podman"
        export COMPOSE_CMD="podman-compose"
        echo -e "${GREEN}✓ Found Podman${NC}"
        
        # Check if podman-compose is available, fallback to podman with docker-compose
        if ! command -v podman-compose &> /dev/null; then
            if command -v docker-compose &> /dev/null; then
                echo -e "${YELLOW}Note: Using docker-compose with podman compatibility${NC}"
                alias docker=podman
                export COMPOSE_CMD="docker-compose"
            else
                echo -e "${YELLOW}Warning: Neither podman-compose nor docker-compose found. Container deployment limited${NC}"
                export COMPOSE_CMD=""
            fi
        fi
    elif command -v docker &> /dev/null; then
        export CONTAINER_CMD="docker"
        export COMPOSE_CMD="docker-compose"
        echo -e "${GREEN}✓ Found Docker${NC}"
    else
        echo -e "${YELLOW}Warning: Neither Podman nor Docker found. Container deployment won't be available${NC}"
        export CONTAINER_CMD=""
        export COMPOSE_CMD=""
    fi
    
    echo -e "${GREEN}✓ Dependencies check passed${NC}"
}

install_models() {
    echo -e "${YELLOW}Installing Ollama models...${NC}"
    
    for model in "${OLLAMA_MODELS[@]}"; do
        echo -e "${YELLOW}Installing model: $model${NC}"
        ollama pull "$model"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Model $model installed successfully${NC}"
        else
            echo -e "${RED}✗ Failed to install model $model${NC}"
        fi
    done
}

start_ollama() {
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    
    # Check if Ollama is already running
    if pgrep -f "ollama serve" > /dev/null; then
        echo -e "${GREEN}✓ Ollama is already running${NC}"
    else
        echo -e "${YELLOW}Starting Ollama server...${NC}"
        nohup ollama serve > ollama.log 2>&1 &
        sleep 3
        
        if pgrep -f "ollama serve" > /dev/null; then
            echo -e "${GREEN}✓ Ollama started successfully${NC}"
        else
            echo -e "${RED}✗ Failed to start Ollama${NC}"
            exit 1
        fi
    fi
}

start_api() {
    echo -e "${YELLOW}Starting RAG Pipeline API...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    
    # Set environment variables
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Start the API
    echo -e "${GREEN}Starting API server on http://$DEFAULT_HOST:$DEFAULT_PORT${NC}"
    uvicorn src.api:app --host $DEFAULT_HOST --port $DEFAULT_PORT --reload
}

start_podman_manual() {
    echo -e "${YELLOW}Starting containers manually with Podman...${NC}"
    
    # Create required directories for volume mounts
    echo -e "${YELLOW}Creating required directories...${NC}"
    mkdir -p chroma_db
    mkdir -p logs
    mkdir -p sample_docs
    mkdir -p ollama_data
    
    # Create sample document if it doesn't exist
    if [ ! -f "sample_docs/sample_document.md" ]; then
        echo "# Sample Document

This is a sample document for testing the RAG pipeline." > sample_docs/sample_document.md
    fi
    
    # Create pod for services
    podman pod create --name rag-pod -p 8000:8000 -p 11434:11434 || true
    
    # Start Ollama container
    echo -e "${YELLOW}Starting Ollama container...${NC}"
    podman run -d --name ollama \
        --pod rag-pod \
        -v ./ollama_data:/root/.ollama \
        -e OLLAMA_ORIGINS=* \
        --restart unless-stopped \
        ollama/ollama:latest
    
    # Build RAG pipeline image
    echo -e "${YELLOW}Building RAG pipeline image...${NC}"
    podman build -t rag-pipeline .
    
    # Start RAG pipeline container
    echo -e "${YELLOW}Starting RAG pipeline container...${NC}"
    podman run -d --name rag-pipeline \
        --pod rag-pod \
        -v ./chroma_db:/app/chroma_db \
        -v ./sample_docs:/app/sample_docs \
        -v ./logs:/app/logs \
        -e OLLAMA_BASE_URL=http://localhost:11434 \
        -e VECTOR_DB_PATH=/app/chroma_db \
        -e LOG_LEVEL=INFO \
        --restart unless-stopped \
        rag-pipeline
}

start_docker() {
    echo -e "${YELLOW}Starting RAG Pipeline with containers...${NC}"
    
    # Check dependencies first to set container runtime variables
    check_dependencies
    
    # Check if we have container runtime
    if [ -z "$CONTAINER_CMD" ]; then
        echo -e "${RED}Error: No container runtime (Podman/Docker) found${NC}"
        exit 1
    fi
    
    # Check if compose file exists
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found${NC}"
        exit 1
    fi
    
    # Create required directories for volume mounts
    echo -e "${YELLOW}Creating required directories...${NC}"
    mkdir -p chroma_db
    mkdir -p logs
    mkdir -p sample_docs
    mkdir -p ollama_data
    
    # Create sample document if it doesn't exist
    if [ ! -f "sample_docs/sample_document.md" ]; then
        echo "# Sample Document

This is a sample document for testing the RAG pipeline.

## Introduction
This document contains sample content that can be used to test document ingestion and querying capabilities.

## Key Features
- Document processing
- Vector embeddings
- Semantic search
- Question answering

## Conclusion
This sample document helps verify that the RAG pipeline is working correctly." > sample_docs/sample_document.md
    fi
    
    # Check if we have compose command
    if [ -z "$COMPOSE_CMD" ]; then
        echo -e "${YELLOW}No compose command available, starting containers manually...${NC}"
        start_podman_manual
        
        # Wait for services to be ready
        echo -e "${YELLOW}Waiting for services to start...${NC}"
        sleep 15
        
        # Install models in Ollama container
        echo -e "${YELLOW}Installing models in Ollama container...${NC}"
        for model in "${OLLAMA_MODELS[@]}"; do
            echo -e "${YELLOW}Checking model: $model${NC}"
            if podman exec ollama ollama list | grep -q "$model"; then
                echo -e "${GREEN}✓ Model $model already available${NC}"
            else
                echo -e "${YELLOW}Installing model: $model${NC}"
                if podman exec ollama ollama pull "$model"; then
                    echo -e "${GREEN}✓ Model $model installed successfully${NC}"
                else
                    echo -e "${RED}✗ Failed to install model $model${NC}"
                fi
            fi
        done
        
        echo -e "${GREEN}✓ Podman containers started successfully${NC}"
        echo -e "${GREEN}API available at: http://localhost:8000${NC}"
        echo -e "${GREEN}API docs available at: http://localhost:8000/docs${NC}"
        return
    fi
    
    # Start services with compose
    echo -e "${YELLOW}Using $COMPOSE_CMD to start services...${NC}"
    $COMPOSE_CMD up -d
    
    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to start...${NC}"
    sleep 10
    
    # Install models in Ollama container
    echo -e "${YELLOW}Installing models in Ollama container...${NC}"
    for model in "${OLLAMA_MODELS[@]}"; do
        echo -e "${YELLOW}Checking model: $model${NC}"
        if $CONTAINER_CMD exec ollama ollama list | grep -q "$model"; then
            echo -e "${GREEN}✓ Model $model already available${NC}"
        else
            echo -e "${YELLOW}Installing model: $model${NC}"
            if $CONTAINER_CMD exec ollama ollama pull "$model"; then
                echo -e "${GREEN}✓ Model $model installed successfully${NC}"
            else
                echo -e "${RED}✗ Failed to install model $model${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}✓ Container services started successfully${NC}"
    echo -e "${GREEN}API available at: http://localhost:8000${NC}"
    echo -e "${GREEN}API docs available at: http://localhost:8000/docs${NC}"
}

stop_services() {
    echo -e "${YELLOW}Stopping services...${NC}"
    
    # Stop API server
    pkill -f "uvicorn src.api:app" || true
    
    # Stop container services
    check_dependencies
    
    if [ -f "docker-compose.yml" ] && [ -n "$COMPOSE_CMD" ]; then
        echo -e "${YELLOW}Stopping services with $COMPOSE_CMD...${NC}"
        $COMPOSE_CMD down
    elif [ "$CONTAINER_CMD" = "podman" ]; then
        echo -e "${YELLOW}Stopping Podman containers...${NC}"
        podman stop ollama rag-pipeline 2>/dev/null || true
        podman rm ollama rag-pipeline 2>/dev/null || true
        podman pod rm rag-pod 2>/dev/null || true
    elif [ "$CONTAINER_CMD" = "docker" ]; then
        echo -e "${YELLOW}Stopping Docker containers...${NC}"
        docker stop ollama rag-pipeline 2>/dev/null || true
        docker rm ollama rag-pipeline 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Services stopped${NC}"
}

run_tests() {
    echo -e "${YELLOW}Running tests...${NC}"
    
    # Activate virtual environment
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Run tests
    python -m pytest tests/ -v
}

show_status() {
    echo -e "${YELLOW}Service Status:${NC}"
    
    check_dependencies
    
    # Check Ollama (local or container)
    if pgrep -f "ollama serve" > /dev/null; then
        echo -e "${GREEN}✓ Ollama (local): Running${NC}"
    elif [ "$CONTAINER_CMD" = "podman" ] && podman ps | grep -q "ollama"; then
        echo -e "${GREEN}✓ Ollama (podman): Running${NC}"
    elif [ "$CONTAINER_CMD" = "docker" ] && docker ps | grep -q "ollama"; then
        echo -e "${GREEN}✓ Ollama (docker): Running${NC}"
    else
        echo -e "${RED}✗ Ollama: Not running${NC}"
    fi
    
    # Check API (local or container)
    if pgrep -f "uvicorn src.api:app" > /dev/null; then
        echo -e "${GREEN}✓ API (local): Running${NC}"
    elif [ "$CONTAINER_CMD" = "podman" ] && podman ps | grep -q "rag-pipeline"; then
        echo -e "${GREEN}✓ API (podman): Running${NC}"
    elif [ "$CONTAINER_CMD" = "docker" ] && docker ps | grep -q "rag-pipeline"; then
        echo -e "${GREEN}✓ API (docker): Running${NC}"
    else
        echo -e "${RED}✗ API: Not running${NC}"
    fi
    
    # Check container orchestration
    if [ -n "$COMPOSE_CMD" ] && $COMPOSE_CMD ps 2>/dev/null | grep -q "Up"; then
        echo -e "${GREEN}✓ Container orchestration ($COMPOSE_CMD): Running${NC}"
    elif [ "$CONTAINER_CMD" = "podman" ] && podman pod ps | grep -q "rag-pod"; then
        echo -e "${GREEN}✓ Container orchestration (podman): Running${NC}"
    else
        echo -e "${RED}✗ Container orchestration: Not running${NC}"
    fi
}

show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start           Start the RAG pipeline (local development)"
    echo "  start-docker    Start with containers (Podman/Docker)"
    echo "  stop            Stop all services"
    echo "  install-models  Install Ollama models"
    echo "  test            Run tests"
    echo "  status          Show service status"
    echo "  help            Show this help message"
    echo ""
    echo "Container Support:"
    echo "  • Podman (preferred) - rootless containers, no daemon"
    echo "  • Docker (fallback) - traditional container runtime"
    echo "  • Automatic detection of available runtime"
    echo ""
    echo "Examples:"
    echo "  $0 start          # Start local development server"
    echo "  $0 start-docker   # Start with containers (auto-detects Podman/Docker)"
    echo "  $0 stop           # Stop all services"
    echo "  $0 status         # Check what's running"
}

# Main script logic
main() {
    print_banner
    
    case "${1:-start}" in
        "start")
            check_dependencies
            start_ollama
            install_models
            start_api
            ;;
        "start-docker")
            start_docker
            ;;
        "stop")
            stop_services
            ;;
        "install-models")
            install_models
            ;;
        "test")
            run_tests
            ;;
        "status")
            show_status
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 