#!/bin/bash

# Feast RAG Pipeline Podman Runner
# This script runs the Podman compose setup from the deploy directory

echo "🚀 Starting Feast RAG Pipeline..."

# Check if Podman is available
if ! command -v podman &> /dev/null; then
    echo "❌ Podman is not installed. Please install Podman first."
    exit 1
fi

# Check if podman-compose is available
if command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "ℹ️  Using docker-compose with podman backend"
else
    echo "❌ Neither podman-compose nor docker-compose found. Please install one of them."
    exit 1
fi

# Pull images first to avoid back-off issues
echo "📦 Pulling container images..."
$COMPOSE_CMD pull

# Build and start services
echo "🔧 Building and starting services..."
$COMPOSE_CMD up --build -d

# Show running containers
echo "📋 Running containers:"
$COMPOSE_CMD ps

echo "✅ Feast RAG Pipeline is running!"
echo "📖 API: http://localhost:8000"
echo "🎯 UI: http://localhost:8000"
echo "🔍 To view logs: $COMPOSE_CMD logs -f"
echo "🛑 To stop: $COMPOSE_CMD down" 