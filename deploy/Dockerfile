FROM python:3.11-slim

# Install system dependencies (keeping single line for podman compatibility)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV OLLAMA_BASE_URL=http://ollama:11434

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python packages as root (ensures global availability)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Verify critical imports work
RUN python -c "import uvicorn; print('uvicorn OK')"

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY static/ ./static/

# Test app import before switching users
RUN python -c "from src.api import app; print('App import OK')"

# Create directories and set proper permissions
RUN mkdir -p chroma_db logs
RUN chown -R app:app /app
RUN chmod -R 755 /app/src /app/config /app/static
RUN chmod 755 /app/chroma_db /app/logs

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Add health check (simple for podman compatibility)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 