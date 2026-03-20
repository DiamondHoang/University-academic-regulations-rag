#!/bin/bash

# Restore Ollama identity if OLLAMA_KEY is provided via Environment Variable
# This avoids the need for manual 'ollama signin' on each restart.
if [ -n "$OLLAMA_KEY" ]; then
    echo "Restoring Ollama identity from environment variable..."
    mkdir -p /root/.ollama
    echo "$OLLAMA_KEY" > /root/.ollama/id_ed25519
    chmod 600 /root/.ollama/id_ed25519
    echo "Identity restored successfully."
fi

# Start the FastAPI application with optimizations for Azure
# Use WEB_CONCURRENCY env var if provided (Azure App Service often sets this), default to 1.
# Reducing workers saves significant RAM (1.5GB-2GB per worker for models).
WORKERS=${WEB_CONCURRENCY:-1}

# --preload: shares memory for heavy models among workers (Embedding, Reranker)
# --timeout 600: prevents worker timeout during long LLM generations
echo "Starting FastAPI application with $WORKERS workers and --preload..."
exec gunicorn server:app -w $WORKERS -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 600 --preload