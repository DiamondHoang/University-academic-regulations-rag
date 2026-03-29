#!/bin/bash

# Start the FastAPI application
# Use WEB_CONCURRENCY env var if provided, default to 1.
# Reducing workers saves significant RAM (1.5GB-2GB per worker for models).
WORKERS=${WEB_CONCURRENCY:-1}

# NOTE: --preload is intentionally removed.
# --preload runs the app in gunicorn's master process (sync context), which breaks
# FastAPI's async lifespan hook — causing startup to hang and never log "Application startup complete".
# --timeout 1200: allows enough time for heavy model loading (bge-m3, reranker, vectorstore) on CPU
echo "Starting FastAPI application with $WORKERS workers on port 7860..."
exec gunicorn server:app -w $WORKERS -k uvicorn.workers.UvicornWorker -b 0.0.0.0:7860 --timeout 1200