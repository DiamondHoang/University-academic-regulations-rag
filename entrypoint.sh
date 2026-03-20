#!/bin/bash

# Start the FastAPI application with optimizations for Azure
# --preload: shares memory for heavy models among workers (Embedding, Reranker)
# --timeout 600: prevents worker timeout during long LLM generations
echo "Starting FastAPI application with 4 workers and --preload..."
exec gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 600 --preload