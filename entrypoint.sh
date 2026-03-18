#!/bin/bash

# Start SSH service for Azure Web SSH
echo "Starting SSH service..."
service ssh start

# Start Ollama in the background
echo "Starting Ollama engine..."

ollama serve &



# Wait for Ollama to become available
echo "Waiting for Ollama to start (timeout 60s)..."
COUNTER=0
MAX_RETRIES=30
until curl -s http://localhost:11434/api/tags > /dev/null || [ $COUNTER -eq $MAX_RETRIES ]; do
    sleep 2
    COUNTER=$((COUNTER+1))
    echo "Waiting for Ollama... ($COUNTER/$MAX_RETRIES)"
done

if [ $COUNTER -eq $MAX_RETRIES ]; then
    echo "Ollama failed to start in time. Proceeding anyway, but LLM features may fail."
else
    echo "Ollama is ready."
fi



# Check if we need to pull the model (optional, usually cloud models are just pointers)

# If using a local model, we would pull it here:

# ollama pull deepseek-v3.1:671b-cloud



# Start the FastAPI application

echo "Preparing vector database..."
python scripts/init_db.py
if [ $? -ne 0 ]; then
    echo "Warning: Database initialization failed. Server may start with missing or stale data."
fi

echo "Starting FastAPI application..."

exec gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000