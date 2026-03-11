#!/bin/bash



# Start Ollama in the background

echo "Starting Ollama engine..."

ollama serve &



# Wait for Ollama to become available

echo "Waiting for Ollama to start..."

until curl -s http://localhost:11434/api/tags > /dev/null; do

    sleep 2

done

echo "Ollama is ready."



# Check if we need to pull the model (optional, usually cloud models are just pointers)

# If using a local model, we would pull it here:

# ollama pull deepseek-v3.1:671b-cloud



# Start the FastAPI application

echo "Starting FastAPI application..."

exec gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000