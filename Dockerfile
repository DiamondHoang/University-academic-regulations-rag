# Stage 1: Build dependencies and cache models
FROM python:3.10-slim AS builder

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH" \
    HF_HOME="/app/.cache/huggingface"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy essential files for model downloading and vectorstore building
COPY config.py .
COPY uni_rag.py .
COPY loader/ loader/
COPY retrieval/ retrieval/
COPY memory/ memory/
COPY md/ md/
COPY scripts/ scripts/

# Pre-download models and pre-build vectorstore to cache them in the image
RUN mkdir -p /app/.cache/huggingface && \
    PYTHONPATH=. python scripts/download_models.py && \
    PYTHONPATH=. python scripts/prebuild_vectorstore.py

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH" \
HF_HOME="/app/.cache/huggingface"

# Install basic packages if needed
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
# Copy cached models and pre-built vectorstore
COPY --from=builder /app/.cache /app/.cache
COPY --from=builder /app/vector_db /app/vector_db

COPY . .

# Install gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Healthcheck - checks if vectorstore is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health | grep -q "ok" || exit 1

# Start with entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
