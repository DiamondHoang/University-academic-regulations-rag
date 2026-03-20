# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download models to cache them in the image
COPY config.py .
COPY scripts/download_models.py scripts/download_models.py
RUN PYTHONPATH=. python scripts/download_models.py

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# Install basic packages if needed
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*


# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Install gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start with entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
