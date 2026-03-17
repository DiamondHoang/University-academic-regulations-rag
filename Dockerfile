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

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# Install Ollama using official script
RUN apt-get update && apt-get install -y curl openssh-server pciutils \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for Azure Web SSH
RUN mkdir -p /var/run/sshd \
    && echo "root:Docker!" | chpasswd \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config \
    && sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Install gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose FastAPI, Ollama, and Azure SSH ports
EXPOSE 8000
EXPOSE 11434
EXPOSE 2222

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start with entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
