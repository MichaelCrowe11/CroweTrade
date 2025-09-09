# Multi-stage Dockerfile for CroweTrade production deployment

FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn[standard] psutil cryptography bcrypt PyJWT

# Production stage
FROM python:3.11-slim AS production

# Create non-root user
RUN groupadd -r crowetrade && useradd -r -g crowetrade -d /app -s /bin/bash crowetrade

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./

# Install the package in editable mode
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data/features /app/data/models /app/logs \
    && chown -R crowetrade:crowetrade /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER crowetrade

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Default command
CMD ["python", "-m", "crowetrade.api.server"]