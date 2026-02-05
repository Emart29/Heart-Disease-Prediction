# Dockerfile for Heart Disease Prediction Application
# Multi-stage build for minimal image size

# =============================================================================
# Stage 1: Builder - Install dependencies and prepare the application
# =============================================================================
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # API configuration (can be overridden)
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    STREAMLIT_PORT=8501

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app
RUN mkdir -p /app/mlflow_artifacts && chown -R appuser:appgroup /app/mlflow_artifacts

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/


# Create necessary directories
RUN mkdir -p /app/reports /app/docs && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose ports for API and Streamlit
EXPOSE ${API_PORT} ${STREAMLIT_PORT}

# Health check for the API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Default command runs the API server
# Can be overridden in docker-compose or at runtime
CMD ["sh", "-c", "uvicorn api.main:app --host ${API_HOST} --port ${API_PORT}"]
