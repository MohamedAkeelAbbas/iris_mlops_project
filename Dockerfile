# Multi-stage build for smaller final image
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/
COPY run_pipeline.py .
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/features \
    models/trained models/registry \
    experiments/plots experiments/mlruns \
    logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command - run the ML pipeline
CMD ["python", "run_pipeline.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.pipeline; print('OK')" || exit 1
