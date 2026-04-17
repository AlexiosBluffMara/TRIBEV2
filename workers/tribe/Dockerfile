FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools && pip install -e .

# Copy worker code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRIBE_DEVICE=cuda:0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD celery -A serve inspect ping || exit 1

# Start Celery worker
CMD ["celery", "-A", "serve", "worker", "--loglevel=info"]
