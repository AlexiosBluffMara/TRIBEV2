# Jemma Discord bot — NVIDIA GPU container
#
# Base: CUDA 12.8 + cuDNN 9 runtime on Ubuntu 22.04
# Python: 3.11 via deadsnakes PPA
#
# Build:  docker build -t jemma .
# Run:    see docker-compose.yml
#
# IMPORTANT: tribev2_weights/, tribev2_src/, and HF_HOME must be bind-mounted
# from the host — they are not copied into the image (too large).

FROM nvidia/cuda:12.8.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /app

# PyTorch cu128 wheels must be fetched from the PyTorch index
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0+cu128 torchvision==0.26.0+cu128 && \
    pip install -r requirements.txt && \
    pip install anthropic

# ── Project source (everything except weights / cache — those are mounted) ───
COPY bot/           bot/
COPY assets/        assets/
COPY .env.example   .env.example

# TRIBE v2 source is bind-mounted at /app/tribev2_src — install editable after
# the container starts (see CMD). We include a placeholder here.
# If you want a fully self-contained image, COPY tribev2_src/ tribev2_src/
# and uncomment the line below.
# RUN pip install --no-deps -e tribev2_src/

# Runtime directories — will be overlaid by bind mounts in docker-compose
RUN mkdir -p uploads outputs logs tribev2_cache

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Install TRIBE editable source on startup (mount must be present by then).
CMD ["sh", "-c", "pip install --no-deps -q -e /app/tribev2_src && python -m bot.bot"]
