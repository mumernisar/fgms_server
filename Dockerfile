# Python 3.10 slim base for small size
FROM python:3.10-slim

# system deps needed for many Python wheels and for running torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt ./requirements.txt

# Install pip, then torch CPU wheel first (so pip doesn't try to pick CUDA wheels)
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only torch (choose versions you pinned)
RUN python -m pip install --no-cache-dir "torch==2.3.1+cpu" "torchvision==0.18.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu

# Now install the rest of requirements
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./ ./server
WORKDIR /app/server

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Use uvicorn as entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info", "--proxy-headers"]
