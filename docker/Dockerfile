# docker/Dockerfile
FROM python:3.9-slim

# Install system dependencies
# Added g++ for C++ extensions and libcurl for genomic data streaming
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pinned dependencies
# Using --upgrade pip ensures the latest wheel support for faster installs
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Default command
CMD ["python3"]