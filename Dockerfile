# Use an ARM-compatible Python base image
FROM python:3.9-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install system dependencies for matplotlib and PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libxft-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to ensure compatibility with specified packages
RUN pip install --no-cache-dir --upgrade pip==23.0.1

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==1.12.0 \
    torchvision==0.13.0 \
    torchaudio==0.12.0 \
    numpy==1.21.6 \
    Pillow==8.2.0 \
    matplotlib==3.4.3 \
    scikit-learn \
    tqdm \
    wandb \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Set environment variables to ensure Python outputs are displayed in the console
ENV PYTHONUNBUFFERED=1
