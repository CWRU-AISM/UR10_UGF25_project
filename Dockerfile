# Dockerfile for Undergraduate Research Project
# ROS2 should be installed natively on the host system
#
# GPU Support Options:
# - For NVIDIA GPU support, use: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
# - For CPU-only, use: ubuntu:22.04 (current)
# - Adjust PyTorch installation URL below to match CUDA version

# Base stage with Python and common dependencies
# FROM ubuntu:22.04 AS base
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    curl \
    wget \
    git \
    build-essential \
    cmake \
    pkg-config \
    software-properties-common \
    # Python
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Azure Kinect dependencies
    libk4a1.4 \
    libk4a1.4-dev \
    k4a-tools \
    # OpenGL for visualization
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Network tools
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python packages from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support 
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install Segment Anything Model 2 (SAM2) from GitHub
# Commented out for now - uncomment when needed
# RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# Install Grounded DINO dependencies and model
# Commented out for now - uncomment when needed
# RUN apt-get update && apt-get install -y \
#     ninja-build \
#     && rm -rf /var/lib/apt/lists/*
#
# RUN pip3 install --no-cache-dir git+https://github.com/IDEA-Research/GroundingDINO.git

# Install COLMAP (for SfM)
RUN apt-get update && apt-get install -y \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Main development stage
FROM base AS development

# Install development tools
RUN apt-get update && apt-get install -y \
    # Code editors support
    vim \
    nano \
    # Debugging tools
    gdb \
    valgrind \
    htop \
    # Documentation
    doxygen \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Additional CV-specific tools
RUN pip3 install --no-cache-dir \
    pyrealsense2 \
    trimesh \
    pymeshlab \
    ipdb \
    jupyter-console \
    nbconvert

# Set environment for development
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Copy source code
COPY src /workspace/src
COPY examples /workspace/examples

# Create data directories
RUN mkdir -p /workspace/data/images \
    /workspace/data/point_clouds \
    /workspace/data/calibration \
    /workspace/logs

# Create mount points for external data
VOLUME ["/workspace/data", "/workspace/logs"]

# Expose ports for various services
# 8888: Jupyter
EXPOSE 8888

# Default to bash shell
CMD ["/bin/bash"]