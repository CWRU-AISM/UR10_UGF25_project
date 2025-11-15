# Dockerfile for UR10e Research Project
# Supports Computer Vision and NeRF Studio workflows
# Includes CUDA 12.6, COLMAP, and all CV dependencies

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
    # Media processing
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
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

# Install COLMAP for Structure from Motion
RUN apt-get update && apt-get install -y \
    colmap \
    && rm -rf /var/lib/apt/lists/*

# Install NeRF Studio and dependencies
RUN pip3 install --no-cache-dir nerfstudio

# Install gsplat for Gaussian Splatting
RUN pip3 install --no-cache-dir gsplat

# Install Segment Anything Model 2 (SAM2) from GitHub
# Uncomment when needed
# RUN pip3 install --no-cache-dir git+https://github.com/facebookresearch/segment-anything-2.git

# Install Grounded DINO dependencies and model
# Uncomment when needed
# RUN apt-get update && apt-get install -y \
#     ninja-build \
#     && rm -rf /var/lib/apt/lists/*
#
# RUN pip3 install --no-cache-dir git+https://github.com/IDEA-Research/GroundingDINO.git

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

# Additional CV and 3D processing tools
RUN pip3 install --no-cache-dir \
    pyrealsense2 \
    trimesh \
    pymeshlab \
    ipdb \
    jupyter-console \
    nbconvert

# Note: Azure Kinect SDK (pyk4a) should be installed on the host system
# Docker container access to USB devices can be unreliable
# For Azure Kinect usage, install natively with conda as described in README

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