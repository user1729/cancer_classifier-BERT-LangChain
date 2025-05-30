# Use official CUDA 12.1 image with Ubuntu
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Rahul R <rahul91indian@gmail.com>"

ARG USERNAME="developer"

# Install basic utilities and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools \
    git curl zip unzip wget vim build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional: Set python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
RUN pip3 install --upgrade pip

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH

# Create and activate a Conda environment
RUN conda update -n base -c defaults conda && \
    conda create -n research python=3.11 cmake=3.14.0 && \
    conda clean -afy

# Make sure the environment is activated in every shell
SHELL ["conda", "run", "-n", "research", "/bin/bash", "-c"]

RUN conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install PyTorch with CUDA 12.1 support (if needed)
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy Python dependencies
ADD requirements.txt .
RUN pip3 install -r requirements.txt 

ENV HOME=/home/$USERNAME
ENV SHELL=/bin/bash

# Set environment variables so that conda is activated on container start
ENV PATH /opt/conda/envs/research/bin:$PATH
ENV CONDA_DEFAULT_ENV=research

#USER $USERNAME:$USERNAME
WORKDIR /home/$USERNAME
VOLUME /home/$USERNAME/
