# ARG for choosing GPU or CPU
ARG PLATFORM="cpu"

# ARG for selecting which environment to set up
ARG ENV_SETUP="base" # Options: "atari", "base", "mujoco", "all"

# ARGs for base images with different Python versions
ARG BASE_IMAGE_ATARI="python:3.9-slim"
ARG BASE_IMAGE_BASE="python:3.10-slim"
ARG BASE_IMAGE_MUJOCO="python:3.10-slim"

# GPU base image
ARG BASE_IMAGE_GPU="pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime"

# Use a multi-stage build to handle GPU/CPU and environment-specific base image selection
FROM ${BASE_IMAGE_GPU} AS base_gpu
FROM ${BASE_IMAGE_ATARI} AS base_atari
FROM ${BASE_IMAGE_BASE} AS base_base
FROM ${BASE_IMAGE_MUJOCO} AS base_mujoco

# Select the appropriate base image
FROM base_${ENV_SETUP} AS base_cpu

FROM base_${PLATFORM} as base

ARG ENV_SETUP

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    swig \
    build-essential \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt /tmp/requirements.txt

ENV ENV_SETUP=$ENV_SETUP

# Atari environment setup
RUN if [ "${ENV_SETUP}" = "atari" ]; then \
    pip install --no-cache-dir gymnasium[atari] gymnasium[accept-rom-license] gymnasium[box2d] && \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# Base environment setup
RUN if [ "$ENV_SETUP" = "base" ]; then \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# Mujoco environment setup
RUN if [ "$ENV_SETUP" = "mujoco" ]; then \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir mujoco gymnasium[mujoco]; \
    fi

# Clean up system dependencies to reduce image size
RUN apt-get purge -y --auto-remove build-essential swig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the shell to bash
SHELL ["/bin/bash", "-c"]