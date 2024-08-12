FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p -d /opt/conda && \
    rm miniconda.sh

# Set path to conda
ENV PATH /opt/conda/bin:$PATH

# Copy the requirements file
COPY requirements.txt /workspaces/satisfia/requirements.txt

# Create and activate environments
RUN conda create -n mujoco python=3.10 -y && \
    conda create -n atari python=3.9 -y && \
    conda create -n default python=3.10 -y

# Install packages in mujoco environment
RUN conda run -n mujoco pip install gymnasium[mujoco] && \
    conda run -n mujoco pip install -r /workspaces/satisfia/requirements.txt

# Install packages in atari environment
RUN conda run -n atari pip install gymnasium[atari] gymnasium[accept-rom-license] && \
    conda run -n atari pip install -r /workspaces/satisfia/requirements.txt

# Install packages in default environment
RUN conda run -n default pip install -r /workspaces/satisfia/requirements.txt

# Copy the rest of the files
COPY ./scripts /workspaces/satisfia/scripts
COPY ./src /workspaces/satisfia/src
COPY . .

# Set the working directory
WORKDIR /workspaces/satisfia/

# Set default environment to 'default'
RUN echo "conda activate default" >> ~/.bashrc

# Set the shell to bash
SHELL ["/bin/bash", "--login", "-c"]