# Stage 1: Build environments
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    swig \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
#ENV CONDA_DIR /opt/conda
#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -O miniconda.sh && \
#    /bin/bash miniconda.sh -b -p $CONDA_DIR && \
#    rm miniconda.sh && \
#    echo "export PATH=$CONDA_DIR/bin:$PATH" > /etc/profile.d/conda.sh

#ENV PATH $CONDA_DIR/bin:$PATH

# Copy the requirements file
COPY requirements.txt /tmp/requirements.txt

# Create and set up atari environment
RUN conda create -n atari python=3.9 -y && \
    conda run -n atari pip install --no-cache-dir gymnasium[atari] gymnasium[accept-rom-license] gymnasium[box2d]&& \
    conda run -n atari pip install --no-cache-dir -r /tmp/requirements.txt

# Create base environment and install common packages

RUN conda create -n base_env python=3.10 -y && \
    conda run -n base_env pip install --no-cache-dir -r /tmp/requirements.txt

# Create and set up mujoco environment
RUN conda create -n mujoco --clone base_env -y && \
    conda run -n mujoco pip install --no-cache-dir gymnasium[mujoco]

# Clean up
RUN conda clean -afy && \
    find $CONDA_DIR -follow -type f -name '*.a' -delete && \
    find $CONDA_DIR -follow -type f -name '*.pyc' -delete && \
    find $CONDA_DIR -follow -type f -name '*.js.map' -delete

# Set the shell to bash
SHELL ["/bin/bash", "--login", "-c"]