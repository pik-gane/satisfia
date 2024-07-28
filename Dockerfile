FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Copy the environment file
COPY ./scripts /workspaces/satisfia/scripts
COPY ./src /workspaces/satisfia/src
COPY requirements.txt /workspaces/satisfia/requirements.txt


# Update base environment
RUN apt-get update && apt purge -y python3 && apt-get install -y python3-pip python3.10 python3.10-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --set python3 /usr/bin/python3.10
RUN pip3 install --upgrade pip
RUN pip3 install -r /workspaces/satisfia/requirements.txt

RUN apt-get install -y git

# Copy all files from the current directory
COPY . .

# Set the working directory
WORKDIR /workspaces/satisfia/
