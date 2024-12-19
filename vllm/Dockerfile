# Use NVIDIA's official CUDA 12.4 base image
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#FROM ubuntu:22.04

# Set up environment variables
ENV PATH /opt/conda/bin:$PATH

# makes sure the shell used for subsequent RUN commands is exactly Bash, as located in /bin.
SHELL ["/bin/bash", "-c"]

# Install dependencies
RUN apt-get update && apt-get install -y \
    nvtop \
    sudo \
    kmod \
    wget \
    vim \
    git \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev
    # Cleanup command to remove the apt cache and reduce the image size: # IMPORTANT: Enforces using sudo apt update when entering the container
    #&& rm -rf /var/lib/apt/lists/*

# Create rog0d user
RUN useradd -ms /bin/bash rog0d && echo "rog0d:rog0d" | chpasswd && usermod -aG sudo rog0d
# From here user rog0d user to execute the following commands
USER rog0d
WORKDIR /home/rog0d

# Cloning the repo
RUN git clone https://github.com/roG0d/gbd_experimentation


# Install Miniconda (a lightweight version of Anaconda)
RUN mkdir -p ~/miniconda3 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh 
RUN /bin/bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
RUN source ~/miniconda3/bin/activate

# Install NVIDIA Drivers

#RUN mkdir -p ~/nvidia-driver
#RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.127.05/NVIDIA-Linux-x86_64-550.127.05.run -O ~/nvidia-driver/NVIDIA-Linux-x86_64-550.127.05.run
#RUN sh ~/nvidia-driver/NVIDIA-Linux-x86_64-550.127.05.run

# Create a new Conda environment and activate it
RUN ~/miniconda3/bin/conda create -n vllm python=3.10 -y
RUN echo "source ~/miniconda3/bin/activate" > ~/.bashrc
RUN echo "conda activate vllm" >> ~/.bashrc

# Install CUDA toolkit within the conda environment
RUN ~/miniconda3/bin/conda install cuda -n vllm -c nvidia/label/cuda-12.4.0 -y

# Install vllm 
RUN ~/miniconda3/envs/vllm/bin/pip install vllm

# Ensure the container uses NVIDIA runtime for CUDA access
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Run apt get update if you wanna install new packages

# docker run commands:
# docker run -it --rm --runtime=nvidia --gpus all vllm nvidia-smi
# docker run -it --rm --runtime=nvidia --gpus all vllm bash

# Initialize the container with complete config, userspace and shared disk:
# docker run -it -d -v ./gbd_experimentation:/home/rog0d/gbd_experimentation --runtime=nvidia --gpus all --name=vllm vllm124 bash 

# Initialize the container with complete config, and .env in the userspace:
# docker run -it -d --runtime=nvidia --gpus all --name=vllm vllm124 bash 