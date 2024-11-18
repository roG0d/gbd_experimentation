# Use NVIDIA's official CUDA 12.4 base image
FROM ubuntu:22.04
#FROM ubuntu:22.04

# Set up environment variables
ENV PATH /opt/conda/bin:$PATH

SHELL ["/bin/bash", "-c"]

# Install dependencies
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit-base \
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
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (a lightweight version of Anaconda)
RUN mkdir -p ~/miniconda3 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh 
RUN /bin/bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh
RUN  source ~/miniconda3/bin/activate

# Install NVIDIA Drivers

#RUN mkdir -p ~/nvidia-driver
#RUN wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.127.05/NVIDIA-Linux-x86_64-550.127.05.run -O ~/nvidia-driver/NVIDIA-Linux-x86_64-550.127.05.run
#RUN sh ~/nvidia-driver/NVIDIA-Linux-x86_64-550.127.05.run

# Create a new Conda environment and activate it
RUN /root/miniconda3/bin/conda create -n vllm python=3.10 -y
RUN echo "source ~/miniconda3/bin/activate" > ~/.bashrc
RUN echo "conda activate vllm" >> ~/.bashrc

# Install CUDA toolkit within the conda environment
RUN /root/miniconda3/bin/conda install cuda -n ironsight -c nvidia/label/cuda-12.4.0 -y

# Install Python packages needed for the project
RUN /root/miniconda3/envs/ironsight/bin/pip install torch==2.4.1 torchvision==0.19.1 torchaudio transformers[torch]==4.44.2  tf-keras==2.17.0 hugsvision==0.75.5

# Ensure the container uses NVIDIA runtime for CUDA access
LABEL com.nvidia.volumes.needed="nvidia_driver"

# docker run command to use gpus
# sudo docker run -it --runtime=nvidia --gpus all ironsight  nvidia-smi
# sudo docker run -it -v ./challenge1:/home/challenge1 --runtime=nvidia --gpus all ironsight  /bin/bash 