FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 AS base 

# Configurations
# Variables specified by ARG can be changed using --build-arg in docker build
# withoug --build-arg, name specified by ARG is used as default
ARG PYTHON_VERSION="3.8.10"
#ARG PYTHON_VERSION="3.7.10"

# Set timezone to avoid Timezone requirement freeze
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libraries via apt-get 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-setuptools python3-dev \
    cmake curl wget unzip libreadline-dev libjpeg-dev libpng-dev ncurses-dev \
    imagemagick gnuplot gnuplot-x11 libssl-dev libzmq3-dev graphviz \
    ca-certificates bzip2 unzip \
    vim git 

RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && pip install --upgrade pip==20.0.2

# Install libraries via apt-get 
# RUN pip install\
#     pytorch_geometric==2.3.0 torch==1.8.0+cu111 torch-cluster==1.6.0\
#     torch-cluster==1.6.0\
#     torch-geometric==2.1.0\
#     torch-scatter==2.1.1\
#     torch-sparse==0.6.10\
#     torch-spline-conv==1.2.2\
#     torchvision==0.9.0+cu111

#この入れ方じゃないと入らない
#公式　https://pytorch.org/get-started/previous-versions/
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html


COPY requirements.txt /root/

# Start container at the home directory
ARG HOME="/root"
WORKDIR ${HOME}

RUN pip install -r /root/requirements.txt