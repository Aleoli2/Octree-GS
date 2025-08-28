FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

COPY . /Octree-GS
WORKDIR /Octree-GS
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y python3.9 python3.9-dev python3-dev python3-pip 
RUN apt update && apt install -y python-is-python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 
RUN pip install --no-cache-dir torch==2.3.0+cu118 torchvision==0.18.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

#install cudatoolkit
RUN apt update && apt install -y wget
RUN pip install --upgrade pip
RUN pip install einops wandb lpips laspy jaxtyping colorama opencv-python plyfile
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html

ARG TORCH_CUDA_ARCH_LIST
RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace