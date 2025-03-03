FROM ubuntu:20.04

# Set timezone and locale
RUN ln -sf /usr/share/zoneinfo/Asia/Jakarta /etc/localtime

# Install necessary packages
RUN apt-get update && apt-get install --no-install-recommends -y \
    libtbb2 \
    libyaml-cpp-dev \
    git \
    vim \
    nano \
    cmake \
    wget \
    libopencv-dev \
    pkg-config \
    g++ \
    gcc \
    libc6-dev \
    make \
    build-essential \
    nlohmann-json3-dev \
    ocl-icd-libopencl1 && \
    apt-get clean

# Download and install OpenVINO toolkit
RUN mkdir /opt/intel && \
    cd /opt/intel && \
    wget -O openvino.tgz https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz && \
    tar -xvf openvino.tgz && \
    rm openvino.tgz && \
    mv l_openvino* openvino_2023.3.0

COPY src/builddir/yolo10detect /usr/bin/yolo10detect
COPY src/builddir/yolov10n_fp32_openvino /root/yolov10n_fp32_openvino
