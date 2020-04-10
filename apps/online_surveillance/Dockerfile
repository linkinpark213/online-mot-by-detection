FROM toolboc/jetson-nano-l4t-cuda-cudnn-opencv:latest

# Install environments with APT
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        make \
        g++ \
        git \
        wget \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3-dev \
        python3-pip \
        python3-setuptools \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        openmpi-bin \
        openmpi-doc \
        libopenmpi-dev \
        freetype2-demos && \
        rm -rf /var/lib/apt/lists/*

# Add repository for InfluxDB
RUN wget https://repos.influxdata.com/influxdb.key && \
        sudo apt-key add influxdb.key && \
        DISTRIB_ID=$(cat /etc/lsb-release  | grep DISTRIB_ID  | cut -d= -f2) && \
        DISTRIB_CODENAME=$(cat /etc/lsb-release  | grep DISTRIB_CODENAME  | cut -d= -f2) && \
        echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list && \
        apt-get update && \
        apt-get install influxdb


# Start InfluxDB service
RUN sudo service influxdb start

# Install Python 3 and pip
RUN rm /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip
# Install Cython, numpy and scipy
RUN pip install setuptools Cython && pip install numpy==1.16 scipy
# Install PyTorch
RUN wget https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl -O torch-1.4.0-cp36-cp36m-linux_aarch64.whl && \
        pip install torch-1.4.0-cp36-cp36m-linux_aarch64.whl
# Install TorchVision
RUN git clone -b v0.3.0 https://github.com/pytorch/vision torchvision && \
        cd torchvision && \
        python setup.py install && \
        cd ..
# Install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git && \
        cd cocoapi/PythonAPI && \
        make && \
        MPLLOCALFREETYPE=1 python setup.py install && \
        cd ..
# Install Detectron2
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore' && \
        git clone https://github.com/facebookresearch/detectron2 && \
        cd detectron2 && \
        FORCE_CUDA="1" pip install --user -e . && \
        cd ..

# Install InfluxDB-Python
RUN pip install influxdb
# Install OMOTBD along with apps
COPY ./online-mot-by-detection omotbd
# Run online surveillance server
RUN cd omotbd && export PYTHONPATH=`pwd` && cd apps/online_surveillance && python server.py configs/tx2.py