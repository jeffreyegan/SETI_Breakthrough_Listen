# SETI Breakthrough Listen Container 

FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    python3-setuptools \
    git && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir "cupy-cuda112==8.6.0" \
    numba numpy scipy torch opencv-python-headless jupyter fastai tqdm timm pandas sklearn

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]