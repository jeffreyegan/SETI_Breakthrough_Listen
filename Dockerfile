# SETI Breakthrough Listen Container 
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ["requirements.txt", "./"]
COPY ["SETI_Breakthrough_Listen_Pytorch.ipynb", "images", "./project/"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir  opencv-python jupyter "cupy-cuda112==8.6.0" fastai
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

ENV PYTHONPATH /usr/local/lib/python3.8/dist-packages/

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

