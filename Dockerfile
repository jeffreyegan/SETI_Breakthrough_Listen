FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
LABEL description="SETI Breakthrough Listen Pytorch CNN Container"
LABEL version="0.2.0"

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install required packages
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
    numba \
    numpy \
    scipy \
    torch \
    opencv-python-headless \
    jupyter \
    fastai \
    tqdm \
    timm \
    pandas \
    scikit-learn

# Set the default shell to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /SETI

# Set the default command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]

