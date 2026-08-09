# AVAGCN environment
# Tested: PyTorch 2.12.0 + CUDA 13.0 + cuDNN 9 (base image below builds and runs
# with GPU access; see README "Environment" for the host driver requirements).
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /workspace

# System deps for matplotlib/spectral image IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# The base image uses a Debian system Python (PEP 668 "externally managed"),
# so pip installs must be explicitly allowed. The container is a dedicated,
# disposable environment for this application, so this is safe here.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt .
# torch is already provided by the base image; install the rest,
# pinned to the versions used for the reported results (requirements-lock.txt)
RUN pip install --no-cache-dir \
      numpy==2.4.4 scipy==1.17.1 scikit-learn==1.8.0 \
      matplotlib==3.10.9 spectral==0.24 hydra-core==1.3.2 omegaconf==2.3.0

COPY . .

# Datasets are NOT bundled (license). Mount them at runtime:
#   docker run --runtime=nvidia --gpus all -it -v $(pwd)/datas:/workspace/datas avagcn \
#     python main.py --config-name IP
# If your host's NVIDIA Container Toolkit is NOT in CDI mode, plain
# "--gpus all" (without --runtime=nvidia) also works.
# CMD ["python", "main.py", "--config-name", "IP"]