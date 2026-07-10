FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# System libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    SimpleITK==2.3.1 \
    nnunetv2==2.5.1 \
    scipy==1.14.0

# Model code
COPY --chown=user:user models/networks.py ./models/networks.py
COPY --chown=user:user models/__init__.py ./models/__init__.py

# Inference entry point
COPY --chown=user:user inference.py ./inference.py

# Weights: Pix2PixHD v32_conditional GAN (3ch, ngf=64, n_blocks=12, 912 MB)
COPY --chown=user:user weights/latest_net_G.pth ./weights/latest_net_G.pth

# Weights: Dataset930 BreastDivider 2D (breast mask, ~120 MB)
COPY --chown=user:user weights/breast_seg_930/ ./weights/breast_seg_930/

# Weights: Dataset940 T1w Tumor Seg (tumor mask, ~257 MB)
COPY --chown=user:user weights/tumor_seg/ ./weights/tumor_seg/

RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]
