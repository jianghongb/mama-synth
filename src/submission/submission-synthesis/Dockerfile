FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

RUN pip install --no-cache-dir \
    SimpleITK==2.3.1 \
    nnunetv2==2.5.1 \
    dynamic-network-architectures==0.3.1

COPY --chown=user:user models/networks.py ./models/networks.py
COPY --chown=user:user inference.py ./
COPY --chown=user:user weights/latest_net_G.pth ./weights/latest_net_G.pth
COPY --chown=user:user weights/breast_seg/ ./weights/breast_seg/

RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "inference.py"]
