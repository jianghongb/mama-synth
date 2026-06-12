FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

RUN pip install --no-cache-dir SimpleITK==2.3.1

COPY --chown=user:user models/networks.py ./models/networks.py
COPY --chown=user:user process.py ./
COPY --chown=user:user weights/latest_net_G.pth ./weights/latest_net_G.pth

RUN mkdir -p /output && chown user:user /output

USER user

ENTRYPOINT ["python", "process.py"]
