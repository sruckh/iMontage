FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git ca-certificates curl build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

COPY scripts/runpod_bootstrap.sh /usr/local/bin/runpod_bootstrap.sh
RUN chmod +x /usr/local/bin/runpod_bootstrap.sh

WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/runpod_bootstrap.sh"]
CMD ["python3.12", "gradio_app.py", "--host", "0.0.0.0", "--port", "7860"]
EXPOSE 7860
