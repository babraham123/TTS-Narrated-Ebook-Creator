# Calliope with Chatterbox TTS for NVIDIA GPUs down to Pascal (Tesla P4, sm_61).
#
#   podman build -t calliope -f Containerfile .
#   podman run --rm --gpus=all -v calliope-cache:/cache -v ./books:/books calliope \
#       IN.epub OUT.epub --voice voice.wav --lang en
#
# torch 2.6.0+cu124 brings its own CUDA 12.4 user-space libraries; the CUDA 12.4 + cuDNN 9 base
# matches them and provides the libraries CTranslate2 (faster-whisper) loads.
FROM docker.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /usr/local/bin/

ARG DEBIAN_FRONTEND=noninteractive

# upgrade: the pinned CUDA base lags Ubuntu's security updates (openssl, gnupg).
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# UV_NO_CACHE: uv's cache would otherwise land in the layer, a second copy of the venv
# (UV_LINK_MODE=copy) that doubles the disk a build needs.
ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH=/opt/venv/bin:$PATH \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/cache \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, from the lock file only, so source edits don't invalidate this layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
# The standalone Python's own pip goes: uv installs everything, and pip's vendored certifi,
# urllib3, msgpack and setuptools lag the locked ones and trip image scans.
RUN uv python install 3.12 \
    && rm -rf /opt/python/cpython-3.12*/lib/python3.12/site-packages/pip* \
        /opt/python/cpython-3.12*/bin/pip* \
    && uv sync --frozen --no-dev --extra tts --extra whisper --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra tts --extra whisper --no-editable \
    && python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/usr/share/nltk_data', quiet=True)" \
    && calliope --help > /dev/null

# The batch runner: one book from an inbox per run, as an alternative entrypoint.
COPY batch/calliope_batch.py /opt/

# Model weights (Chatterbox, faster-whisper) download to $HF_HOME on first use: mount /cache.
VOLUME ["/cache"]
WORKDIR /books

ENTRYPOINT ["calliope"]
