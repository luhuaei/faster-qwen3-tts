ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.08-py3
ARG PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple
ARG PIP_EXTRA_INDEX_URL=https://pypi.org/simple
ARG TORCHAUDIO_SPEC=torchaudio==2.8.0
ARG MODEL_NAME=Qwen3-TTS-12Hz-0.6B-Base
ARG MODEL_DIR=/opt/models/Qwen3-TTS-12Hz-0.6B-Base
ARG QWEN_TTS_MODEL=/opt/models/Qwen3-TTS-12Hz-0.6B-Base
ARG QWEN_TTS_MODE=clone
ARG QWEN_TTS_DEFAULT_VOICE=vivian
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG TORCHAUDIO_SPEC
ARG MODEL_NAME
ARG MODEL_DIR
ARG QWEN_TTS_MODEL
ARG QWEN_TTS_MODE
ARG QWEN_TTS_DEFAULT_VOICE

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PIP_CONSTRAINT= \
    UV_CONSTRAINT= \
    UV_BUILD_CONSTRAINT= \
    APP_VENV=/opt/faster-qwen3-tts-venv \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PATH=/opt/faster-qwen3-tts-venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache \
    QWEN_TTS_MODEL=${QWEN_TTS_MODEL} \
    QWEN_TTS_MODE=${QWEN_TTS_MODE} \
    QWEN_TTS_VOICES=/opt/build/faster-qwen3-tts/voices.json \
    QWEN_TTS_DEFAULT_VOICE=${QWEN_TTS_DEFAULT_VOICE} \
    QWEN_TTS_LANGUAGE=Auto \
    QWEN_TTS_CHUNK_SIZE=8 \
    QWEN_TTS_WARMUP_MAX_NEW_TOKENS=32

WORKDIR /opt/build

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 opus-tools sox \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

COPY faster-qwen3-tts-requirements.txt /tmp/faster-qwen3-tts-requirements.txt
RUN grep -v '^torchaudio$' /tmp/faster-qwen3-tts-requirements.txt > /tmp/faster-qwen3-tts-requirements-no-torchaudio.txt \
    && uv venv "${APP_VENV}" --python /usr/bin/python3 --system-site-packages --seed \
    && PIP_INDEX_URL="${PIP_INDEX_URL}" PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}" \
       "${APP_VENV}/bin/python3" -m pip install --no-cache-dir -r /tmp/faster-qwen3-tts-requirements-no-torchaudio.txt \
    && PIP_INDEX_URL="${PIP_INDEX_URL}" PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}" \
       "${APP_VENV}/bin/python3" -m pip install --no-cache-dir --no-deps "accelerate==1.12.0" \
    && PIP_INDEX_URL="${PIP_INDEX_URL}" PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL}" \
       "${APP_VENV}/bin/python3" -m pip install --no-cache-dir --no-deps --force-reinstall "${TORCHAUDIO_SPEC}" \
    && rm -rf /root/.cache/uv

COPY MANIFEST.in README.md pyproject.toml /opt/build/faster-qwen3-tts/
COPY voices.json /opt/build/faster-qwen3-tts/voices.json
COPY voices /opt/build/faster-qwen3-tts/voices
COPY faster_qwen3_tts /opt/build/faster-qwen3-tts/faster_qwen3_tts
COPY examples/openai_server.py /opt/build/faster-qwen3-tts/examples/openai_server.py
COPY models/${MODEL_NAME} ${MODEL_DIR}

RUN "${APP_VENV}/bin/python3" -m pip install --no-cache-dir --no-deps "qwen-tts>=0.1.1" \
    && "${APP_VENV}/bin/python3" -m pip install --no-cache-dir "fastapi>=0.100.0" "uvicorn[standard]>=0.24.0" "python-multipart>=0.0.7" \
    && "${APP_VENV}/bin/python3" -m pip install --no-cache-dir --no-deps --no-build-isolation -e "/opt/build/faster-qwen3-tts[demo]" \
    && "${APP_VENV}/bin/python3" -c 'import sys, torch, torchaudio; torch_version = torch.__version__; torchaudio_version = torchaudio.__version__; (torch_version.startswith("2.8.0a0") and "nv25.08" in torch_version) or sys.exit(f"expected NGC PyTorch 2.8.0a0 nv25.08, got {torch_version}"); torchaudio_version == "2.8.0" or sys.exit(f"expected torchaudio 2.8.0, got {torchaudio_version}"); print(f"validated torch {torch_version}, torchaudio {torchaudio_version}")' \
    && find /opt/build -name '__pycache__' -type d -prune -exec rm -rf '{}' + \
    && find /root/.cache -type d -name '.locks' -prune -exec rm -rf '{}' + \
    && rm -rf /root/.cache/uv

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace
EXPOSE 8000
ENTRYPOINT ["python3", "/opt/build/faster-qwen3-tts/examples/openai_server.py"]
