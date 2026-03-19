# =============================================================================
# vast-comfyui-wan — Production Dockerfile
# Base: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9 (runtime)
# Models are NOT baked in — downloaded at provisioning time via provisioning.sh
# Build & push only via GitHub Actions (.github/workflows/build-and-push.yml)
# =============================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG GITHUB_REPOSITORY="nikitowie/vast-uploader"
LABEL org.opencontainers.image.source="https://github.com/${GITHUB_REPOSITORY}"
LABEL org.opencontainers.image.description="ComfyUI Wan 2.2 I2V + MMAudio — Vast.ai Serverless"

ARG BUILD_DATE=""
ARG GIT_SHA=""
ENV BUILD_DATE=${BUILD_DATE} \
    GIT_SHA=${GIT_SHA}

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install ComfyUI
# ---------------------------------------------------------------------------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Install handler dependencies
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    httpx \
    websockets \
    pydantic \
    runpod

# ---------------------------------------------------------------------------
# Install custom node dependencies (Python packages)
# Some nodes install their own deps via requirements.txt during clone step
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    # VHS (VideoHelperSuite)
    imageio[ffmpeg] \
    imageio-ffmpeg \
    av \
    # KJNodes / color processing
    kornia \
    # WanVideoWrapper / MMAudio
    einops \
    accelerate \
    transformers \
    # Audio processing
    soundfile \
    librosa \
    torchaudio \
    # GGUF support
    gguf \
    # Comfyroll / misc
    scipy \
    # Image processing
    Pillow \
    # requests
    requests \
    tqdm

# ---------------------------------------------------------------------------
# Install custom nodes
# ---------------------------------------------------------------------------
COPY scripts/install_custom_nodes.sh /opt/scripts/install_custom_nodes.sh
RUN chmod +x /opt/scripts/install_custom_nodes.sh && \
    /opt/scripts/install_custom_nodes.sh

# ---------------------------------------------------------------------------
# Set up ComfyUI model directories
# ---------------------------------------------------------------------------
RUN mkdir -p \
      /opt/ComfyUI/models/checkpoints \
      /opt/ComfyUI/models/vae \
      /opt/ComfyUI/models/loras \
      /opt/ComfyUI/models/controlnet \
      /opt/ComfyUI/models/upscale_models \
      /opt/ComfyUI/models/clip_vision \
      /opt/ComfyUI/models/ipadapter \
      /opt/ComfyUI/models/mmaudio \
      /opt/ComfyUI/models/vfi_models \
      /opt/ComfyUI/models/text_encoders \
      /opt/ComfyUI/models/unet \
      /opt/ComfyUI/models/diffusion_models \
      /opt/ComfyUI/output \
      /opt/ComfyUI/input \
      /opt/ComfyUI/user/default/workflows

# ---------------------------------------------------------------------------
# Copy project files
# ---------------------------------------------------------------------------
COPY extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml
COPY handler.py /opt/handler.py
COPY provisioning.sh /opt/provisioning.sh
COPY scripts/ /opt/scripts/

# Make scripts executable
RUN chmod +x /opt/provisioning.sh /opt/scripts/*.sh 2>/dev/null || true && \
    chmod +x /opt/scripts/*.py 2>/dev/null || true

# ---------------------------------------------------------------------------
# Copy workflow for optional local smoke test
# ---------------------------------------------------------------------------
COPY workflows/ /opt/ComfyUI/user/default/workflows/

# ---------------------------------------------------------------------------
# Build info
# ---------------------------------------------------------------------------
RUN /opt/scripts/build-info.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
ENV COMFYUI_PATH=/opt/ComfyUI \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    HANDLER_PORT=8000 \
    # Model cache dir — override to /workspace/models if using Vast network volume
    MODEL_CACHE_DIR=/opt/ComfyUI/models \
    LORA_DIR=/opt/ComfyUI/models/loras \
    OUTPUT_DIR=/opt/ComfyUI/output \
    # Max time (seconds) to wait for one inference job
    MAX_WAIT_SECONDS=600 \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8188

# start.sh: runs provisioning → launches ComfyUI → launches handler
CMD ["/opt/scripts/start.sh"]
