#!/usr/bin/env bash
# =============================================================================
# provisioning.sh — Downloads models at container startup (provisioning time)
#
# Design:
#   - Called by start.sh before ComfyUI launches
#   - Checks each model individually by filename — no global lock file
#   - Supports persistent Vast.ai network volume mounted at /workspace
#   - If /workspace is mounted, models cached there and reused across cold starts
#   - If not mounted, models downloaded to /opt/ComfyUI/models (ephemeral)
#   - Supports PROVISIONING_SCRIPT env var override (Vast.ai native mechanism)
#
# Model URLs — fill in before use.
# See README.md → "TODO: Model Download URLs"
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { echo "[provisioning] $*"; }
warn() { echo "[provisioning][WARN] $*" >&2; }
die()  { echo "[provisioning][ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Determine model cache directory
# Use persistent /workspace volume if available, fall back to ephemeral
# ---------------------------------------------------------------------------
COMFYUI_PATH="${COMFYUI_PATH:-/opt/ComfyUI}"

if mountpoint -q /workspace 2>/dev/null; then
    log "Persistent /workspace volume detected — using /workspace/models as cache"
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/models}"
    mkdir -p "$MODEL_CACHE_DIR"
    # Redirect ComfyUI models dir to persistent volume
    if [ -d "$COMFYUI_PATH/models" ] && [ ! -L "$COMFYUI_PATH/models" ]; then
        # Copy any existing dirs structure (no files), then symlink
        cp -r --no-clobber "$COMFYUI_PATH/models/." "$MODEL_CACHE_DIR/" 2>/dev/null || true
        rm -rf "$COMFYUI_PATH/models"
        ln -sfn "$MODEL_CACHE_DIR" "$COMFYUI_PATH/models"
        log "Symlinked $COMFYUI_PATH/models → $MODEL_CACHE_DIR"
    fi
else
    log "No persistent volume — models will be downloaded to $COMFYUI_PATH/models (ephemeral)"
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$COMFYUI_PATH/models}"
fi

# ---------------------------------------------------------------------------
# Create all expected model subdirs
# ---------------------------------------------------------------------------
for d in checkpoints vae loras controlnet upscale_models clip_vision \
          ipadapter mmaudio vfi_models text_encoders unet diffusion_models; do
    mkdir -p "$MODEL_CACHE_DIR/$d"
done

# Ensure all model subdirs exist
mkdir -p "$MODEL_CACHE_DIR/diffusion_models"

# ---------------------------------------------------------------------------
# Download helper with resume support
# Usage: download_model <dest_dir> <filename> <url> [sha256_checksum]
# ---------------------------------------------------------------------------
download_model() {
    local dest_dir="$1"
    local filename="$2"
    local url="$3"
    local checksum="${4:-}"

    local dest="$dest_dir/$filename"

    if [ -f "$dest" ]; then
        log "✓ Already exists: $filename"
        return 0
    fi

    if [ -z "$url" ] || [ "$url" = "TODO" ]; then
        warn "⚠ No URL for $filename — skipping (set URL in provisioning.sh)"
        return 0
    fi

    log "↓ Downloading: $filename"
    log "  from: $url"
    log "  to:   $dest"

    mkdir -p "$dest_dir"

    # wget with resume (-c), quiet progress, retry 3x
    if ! wget -c --tries=3 --timeout=60 --progress=bar:force \
              -O "${dest}.tmp" "$url" 2>&1; then
        warn "Download failed for $filename"
        rm -f "${dest}.tmp"
        return 1
    fi

    # Optional checksum verification
    if [ -n "$checksum" ]; then
        actual=$(sha256sum "${dest}.tmp" | cut -d' ' -f1)
        if [ "$actual" != "$checksum" ]; then
            warn "Checksum mismatch for $filename (expected: $checksum, got: $actual)"
            rm -f "${dest}.tmp"
            return 1
        fi
        log "✓ Checksum OK: $filename"
    fi

    mv "${dest}.tmp" "$dest"
    local size
    size=$(du -sh "$dest" | cut -f1)
    log "✓ Downloaded $filename ($size)"
}

# ---------------------------------------------------------------------------
# CivitAI helper — injects API token into URL if CIVITAI_API_TOKEN is set
# CivitAI requires auth for model downloads.
# Set CIVITAI_API_TOKEN in Vast.ai template env vars.
# ---------------------------------------------------------------------------
civitai_url() {
    local base_url="$1"
    if [ -n "${CIVITAI_API_TOKEN:-}" ]; then
        # Append token as query param (CivitAI's supported auth method)
        if echo "$base_url" | grep -q "?"; then
            echo "${base_url}&token=${CIVITAI_API_TOKEN}"
        else
            echo "${base_url}?token=${CIVITAI_API_TOKEN}"
        fi
    else
        warn "CIVITAI_API_TOKEN is not set — CivitAI download may fail (401)"
        echo "$base_url"
    fi
}

# ============================================================
# MODEL DOWNLOAD SECTION
# URLs are hardcoded below. All sources verified.
# CivitAI models require CIVITAI_API_TOKEN env var.
# ============================================================

# ---------------------------------------------------------------------------
# 1. Diffusion models (safetensors) → models/diffusion_models/
#    Loader: UNETLoader (core) — nodes 197 (HIGH) and 186 (LOW) in workflow
#    Source: CivitAI — requires CIVITAI_API_TOKEN
# ---------------------------------------------------------------------------
DIFFUSION_DIR="$MODEL_CACHE_DIR/diffusion_models"
mkdir -p "$DIFFUSION_DIR"

download_model "$DIFFUSION_DIR" \
    "smoothMixWan2214BI2V_i2vV20High.safetensors" \
    "$(civitai_url 'https://civitai.com/api/download/models/2513182?type=Model&format=SafeTensor&size=pruned&fp=fp8')"

download_model "$DIFFUSION_DIR" \
    "smoothMixWan2214BI2V_i2vV20Low.safetensors" \
    "$(civitai_url 'https://civitai.com/api/download/models/2513186?type=Model&format=SafeTensor&size=pruned&fp=fp8')"

# ---------------------------------------------------------------------------
# 2. VAE → models/vae/
#    Loader: VAELoader (core)
#    Source: https://huggingface.co/Kijai/WanVideo_comfy
# ---------------------------------------------------------------------------
VAE_DIR="$MODEL_CACHE_DIR/vae"

download_model "$VAE_DIR" \
    "Wan2_1_VAE_fp32.safetensors" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors"

# ---------------------------------------------------------------------------
# 3. Text encoder → models/text_encoders/
#    Loader: CLIPLoader (core, type="wan")
#    Source: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
# ---------------------------------------------------------------------------
TEXT_ENC_DIR="$MODEL_CACHE_DIR/text_encoders"

download_model "$TEXT_ENC_DIR" \
    "umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# ---------------------------------------------------------------------------
# 4. CLIP Vision → models/clip_vision/
#    Loader: CLIPVisionLoader (core)
#    Source: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
# ---------------------------------------------------------------------------
CLIP_VISION_DIR="$MODEL_CACHE_DIR/clip_vision"

download_model "$CLIP_VISION_DIR" \
    "clip_vision_h.safetensors" \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"

# ---------------------------------------------------------------------------
# 5. MMAudio models → models/mmaudio/
#    Loader: MMAudioModelLoader, MMAudioFeatureUtilsLoader (comfyui-mmaudio)
# ---------------------------------------------------------------------------
MMAUDIO_DIR="$MODEL_CACHE_DIR/mmaudio"

# Main audio model — NSFW-capable variant (used in workflow node 211)
# Source: https://huggingface.co/phazei/NSFW_MMaudio
download_model "$MMAUDIO_DIR" \
    "mmaudio_large_44k_nsfw_gold_8.5k_final_fp16.safetensors" \
    "https://huggingface.co/phazei/NSFW_MMaudio/resolve/main/mmaudio_large_44k_nsfw_gold_8.5k_final_fp16.safetensors"

# Feature utils — VAE, Synchformer, CLIP encoder (used in workflow node 204)
# Source: https://huggingface.co/Kijai/MMAudio_safetensors
download_model "$MMAUDIO_DIR" \
    "mmaudio_vae_44k_fp16.safetensors" \
    "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_vae_44k_fp16.safetensors"

download_model "$MMAUDIO_DIR" \
    "mmaudio_synchformer_fp16.safetensors" \
    "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/mmaudio_synchformer_fp16.safetensors"

download_model "$MMAUDIO_DIR" \
    "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors" \
    "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"

# ---------------------------------------------------------------------------
# 6. VFI model (RIFE flownet) → models/vfi_models/
#    Loader: RIFEInterpolation (ComfyUI-VFI)
#    Source: https://huggingface.co/LeonJoe13/Sonic (SHA256: fe854fc8...)
# ---------------------------------------------------------------------------
VFI_DIR="$MODEL_CACHE_DIR/vfi_models"

download_model "$VFI_DIR" \
    "flownet.pkl" \
    "https://huggingface.co/LeonJoe13/Sonic/resolve/main/RIFE/flownet.pkl"

# ---------------------------------------------------------------------------
# 7. LoRA models → models/loras/
#    Loader: Power Lora Loader (rgthree)
#    lightx2v: speed LoRA, used every run (baked into provisioning)
# ---------------------------------------------------------------------------
LORA_DIR="$MODEL_CACHE_DIR/loras"

# Speed LoRA — used in both HIGH and LOW Lora Loader nodes
# Source: https://huggingface.co/Kijai/WanVideo_comfy
download_model "$LORA_DIR" \
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors" \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/709844db75d2e15582cf204e9a0b5e12b23a35dd/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=== Provisioning complete ==="
log "Model cache: $MODEL_CACHE_DIR"
log ""
log "Model inventory:"
for d in diffusion_models vae text_encoders clip_vision mmaudio vfi_models loras; do
    count=$(find "$MODEL_CACHE_DIR/$d" -type f 2>/dev/null | wc -l)
    log "  $d/: $count files"
done
log ""
