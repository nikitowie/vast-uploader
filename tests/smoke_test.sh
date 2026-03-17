#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Basic structural smoke tests for the container image
#
# Run inside the container or in CI after build (with SKIP_PROVISIONING=1).
# Tests:
#   1. ComfyUI is installed and importable
#   2. All required custom nodes are present
#   3. Model directories exist with expected structure
#   4. Handler can be imported (syntax check)
#   5. lora_helper works (local path, no network)
#   6. ComfyUI health endpoint responds
# =============================================================================

set -euo pipefail

COMFYUI_PATH="${COMFYUI_PATH:-/opt/ComfyUI}"
PASS=0
FAIL=0
WARN=0

green='\033[0;32m'
red='\033[0;31m'
yellow='\033[0;33m'
reset='\033[0m'

pass() { echo -e "${green}[PASS]${reset} $*"; PASS=$((PASS+1)); }
fail() { echo -e "${red}[FAIL]${reset} $*"; FAIL=$((FAIL+1)); }
warn() { echo -e "${yellow}[WARN]${reset} $*"; WARN=$((WARN+1)); }

echo "=============================="
echo " Smoke Test: vast-comfyui-wan"
echo "=============================="
echo ""

# ---------------------------------------------------------------------------
# 1. ComfyUI installation
# ---------------------------------------------------------------------------
echo "--- 1. ComfyUI installation ---"

if [ -f "$COMFYUI_PATH/main.py" ]; then
    pass "ComfyUI main.py found at $COMFYUI_PATH"
else
    fail "ComfyUI main.py NOT found at $COMFYUI_PATH"
fi

if python -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
    pass "PyTorch importable"
else
    fail "PyTorch import failed"
fi

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    pass "CUDA available"
else
    warn "CUDA not available (expected in GPU container, may be OK in CI)"
fi

# ---------------------------------------------------------------------------
# 2. Custom nodes presence
# ---------------------------------------------------------------------------
echo ""
echo "--- 2. Custom nodes ---"

REQUIRED_NODES=(
    "comfy_mtb"
    "ComfyUI-VideoHelperSuite"
    "ComfyUI-WanVideoWrapper"
    "ComfyUI-MMAudio"
    "ComfyUI-KJNodes"
    "ComfyUI-Easy-Use"
    "rgthree-comfy"
    "ComfyUI_Comfyroll_CustomNodes"
    "ComfyUI-GGUF"
    "ComfyUI-VFI"
)

for node in "${REQUIRED_NODES[@]}"; do
    if [ -d "$COMFYUI_PATH/custom_nodes/$node" ]; then
        pass "custom_nodes/$node present"
    else
        fail "custom_nodes/$node MISSING"
    fi
done

# ---------------------------------------------------------------------------
# 3. Model directories
# ---------------------------------------------------------------------------
echo ""
echo "--- 3. Model directories ---"

MODEL_DIRS=(
    "models/unet"
    "models/vae"
    "models/text_encoders"
    "models/clip_vision"
    "models/loras"
    "models/mmaudio"
    "models/vfi_models"
    "models/checkpoints"
    "models/controlnet"
)

for d in "${MODEL_DIRS[@]}"; do
    full="$COMFYUI_PATH/$d"
    if [ -d "$full" ] || [ -L "$full" ]; then
        pass "$d/ exists"
    else
        fail "$d/ MISSING"
    fi
done

# Check expected base model files (only if models were provisioned)
echo ""
echo "--- 3b. Base model files (skipped if SKIP_PROVISIONING=1) ---"
if [ "${SKIP_PROVISIONING:-0}" = "1" ]; then
    warn "SKIP_PROVISIONING=1 — skipping model file checks"
else
    EXPECTED_MODELS=(
        "models/unet/wan2.2_i2v_low_noise_14B_Q8_0.gguf"
        "models/unet/wan2.2_i2v_high_noise_14B_Q8_0.gguf"
        "models/vae/Wan2_1_VAE_fp32.safetensors"
        "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        "models/clip_vision/clip_vision_h.safetensors"
        "models/mmaudio/mmaudio_large_44k_v2_fp16.safetensors"
        "models/mmaudio/mmaudio_vae_44k_fp16.safetensors"
        "models/mmaudio/mmaudio_synchformer_fp16.safetensors"
        "models/mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors"
        "models/vfi_models/flownet.pkl"
        "models/loras/SmoothMix_T2V_High_v3.safetensors"
        "models/loras/SmoothMix_T2V_Low_v3.safetensors"
    )
    for f in "${EXPECTED_MODELS[@]}"; do
        full="$COMFYUI_PATH/$f"
        if [ -f "$full" ]; then
            size=$(du -sh "$full" | cut -f1)
            pass "$f ($size)"
        else
            warn "$f not present (will be downloaded at provisioning time)"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 4. Handler syntax check
# ---------------------------------------------------------------------------
echo ""
echo "--- 4. Handler syntax ---"

if python -m py_compile /opt/handler.py 2>/dev/null; then
    pass "handler.py syntax OK"
else
    fail "handler.py syntax ERROR"
fi

# ---------------------------------------------------------------------------
# 5. lora_helper self-test
# ---------------------------------------------------------------------------
echo ""
echo "--- 5. lora_helper ---"

if python -c "
import sys
sys.path.insert(0, '/opt/scripts')
from lora_helper import ensure_lora, _verify_checksum
import tempfile, pathlib, hashlib

# Create a fake lora file and verify checksum logic
with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as f:
    f.write(b'fake lora data for testing')
    tmp = f.name

p = pathlib.Path(tmp)
expected = hashlib.sha256(b'fake lora data for testing').hexdigest()
assert _verify_checksum(p, expected), 'checksum verify failed'
p.unlink()
print('lora_helper self-test passed')
" 2>/dev/null; then
    pass "lora_helper checksum logic OK"
else
    fail "lora_helper self-test FAILED"
fi

# ---------------------------------------------------------------------------
# 6. ComfyUI health (only if running)
# ---------------------------------------------------------------------------
echo ""
echo "--- 6. ComfyUI health endpoint ---"

if curl -sf "http://127.0.0.1:${COMFYUI_PORT:-8188}/system_stats" > /dev/null 2>&1; then
    pass "ComfyUI /system_stats responds"
else
    warn "ComfyUI not running (expected in pre-start smoke test)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================="
echo " Results: PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
echo "=============================="

if [ $FAIL -gt 0 ]; then
    echo -e "${red}Smoke test FAILED${reset}"
    exit 1
else
    echo -e "${green}Smoke test PASSED${reset}"
    exit 0
fi
