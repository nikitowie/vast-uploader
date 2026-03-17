#!/usr/bin/env bash
# =============================================================================
# build-info.sh — Writes build metadata to /opt/build-info.json
# Called during Docker build to stamp the image with reproducible info.
# =============================================================================

set -euo pipefail

COMFYUI_PATH="${COMFYUI_PATH:-/opt/ComfyUI}"
OUTPUT="/opt/build-info.json"

# ComfyUI git info
COMFYUI_SHA=$(git -C "$COMFYUI_PATH" rev-parse HEAD 2>/dev/null || echo "unknown")

# Custom nodes info
NODES_INFO=""
for d in "$COMFYUI_PATH/custom_nodes"/*/; do
    name=$(basename "$d")
    sha=$(git -C "$d" rev-parse HEAD 2>/dev/null || echo "unknown")
    NODES_INFO="${NODES_INFO}\"$name\": \"$sha\","
done
# Remove trailing comma
NODES_INFO="${NODES_INFO%,}"

cat > "$OUTPUT" <<EOF
{
  "build_date": "${BUILD_DATE:-unknown}",
  "git_sha": "${GIT_SHA:-unknown}",
  "comfyui_sha": "$COMFYUI_SHA",
  "python_version": "$(python --version 2>&1)",
  "torch_version": "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')",
  "cuda_version": "$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')",
  "custom_nodes": {
    $NODES_INFO
  }
}
EOF

echo "[build-info] Written to $OUTPUT"
cat "$OUTPUT"
