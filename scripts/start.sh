#!/usr/bin/env bash
# =============================================================================
# start.sh — Container entrypoint
#
# Sequence:
#   1. Run provisioning.sh (downloads models if needed)
#   2. Start ComfyUI in background
#   3. Wait for ComfyUI to be ready (health check loop)
#   4. Start handler.py (Vast.ai serverless HTTP handler)
#
# Environment variables (all have sensible defaults):
#   COMFYUI_PATH       — path to ComfyUI install (default: /opt/ComfyUI)
#   COMFYUI_PORT       — ComfyUI internal port (default: 8188)
#   HANDLER_PORT       — handler API port (default: 8000)
#   SKIP_PROVISIONING  — set to "1" to skip model downloads (dev/test only)
#   COMFYUI_ARGS       — extra args forwarded to ComfyUI --extra-model-paths etc.
# =============================================================================

set -euo pipefail

COMFYUI_PATH="${COMFYUI_PATH:-/opt/ComfyUI}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
HANDLER_PORT="${HANDLER_PORT:-8000}"
SKIP_PROVISIONING="${SKIP_PROVISIONING:-0}"

log()  { echo "[start] $*"; }
die()  { echo "[start][ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Provisioning
# ---------------------------------------------------------------------------
if [ "$SKIP_PROVISIONING" = "1" ]; then
    log "SKIP_PROVISIONING=1 — skipping model downloads"
else
    log "Running provisioning.sh..."
    bash /opt/provisioning.sh
fi

# ---------------------------------------------------------------------------
# 2. Start ComfyUI in background
# ---------------------------------------------------------------------------
log "Starting ComfyUI on port $COMFYUI_PORT..."

cd "$COMFYUI_PATH"

python main.py \
    --listen 127.0.0.1 \
    --port "$COMFYUI_PORT" \
    --output-directory "$COMFYUI_PATH/output" \
    --disable-xformers \
    --preview-method none \
    ${COMFYUI_ARGS:-} \
    >> /var/log/comfyui.log 2>&1 &

COMFYUI_PID=$!
log "ComfyUI started (PID: $COMFYUI_PID)"

# ---------------------------------------------------------------------------
# 3. Wait for ComfyUI to be ready
# ---------------------------------------------------------------------------
log "Waiting for ComfyUI to become ready..."
MAX_WAIT=120
ELAPSED=0

until curl -sf "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        log "ComfyUI failed to start within ${MAX_WAIT}s. Last log lines:"
        tail -20 /var/log/comfyui.log >&2
        die "ComfyUI startup timeout"
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

log "ComfyUI ready after ${ELAPSED}s"

# ---------------------------------------------------------------------------
# 4. Start handler (foreground — keeps container alive)
# ---------------------------------------------------------------------------
log "Starting handler on port $HANDLER_PORT..."
exec python /opt/handler.py
