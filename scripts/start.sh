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

# Note: no set -e so container stays alive for debugging on failures
set -uo pipefail

COMFYUI_PATH="${COMFYUI_PATH:-/opt/ComfyUI}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
HANDLER_PORT="${HANDLER_PORT:-8000}"
SKIP_PROVISIONING="${SKIP_PROVISIONING:-0}"
START_LOG="/tmp/start.log"

log()  { echo "[start] $*" | tee -a "$START_LOG"; }
err()  { echo "[start][ERROR] $*" | tee -a "$START_LOG" >&2; }

# Redirect all stdout/stderr to log file as well
exec > >(tee -a "$START_LOG") 2>&1

log "=== Container starting at $(date) ==="
log "COMFYUI_PATH=$COMFYUI_PATH"
log "COMFYUI_PORT=$COMFYUI_PORT"
log "HANDLER_PORT=$HANDLER_PORT"
log "SKIP_PROVISIONING=$SKIP_PROVISIONING"

# ---------------------------------------------------------------------------
# Fallback debug HTTP server — starts immediately, serves /tmp/start.log
# on port 8000 ONLY if the handler isn't started yet.
# This means on errors you can hit /health and get container logs.
# ---------------------------------------------------------------------------
start_debug_server() {
    python3 -c "
import http.server, os, time

class DebugHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # suppress access log
    def do_GET(self):
        try:
            with open('$START_LOG') as f:
                body = f.read()
        except Exception as e:
            body = 'Log not available: ' + str(e)
        encoded = body.encode('utf-8', errors='replace')
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

server = http.server.HTTPServer(('0.0.0.0', int('$HANDLER_PORT')), DebugHandler)
server.serve_forever()
" &
    DEBUG_SERVER_PID=$!
    log "Debug HTTP server started on port $HANDLER_PORT (PID: $DEBUG_SERVER_PID)"
}

kill_debug_server() {
    if [ -n "${DEBUG_SERVER_PID:-}" ]; then
        kill "$DEBUG_SERVER_PID" 2>/dev/null || true
        log "Debug HTTP server stopped"
    fi
}

start_debug_server

# ---------------------------------------------------------------------------
# 1. Provisioning
# ---------------------------------------------------------------------------
if [ "$SKIP_PROVISIONING" = "1" ]; then
    log "SKIP_PROVISIONING=1 — skipping model downloads"
else
    log "Running provisioning.sh..."
    if ! bash /opt/provisioning.sh; then
        err "provisioning.sh failed — container will stay alive for debugging"
        err "Check port $HANDLER_PORT for logs"
        wait  # keep container alive
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 1b. NVMe acceleration — copy heavy models from network volume to local SSD
#     /tmp is on local NVMe (~3-7 GB/s) vs network volume (~200-500 MB/s)
#     Only copy diffusion_models (~28GB) — the biggest bottleneck.
#     Other models (VAE, text_enc, etc.) are small and fine on network volume.
#     Uses rsync with --drop-cache to avoid filling RAM.
# ---------------------------------------------------------------------------
NVME_CACHE="/tmp/nvme_models"
WORKSPACE_MODELS="/workspace/models"

if mountpoint -q /workspace 2>/dev/null && [ -d "$WORKSPACE_MODELS" ]; then
    # Create NVMe cache dir structure matching workspace
    mkdir -p "$NVME_CACHE/diffusion_models"

    if [ -d "$NVME_CACHE/diffusion_models" ] && [ "$(ls -A "$NVME_CACHE/diffusion_models" 2>/dev/null)" ]; then
        log "NVMe cache already has diffusion_models — skipping copy"
    else
        log "Copying diffusion_models to local NVMe (bypassing page cache)..."
        COPY_START=$(date +%s)
        # Use dd-style copy with direct I/O to avoid RAM pressure
        for f in "$WORKSPACE_MODELS/diffusion_models"/*; do
            [ -f "$f" ] || continue
            fname=$(basename "$f")
            if [ -f "$NVME_CACHE/diffusion_models/$fname" ]; then
                log "  ✓ $fname already cached"
            else
                log "  ↓ Copying $fname..."
                cp --reflink=auto "$f" "$NVME_CACHE/diffusion_models/$fname"
                # Drop page cache for this file to free RAM
                python3 -c "
import os, ctypes, ctypes.util
f = os.open('$NVME_CACHE/diffusion_models/$fname', os.O_RDONLY)
try:
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
    libc.posix_fadvise(f, 0, 0, 4)  # POSIX_FADV_DONTNEED=4
finally:
    os.close(f)
" 2>/dev/null || true
            fi
        done
        COPY_END=$(date +%s)
        COPY_ELAPSED=$((COPY_END - COPY_START))
        COPY_SIZE=$(du -sh "$NVME_CACHE/diffusion_models" | cut -f1)
        log "NVMe copy done: $COPY_SIZE in ${COPY_ELAPSED}s"
    fi

    # Symlink only diffusion_models to NVMe, keep rest on network volume
    if [ -L "$COMFYUI_PATH/models" ]; then
        # models is already symlinked to /workspace/models by provisioning.sh
        # Just override diffusion_models subdir
        rm -rf "$COMFYUI_PATH/models/diffusion_models" 2>/dev/null || true
        ln -sfn "$NVME_CACHE/diffusion_models" "$COMFYUI_PATH/models/diffusion_models"
        log "Symlinked diffusion_models → NVMe, rest stays on network volume"
    elif [ -d "$COMFYUI_PATH/models" ]; then
        rm -rf "$COMFYUI_PATH/models/diffusion_models" 2>/dev/null || true
        ln -sfn "$NVME_CACHE/diffusion_models" "$COMFYUI_PATH/models/diffusion_models"
        log "Symlinked diffusion_models → NVMe"
    fi
else
    log "No /workspace volume or no models — ComfyUI will use default model paths"
fi

# ---------------------------------------------------------------------------
# 2. Start ComfyUI in background
# ---------------------------------------------------------------------------
log "Starting ComfyUI on port $COMFYUI_PORT..."

if [ ! -d "$COMFYUI_PATH" ]; then
    err "ComfyUI directory not found: $COMFYUI_PATH"
    err "Contents of /opt/: $(ls /opt/ 2>&1)"
    wait
    exit 1
fi

cd "$COMFYUI_PATH"
log "Working dir: $(pwd)"
log "Python: $(python --version 2>&1)"

mkdir -p /var/log

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
MAX_WAIT=300
ELAPSED=0

until curl -sf "http://127.0.0.1:$COMFYUI_PORT/system_stats" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        err "ComfyUI failed to start within ${MAX_WAIT}s"
        err "=== Last 30 lines of ComfyUI log ==="
        tail -30 /var/log/comfyui.log | tee -a "$START_LOG" >&2
        err "Container staying alive for debugging — check port $HANDLER_PORT"
        wait
        exit 1
    fi
    # Check if ComfyUI process died
    if ! kill -0 "$COMFYUI_PID" 2>/dev/null; then
        err "ComfyUI process (PID $COMFYUI_PID) died unexpectedly!"
        err "=== ComfyUI log ==="
        cat /var/log/comfyui.log | tee -a "$START_LOG" >&2
        err "Container staying alive for debugging — check port $HANDLER_PORT"
        wait
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

log "ComfyUI ready after ${ELAPSED}s"

# ---------------------------------------------------------------------------
# 4. Start handler (kill debug server first, then start handler in foreground)
# ---------------------------------------------------------------------------
kill_debug_server
log "Starting handler on port $HANDLER_PORT..."
exec python /opt/handler.py
