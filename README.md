# vast-comfyui-wan

Production-ready ComfyUI image for **Vast.ai Serverless** — Wan 2.2 I2V + MMAudio audio-to-video workflow.

Build and push exclusively via **GitHub Actions**. No local Docker required.

---

## Architecture overview

```
┌────────────────────────────────────────────────────────────────┐
│  Docker image (built by GitHub Actions)                        │
│  ├── ComfyUI + all 13 custom nodes + Python deps              │
│  ├── No model files (image stays ~12–15 GB)                    │
│  └── handler.py (FastAPI HTTP server for Vast.ai)              │
└────────────────────────────────────────────────────────────────┘
         │ deployed to Vast.ai Serverless
         ▼
┌────────────────────────────────────────────────────────────────┐
│  Container start (provisioning time)                           │
│  └── provisioning.sh downloads models to MODEL_CACHE_DIR      │
│      - SmoothMix Wan 2.2 14B I2V High + Low safetensors       │
│      - VAE, text encoder (umt5), CLIP vision                  │
│      - MMAudio NSFW + 3 feature files, VFI flownet.pkl        │
│      - lightx2v speed LoRA (always used)                      │
└────────────────────────────────────────────────────────────────┘
         │ runtime
         ▼
┌────────────────────────────────────────────────────────────────┐
│  POST /generate/sync                                           │
│  ├── optional: loras[] → lora_helper.py downloads per-request │
│  ├── submits workflow_json to ComfyUI                         │
│  ├── waits for completion (WebSocket)                         │
│  └── returns output files as base64                           │
└────────────────────────────────────────────────────────────────┘
```

### What lives where

| Layer | Contents | When |
|---|---|---|
| **Image** (baked) | ComfyUI, custom nodes, Python deps | Build time (GitHub Actions) |
| **Provisioning** | All model files (GGUF, VAE, MMAudio, etc.) | Container start |
| **Runtime** | Ad-hoc LoRA via API payload | Per-request |

---

## Quick start

### 1. Fork / clone this repo

```bash
git clone https://github.com/youruser/vast-comfyui-wan
cd vast-comfyui-wan
```

### 2. Set CivitAI API token

Two models (SmoothMix High and Low) are downloaded from CivitAI and require authentication.

1. Get your token at [civitai.com/user/account](https://civitai.com/user/account) → API Keys
2. Add it as a Vast.ai template env var: `CIVITAI_API_TOKEN=your_token_here`

All other models are from HuggingFace and download without auth.

### 3. Model manifest (all URLs hardcoded — no manual config needed)

All download URLs are already filled in `provisioning.sh`. The models downloaded at container start:

| File | Dir | Source |
|---|---|---|
| `smoothMixWan2214BI2V_i2vV20High.safetensors` | `diffusion_models/` | CivitAI (needs token) |
| `smoothMixWan2214BI2V_i2vV20Low.safetensors` | `diffusion_models/` | CivitAI (needs token) |
| `Wan2_1_VAE_fp32.safetensors` | `vae/` | HuggingFace / Kijai |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | HuggingFace / Comfy-Org |
| `clip_vision_h.safetensors` | `clip_vision/` | HuggingFace / Comfy-Org |
| `mmaudio_large_44k_nsfw_gold_8.5k_final_fp16.safetensors` | `mmaudio/` | HuggingFace / phazei |
| `mmaudio_vae_44k_fp16.safetensors` | `mmaudio/` | HuggingFace / Kijai |
| `mmaudio_synchformer_fp16.safetensors` | `mmaudio/` | HuggingFace / Kijai |
| `apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors` | `mmaudio/` | HuggingFace / Kijai |
| `flownet.pkl` | `vfi_models/` | HuggingFace / LeonJoe13 |
| `lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors` | `loras/` | HuggingFace / Kijai |

### 4. Add GitHub Secrets

Go to your repo → **Settings → Secrets and variables → Actions → Secrets**

| Secret | Value |
|---|---|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub access token (generate at hub.docker.com → Account Settings → Security) |

For GHCR (GitHub Container Registry), no secrets needed — `GITHUB_TOKEN` is used automatically. Skip `DOCKER_USERNAME` and `DOCKER_PASSWORD`.

### 5. Set GitHub Variables

Go to **Settings → Secrets and variables → Actions → Variables**

| Variable | Example value | Notes |
|---|---|---|
| `REGISTRY` | `docker.io` | or `ghcr.io` |
| `IMAGE_NAME` | `youruser/vast-comfyui-wan` | Docker Hub format |

For GHCR: `IMAGE_NAME` = `yourghuser/vast-comfyui-wan` (lowercase).

### 6. Trigger a build

**Option A — Push to main:**
```bash
git add .
git commit -m "Initial setup"
git push origin main
```

**Option B — Manual dispatch (GitHub UI):**
1. Go to **Actions → Build and Push Docker Image**
2. Click **Run workflow**
3. Optionally set a custom tag (default: `latest`)
4. Click **Run workflow**

**Option C — Push a version tag:**
```bash
git tag v1.0.0
git push origin v1.0.0
```

### 7. Get the image path

After the build succeeds, go to the workflow run → summary. You'll see:

```
Image ready for Vast.ai:
  docker.io/youruser/vast-comfyui-wan:abc1234
```

### 8. Configure Vast.ai template

1. Go to [vast.ai](https://vast.ai) → **Templates** → **New Template** (or edit existing)
2. Set **Image Path:Tag**:
   ```
   youruser/vast-comfyui-wan:latest
   ```
3. Set **On-start script** (optional, alternative way to run provisioning):
   ```
   /opt/provisioning.sh
   ```
4. Set **Environment variables** — required:
   ```
   CIVITAI_API_TOKEN=your_civitai_token_here
   ```
   Optional overrides (all URLs already hardcoded in `provisioning.sh`):
   ```
   SKIP_PROVISIONING=0        # set to 1 to skip downloads (dev only)
   MAX_WAIT_SECONDS=600       # max inference wait time
   COMFYUI_ARGS=              # extra args for ComfyUI (e.g. --highvram)
   ```
5. Set **Exposed ports**: `8000` (handler) and optionally `8188` (ComfyUI direct)
6. **Instance type**: Serverless
7. Save and deploy

> **Persistent volume (recommended):** Mount a Vast network volume at `/workspace`. Models will be cached there and reused across cold starts — prevents 10–20 min re-download on every start.

---

## API usage

### Health check
```bash
curl http://<worker-ip>:8000/health
```

### Run inference
```bash
curl -X POST http://<worker-ip>:8000/generate/sync \
  -H "Content-Type: application/json" \
  -d @examples/payload.json
```

**Request body:**
```json
{
  "workflow_json": { ... },
  "loras": [
    {
      "name": "my_lora.safetensors",
      "url": "https://...",
      "checksum": "optional_sha256"
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "job_id": "a1b2c3d4",
  "elapsed_seconds": 42.1,
  "outputs": [
    {
      "type": "gifs",
      "filename": "output_00001.mp4",
      "data": "<base64-encoded video>",
      "format": "video/mp4",
      "size_bytes": 1234567
    }
  ]
}
```

To decode the video:
```python
import base64, json
resp = json.load(open("response.json"))
video_bytes = base64.b64decode(resp["outputs"][0]["data"])
open("output.mp4", "wb").write(video_bytes)
```

### Download a specific LoRA at runtime

Include it in `loras[]` in the request. `lora_helper.py` will:
1. Check if already in `models/loras/`
2. Download and cache if missing
3. Optionally verify SHA-256

```json
{
  "workflow_json": { ... },
  "loras": [
    {
      "name": "new_lora_v2.safetensors",
      "url": "https://civitai.com/api/download/models/99999",
      "checksum": "abc123..."
    }
  ]
}
```

---

## Adding a new LoRA (without rebuilding)

You do **not** need to rebuild the image to add a new LoRA. Just:

1. Add the URL to your request payload `loras[]` array
2. On first request, the LoRA is downloaded and cached
3. Subsequent requests use the cached file

For LoRAs that are always needed (like `SmoothMix`), add them to `provisioning.sh` — they'll be downloaded at container start.

---

## Updating custom nodes (requires rebuild)

1. Edit `scripts/install_custom_nodes.sh`
2. Update the commit SHA for the relevant node
3. Push to main → GitHub Actions rebuilds the image

---

## Running smoke tests

Inside the container (or in CI with `SKIP_PROVISIONING=1`):

```bash
bash /opt/tests/smoke_test.sh
```

---

## Project structure

```
├── Dockerfile                       # Image definition (build-only)
├── provisioning.sh                  # Model downloads (run at container start)
├── handler.py                       # FastAPI serverless handler
├── extra_model_paths.yaml           # ComfyUI model path config
├── .dockerignore
├── .env.example                     # All env variables documented
├── .github/
│   └── workflows/
│       └── build-and-push.yml       # GitHub Actions CI/CD
├── scripts/
│   ├── install_custom_nodes.sh      # Custom nodes install (build time)
│   ├── lora_helper.py               # Runtime LoRA downloader
│   ├── start.sh                     # Container entrypoint
│   └── build-info.sh                # Stamps image with build metadata
├── examples/
│   └── payload.json                 # Example API request
├── tests/
│   └── smoke_test.sh                # Structural smoke test
└── workflows/
    └── wan-audio.json               # Source workflow (reference)
```

---

## TODO / Open questions

- [ ] **`umt5_xxl_fp8_e4m3fn_scaled.safetensors` URL** — used URL from `Comfy-Org/Wan_2.1_ComfyUI_repackaged`. Verify on first run that the file is found by CLIPLoader. If 404, check the repo's `split_files/text_encoders/` directory.
- [ ] **ComfyUI-NAG commit** — not pinned. After first successful build, capture SHA from `/build-info.json` and pin in `install_custom_nodes.sh`.
- [ ] **ComfyUI-mxToolkit commit** — not pinned. Same as above.
- [ ] **ComfyUI version pin** — currently using `--depth 1` (latest HEAD). For full reproducibility, pin to a specific commit matching the workflow's `comfy-core` version (e.g. `0.3.47`).

---

## Known issues and risks

### Cold start time
First start downloads ~25 GB of models (SmoothMix × 2 + VAE + encoders + MMAudio + lightx2v). Without a persistent volume this takes **10–15 minutes**. **Mitigation:** Mount a Vast network volume at `/workspace` — models persist between cold starts.

### CivitAI auth
`CIVITAI_API_TOKEN` is required for the two SmoothMix diffusion models. Without it, provisioning warns but continues — those two models will be missing and the workflow will fail. Set the token in Vast.ai template env vars.

### ComfyUI-NAG and ComfyUI-mxToolkit not pinned
These two nodes are cloned at latest HEAD. If a breaking change is pushed to those repos, the image build may fail or the nodes may malfunction. Pin commits in `install_custom_nodes.sh` after the first working build.

### Video output size
Wan 2.2 I2V outputs can be 100–500 MB per video. These are returned base64-encoded in the JSON response. For production, consider adding S3/R2 upload support in `handler.py`.

### UNETLoader vs UnetLoaderGGUF
The updated workflow uses `UNETLoader` (safetensors, core) for both HIGH and LOW models. The GGUF loader nodes (`UnetLoaderGGUF`) are present in the workflow but disconnected — not used. `extra_model_paths.yaml` still configures `models/unet/` for the GGUF loader in case those nodes are re-enabled.

---

## GitHub Secrets reference

| Secret | Required for | Description |
|---|---|---|
| `DOCKER_USERNAME` | Docker Hub | Your Docker Hub login |
| `DOCKER_PASSWORD` | Docker Hub | Docker Hub access token |

For GHCR: no additional secrets needed — `GITHUB_TOKEN` is used automatically.

## GitHub Variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `REGISTRY` | Yes | `docker.io` | `docker.io` or `ghcr.io` |
| `IMAGE_NAME` | Yes | _(none)_ | `user/repo` for the image |

## Vast.ai template env vars reference

| Variable | Required | Description |
|---|---|---|
| `CIVITAI_API_TOKEN` | **Yes** | CivitAI API key for SmoothMix model downloads |
| `SKIP_PROVISIONING` | No | Set `1` to skip model downloads (dev/debug) |
| `MAX_WAIT_SECONDS` | No | Max inference wait (default: 600) |
| `COMFYUI_ARGS` | No | Extra ComfyUI flags (e.g. `--highvram --fp8_e4m3fn`) |
