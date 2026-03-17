#!/usr/bin/env python3
"""
handler.py — Vast.ai Serverless HTTP handler for ComfyUI
=========================================================

Exposes:
  GET  /health          — liveness check (returns ComfyUI status)
  GET  /build-info      — image metadata
  POST /generate/sync   — synchronous inference (blocks until done)

Request body for /generate/sync:
  {
    "workflow_json": { ... },     // ComfyUI prompt — API format OR UI format (auto-detected)
    "input_image":  "<base64>",   // optional: PNG/JPEG to inject into all LoadImage nodes
    "loras": [                    // optional runtime LoRAs to download
      {
        "name": "my_lora.safetensors",
        "url":  "https://...",
        "checksum": "sha256hex"   // optional
      }
    ]
  }

Workflow format notes:
  - API format:  {"1": {"class_type": "...", "inputs": {...}}, ...}
  - UI format:   {"nodes": [...], "links": [...], ...}
  Both are accepted. UI format is automatically converted to API format.

Response:
  {
    "status": "success",
    "outputs": [
      {
        "type": "gifs" | "videos" | "images",
        "filename": "output.mp4",
        "data": "<base64>",
        "format": "video/mp4"
      }
    ]
  }

Error:
  {
    "status": "error",
    "error": "message"
  }
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("handler")

# ---------------------------------------------------------------------------
# Add scripts/ to path for lora_helper import
# ---------------------------------------------------------------------------
sys.path.insert(0, "/opt/scripts")
from lora_helper import ensure_lora  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_URL  = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_WS   = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}"

OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR", "/opt/ComfyUI/output"))
MAX_WAIT        = int(os.getenv("MAX_WAIT_SECONDS", "600"))
HANDLER_PORT    = int(os.getenv("HANDLER_PORT", "8000"))
BUILD_INFO_PATH = Path("/opt/build-info.json")

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LoraSpec(BaseModel):
    name: str
    url: str
    checksum: Optional[str] = None


class GenerateRequest(BaseModel):
    workflow_json: dict
    input_image: Optional[str] = None   # base64-encoded PNG or JPEG
    loras: Optional[list[LoraSpec]] = []


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ComfyUI Serverless Handler", version="1.1.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{COMFYUI_URL}/system_stats", timeout=5.0)
        stats = resp.json()
        return {
            "status": "ok",
            "comfyui": "running",
            "system": {
                "vram_total": stats.get("devices", [{}])[0].get("vram_total", "?"),
                "vram_free":  stats.get("devices", [{}])[0].get("vram_free", "?"),
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "comfyui": "unavailable", "detail": str(e)},
        )


# ---------------------------------------------------------------------------
# Build info
# ---------------------------------------------------------------------------
@app.get("/build-info")
async def get_build_info():
    if BUILD_INFO_PATH.exists():
        return json.loads(BUILD_INFO_PATH.read_text())
    return {"status": "no build info"}


# ---------------------------------------------------------------------------
# Main inference endpoint
# ---------------------------------------------------------------------------
@app.post("/generate/sync")
async def generate_sync(request: GenerateRequest):
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] New job received")
    t_start = time.time()

    async with httpx.AsyncClient() as client:

        # 1. Ensure LoRAs are locally available
        if request.loras:
            logger.info(f"[{job_id}] Ensuring {len(request.loras)} LoRA(s)...")
            for lora in request.loras:
                try:
                    ensure_lora(lora.name, lora.url, lora.checksum)
                except RuntimeError as e:
                    raise HTTPException(status_code=400, detail=f"LoRA prep failed: {e}")

        # 2. Detect and convert workflow format (UI → API if needed)
        try:
            api_workflow = await _prepare_workflow(request.workflow_json, client, job_id)
        except Exception as e:
            logger.error(f"[{job_id}] Workflow conversion failed: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Workflow conversion failed: {e}")

        # 3. Upload input_image and inject into LoadImage nodes
        if request.input_image:
            try:
                img_filename = await _upload_image_b64(request.input_image, client, job_id)
                injected = 0
                for node_id, node in api_workflow.items():
                    if node.get("class_type") == "LoadImage":
                        node["inputs"]["image"] = img_filename
                        injected += 1
                logger.info(f"[{job_id}] Injected image '{img_filename}' into {injected} LoadImage node(s)")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Image upload failed: {e}")

        # 4. Submit prompt to ComfyUI
        client_id = str(uuid.uuid4())
        prompt_payload = {
            "prompt": api_workflow,
            "client_id": client_id,
        }

        try:
            resp = await client.post(
                f"{COMFYUI_URL}/prompt",
                json=prompt_payload,
                timeout=30.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:500]
            logger.error(f"[{job_id}] Prompt submit failed: {detail}")
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected prompt: {detail}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"ComfyUI unreachable: {e}")

    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=502, detail="ComfyUI returned no prompt_id")

    logger.info(f"[{job_id}] Submitted — prompt_id={prompt_id}, client_id={client_id}")

    # 5. Wait for completion via WebSocket
    try:
        outputs = await _wait_for_completion(job_id, client_id, prompt_id)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Timeout after {MAX_WAIT}s")

    t_elapsed = time.time() - t_start
    logger.info(f"[{job_id}] Done in {t_elapsed:.1f}s — {len(outputs)} output(s)")

    return {
        "status": "success",
        "job_id": job_id,
        "elapsed_seconds": round(t_elapsed, 1),
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Workflow format detection and conversion
# ---------------------------------------------------------------------------

async def _prepare_workflow(workflow: dict, client: httpx.AsyncClient, job_id: str) -> dict:
    """
    Detect whether workflow is in UI format (nodes list) or API format (node dict).
    Convert from UI format to API format if needed.
    """
    if "nodes" in workflow and isinstance(workflow.get("nodes"), list):
        logger.info(f"[{job_id}] UI-format workflow detected — converting to API format")
        api = await _ui_to_api(workflow, client, job_id)
        logger.info(f"[{job_id}] Converted {len(api)} nodes to API format")
        return api
    return workflow


async def _ui_to_api(ui_workflow: dict, client: httpx.AsyncClient, job_id: str) -> dict:
    """
    Convert a ComfyUI UI-format workflow (exported from the web interface) to
    the API-format prompt dict that /prompt expects.

    UI format:  {"nodes": [{id, type, widgets_values, inputs:[{name, link}]}, ...], "links": [...]}
    API format: {"<node_id>": {"class_type": "...", "inputs": {"input_name": value_or_link}}}

    Links in API format are [source_node_id, source_output_slot] lists.
    """
    nodes = ui_workflow.get("nodes", [])
    links = ui_workflow.get("links", [])

    # link_id -> [source_node_id, source_output_slot]
    link_map: dict[int, list] = {
        link[0]: [link[1], link[2]] for link in links
    }

    # Fetch object_info to resolve widget input names
    all_info: dict = {}
    try:
        resp = await client.get(f"{COMFYUI_URL}/object_info", timeout=30.0)
        all_info = resp.json()
        logger.debug(f"[{job_id}] object_info fetched for {len(all_info)} node types")
    except Exception as e:
        logger.warning(f"[{job_id}] Could not fetch object_info: {e} — widget mapping may be incomplete")

    def get_widget_names(node_type: str) -> list[str]:
        """
        Return ordered list of widget (non-linkable) input names for a node type.
        Widget inputs are STRING, INT, FLOAT, BOOLEAN, COMBO (list), IMAGEUPLOAD.
        Linkable inputs (IMAGE, LATENT, MODEL, etc.) are excluded.
        """
        info = all_info.get(node_type)
        if not info:
            return []
        required = info.get("input", {}).get("required", {}) or {}
        optional = info.get("input", {}).get("optional", {}) or {}
        names = []
        for name, inp_def in {**required, **optional}.items():
            if not isinstance(inp_def, (list, tuple)) or not inp_def:
                continue
            inp_type = inp_def[0]
            # COMBO (dropdown) is a list; primitives are named types
            if isinstance(inp_type, list) or inp_type in (
                "STRING", "INT", "FLOAT", "BOOLEAN", "IMAGEUPLOAD"
            ):
                names.append(name)
        return names

    api_prompt: dict[str, dict] = {}

    for node in nodes:
        node_id   = str(node.get("id", ""))
        node_type = node.get("type", "")

        # Skip non-executable nodes
        if not node_type or node_type in ("Note", "Reroute"):
            continue

        widgets_values = node.get("widgets_values") or []
        node_inputs    = node.get("inputs") or []

        inputs: dict = {}

        # Resolve linked inputs
        for inp in node_inputs:
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                inputs[inp["name"]] = link_map[link_id]

        # Map widget values → input names (in schema order, skipping linked inputs)
        widget_names = get_widget_names(node_type)
        widget_idx = 0
        for wname in widget_names:
            if wname not in inputs:
                if widget_idx < len(widgets_values):
                    inputs[wname] = widgets_values[widget_idx]
                widget_idx += 1

        api_prompt[node_id] = {
            "class_type": node_type,
            "inputs": inputs,
        }

    return api_prompt


# ---------------------------------------------------------------------------
# Image upload helper
# ---------------------------------------------------------------------------

async def _upload_image_b64(b64_data: str, client: httpx.AsyncClient, job_id: str) -> str:
    """
    Decode a base64 image and upload it to ComfyUI's /upload/image endpoint.
    Returns the filename assigned by ComfyUI.
    """
    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")

    # Detect image format from magic bytes
    if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        ext, mime = "png", "image/png"
    elif img_bytes[:3] == b'\xff\xd8\xff':
        ext, mime = "jpg", "image/jpeg"
    elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        ext, mime = "webp", "image/webp"
    else:
        ext, mime = "png", "image/png"

    filename = f"input_{uuid.uuid4().hex[:12]}.{ext}"
    logger.info(f"[{job_id}] Uploading input image as '{filename}' ({len(img_bytes)//1024}KB)")

    resp = await client.post(
        f"{COMFYUI_URL}/upload/image",
        files={"image": (filename, img_bytes, mime)},
        data={"overwrite": "true"},
        timeout=30.0,
    )
    resp.raise_for_status()
    result = resp.json()
    assigned_name = result.get("name", filename)
    logger.info(f"[{job_id}] Image uploaded → '{assigned_name}'")
    return assigned_name


# ---------------------------------------------------------------------------
# WebSocket wait + output collection
# ---------------------------------------------------------------------------
async def _wait_for_completion(
    job_id: str,
    client_id: str,
    prompt_id: str,
) -> list[dict]:
    """
    Connect to ComfyUI websocket and wait for the prompt to finish.
    Returns list of output file dicts with base64-encoded data.
    """
    uri = f"{COMFYUI_WS}/ws?clientId={client_id}"

    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=30) as ws:
            deadline = time.time() + MAX_WAIT

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError()

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(30.0, remaining))
                except asyncio.TimeoutError:
                    if time.time() >= deadline:
                        raise asyncio.TimeoutError()
                    continue

                # Binary frames = preview images — skip
                if isinstance(raw, bytes):
                    continue

                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "executing":
                    node = data["data"].get("node")
                    if node is None and data["data"].get("prompt_id") == prompt_id:
                        logger.info(f"[{job_id}] Execution complete signal received")
                        break

                elif msg_type == "execution_error":
                    err = data.get("data", {})
                    raise HTTPException(
                        status_code=500,
                        detail=f"ComfyUI execution error: {err}",
                    )

                elif msg_type == "status":
                    queue_remaining = (
                        data.get("data", {})
                        .get("status", {})
                        .get("exec_info", {})
                        .get("queue_remaining", "?")
                    )
                    logger.debug(f"[{job_id}] Queue remaining: {queue_remaining}")

    except websockets.exceptions.ConnectionClosed as e:
        raise HTTPException(status_code=502, detail=f"WebSocket closed unexpectedly: {e}")

    return await _collect_outputs(job_id, prompt_id)


async def _collect_outputs(job_id: str, prompt_id: str) -> list[dict]:
    """Fetch job history from ComfyUI and return encoded output files."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10.0)
        history = resp.json()

    outputs = []

    if prompt_id not in history:
        logger.warning(f"[{job_id}] No history entry for prompt_id={prompt_id}")
        return outputs

    job_outputs = history[prompt_id].get("outputs", {})

    for node_id, node_out in job_outputs.items():
        for output_type, items in node_out.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "filename" not in item:
                    continue

                subfolder = item.get("subfolder", "")
                filename  = item["filename"]
                file_path = OUTPUT_DIR / subfolder / filename if subfolder else OUTPUT_DIR / filename

                if not file_path.exists():
                    logger.warning(f"[{job_id}] Output file not found: {file_path}")
                    continue

                file_size = file_path.stat().st_size
                logger.info(f"[{job_id}] Encoding output: {filename} ({file_size / 1024**2:.1f} MB)")

                with open(file_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                outputs.append({
                    "type": output_type,
                    "filename": filename,
                    "data": encoded,
                    "format": item.get("format", _guess_mime(filename)),
                    "size_bytes": file_size,
                })

    return outputs


def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".mp4":  "video/mp4",
        ".webm": "video/webm",
        ".gif":  "image/gif",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".wav":  "audio/wav",
        ".mp3":  "audio/mpeg",
    }.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Handler starting on port {HANDLER_PORT}")
    logger.info(f"ComfyUI target: {COMFYUI_URL}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=HANDLER_PORT,
        log_level="info",
        access_log=True,
    )
