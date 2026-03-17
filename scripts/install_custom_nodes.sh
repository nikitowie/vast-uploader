#!/usr/bin/env bash
# =============================================================================
# install_custom_nodes.sh — Clones and installs all custom nodes into ComfyUI
#
# Run during Docker BUILD phase (not runtime).
# Each node is pinned to a specific commit for reproducibility.
# If a node has requirements.txt, it's installed via pip.
# =============================================================================

set -euo pipefail

COMFYUI_CUSTOM_NODES="/opt/ComfyUI/custom_nodes"
mkdir -p "$COMFYUI_CUSTOM_NODES"
cd "$COMFYUI_CUSTOM_NODES"

log()  { echo "[custom_nodes] $*"; }
die()  { echo "[custom_nodes][ERROR] $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# clone_node <repo_url> <dir_name> [commit_sha]
# Clones a repo and optionally pins to a specific commit.
# If commit_sha is omitted, uses HEAD of default branch.
# ---------------------------------------------------------------------------
clone_node() {
    local url="$1"
    local name="$2"
    local commit="${3:-}"

    if [ -d "$name" ]; then
        log "✓ Already present: $name"
        return 0
    fi

    log "Cloning: $name"
    git clone --depth 1 "$url" "$name" 2>&1 | tail -1

    if [ -n "$commit" ]; then
        log "  Pinning to $commit"
        cd "$name"
        git fetch --depth 1 origin "$commit" 2>/dev/null || true
        git checkout "$commit" 2>/dev/null || true
        cd ..
    fi

    # Install Python requirements if present
    if [ -f "$name/requirements.txt" ]; then
        log "  Installing requirements for $name"
        pip install --no-cache-dir -r "$name/requirements.txt" || \
            log "  WARNING: some requirements failed for $name"
    fi

    # Run install.py if present (some nodes use this)
    if [ -f "$name/install.py" ]; then
        log "  Running install.py for $name"
        python "$name/install.py" || log "  WARNING: install.py failed for $name"
    fi

    log "✓ Installed: $name"
}

# =============================================================================
# CUSTOM NODES — cnr_id mapping to GitHub repos
# Commits pinned to versions seen in workflow JSON (best effort).
# =============================================================================

# 1. comfy-mtb (melMass)
#    Nodes: Pick From Batch (mtb), Note Plus (mtb)
#    cnr_id: comfy-mtb | commit: d87e52e
clone_node \
    "https://github.com/melMass/comfy_mtb" \
    "comfy_mtb" \
    "d87e52ea2c112fd95f257dcd6a54a5db77a34fc3"

# 2. ComfyUI-VideoHelperSuite (Kosinkadink)
#    Nodes: VHS_VideoCombine
#    cnr_id: comfyui-videohelpersuite | commit: 4c7858d (from workflow)
clone_node \
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" \
    "ComfyUI-VideoHelperSuite" \
    "4c7858ddd5126f7293dc3c9f6e0fc4c263cde079"

# 3. ComfyUI-WanVideoWrapper (kijai)
#    Nodes: NormalizeAudioLoudness (+ full Wan video pipeline)
#    cnr_id: ComfyUI-WanVideoWrapper | aux_id: kijai/ComfyUI-WanVideoWrapper
#    commit: 90c3bbb
clone_node \
    "https://github.com/kijai/ComfyUI-WanVideoWrapper" \
    "ComfyUI-WanVideoWrapper" \
    "90c3bbb6c2e4ff5e05305e765d007d5e58428ce4"

# 4. ComfyUI-MMAudio (kijai)
#    Nodes: MMAudioModelLoader, MMAudioFeatureUtilsLoader
#    cnr_id: comfyui-mmaudio | commit: 841b307
#    NOTE: if this repo moves or changes name, update URL here
clone_node \
    "https://github.com/kijai/ComfyUI-MMAudio" \
    "ComfyUI-MMAudio" \
    "841b307095f3cdc448d940a5787bc36706e49f1d"

# 5. ComfyUI-KJNodes (kijai)
#    Nodes: ColorMatch, ImageResizeKJv2
#    cnr_id: comfyui-kjnodes | commit: 07b804c
clone_node \
    "https://github.com/kijai/ComfyUI-KJNodes" \
    "ComfyUI-KJNodes" \
    "07b804cb3ff3b3eb8d2c5fdadd62d0822bebe4e8"

# 6. ComfyUI-Easy-Use (yolain)
#    Nodes: easy cleanGpuUsed
#    cnr_id: comfyui-easy-use | commit: de92038
clone_node \
    "https://github.com/yolain/ComfyUI-Easy-Use" \
    "ComfyUI-Easy-Use" \
    "de92038f88317699f314be85e5c7af84f1ab9c3a"

# 7. rgthree-comfy (rgthree)
#    Nodes: Power Lora Loader, Label, Seed, Fast Groups Bypasser
#    cnr_id: rgthree-comfy | commit: 8ff50e4 (from workflow widgets_values)
clone_node \
    "https://github.com/rgthree/rgthree-comfy" \
    "rgthree-comfy" \
    "8ff50e4521881eca1fe26aec9615fc9362474931"

# 8. ComfyUI_Comfyroll_CustomNodes (Suzie1)
#    Nodes: CR Float To Integer
#    cnr_id: ComfyUI_Comfyroll_CustomNodes | commit: d78b780
clone_node \
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes" \
    "ComfyUI_Comfyroll_CustomNodes" \
    "d78b780ae43fcf8c6b7c6505e6ffb4584281ceca"

# 9. ComfyUI-GGUF (city96)
#    Nodes: UnetLoaderGGUF
#    cnr_id: comfyui-gguf | commit: b3ec875
clone_node \
    "https://github.com/city96/ComfyUI-GGUF" \
    "ComfyUI-GGUF" \
    "b3ec875a68d94b758914fd48d30571d953bb7a54"

# 10. ComfyUI-VFI (GACLove)
#     Nodes: RIFEInterpolation (flownet.pkl — RIFE frame interpolation)
#     aux_id: GACLove/ComfyUI-VFI | commit: 6176a43 (from workflow)
clone_node \
    "https://github.com/GACLove/ComfyUI-VFI" \
    "ComfyUI-VFI" \
    "6176a430f12cd16003f4664c1e3c6af8e96cc3c6"

# 11. ComfyUI-NAG (scottmudge)
#     Nodes: NAG Sampler (nag_scale parameter, used in sampling subgraph)
#     Listed in workflow Extension List; required for NAG-based sampling nodes
clone_node \
    "https://github.com/scottmudge/ComfyUI-NAG" \
    "ComfyUI-NAG"
    # No commit pinned — not present in workflow properties.
    # TODO: pin after first successful build by capturing SHA from build-info.json

# 12. comfyui-adaptiveprompts (Alectriciti)
#     Nodes: PromptGenerator (used inside "Positive" subgraph, node 240)
#     aux_id: Alectriciti/comfyui-adaptiveprompts | commit: aa0d896 (from workflow)
clone_node \
    "https://github.com/Alectriciti/comfyui-adaptiveprompts" \
    "comfyui-adaptiveprompts" \
    "aa0d896b5e5d9c7031c3aa995e028f2507dfc63c"

# 13. ComfyUI-mxToolkit (Smirnov75)
#     Nodes: mxSlider2D (VIDEO Width x Height control, node 208 in workflow)
#     Required — affects video dimensions
clone_node \
    "https://github.com/Smirnov75/ComfyUI-mxToolkit" \
    "ComfyUI-mxToolkit"
    # No commit pinned — not present in workflow properties.
    # TODO: pin after first successful build

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "[custom_nodes] Installation complete:"
for d in "$COMFYUI_CUSTOM_NODES"/*/; do
    name=$(basename "$d")
    echo "  ✓ $name"
done
echo ""
