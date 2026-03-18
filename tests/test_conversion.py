#!/usr/bin/env python3
"""
Offline unit test for _ui_to_api workflow conversion.

Runs the full conversion logic with a mocked object_info (built from real
source code) and validates every key node in wan-audio.json produces the
correct API-format inputs.

Usage:
    python tests/test_conversion.py
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Build mock object_info from real node schemas
# This mirrors exactly what a live ComfyUI /object_info would return.
# ---------------------------------------------------------------------------

# KSampler SAMPLERS / SCHEDULERS (abbreviated — full lists not needed for tests)
SAMPLERS = ["euler", "euler_cfg_pp", "euler_ancestral", "dpm_2", "dpm_fast", "dpmpp_2m"]
SCHEDULERS = ["simple", "normal", "karras", "sgm_uniform", "simple_hyper"]

MOCK_OBJECT_INFO = {
    # --- ComfyUI core nodes ---
    "LoadImage": {
        "input": {"required": {"image": ("IMAGE_UPLOAD",), "upload": (["image"],)}}
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP",)}}
    },
    "CLIPLoader": {
        "input": {"required": {
            "clip_name": (["clip1.safetensors"],),
            "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
                      "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace"],),
            "device": (["default", "cpu"],),
        }}
    },
    "CLIPVisionLoader": {
        "input": {"required": {"clip_name": (["model.safetensors"],)}}
    },
    "CLIPVisionEncode": {
        "input": {"required": {"clip_vision": ("CLIP_VISION",), "image": ("IMAGE",),
                               "crop": (["center", "none"],)}}
    },
    "UNETLoader": {
        "input": {"required": {
            "unet_name": (["model.safetensors"],),
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
        }}
    },
    "VAELoader": {
        "input": {"required": {"vae_name": (["vae.safetensors"],)}}
    },
    "VAEDecode": {
        "input": {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}
    },
    "ImageScaleBy": {
        "input": {"required": {
            "image": ("IMAGE",),
            "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
        }}
    },
    "PreviewImage": {
        "input": {"required": {"images": ("IMAGE",)}}
    },
    "SaveImage": {
        "input": {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "ComfyUI"})}}
    },
    "KSamplerAdvanced": {
        "input": {"required": {
            "model": ("MODEL",),
            "add_noise": (["enable", "disable"],),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                   "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "sampler_name": (SAMPLERS,),
            "scheduler": (SCHEDULERS,),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "return_with_leftover_noise": (["disable", "enable"],),
        }}
    },
    "ModelSamplingSD3": {
        "input": {"required": {
            "model": ("MODEL",),
            "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
        }, "optional": {"sigmas": ("SIGMAS",)}}
    },
    "PreviewAny": {
        "input": {"required": {"source": ("*",)}}
    },
    "StringConcatenate": {
        "input": {"required": {
            "string1": ("STRING", {"default": "", "forceInput": True}),
            "string2": ("STRING", {"default": "", "forceInput": True}),
            "delimiter": ("STRING", {"default": " "}),
        }}
    },

    # --- ComfyUI-KJNodes ---
    "ImageResizeKJv2": {
        "input": {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 16384, "step": 1}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],),
                "keep_proportion": (["stretch", "resize", "pad", "pad_edge", "crop"],),
                "pad_color": ("STRING", {"default": "0, 0, 0"}),
                "crop_position": (["center", "top", "bottom", "left", "right"],),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {"mask": ("MASK",), "device": (["cpu", "gpu"],)},
        }
    },
    "ColorMatch": {
        "input": {"required": {
            "image_ref": ("IMAGE",),
            "image_target": ("IMAGE",),
            "method": (["mkl", "hm", "reinhard", "mvgd", "hm-mvgd-hm", "rhs"],),
        }, "optional": {"strength": ("FLOAT", {"default": 1.0})}}
    },

    # --- ComfyUI-NAG ---
    "KSamplerWithNAG (Advanced)": {
        "input": {"required": {
            "model": ("MODEL",),
            "add_noise": (["enable", "disable"],),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                   "control_after_generate": True}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "nag_tau": ("FLOAT", {"default": 2.5, "min": 1.0, "max": 10.0, "step": 0.1}),
            "nag_alpha": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            "nag_sigma_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01}),
            "sampler_name": (SAMPLERS,),
            "scheduler": (SCHEDULERS,),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "nag_negative": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            "return_with_leftover_noise": (["disable", "enable"],),
        }}
    },

    # --- ComfyUI-WanVideoWrapper ---
    "WanImageToVideo": {
        "input": {"required": {
            "model": ("MODEL",),
            "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            "text_encodings_pos": ("CONDITIONING",),
            "text_encodings_neg": ("CONDITIONING",),
            "image": ("IMAGE",),
            "vae": ("VAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 16}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 257, "step": 4}),
            "generation_type": (["image_to_video"],),
            "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0}),
        }}
    },
    "NormalizeAudioLoudness": {
        "input": {"required": {
            "audio": ("AUDIO",),
            "lufs_target": ("FLOAT", {"default": -23.0, "min": -70.0, "max": 0.0, "step": 0.5}),
        }}
    },

    # --- ComfyUI-MMAudio ---
    "MMAudioModelLoader": {
        "input": {"required": {"mmaudio_model": (["mmaudio_small_16k.pth"],)}}
    },
    "MMAudioFeatureUtilsLoader": {
        "input": {"required": {"feature_utils_model": (["mmaudio_small_16k.pth"],)}}
    },
    "MMAudioSampler": {
        "input": {"required": {
            "mmaudio_model": ("MMAUDIO_MODEL",),
            "feature_utils": ("MMAUDIO_FEATURE_UTILS",),
            "video": ("IMAGE",),
            "duration": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            "steps": ("INT", {"default": 25, "min": 1, "max": 200}),
            "cfg_strength": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
        }, "optional": {
            "clip_video": ("IMAGE",),
            "clip_vision_model": ("CLIP_VISION",),
        }}
    },

    # --- ComfyUI-VideoHelperSuite ---
    # VHS_VideoCombine uses dict widgets_values — handled separately
    "VHS_VideoCombine": {
        "input": {"required": {
            "images": ("IMAGE",),
            "frame_rate": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 1000.0}),
            "loop_count": ("INT", {"default": 0, "min": 0, "max": 100}),
            "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
            "format": (["video/h264-mp4"],),
            "pingpong": ("BOOLEAN", {"default": False}),
            "save_output": ("BOOLEAN", {"default": True}),
        }, "optional": {"audio": ("AUDIO",), "meta_batch": ("VHS_BatchManager",)}}
    },

    # --- ComfyUI-GGUF ---
    "UnetLoaderGGUF": {
        "input": {"required": {
            "unet_name": (["model.gguf"],),
        }}
    },

    # --- ComfyUI-VFI ---
    "RIFEInterpolation": {
        "input": {"required": {
            "frames": ("IMAGE",),
            "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
            "ckpt_name": (["rife49.pth"],),
            "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
            "multiplier_str": ("STRING", {"default": "1,1,1,1"}),
        }, "optional": {"optional_interpolation_states": ("INTERPOLATION_STATES",)}}
    },

    # --- ComfyUI-Easy-Use ---
    "easy cleanGpuUsed": {
        "input": {"required": {"anything": ("*",), "unique_id": ("UNIQUE_ID",)},
                  "optional": {}}
    },

    # --- rgthree-comfy ---
    "Power Lora Loader (rgthree)": {
        "input": {"required": {"model": ("MODEL",), "clip": ("CLIP",)},
                  "optional": {}}
    },
    "Seed (rgthree)": {
        "input": {"required": {"seed": ("INT", {"default": 0, "min": -1,
                                               "max": 0xffffffffffffffff,
                                               "control_after_generate": True})}}
    },

    # --- ComfyUI_Comfyroll_CustomNodes ---
    "CR Float To Integer": {
        "input": {"required": {
            "float_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.99, "step": 0.01}),
            "rounding": (["round", "floor", "ceiling"],),
        }}
    },

    # --- comfyui-adaptiveprompts ---
    "PromptGenerator": {
        "input": {"required": {
            "context": ("DICT",),
            "text": ("STRING", {"default": "", "multiline": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                             "control_after_generate": True}),
        }}
    },

    # --- comfy_mtb ---
    "Pick From Batch (mtb)": {
        "input": {"required": {
            "images": ("IMAGE",),
            "index": ("INT", {"default": 0, "min": 0, "max": 99}),
        }}
    },

    # --- ComfyUI-mxToolkit ---
    "mxSlider2D": {
        "input": {"required": {
            "x_min": ("FLOAT", {"default": 0.0}),
            "x_max": ("FLOAT", {"default": 1.0}),
            "y_min": ("FLOAT", {"default": 0.0}),
            "y_max": ("FLOAT", {"default": 1.0}),
            "x": ("FLOAT", {"default": 0.5}),
            "y": ("FLOAT", {"default": 0.5}),
        }}
    },

    # --- ComfyUI built-in primitives ---
    "PrimitiveFloat": {
        "input": {"required": {"value": ("FLOAT", {"default": 0.0})}}
    },
    "ComfyMathExpression": {
        "input": {"required": {
            "expression": ("STRING", {"default": "a+b"}),
        }, "optional": {
            "a": ("FLOAT", {"default": 0.0}),
            "b": ("FLOAT", {"default": 0.0}),
            "c": ("FLOAT", {"default": 0.0}),
        }}
    },
}


# ---------------------------------------------------------------------------
# Import the conversion function from handler.py
# ---------------------------------------------------------------------------
# We need to patch the httpx call inside _ui_to_api so it returns our mock.
# The cleanest way: monkeypatch AsyncClient.get to return mock data.

# Stub out lora_helper before importing handler
import sys as _sys, types as _types
_lh = _types.ModuleType("lora_helper")
_lh.ensure_lora = lambda *a, **kw: None
_sys.modules["lora_helper"] = _lh
# Also add scripts/ to path for the real import path handler uses at runtime
_sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import handler  # noqa: E402 (after sys.path manipulation above)


async def convert_with_mock(workflow: dict) -> dict:
    """Run _ui_to_api with MOCK_OBJECT_INFO instead of a live ComfyUI."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = MOCK_OBJECT_INFO

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    return await handler._ui_to_api(workflow, mock_client, "test")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0

def check(label: str, actual, expected):
    global PASS, FAIL
    if actual == expected:
        print(f"  ✓ {label}")
        PASS += 1
    else:
        print(f"  ✗ {label}: got {actual!r}, expected {expected!r}")
        FAIL += 1


def check_link(label: str, actual, src_node: str, src_slot: int):
    global PASS, FAIL
    if isinstance(actual, list) and len(actual) == 2 and actual[0] == src_node and actual[1] == src_slot:
        print(f"  ✓ {label}")
        PASS += 1
    else:
        print(f"  ✗ {label}: got {actual!r}, expected ['{src_node}', {src_slot}]")
        FAIL += 1


async def run_tests():
    with open("workflows/wan-audio.json") as f:
        wf = json.load(f)

    print("Converting wan-audio.json workflow...")
    api = await convert_with_mock(wf)
    print(f"Conversion done: {len(api)} nodes in API prompt\n")

    # -----------------------------------------------------------------------
    # 1. Presence check — key nodes must exist
    # -----------------------------------------------------------------------
    print("=== 1. Node presence ===")
    required_types = {
        "KSamplerAdvanced", "KSamplerWithNAG (Advanced)", "WanImageToVideo",
        "ImageResizeKJv2", "UnetLoaderGGUF", "VHS_VideoCombine",
        "MMAudioSampler", "MMAudioModelLoader", "NormalizeAudioLoudness",
        "CLIPTextEncode", "CLIPLoader", "VAEDecode",
        "PromptGenerator",  # inside subgraph
    }
    present_types = {v["class_type"] for v in api.values()}

    for t in sorted(required_types):
        global PASS, FAIL
        if t in present_types:
            print(f"  ✓ {t}")
            PASS += 1
        else:
            print(f"  ✗ {t} MISSING from API prompt")
            FAIL += 1

    # UI-only nodes must NOT appear
    ui_only = {"Label (rgthree)", "Note Plus (mtb)", "Fast Groups Bypasser (rgthree)"}
    for t in ui_only:
        if t not in present_types:
            print(f"  ✓ {t} correctly excluded")
            PASS += 1
        else:
            print(f"  ✗ {t} should have been excluded")
            FAIL += 1

    # -----------------------------------------------------------------------
    # 2. Node 182 — ImageResizeKJv2: linked width/height + widget values
    # -----------------------------------------------------------------------
    print("\n=== 2. Node 182 (ImageResizeKJv2) ===")
    node_182 = api.get("182")
    if not node_182:
        print("  ✗ node 182 missing")
        FAIL += 1
    else:
        inp = node_182["inputs"]
        check_link("width linked from node 208 slot 0", inp.get("width"), "208", 0)
        check_link("height linked from node 208 slot 1", inp.get("height"), "208", 1)
        check("upscale_method = lanczos", inp.get("upscale_method"), "lanczos")
        check("keep_proportion = resize", inp.get("keep_proportion"), "resize")
        check("pad_color = '0, 0, 0'", inp.get("pad_color"), "0, 0, 0")
        check("crop_position = center", inp.get("crop_position"), "center")
        check("divisible_by = 16", inp.get("divisible_by"), 16)
        check("device = cpu", inp.get("device"), "cpu")

    # -----------------------------------------------------------------------
    # 3. Node 206 — KSamplerAdvanced (inner, subgraph 8d040fd1)
    #    noise_seed has widget field AND is in widget_names — control_mode slot
    # -----------------------------------------------------------------------
    print("\n=== 3. Node 206 (KSamplerAdvanced, inner) ===")
    node_206 = api.get("206")
    if not node_206:
        print("  ✗ node 206 missing from API prompt")
        FAIL += 1
    else:
        inp = node_206["inputs"]
        # Linked inputs (from outer boundary -10 slots resolved via link_map)
        check("noise_seed linked", isinstance(inp.get("noise_seed"), list), True)
        check("add_noise = enable", inp.get("add_noise"), "enable")
        # control_after_generate slot should be SKIPPED
        check("steps = 6", inp.get("steps"), 6)
        check("cfg = 1", inp.get("cfg"), 1)
        check("sampler_name = euler", inp.get("sampler_name"), "euler")
        check("scheduler = simple", inp.get("scheduler"), "simple")
        check("start_at_step = 0", inp.get("start_at_step"), 0)
        check("end_at_step = 3", inp.get("end_at_step"), 3)
        check("return_with_leftover_noise = enable", inp.get("return_with_leftover_noise"), "enable")

    # -----------------------------------------------------------------------
    # 4. Node 231 — KSamplerWithNAG (inner, subgraph c0ea7e28)
    # -----------------------------------------------------------------------
    print("\n=== 4. Node 231 (KSamplerWithNAG Advanced, inner) ===")
    node_231 = api.get("231")
    if not node_231:
        print("  ✗ node 231 missing")
        FAIL += 1
    else:
        inp = node_231["inputs"]
        # wv = ['disable', 0, 'fixed', 6, 1, 30, 2.5, 0.25, 1, 'euler', 'simple', 3, 10000, 'disable']
        check("add_noise = disable", inp.get("add_noise"), "disable")
        check("noise_seed = 0", inp.get("noise_seed"), 0)
        # 'fixed' at [2] = control_after_generate placeholder → skipped
        check("steps = 6", inp.get("steps"), 6)
        check("cfg = 1", inp.get("cfg"), 1)
        check("nag_scale = 30", inp.get("nag_scale"), 30)
        check("nag_tau = 2.5", inp.get("nag_tau"), 2.5)
        check("nag_alpha = 0.25", inp.get("nag_alpha"), 0.25)
        check("nag_sigma_end = 1", inp.get("nag_sigma_end"), 1)
        check("sampler_name = euler", inp.get("sampler_name"), "euler")
        check("scheduler = simple", inp.get("scheduler"), "simple")
        check("start_at_step = 3", inp.get("start_at_step"), 3)
        check("end_at_step = 10000", inp.get("end_at_step"), 10000)
        check("return_with_leftover_noise = disable", inp.get("return_with_leftover_noise"), "disable")

    # -----------------------------------------------------------------------
    # 5. Node 239 — PreviewAny (inner, subgraph 493dddb9)
    #    source must be linked to inner node 240 via inner-to-inner link 373
    # -----------------------------------------------------------------------
    print("\n=== 5. Node 239 (PreviewAny, inner) ===")
    node_239 = api.get("239")
    if not node_239:
        print("  ✗ node 239 missing")
        FAIL += 1
    else:
        inp = node_239["inputs"]
        check_link("source linked to node 240 slot 0", inp.get("source"), "240", 0)

    # -----------------------------------------------------------------------
    # 6. Node 240 — PromptGenerator (inner, subgraph 493dddb9)
    #    seed has control_after_generate → slot [1]=seed, [2]='randomize' skipped
    # -----------------------------------------------------------------------
    print("\n=== 6. Node 240 (PromptGenerator, inner) ===")
    node_240 = api.get("240")
    if not node_240:
        print("  ✗ node 240 missing")
        FAIL += 1
    else:
        inp = node_240["inputs"]
        # wv = ['moves sexily, laughs sexily', 716195816920133, 'randomize']
        check("text = 'moves sexily...'", inp.get("text"), "moves sexily, laughs sexily")
        check("seed = 716195816920133", inp.get("seed"), 716195816920133)
        # 'randomize' at [2] is control_after_generate → not stored

    # -----------------------------------------------------------------------
    # 7. VHS_VideoCombine — dict widgets_values
    # -----------------------------------------------------------------------
    print("\n=== 7. VHS_VideoCombine (dict widgets_values) ===")
    vhs_nodes = [v for v in api.values() if v["class_type"] == "VHS_VideoCombine"]
    if not vhs_nodes:
        print("  ✗ VHS_VideoCombine missing")
        FAIL += 1
    else:
        inp = vhs_nodes[0]["inputs"]
        check("frame_rate is set", "frame_rate" in inp, True)
        check("filename_prefix is set", "filename_prefix" in inp, True)
        check("format is set", "format" in inp, True)

    # -----------------------------------------------------------------------
    # 8. Seed (rgthree) — seed with control_after_generate
    # -----------------------------------------------------------------------
    print("\n=== 8. Node 210 (Seed rgthree) ===")
    node_210 = api.get("210")
    if not node_210:
        print("  ✗ node 210 missing")
        FAIL += 1
    else:
        inp = node_210["inputs"]
        # wv = [-1, '', '', '']  — wait, that's 4 values for 1 INT + 1 control slot = 2 slots
        # Actually Seed (rgthree) might have extra internal widgets
        # Just check seed is present and is -1
        check("seed = -1", inp.get("seed"), -1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    return FAIL


if __name__ == "__main__":
    fail_count = asyncio.run(run_tests())
    sys.exit(1 if fail_count else 0)
