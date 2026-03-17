#!/usr/bin/env python3
"""
lora_helper.py — Runtime LoRA download helper

Ensures LoRA files are available locally before inference runs.
Designed to be imported by handler.py.

Flow:
  1. Check if LoRA file exists in LORA_DIR
  2. Optionally verify SHA-256 checksum
  3. Download from URL if missing (with .tmp → rename pattern)
  4. Protect against concurrent double-downloads via file lock
  5. Log everything for cold-start debugging

Thread-safe: uses threading.Lock per filename.
"""

import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("lora_helper")

LORA_DIR = os.getenv("LORA_DIR", "/opt/ComfyUI/models/loras")

# Per-filename locks to prevent concurrent duplicate downloads
_locks: dict[str, threading.Lock] = {}
_locks_mu = threading.Lock()


def _get_lock(name: str) -> threading.Lock:
    with _locks_mu:
        if name not in _locks:
            _locks[name] = threading.Lock()
        return _locks[name]


def ensure_lora(
    name: str,
    url: str,
    checksum: Optional[str] = None,
    lora_dir: Optional[str] = None,
) -> str:
    """
    Ensure a LoRA file is present locally. Downloads if missing.

    Args:
        name:     Filename, e.g. "my_lora_v1.safetensors"
        url:      Direct download URL
        checksum: Optional SHA-256 hex digest for integrity check
        lora_dir: Override default LORA_DIR

    Returns:
        Absolute path to the local LoRA file.

    Raises:
        RuntimeError: If download fails or checksum mismatch.
    """
    dest_dir = Path(lora_dir or LORA_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    lock = _get_lock(name)
    with lock:
        # 1. Already present
        if dest.exists():
            if checksum:
                if _verify_checksum(dest, checksum):
                    logger.info(f"LoRA cached and verified: {name}")
                    return str(dest)
                else:
                    logger.warning(f"LoRA {name} checksum mismatch — re-downloading")
                    dest.unlink()
            else:
                size_mb = dest.stat().st_size / (1024 ** 2)
                logger.info(f"LoRA cached: {name} ({size_mb:.1f} MB)")
                return str(dest)

        # 2. Download
        logger.info(f"Downloading LoRA: {name}")
        logger.info(f"  URL: {url}")
        tmp = dest.with_suffix(".tmp")

        try:
            _download_with_progress(url, tmp)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Download failed for {name}: {e}") from e

        # 3. Checksum verification
        if checksum:
            if not _verify_checksum(tmp, checksum):
                tmp.unlink()
                raise RuntimeError(
                    f"Checksum mismatch for {name}. "
                    f"Expected: {checksum}. "
                    f"Got: {_sha256(tmp)}"
                )
            logger.info(f"Checksum OK: {name}")

        # 4. Atomic rename
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 ** 2)
        logger.info(f"LoRA ready: {name} ({size_mb:.1f} MB)")
        return str(dest)


def _download_with_progress(url: str, dest: Path) -> None:
    """Download URL to dest, logging progress every 10%."""

    class _ProgressReporter:
        def __init__(self):
            self.last_pct = -1

        def __call__(self, block_num, block_size, total_size):
            if total_size <= 0:
                return
            pct = int(block_num * block_size * 100 / total_size)
            pct = min(pct, 100)
            if pct >= self.last_pct + 10:
                self.last_pct = pct
                downloaded_mb = block_num * block_size / (1024 ** 2)
                total_mb = total_size / (1024 ** 2)
                logger.info(f"  {pct}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)")

    urllib.request.urlretrieve(url, dest, reporthook=_ProgressReporter())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, expected: str) -> bool:
    return _sha256(path) == expected.lower()


# ---------------------------------------------------------------------------
# CLI usage: python lora_helper.py <name> <url> [checksum]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 3:
        print("Usage: lora_helper.py <filename> <url> [sha256_checksum]")
        sys.exit(1)

    name = sys.argv[1]
    url = sys.argv[2]
    checksum = sys.argv[3] if len(sys.argv) > 3 else None

    path = ensure_lora(name, url, checksum)
    print(f"Ready: {path}")
