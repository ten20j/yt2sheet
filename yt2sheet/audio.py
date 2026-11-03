"""Decode any audio/video file to a float32 numpy array using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np


def load_audio(
    path: str,
    sr: int = 44100,
    channels: int = 1,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Return audio as float32 array of shape (samples, channels)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but was not found on PATH")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(path), "-vn", "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    # -ss/-to before -i seek the input, so -to is an absolute timestamp of the input.
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"No audio decoded from {path}")
    return audio.reshape(-1, channels).copy()
