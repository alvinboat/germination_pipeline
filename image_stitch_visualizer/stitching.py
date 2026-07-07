"""
Shared low-level pieces used by every bin_to_image_*.py variant: unpacking
Mono12Packed lines, stitching a capture directory into a cube, and rendering
the plain mean-across-bands pseudo image. Kept dependency-free (numpy only)
so it stays usable even from a mode-specific script that pulls in heavier
deps (e.g. OpenCV for checkerboard detection).
"""

import os
from pathlib import Path

import numpy as np

from hsi_save_load import load_hsi

WIDTH = 640       # spatial pixels per line
CHANNELS = 224    # spectral bands per line


def load_line(raw, w=WIDTH, c=CHANNELS):
    """Unpack one raw Mono12Packed line into an oriented (w, c) uint16 frame.

    A corrupt/unparseable line becomes a zero frame instead of aborting the
    whole stitch (mirrors the parent repo's stitch_grain.load_line).
    """
    try:
        line = load_hsi(raw)
        return line.reshape([c, w]).swapaxes(0, 1)[::-1]
    except Exception:
        return np.zeros((w, c), dtype=np.uint16)


def stitch(directory, w=WIDTH, c=CHANNELS):
    """Stitch a directory of per-line .bin captures into a (w, n_lines, c) cube."""
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise ValueError(f"no .bin files found in {directory}")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    bad = 0
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        frame = load_line(raw, w, c)
        if not frame.any():
            bad += 1
        cube[:, i, :] = frame
    print(f"  {directory.name}: {len(files)} lines stitched ({bad} blank/bad)")
    return cube


def pseudo_image(cube):
    """(width, n_lines) uint8 greyscale: mean intensity across all bands, min-max normalised."""
    mean = cube.mean(axis=2)
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)
