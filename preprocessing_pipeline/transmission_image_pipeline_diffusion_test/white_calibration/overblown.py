"""
Percentage of pixels at ADC saturation in a raw white capture.

Stitches a directory of per-line Mono12Packed .bin captures (same format as
process_image.py's raw_white/<sample>/) and reports what fraction of pixels
sit at the 12-bit ADC ceiling (4095) -- a fully blown-out white reference
gives no dynamic range to divide by, so this is a quick sanity check before
running a sample through process_image.py.

Usage:
    python3 white_calibration/overblown.py raw_white/slit_diffuse_white_10k
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "loadstich"))

WIDTH = 640
CHANNELS = 224
SATURATION = 4095  # 12-bit ADC ceiling (Mono12Packed)

def load_hsi(arr):
    """
    Loads a 12-bit HSI image stored with 2 12-bit pixels in 3 8-bit bytes. Assumes the input array is of dtype np.uint8. channels must be a multiple of 3.
    Returns the loaded image with dtype np.uint16
    """
    if not arr.dtype == np.uint8:
        raise ValueError("Input array must have dtype np.uint8")
    shape = arr.shape
    if len(shape) == 3:
        img_shape = True
    else:
        img_shape = False

    arr = arr.flatten()
    flat_len = arr.shape[0]

    idx_arr = np.arange(flat_len // 3) * 3
    fst_uint8 = np.uint16(arr[idx_arr])
    mid_uint8 = np.uint16(arr[idx_arr + 1])
    lst_uint8 = np.uint16(arr[idx_arr + 2])

    arr = np.empty(flat_len * 2 // 3, dtype=np.uint16)
    arr[0::2] = (fst_uint8 << 4) + (mid_uint8 >> 4)
    # arr[..., 1::2] = (lst_uint8 << 4) + ((mid_uint8 & 0xF0) >> 4) # This shit is fucking wrong.
    arr[1::2] = (lst_uint8 << 4) + (mid_uint8 & 0xF)

    if img_shape:
        arr = arr.reshape(shape[0], shape[1], -1)
    return arr

def load_line(raw, w=WIDTH, c=CHANNELS):
    try:
        line = load_hsi(raw)
        return line.reshape([c, w]).swapaxes(0, 1)[::-1]
    except Exception:
        return np.zeros((w, c), dtype=np.uint16)


def stitch(directory, w=WIDTH, c=CHANNELS):
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise ValueError(f"no .bin files found in {directory}")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        cube[:, i, :] = load_line(raw, w, c)
    return cube


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Raw capture directory of per-line .bin files, "
                                      "e.g. raw_white/slit_diffuse_white_10k")
    args = parser.parse_args()

    cube = stitch(args.path)
    saturated = int((cube >= SATURATION).sum())
    pct = 100 * saturated / cube.size
    print(f"{pct:.2f}% of pixels saturated ({saturated}/{cube.size}, >= {SATURATION})")


if __name__ == "__main__":
    main()
