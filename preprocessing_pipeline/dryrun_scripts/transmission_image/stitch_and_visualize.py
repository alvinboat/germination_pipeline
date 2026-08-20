"""
Stitch a directory of per-line Mono12Packed .bin captures into a hyperspectral
cube and render a quick-look visualization -- no dark/white/geometry
correction, just "what did we capture". Backup visualizer kept alongside
reflectance_image_pipeline/ and transmission_image_pipeline/ for a quick look
at a raw capture without running either full correction pipeline.
Non-destructive: source .bin files are never modified.

Uses the local loadstich/hsi_save_load.py codec -- the Mono12Packed
byte-unpacking math is subtle and that module is the single source of truth
for it (see CLAUDE.md), so it's imported rather than reimplemented here.

Run:
    python3 stitch_and_visualize.py path/to/capture_dir
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Push-broom scan axis (n_lines) is heavily oversampled relative to the 640px
# spatial axis (stage speed vs frame rate), so the raw stitch reads as an
# extremely elongated filmstrip. Undoing that properly needs a checkerboard
# geometry correction (see generate_viable_reflectance.py) which this quick
# script doesn't do; instead the preview is additionally squashed by this
# factor purely so the dish/kernels are recognisable by eye.
SCAN_SQUASH = 10

STORAGE_DIR = "storage_image"  # all outputs are written here
1. 
WIDTH = 640     # spatial pixels per line
CHANNELS = 224  # spectral bands per line

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
    """Unpack one raw Mono12Packed line into an oriented (w, c) uint16 frame.

    A corrupt/unparseable line becomes a zero frame instead of aborting the
    whole stitch.
    """
    try:
        line = load_hsi(raw)
        return line.reshape([c, w]).swapaxes(0, 1)[::-1]
    except Exception:
        return np.zeros((w, c), dtype=np.uint16)


def stitch(directory, w=WIDTH, c=CHANNELS):
    """Stitch a directory of per-line .bin captures into a (w, n_lines, c) cube."""
    directory = Path(directory)
    files = sorted(f for f in directory.iterdir() if f.suffix == ".bin")
    if not files:
        raise ValueError(f"no .bin files found in {directory}")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    bad = 0
    for i, f in enumerate(files):
        raw = np.fromfile(f, dtype=np.uint8)
        frame = load_line(raw, w, c)
        if not frame.any():
            bad += 1
        cube[:, i, :] = frame
    print(f"  {directory.name}: {len(files)} lines stitched ({bad} blank/bad).")
    return cube


def intensity_preview(cube):
    """Greyscale (n_lines, width) uint8 preview: mean across all bands, min-max normalised."""
    mean = cube.mean(axis=2).T.astype(np.float32)  # (n_lines, width)
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


def squash_for_viewing(preview, factor=SCAN_SQUASH):
    """Downsample the scan (n_lines) axis by `factor` so the dish/kernels are
    recognisable by eye -- NOT a physically accurate geometry correction."""
    im = Image.fromarray(preview, mode="L")
    w, h = im.size  # (width, n_lines)
    return im.resize((w, max(1, round(h / factor))), Image.BOX)


def squash_cube(cube, factor=SCAN_SQUASH):
    """Same scan-axis squash as squash_for_viewing, applied to the full
    (width, n_lines, channels) cube instead of just the mean-band preview.

    cv2.resize only accepts <=4 channels per call, so the 224-band cube is
    resized band-by-band. Not a physically accurate geometry correction (see
    SCAN_SQUASH) -- purely matches the squashed preview for a quick look at
    the actual spectral data in that aspect ratio.
    """
    w, n_lines, c = cube.shape
    new_n_lines = max(1, round(n_lines / factor))
    out = np.empty((w, new_n_lines, c), dtype=cube.dtype)
    for b in range(c):
        out[:, :, b] = cv2.resize(cube[:, :, b], (new_n_lines, w), interpolation=cv2.INTER_AREA)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_dir", help="Path to a directory of per-line .bin captures.")
    parser.add_argument("--out", default=None,
                        help=f"Output basename (default: <capture_dir>'s name). Writes "
                             f"<out>_cube.npy and <out>_preview.png under {STORAGE_DIR}/.")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    out_dir = Path(STORAGE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / (args.out if args.out else capture_dir.name)

    print(f"Stitching {capture_dir} ...")
    cube = stitch(capture_dir)
    print(f"  cube shape {cube.shape}, dtype {cube.dtype}")

    cube_path = out_base.with_name(out_base.name + "_cube.npy")
    np.save(cube_path, cube)
    print(f"wrote {cube_path} ({cube.nbytes / 1e6:.0f} MB)")

    preview = intensity_preview(cube)
    preview_path = out_base.with_name(out_base.name + "_preview.png")
    Image.fromarray(preview, mode="L").save(preview_path)
    print(f"wrote {preview_path} ({cube.shape[1]}x{cube.shape[0]}, raw scan-axis aspect)")

    squashed_path = out_base.with_name(out_base.name + "_preview_squashed.png")
    squash_for_viewing(preview).save(squashed_path)
    print(f"wrote {squashed_path} (scan axis squashed x{SCAN_SQUASH} for viewing, not geometry-corrected)")

    corrected_cube = squash_cube(cube)
    corrected_path = out_base.with_name(out_base.name + "_cube_corrected.npy")
    np.save(corrected_path, corrected_cube)
    print(f"wrote {corrected_path} ({corrected_cube.nbytes / 1e6:.0f} MB, shape {corrected_cube.shape}, "
          f"scan axis squashed x{SCAN_SQUASH} -- not geometry-corrected)")

if __name__ == "__main__":
    main()

