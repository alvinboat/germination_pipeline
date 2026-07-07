"""
Stitch the grain HSI captures (reflectance and transmittance) into hyperspectral
cubes and render a greyscale false-colour image (mean signal intensity across all
spectral bands) for each.

Outputs are written to a subfolder (default: 07012026/stitched/):
    grain_ref_exp_2500_cube.npy      stitched uint16 cube, shape (width, n_lines, channels)
    grain_ref_exp_2500_intensity.png greyscale mean-across-bands image
    grain_trans_exp_100k_cube.npy
    grain_trans_exp_100k_intensity.png

Run:
    python3 stitch_grain.py                 # both grain captures, default paths
    python3 stitch_grain.py --no-cube       # only write the PNGs
    python3 stitch_grain.py --dirs 07012026/grain_ref_exp_2500 ...
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# hsi_save_load lives in ./loadstich (the module's own import path assumes a wider
# jarvis_gui package that is not present in this checkout, so import it locally).
sys.path.insert(0, str(Path(__file__).resolve().parent / "loadstich"))
from hsi_save_load import load_hsi  # noqa: E402

WIDTH = 640       # spatial pixels per line
CHANNELS = 224    # spectral bands


def load_line(raw, w=WIDTH, c=CHANNELS):
    """Unpack one raw Mono12Packed line into an oriented (w, c) uint16 frame.

    Mirrors SpecimStitcher.load_line: on any parse failure the line is replaced
    by a zero frame so a single corrupt file does not abort the whole stitch.
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
        raise ValueError(f"No .bin files found in {directory}")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    bad = 0
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        frame = load_line(raw, w, c)
        if not frame.any():
            bad += 1
        cube[:, i, :] = frame
    print(f"  {directory.name}: {len(files)} lines stitched ({bad} blank/bad).")
    return cube


def intensity_image(cube):
    """Greyscale false-colour image: mean intensity across all bands, 0-255 uint8.

    Returns a (width, n_lines) array. Normalised per-image via min-max so the full
    dynamic range is visible regardless of exposure/reflectance level.
    """
    mean = cube.mean(axis=2)  # (width, n_lines), float
    lo, hi = float(mean.min()), float(mean.max())
    if hi > lo:
        norm = (mean - lo) / (hi - lo)
    else:
        norm = np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+",
                        default=["07012026/grain_ref_exp_2500",
                                 "07012026/grain_trans_exp_100k",
                                 "07012026/07022026/grain_trans_10k",
                                 "07012026/07022026/grain_trans_20k",
                                 "07012026/lets_get_this_money/grain_trans_exp_10k",
                                 "07012026/lets_get_this_money/grain_trans_exp_20k",
                                 "07012026/hsi_test"
    
                                 ],
                        help="Capture directories to stitch.")
    parser.add_argument("--out", default="07012026/stitched",
                        help="Output subfolder (created if missing).")
    parser.add_argument("--no-cube", action="store_true",
                        help="Skip writing the stitched .npy cubes; only write PNGs.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in args.dirs:
        name = Path(d).name
        print(f"Processing {name} ...")
        cube = stitch(d)

        if not args.no_cube:
            cube_path = out_dir / f"{name}_cube.npy"
            np.save(cube_path, cube)
            print(f"  wrote {cube_path} ({cube.nbytes / 1e6:.0f} MB, shape {cube.shape})")

        img = intensity_image(cube)
        png_path = out_dir / f"{name}_intensity.png"
        Image.fromarray(img, mode="L").save(png_path)
        print(f"  wrote {png_path} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
