"""
Dark-current correction for a stitched reflectance grain cube.

corrected = clip(raw - dark, 0)

There is no white reference for reflectance-mode captures in this checkout (only
white_ref_trans_exp_10000, which is transmittance-mode at a different exposure),
so this only removes the fixed per-pixel/per-band dark-current offset — it does
NOT normalize for illumination or sensor gain. The dark reference is a directory
of scan lines captured at the same exposure with no illuminated scene; its lines
are stitched and averaged across the scan-line axis into one (width, channels)
dark frame, then broadcast-subtracted from every line of the grain cube.

Outputs (into 07012026/stitched/ by default):
    <name>_darksub_cube.npy       corrected uint16 cube, same shape as input
    <name>_darksub_intensity.png  greyscale mean-across-bands image

Run:
    python3 correction.py                              # default grain/dark pair
    python3 correction.py --grain <dir_or_cube.npy> --dark <dir_or_cube.npy>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from stitch_grain import stitch, intensity_image

DEFAULT_GRAIN = "07012026/stitched/grain_ref_exp_2500_cube.npy"
DEFAULT_DARK = "07012026/dark_ref_exp_2500"


def load_cube(src):
    """Load a cube from a stitched .npy, or stitch it fresh from a capture dir."""
    src = Path(src)
    if src.is_file() and src.suffix == ".npy":
        return np.load(src)
    return stitch(src)


def dark_frame(dark_cube):
    """Mean (width, channels) dark frame, averaged across the dark cube's lines."""
    return dark_cube.mean(axis=1)


def apply_dark_correction(grain_cube, dark_cube):
    """clip(grain - mean(dark), 0), cast back to the grain cube's dtype."""
    frame = dark_frame(dark_cube)
    corrected = grain_cube.astype(np.float32) - frame[:, None, :]
    return np.clip(corrected, 0, None).astype(grain_cube.dtype)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grain", default=DEFAULT_GRAIN,
                        help="Grain capture: stitched cube .npy or raw capture dir.")
    parser.add_argument("--dark", default=DEFAULT_DARK,
                        help="Dark reference: stitched cube .npy or raw capture dir.")
    parser.add_argument("--out", default="07012026/stitched",
                        help="Output subfolder (created if missing).")
    args = parser.parse_args()

    print(f"Loading grain cube from {args.grain} ...")
    grain_cube = load_cube(args.grain)
    print(f"  shape {grain_cube.shape}, dtype {grain_cube.dtype}")

    print(f"Loading dark reference from {args.dark} ...")
    dark_cube = load_cube(args.dark)
    print(f"  shape {dark_cube.shape}, dtype {dark_cube.dtype}")

    if dark_cube.shape[0] != grain_cube.shape[0] or dark_cube.shape[2] != grain_cube.shape[2]:
        sys.exit(f"dark/grain shape mismatch: dark {dark_cube.shape} vs grain {grain_cube.shape}")

    corrected = apply_dark_correction(grain_cube, dark_cube)
    n_clipped = int((grain_cube.astype(np.float32) - dark_frame(dark_cube)[:, None, :] < 0).sum())
    print(f"Applied dark subtraction ({n_clipped} px clipped at 0).")

    name = Path(args.grain).name.removesuffix("_cube.npy")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube_path = out_dir / f"{name}_darksub_cube.npy"
    np.save(cube_path, corrected)
    print(f"wrote {cube_path} ({corrected.nbytes / 1e6:.0f} MB, shape {corrected.shape})")

    img = intensity_image(corrected)
    png_path = out_dir / f"{name}_darksub_intensity.png"
    Image.fromarray(img, mode="L").save(png_path)
    print(f"wrote {png_path} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
