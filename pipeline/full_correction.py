"""
One-step correction: dark subtraction -> tape-blob white correction ->
checkerboard scan-axis geometry correction, chained end to end without
writing the (multi-GB) intermediate darksub/whitecorr cubes to disk.

corrected[w, n, c] = resample( clip(raw - dark, 0)[w, n, c] / white_ref[c] * 4095, fx, fy )

This is exactly correction.py -> white_correction.py -> generate_viable_reflectance.py
in sequence -- every step below calls into those modules' functions rather than
reimplementing them. See their docstrings for the derivation/rationale of each
stage; run them individually if you need to inspect or tune one stage (e.g.
--tape-pct or --manual-scale) without repeating the others.

Outputs (into 07012026/stitched/ by default):
    <name>_corrected_cube.npy       float32 cube, geometry- and radiometrically-corrected
    <name>_corrected_intensity.png  greyscale mean-across-bands preview

Run:
    python3 full_correction.py                              # default grain/dark pair
    python3 full_correction.py --grain <dir_or_cube> --dark <dir_or_cube>
    python3 full_correction.py --checkerboard 3 3 --manual-scale 0.14
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from correction import DEFAULT_DARK, DEFAULT_GRAIN, apply_dark_correction, load_cube
from generate_viable_reflectance import measure_scan_scale, render_gray8, resample, scan_scale_factor
from white_correction import apply_white_correction, compute_white_reference, find_tape_blobs, intensity_preview


def full_correction(grain_cube, dark_cube, tape_pct=85.0, checkerboard=(3, 3),
                    reference="spatial", manual_scale=None):
    """Run all three correction stages; returns (corrected_cube, (fx, fy))."""
    darksub = apply_dark_correction(grain_cube, dark_cube)

    gray_mean = np.asarray(darksub.mean(axis=2)).astype(np.float32)
    masks = find_tape_blobs(gray_mean, pct=tape_pct)
    white_ref = compute_white_reference(darksub, masks)
    whitecorr = apply_white_correction(darksub, white_ref)

    if manual_scale is not None:
        fx, fy = manual_scale, 1.0
    else:
        gray8 = render_gray8(whitecorr, None)
        px_scan, px_spatial, k, corners, method = measure_scan_scale(gray8, tuple(checkerboard))
        if px_scan is None:
            sys.exit(f"Checkerboard {checkerboard[0]}x{checkerboard[1]} NOT detected at any "
                     f"pre-compression.\n  Re-run with --manual-scale FLOAT to force the correction.")
        fx, fy = scan_scale_factor(px_scan, px_spatial, reference)
        print(f"Detected via {method}: scan x{fx:.5f}, spatial x{fy:.5f} ({reference} reference).")

    corrected = resample(whitecorr, fx, fy)
    return corrected, (fx, fy)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grain", default=DEFAULT_GRAIN,
                        help="Grain capture: stitched cube .npy or raw capture dir.")
    parser.add_argument("--dark", default=DEFAULT_DARK,
                        help="Dark reference: stitched cube .npy or raw capture dir.")
    parser.add_argument("--out", default="07012026/stitched",
                        help="Output subfolder (created if missing).")
    parser.add_argument("--name", default=None,
                        help="Output basename (default: derived from --grain).")
    parser.add_argument("--tape-pct", type=float, default=85.0,
                        help="Brightness percentile a tape blob must exceed.")
    parser.add_argument("--checkerboard", nargs=2, type=int, default=(3, 3),
                        metavar=("COLS", "ROWS"),
                        help="Inner-corner count (default 3 3 = a 4x4-square board).")
    parser.add_argument("--reference", choices=["spatial", "scan", "min", "max"],
                        default="spatial",
                        help="Which axis to trust when equalising (default: spatial).")
    parser.add_argument("--manual-scale", type=float, default=None,
                        help="Skip checkerboard detection; force this scan-axis (x) scale factor.")
    args = parser.parse_args()

    print(f"Loading grain cube from {args.grain} ...")
    grain_cube = load_cube(args.grain)
    print(f"  shape {grain_cube.shape}, dtype {grain_cube.dtype}")

    print(f"Loading dark reference from {args.dark} ...")
    dark_cube = load_cube(args.dark)
    print(f"  shape {dark_cube.shape}, dtype {dark_cube.dtype}")

    if dark_cube.shape[0] != grain_cube.shape[0] or dark_cube.shape[2] != grain_cube.shape[2]:
        sys.exit(f"dark/grain shape mismatch: dark {dark_cube.shape} vs grain {grain_cube.shape}")

    corrected, (fx, fy) = full_correction(
        grain_cube, dark_cube, tape_pct=args.tape_pct, checkerboard=args.checkerboard,
        reference=args.reference, manual_scale=args.manual_scale)
    print(f"Corrected: shape {corrected.shape}, dtype {corrected.dtype} (scan x{fx:.5f}, spatial x{fy:.5f}).")

    name = args.name or Path(str(args.grain).removesuffix("_cube.npy")).name
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube_path = out_dir / f"{name}_corrected_cube.npy"
    np.save(cube_path, corrected)
    print(f"wrote {cube_path} ({corrected.nbytes / 1e6:.0f} MB, shape {corrected.shape})")

    png_path = out_dir / f"{name}_corrected_intensity.png"
    Image.fromarray(intensity_preview(corrected), mode="L").save(png_path)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
