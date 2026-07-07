"""
Render the transmittance (or any) HSI cube as a greyscale image with a labelled
pixel grid overlaid, so kernels can be located by eye and outlined by coordinate.

The axes are labelled in PIXEL coordinates. A coordinate is (x, y) where:
    x = horizontal = scan-line axis   (0 .. n_lines-1)
    y = vertical   = spatial px axis  (0 .. width-1)
Read (x, y) pairs straight off the grid to build kernel outlines for
analyze_kernels.py.
07012026/stitched/hsi_test_cube.npy
Run:
    python3 grid_overlay.py 07012026/stitched/hsi_test_cube.npy
    python3 grid_overlay.py <cube> --step 50 --band 100 --save grid.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from segment_kernels import display_band


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", help="Path to a stitched uint16 cube .npy.")
    parser.add_argument("--step", type=int, default=50,
                        help="Grid spacing in pixels (default: 50).")
    parser.add_argument("--band", type=int, default=None,
                        help="Band index to display (default: mean across bands).")
    parser.add_argument("--save", default=None,
                        help="Output PNG (default: <cube w/o _cube.npy>_grid.png).")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    args = parser.parse_args()

    cube_path = Path(args.cube)
    cube = np.load(cube_path)
    if cube.ndim != 3:
        sys.exit(f"Expected a 3D cube, got shape {cube.shape}")
    disp = display_band(cube, args.band)
    H, W = disp.shape  # H = width axis (y), W = n_lines axis (x)
    print(f"Loaded {cube_path.name}: shape {cube.shape}. Grid step {args.step}px.")

    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, don't try to open a window
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # Scale the figure so grid cells stay roughly square and labels stay legible.
    fig, ax = plt.subplots(figsize=(W / 100 + 2, H / 100 + 1))
    ax.imshow(disp, cmap="gray", interpolation="nearest")

    ax.xaxis.set_major_locator(MultipleLocator(args.step))
    ax.yaxis.set_major_locator(MultipleLocator(args.step))
    ax.grid(which="major", color="lime", linewidth=0.4, alpha=0.6)
    ax.tick_params(labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel("x  (scan line)")
    ax.set_ylabel("y  (spatial px)")
    ax.set_title(f"{cube_path.name}  —  grid step {args.step}px")

    out = Path(args.save) if args.save else Path(
        str(cube_path).removesuffix("_cube.npy")).with_name(
        Path(str(cube_path).removesuffix("_cube.npy")).name + "_grid.png")
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}  ({W}x{H}px image, view it to read off kernel coords)")


if __name__ == "__main__":
    main()
