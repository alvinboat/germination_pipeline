"""
Renders test10's corrected hyperspectral cube as a greyscale intensity image
(same mean-across-bands normalisation process_image.py uses for
corrected_image/*.png) with a labelled pixel-coordinate grid overlaid, so a
region's corners can be read off by eye before being passed to
plot_region_spectrum.py.

x = width/spatial axis (0-639), y = scan-line axis (0-n_lines-1) -- matches
the (width, n_lines, channels) cube process_image.py writes to
corrected_file/*.npy.

Usage:
    python3 grid_overlay.py
    python3 grid_overlay.py --cube ../transmission_image_pipeline_production/corrected_file/test10.npy --spacing 25
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, this script only saves a PNG
import matplotlib.pyplot as plt
import numpy as np

# Reuse the exact greyscale normalisation process_image.py uses for corrected_image/*.png
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transmission_image_pipeline_diffusion_test"))
from process_image import intensity_preview  # noqa: E402

DEFAULT_CUBE = (Path(__file__).resolve().parent.parent
                 / "transmission_image_pipeline_production" / "corrected_file" / "test1_cube_corrected.npy")
DEFAULT_OUT = Path(__file__).resolve().parent / "test1_grid.png"

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cube", default=str(DEFAULT_CUBE),
                        help="Corrected .npy cube (width, n_lines, channels).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output grid overlay PNG path.")
    parser.add_argument("--spacing", type=int, default=25, help="Grid line/label spacing in pixels.")
    args = parser.parse_args()

    cube = np.load(args.cube)
    print(f"Loaded {args.cube}: shape {cube.shape}")
    width, n_lines = cube.shape[0], cube.shape[1]

    gray = intensity_preview(cube)  # (width, n_lines) uint8
    img = gray.T                    # -> (n_lines, width) so imshow rows=y, cols=x

    fig, ax = plt.subplots(figsize=(14, 14 * n_lines / width), dpi=200)
    ax.imshow(img, cmap="gray", origin="upper")

    xt = np.arange(0, width, args.spacing)
    yt = np.arange(0, n_lines, args.spacing)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels(xt, rotation=90, fontsize=10)
    ax.set_yticklabels(yt, fontsize=10)
    ax.grid(which="major", color="red", linewidth=0.3, alpha=0.6)
    ax.set_xlabel(f"x (width axis, 0-{width - 1})")
    ax.set_ylabel(f"y (scan-line axis, 0-{n_lines - 1})")
    ax.set_title(Path(args.cube).stem)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
