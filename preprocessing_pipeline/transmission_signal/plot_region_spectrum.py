"""
Plots mean +/- 1 std intensity per spectral band for one or more rectangular
regions of test10's corrected cube, given each region's two opposite pixel
corners read off grid_overlay.py's test10_grid.png (x = width axis 0-639,
y = scan-line axis). All regions are plotted together on one chart. Also
renders every region as a bright, translucent box on the original intensity
image -- each box uses the same color as its corresponding spectrum line, so
a region is easy to trace between the two outputs.

The two corners within a box don't need to be given in any particular order
-- the region used is just their bounding box.

Usage:
    python3 plot_region_spectrum.py --box X0 Y0 X1 Y1 [--box X0 Y0 X1 Y1 ...]
    python3 plot_region_spectrum.py --box 100 200 150 250 --box 300 400 350 450 \
        --cube ../transmission_image_pipeline_production/corrected_file/test10.npy
"""
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict, deque
from typing import List


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

# Reuse the exact greyscale normalisation process_image.py uses for corrected_image/*.png
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transmission_image_pipeline_diffusion_test"))
from process_image import intensity_preview  # noqa: E402

DEFAULT_CUBE = (Path(__file__).resolve().parent.parent
                 / "transmission_image_pipeline_production" / "corrected_file" / "test1_cube_corrected.npy")

# Camera's 224 bands are linearly spaced across its NIR spectral range -- no
# per-band factory calibration is available, so this is a linear estimate.
WAVELENGTH_MIN_NM = 900.0
WAVELENGTH_MAX_NM = 1700.0

# Hand-picked regions, read off test10_grid.png -- used whenever --box isn't
# passed on the command line. Corner order within a box doesn't matter.
def genbox(nums):
    x, y, x_off, y_off = nums[0], nums[1], nums[2], nums[3]
    hold = []

    for i in range(6):
        x_base = x + i * x_off
        y_base = y + i * y_off
        x1, x2 = x_base-5, x_base+5
        y1, y2 = y_base+5, y_base-5
        hold.append([x1,y1,x2,y2])
    return hold
nums_40k = [110, 915, 75, 10]
nums_60k = [110, 760, 75, 8]

DEFAULT_BOXES = [[250,670, 260,660],
                 [250, 775, 260, 750]]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--box", dest="boxes", action="append", nargs=4, type=int, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="Region corners, repeatable -- one per square, e.g. "
                             "--box 100 200 150 250 --box 300 400 350 450. "
                             "Defaults to DEFAULT_BOXES in this script if omitted.")
    parser.add_argument("--cube", default=str(DEFAULT_CUBE),
                        help="Corrected .npy cube (width, n_lines, channels).")
    parser.add_argument("--out", default=None,
                        help="Output spectrum PNG path (default: <cube>_spectrum_<...>.png "
                             "next to this script).")
    args = parser.parse_args()

    cube = np.load(args.cube)
    width, n_lines, n_bands = cube.shape

    boxes = []
    for x0_, y0_, x1_, y1_ in (args.boxes if args.boxes is not None else DEFAULT_BOXES):
        x0, x1 = sorted((x0_, x1_))
        y0, y1 = sorted((y0_, y1_))
        if x0 < 0 or x1 >= width or y0 < 0 or y1 >= n_lines:
            raise SystemExit(f"region x[{x0}:{x1}] y[{y0}:{y1}] out of bounds for cube shape {cube.shape}")
        boxes.append((x0, y0, x1, y1))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # short, informative filename tag for a few boxes; a bare count once it'd get unwieldy
    tag = ("_".join(f"{x0}_{y0}_{x1}_{y1}" for x0, y0, x1, y1 in boxes)
           if len(boxes) <= 2 else f"{len(boxes)}boxes")

    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n_bands)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        color = colors[i % len(colors)]
        region = cube[x0:x1 + 1, y0:y1 + 1, :]
        n_px = region.shape[0] * region.shape[1]
        mean = region.mean(axis=(0, 1))
        std = region.std(axis=(0, 1))
        print(f"Region {i}: x[{x0}:{x1}] y[{y0}:{y1}] -- {n_px} px")
        ax.plot(wavelengths, mean, color=color, label=f"x[{x0}:{x1}] y[{y0}:{y1}]")
        ax.fill_between(wavelengths, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel(f"wavelength (nm, {WAVELENGTH_MIN_NM:.0f}-{WAVELENGTH_MAX_NM:.0f} linear estimate)")
    ax.set_ylabel("intensity")
    ax.set_title(f"{Path(args.cube).stem}: {len(boxes)} region(s)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out = args.out or str(Path(__file__).resolve().parent / f"{Path(args.cube).stem}_spectrum_{tag}.png")
    fig.savefig(out)
    print(f"wrote {out}")

    gray = intensity_preview(cube)  # (width, n_lines)
    img = gray.T                    # -> (n_lines, width) so imshow rows=y, cols=x

    fig2, ax2 = plt.subplots(figsize=(14, 14 * n_lines / width), dpi=200)
    ax2.imshow(img, cmap="gray", origin="upper")
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        color = colors[i % len(colors)]
        rect = Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                          facecolor=mcolors.to_rgba(color, alpha=0.35),
                          edgecolor=mcolors.to_rgba(color, alpha=0.9), linewidth=2)
        ax2.add_patch(rect)
    ax2.set_xlabel(f"x (width axis, 0-{width - 1})")
    ax2.set_ylabel(f"y (scan-line axis, 0-{n_lines - 1})")
    ax2.set_title(f"{Path(args.cube).stem}: {len(boxes)} region(s)")
    fig2.tight_layout()

    region_out = str(Path(__file__).resolve().parent / f"{Path(args.cube).stem}_region_{tag}.png")
    fig2.savefig(region_out)
    print(f"wrote {region_out}")

    plt.show()


if __name__ == "__main__":
    main()
