"""
Plots intensity against wavelength for one or more rectangular regions of a
corrected cube, given each region's two opposite pixel corners read off
grid_overlay.py's test1_grid.png (x = width axis 0-639, y = scan-line axis).

Where plot_region_spectrum.py summarises a region by its mean +/- 1 std, this
script marks the *maximum* intensity each band reaches anywhere inside the
region: one marker per band on a dashed max curve, with the region mean drawn
underneath for reference. The brightest band overall is annotated with the pixel
that produced it, and that pixel is crosshaired on the region image -- so a hot
band can be traced straight back to the pixel responsible.

All regions are plotted together on one chart, and every region is drawn as a
bright translucent box on the intensity image in the same color as its curve.

The two corners within a box don't need to be given in any particular order
-- the region used is just their bounding box.

Usage:
    python3 plot_region_max_spectrum.py --box X0 Y0 X1 Y1 [--box X0 Y0 X1 Y1 ...]
    python3 plot_region_max_spectrum.py --box 315 485 325 475 --no-mean
    python3 plot_region_max_spectrum.py --box 100 200 150 250 --csv peaks.csv \
        --cube ../transmission_image_pipeline_production/corrected_file/test10.npy
"""
import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle

DEFAULT_CUBE = (Path(__file__).resolve().parent.parent
                 / "transmission_image_pipeline_production" / "corrected_file" / "test1_cube_corrected.npy")

# Camera's 224 bands are linearly spaced across its NIR spectral range -- no
# per-band factory calibration is available, so this is a linear estimate.
WAVELENGTH_MIN_NM = 900.0
WAVELENGTH_MAX_NM = 1700.0

# Hand-picked regions, read off test1_grid.png -- used whenever --box isn't
# passed on the command line. Corner order within a box doesn't matter.
DEFAULT_BOXES = [
    [385, 1160, 395, 1150],
    [345, 755, 355, 745]
]


def intensity_preview(cube, lo_pct=1.0, hi_pct=99.0):
    """Greyscale (width, n_lines) uint8 preview: mean across all bands, percentile-clip normalised.

    Copied verbatim from transmission_image_pipeline_diffusion_test/process_image.py
    to keep this script standalone -- it is the same normalisation that pipeline
    writes corrected_image/*.png with, so the region image below lines up
    pixel-for-pixel with those previews. If the pipeline's version ever changes,
    this one has to be re-copied by hand.

    A handful of bad-white-reference pixels produce reflectance values orders of
    magnitude above the rest of the scene -- a true min-max stretch spends nearly
    the whole 0-255 range on those outliers and crushes everything else to
    near-black. Clipping to the 1st/99th percentile instead keeps the stretch
    anchored to the actual scene content.

    nan-aware, because transmission_imagev2 writes nan where it has no usable
    white reference and every pixel of such a cube carries some (the lamp is
    dead in the outermost bands). A plain mean propagates that to the whole
    frame, the percentiles follow, and the uint8 cast turns nan into 0 -- a
    silently, solidly black preview. On a cube with no nan these behave
    identically to the plain versions.
    """
    mean = np.asarray(np.nanmean(cube, axis=2))
    if not np.isfinite(mean).any():
        return np.zeros(mean.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(mean, [lo_pct, hi_pct])
    norm = np.clip((mean - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(mean)
    return (np.nan_to_num(norm, nan=0.0) * 255.0).round().astype(np.uint8)


def band_stats(region):
    """Per-band mean, std, max and the (x, y) offset of each band's max pixel.

    nan-aware throughout: transmission_imagev2 writes nan wherever it has no
    usable white reference (the lamp is dead in the outermost bands), so a plain
    max would report nan for those bands and a plain argmax would point at an
    arbitrary pixel. Bands with no finite pixel at all get nan stats and an
    offset of -1.
    """
    n_bands = region.shape[2]
    flat = region.reshape(-1, n_bands).astype(np.float64)  # rows ordered x-major, y fastest
    finite = np.isfinite(flat)
    has_data = finite.any(axis=0)

    with warnings.catch_warnings():  # all-nan bands are expected, not worth a warning per band
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        peak = np.nanmax(flat, axis=0)

    # -inf fill so argmax can't land on a nan pixel; bands with no data are masked out after
    idx = np.where(finite, flat, -np.inf).argmax(axis=0)
    idx = np.where(has_data, idx, -1)
    dx, dy = np.divmod(idx, region.shape[1])
    return mean, std, peak, np.where(has_data, dx, -1), np.where(has_data, dy, -1)


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
    parser.add_argument("--no-mean", action="store_true",
                        help="Plot only the per-band max curve, without the mean +/- 1 std reference.")
    parser.add_argument("--csv", default=None,
                        help="Also write per-band mean/std/max and each max's pixel to this CSV.")
    parser.add_argument("--out", default=None,
                        help="Output spectrum PNG path (default: <cube>_maxspectrum_<...>.png "
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
    csv_rows = []
    hot_pixels = []   # (x, y, color) of each region's brightest pixel, crosshaired on the region image
    peak_labels = []  # (wavelength, value, text, color) -- annotated after the loop, once ylim is known

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        color = colors[i % len(colors)]
        region = cube[x0:x1 + 1, y0:y1 + 1, :]
        n_px = region.shape[0] * region.shape[1]
        mean, std, peak, dx, dy = band_stats(region)
        label = f"x[{x0}:{x1}] y[{y0}:{y1}]"

        if not args.no_mean:
            ax.plot(wavelengths, mean, color=color, linewidth=1.2, alpha=0.55,
                    label=f"{label} mean")
            ax.fill_between(wavelengths, mean - std, mean + std, color=color, alpha=0.15)

        peak_note = ""
        if np.isfinite(peak).any():
            b = int(np.nanargmax(peak))
            hx, hy = x0 + int(dx[b]), y0 + int(dy[b])
            hot_pixels.append((hx, hy, color))
            peak_note = f" -- peak {peak[b]:.4g} @ {wavelengths[b]:.0f} nm"
            print(f"Region {i}: x[{x0}:{x1}] y[{y0}:{y1}] -- {n_px} px, "
                  f"{int(np.isfinite(peak).sum())}/{n_bands} bands with data")
            print(f"  brightest band {b} ({wavelengths[b]:.1f} nm): max {peak[b]:.4g} "
                  f"at pixel (x={hx}, y={hy}), band mean {mean[b]:.4g}")
            peak_labels.append((wavelengths[b], peak[b],
                                f"band {b}, {wavelengths[b]:.0f} nm\nmax {peak[b]:.4g} @ ({hx}, {hy})",
                                color))
        else:
            print(f"Region {i}: x[{x0}:{x1}] y[{y0}:{y1}] -- {n_px} px, no finite pixels in any band")

        # one marker per band: the max intensity that band reaches inside the region
        ax.plot(wavelengths, peak, color=color, linewidth=1.0, linestyle="--",
                marker="o", markersize=3.5, markerfacecolor=color, markeredgecolor="none",
                label=f"{label} per-band max{peak_note}")
        if peak_note:  # star the brightest band of the lot, white-ringed so it reads over the curve
            ax.plot(wavelengths[b], peak[b], marker="*", markersize=13, color=color,
                    markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=5)

        for b in range(n_bands):
            csv_rows.append([i, label, b, f"{wavelengths[b]:.3f}", mean[b], std[b], peak[b],
                             x0 + int(dx[b]) if dx[b] >= 0 else "",
                             y0 + int(dy[b]) if dy[b] >= 0 else ""])

    # Spell the winning pixel out next to the star, but only for a single region:
    # several regions peak within a band or two of each other on the same target,
    # so the labels would land on top of one another. Their numbers ride in the
    # legend instead, and stdout has the full detail either way.
    if len(peak_labels) == 1:
        wl, val, text, color = peak_labels[0]
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))  # headroom so the text clears the title
        x_lo, x_hi = ax.get_xlim()
        right_edge = wl > x_lo + 0.75 * (x_hi - x_lo)  # keep the text off the right spine
        ax.annotate(text, xy=(wl, val), xycoords="data",
                    xytext=(-10 if right_edge else 10, 14), textcoords="offset points",
                    ha="right" if right_edge else "left", fontsize=7, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8, alpha=0.7))

    ax.set_xlabel(f"wavelength (nm, {WAVELENGTH_MIN_NM:.0f}-{WAVELENGTH_MAX_NM:.0f} linear estimate)")
    ax.set_ylabel("intensity")
    ax.set_title(f"{Path(args.cube).stem}: per-band max intensity, {len(boxes)} region(s)")
    ax.grid(True, color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7)
    fig.tight_layout()

    out = args.out or str(Path(__file__).resolve().parent / f"{Path(args.cube).stem}_maxspectrum_{tag}.png")
    fig.savefig(out)
    print(f"wrote {out}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["region", "box", "band", "wavelength_nm", "mean", "std", "max",
                        "max_x", "max_y"])
            w.writerows(csv_rows)
        print(f"wrote {args.csv}")

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
    for hx, hy, color in hot_pixels:
        ax2.plot(hx, hy, marker="+", color=color, markersize=10, markeredgewidth=1.5)
    ax2.set_xlabel(f"x (width axis, 0-{width - 1})")
    ax2.set_ylabel(f"y (scan-line axis, 0-{n_lines - 1})")
    ax2.set_title(f"{Path(args.cube).stem}: {len(boxes)} region(s), + marks each region's brightest pixel")
    fig2.tight_layout()

    region_out = str(Path(__file__).resolve().parent / f"{Path(args.cube).stem}_maxregion_{tag}.png")
    fig2.savefig(region_out)
    print(f"wrote {region_out}")

    plt.show()


if __name__ == "__main__":
    main()
