"""
Dark-reference intensity heatmap (spatial pixel x wavelength).

Reads a folder of per-line Mono12Packed .bin captures (one dark-reference
scan line per file) and renders the full median dark frame as a 2D heatmap:
spatial pixel down the y-axis, wavelength across the x-axis, colour = raw
intensity. Modelled on the "Dark reference intensity" figure (Fig. 7f) in
Engstrom's NIR-HSI food-analysis thesis (arXiv:2510.13452) -- there it's the
row-wise median of a shutter-closed dark image; here it's the median across
the .bin files, taken per (pixel, wavelength) so a few outlier/blank lines
don't skew it. Same inferno colormap and "Intensity" colourbar.

A dark reference is the sensor's own current/offset with no light, so this is
the sensor's fixed-pattern dark signal in full: the spatial dome down the pixel
axis (non-uniform dark current) AND its variation across the 224 bands, plus
any hot pixels/columns -- exactly the per-(pixel, band) frame the pipeline's
mean_frame() builds and later subtracts (raw - dark) in process_image.py.

By default the colour scale is clipped to the 1st/99th percentile of the frame
so hot pixels don't wash the structure out to a flat wash (the thesis figure,
unclipped, reads as uniform magenta for that reason); pass --pmin/--pmax or
--full-scale to change that. The wavelength axis is a nominal linear 900-1700nm
map (this transmission camera's NIR range; no factory per-band calibration is
on file) -- pass --wl-range, or --band-axis to label raw band index instead.

Reuses the loadstich/hsi_save_load codec (the byte-packing math is subtle --
do not reimplement it here) via the same sys.path shim process_image.py uses,
and orients each line identically to process_image.py:load_line.

    python3 visualize_dark_intensity.py raw_dark/raws/slit_diffuse_dark_25k
    python3 visualize_dark_intensity.py "<dark_dir>" --full-scale --out dark_qc.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file; no interactive display needed
import matplotlib.pyplot as plt
import numpy as np

# hsi_save_load's own import path assumes a wider jarvis_gui package not
# present in this checkout, so import the local copy directly instead
# (mirrors process_image.py).
sys.path.insert(0, str(Path(__file__).resolve().parent / "loadstich"))
from hsi_save_load import load_hsi  # noqa: E402

WIDTH = 640       # spatial pixels per line
CHANNELS = 224    # spectral bands per line
WL_MIN_NM = 900   # nominal NIR range for this transmission camera (linearly
WL_MAX_NM = 1700  # spaced across the 224 bands; no factory calibration on file)


def load_line(raw, w=WIDTH, c=CHANNELS):
    """Unpack one raw Mono12Packed line into an oriented (w, c) uint16 frame.

    A corrupt/unparseable line becomes a zero frame rather than aborting the
    whole scan (mirrors process_image.py:load_line). Keeping the orientation
    identical means the pixel (0..639) and band (0..223) axes here match the
    rest of the pipeline exactly.
    """
    try:
        line = load_hsi(raw)
        return line.reshape([c, w]).swapaxes(0, 1)[::-1]
    except Exception:
        return np.zeros((w, c), dtype=np.uint16)


def median_frame(directory, w=WIDTH, c=CHANNELS):
    """Median dark frame (w, c) across every .bin line in `directory`.

    Stacks the per-line (w, c) frames and takes the median over the file axis
    per (pixel, band) -- the analog of the thesis figure's row-wise median,
    robust to a handful of outlier/blank lines. Returns (files, frame,
    blank_idx); blank_idx lists any all-zero (unparseable/dead) lines, which
    are kept in the median but reported.
    """
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise SystemExit(f"no .bin files found in {directory}")

    stack = np.empty((len(files), w, c), dtype=np.float32)
    blank_idx = []
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        frame = load_line(raw, w, c)
        if not frame.any():
            blank_idx.append(i)
        stack[i] = frame

    return files, np.median(stack, axis=0), blank_idx


def plot_heatmap(directory, frame, n_files, vmin, vmax, cmap, band_axis, wl_range, out_path):
    """Render the (w, c) median dark frame as a pixel x wavelength heatmap."""
    w, c = frame.shape
    if band_axis:
        extent = [0, c - 1, w - 1, 0]      # x = raw band index
        xlabel = "wavelength (band index, 0..%d)" % (c - 1)
    else:
        extent = [wl_range[0], wl_range[1], w - 1, 0]
        xlabel = f"wavelength (nm, nominal {wl_range[0]}–{wl_range[1]})"

    fig, ax = plt.subplots(figsize=(7, 7.5))
    im = ax.imshow(frame, aspect="auto", origin="upper", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("pixel index (spatial pixel along line, 0..%d)" % (w - 1))
    ax.set_title(f"Dark reference intensity  —  {Path(directory).name}\n"
                 f"({n_files} files, median across files)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("intensity (ADC counts, 12-bit)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dark_dir",
                        help="Folder of per-line dark-reference .bin captures.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path. Defaults to <dark_dir_name>_dark_heatmap.png "
                             "in the current directory.")
    parser.add_argument("--cmap", default="inferno",
                        help="Matplotlib colormap (default inferno, matching the thesis figure).")
    parser.add_argument("--pmin", type=float, default=1.0,
                        help="Lower colour-scale percentile clip (default 1). Ignored with --full-scale.")
    parser.add_argument("--pmax", type=float, default=99.0,
                        help="Upper colour-scale percentile clip (default 99). Ignored with --full-scale.")
    parser.add_argument("--full-scale", action="store_true",
                        help="Fix the colour scale to the full 12-bit range 0..4095 instead of "
                             "percentile-clipping to the frame's own content.")
    parser.add_argument("--band-axis", action="store_true",
                        help="Label the x-axis with raw band index (0..223) instead of nominal nm.")
    parser.add_argument("--wl-range", nargs=2, type=float, default=(WL_MIN_NM, WL_MAX_NM),
                        metavar=("MIN_NM", "MAX_NM"),
                        help=f"Nominal wavelength range for the x-axis (default {WL_MIN_NM} {WL_MAX_NM}).")
    args = parser.parse_args()

    dark_dir = Path(args.dark_dir)
    out_path = Path(args.out) if args.out else Path(f"{dark_dir.name.strip()}_dark_heatmap.png")

    files, frame, blank_idx = median_frame(dark_dir)

    if args.full_scale:
        vmin, vmax = 0.0, 4095.0
    else:
        vmin, vmax = np.percentile(frame, [args.pmin, args.pmax])

    print(f"{dark_dir.name}: {len(files)} files, median frame shape {frame.shape} (pixel x band).")
    print(f"  intensity: min {frame.min():.1f}, max {frame.max():.1f}, median {np.median(frame):.1f}")
    print(f"  colour scale: {vmin:.1f} .. {vmax:.1f} "
          f"({'full 12-bit range' if args.full_scale else f'{args.pmin:g}–{args.pmax:g} pct clip'})")
    if blank_idx:
        print(f"  warning: {len(blank_idx)} blank/unparseable line(s) at indices {blank_idx} "
              f"(kept in the median, which is robust to a few)")

    plot_heatmap(dark_dir, frame, len(files), vmin, vmax, args.cmap,
                 args.band_axis, args.wl_range, out_path)
    print(f"wrote {out_path.resolve()}")

if __name__ == "__main__":
    main()
