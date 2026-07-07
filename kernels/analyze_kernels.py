"""
Ingest hand-picked kernel outlines, mask each kernel out of the HSI cube, and
plot the average spectral signature per kernel. Built for transmittance QC:
confirm that light actually passes through a kernel and yields usable signal.

Coordinate convention (matches grid_overlay.py):
    each kernel is a list of (x, y) pixel vertices, x = scan line, y = spatial px.
    >=3 points -> filled polygon;  exactly 2 points -> rectangle (opposite corners).

Supply the outlines as a file:
    .json : [[[x,y],[x,y],...], [[x,y],...], ...]
    .py   : a variable  KERNELS = [ [(x,y), ...], ... ]

Run:
    python3 analyze_kernels.py <cube> kernels.py
    python3 analyze_kernels.py <cube> kernels.json --band 100 --save spectra.png
    python3 analyze_kernels.py <cube> kernels.py --save-mask   # also dump _masks.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from segment_kernels import display_band


def load_outlines(path):
    """Load kernel outlines from a .json or .py file into a list of (N,2) arrays."""
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text())
    elif path.suffix == ".py":
        ns = {}
        exec(path.read_text(), ns)
        if "KERNELS" not in ns:
            sys.exit(f"{path} must define a variable named KERNELS")
        raw = ns["KERNELS"]
    else:
        sys.exit("Outline file must be .json or .py")
    outlines = [np.asarray(k, dtype=float) for k in raw]
    for i, o in enumerate(outlines, 1):
        if o.ndim != 2 or o.shape[1] != 2 or len(o) < 2:
            sys.exit(f"kernel {i}: expected >=2 (x,y) points, got shape {o.shape}")
    return outlines


def rasterize(outline, H, W, pix):
    """Turn one outline into a boolean (H, W) mask.

    2 points -> rectangle between them; >=3 -> filled polygon. pix is the
    precomputed (H*W, 2) array of (x, y) pixel centres.
    """
    from matplotlib.path import Path as MplPath

    if len(outline) == 2:
        (x0, y0), (x1, y1) = outline
        verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        verts = outline
    return MplPath(verts).contains_points(pix).reshape(H, W)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", help="Path to a stitched uint16 cube .npy.")
    parser.add_argument("outlines", help="Kernel outlines file (.json or .py).")
    parser.add_argument("--band", type=int, default=None,
                        help="Band index for the backdrop (default: mean across bands).")
    parser.add_argument("--no-std", action="store_true",
                        help="Hide the +/-1 std shaded band on the spectra.")
    parser.add_argument("--save", default=None,
                        help="Figure PNG (default: <outlines stem>_spectra.png).")
    parser.add_argument("--save-mask", action="store_true",
                        help="Also write a labelled <outlines stem>_masks.npy.")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    args = parser.parse_args()

    cube_path = Path(args.cube)
    cube = np.load(cube_path)
    if cube.ndim != 3:
        sys.exit(f"Expected a 3D cube, got shape {cube.shape}")
    H, W = cube.shape[:2]
    outlines = load_outlines(args.outlines)
    print(f"Loaded {cube_path.name} {cube.shape}; {len(outlines)} kernel outline(s).")

    yy, xx = np.mgrid[0:H, 0:W]
    pix = np.vstack((xx.ravel(), yy.ravel())).T

    labels = np.zeros((H, W), dtype=np.int32)
    means, stds, kept = [], [], []
    for i, outline in enumerate(outlines, 1):
        mask = rasterize(outline, H, W, pix)
        n = int(mask.sum())
        if n == 0:
            print(f"  kernel {i}: 0 px inside image bounds — skipped")
            continue
        px = cube[mask]  # (n, channels)
        labels[mask] = i
        means.append(px.mean(axis=0))
        stds.append(px.std(axis=0))
        kept.append(i)
        print(f"  kernel {i}: {n} px")

    if not kept:
        sys.exit("No kernels produced any pixels; check your coordinates.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    ax_img.imshow(display_band(cube, args.band), cmap="gray")
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    for k, i in enumerate(kept):
        m = labels == i
        overlay[m] = (*cmap(k % 10)[:3], 0.4)
        ys, xs = np.nonzero(m)
        ax_img.text(xs.mean(), ys.mean(), str(i), color="white",
                    ha="center", va="center", fontweight="bold")
    ax_img.imshow(overlay)
    ax_img.set_title(f"{len(kept)} kernel(s)")
    ax_img.set_xlabel("x (scan line)")
    ax_img.set_ylabel("y (spatial px)")

    bands = np.arange(cube.shape[2])
    for k, i in enumerate(kept):
        c = cmap(k % 10)
        ax_spec.plot(bands, means[k], color=c, label=f"#{i}")
        if not args.no_std:
            ax_spec.fill_between(bands, means[k] - stds[k], means[k] + stds[k],
                                 color=c, alpha=0.15)
    ax_spec.set_xlabel("band index")
    ax_spec.set_ylabel("mean intensity (DN)")
    ax_spec.set_title("Average spectral signature per kernel")
    ax_spec.legend(fontsize="small", ncol=2)

    out = Path(args.save) if args.save else Path(args.outlines).with_suffix("").with_name(
        Path(args.outlines).stem + "_spectra.png")
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")

    if args.save_mask:
        mask_out = Path(args.outlines).with_suffix("").with_name(
            Path(args.outlines).stem + "_masks.npy")
        np.save(mask_out, labels)
        print(f"wrote {mask_out}")


if __name__ == "__main__":
    main()
