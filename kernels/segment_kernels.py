"""
Manually segment individual grain kernels out of a stitched HSI cube.

Opens an interactive matplotlib window showing one band (or the mean-across-bands
image) of the cube. You trace around each kernel with the mouse; every completed
outline becomes one labelled region. On close, a labelled mask and (optionally)
the per-kernel mean spectra are written to disk.

Controls (lasso mode, the default):
    drag around a kernel, release   -> commits one kernel
Controls (polygon mode, --selector polygon):
    click vertices, then Enter/double-click -> commits one kernel
    Esc                             -> discard the in-progress polygon
General:
    close the window                -> finish and save

Outputs (next to the cube, or --out):
    <name>_masks.npy       int32 labelled mask, shape (width, n_lines);
                           0 = background, 1..N = kernel index
    <name>_spectra.npy     float64 mean spectra, shape (N, channels)  [--spectra]

Run:
    python3 segment_kernels.py 07012026/stitched/grain_trans_exp_100k_cube.npy
    python3 segment_kernels.py <cube> --band 100 --spectra
    python3 segment_kernels.py <cube> --selector polygon

In Jupyter, use `%matplotlib widget` (or `qt`) first, then call
KernelSegmenter(cube).run() directly — the inline backend can't capture the mouse.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def display_band(cube, band=None):
    """Return a (width, n_lines) float image in [0, 1] to trace on.

    band=None uses the mean across all spectral channels; otherwise the single
    band index. Min-max normalised so the full dynamic range is visible.
    """
    img = cube.mean(axis=2) if band is None else cube[:, :, band]
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


class KernelSegmenter:
    """Interactive freehand/polygon segmentation of a hyperspectral cube.

    args:
        cube: (width, n_lines, channels) array (uint16 or float).
        band: band index to display, or None for the mean-across-bands image.
        selector: "lasso" (freehand drag) or "polygon" (click vertices).
    """

    def __init__(self, cube, band=None, selector="lasso"):
        self.cube = cube
        self.H, self.W = cube.shape[:2]
        self.disp = display_band(cube, band)
        self.selector_kind = selector

        # Every pixel coordinate as (x, y), tested once per selection against the
        # traced path. x = column (n_lines axis), y = row (width axis).
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self._pix = np.vstack((xx.ravel(), yy.ravel())).T

        # Labelled mask: 0 = background, 1..N = kernel. Later traces win on overlap.
        self.labels = np.zeros((self.H, self.W), dtype=np.int32)
        self.n = 0

    def _commit(self, verts):
        from matplotlib.path import Path as MplPath

        if verts is None or len(verts) < 3:
            return
        mask = MplPath(verts).contains_points(self._pix).reshape(self.H, self.W)
        if not mask.any():
            return
        self.n += 1
        self.labels[mask] = self.n
        print(f"  kernel {self.n}: {int(mask.sum())} px")
        self._refresh_overlay()

    def _refresh_overlay(self):
        # Tint committed regions so the user sees what's already captured.
        overlay = np.zeros((self.H, self.W, 4), dtype=np.float32)
        overlay[self.labels > 0] = (1.0, 0.0, 0.0, 0.35)
        self._ov.set_data(overlay)
        self._ax.figure.canvas.draw_idle()

    def run(self):
        """Open the window, block until closed, and return the labelled mask."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import LassoSelector, PolygonSelector

        fig, ax = plt.subplots()
        self._ax = ax
        ax.imshow(self.disp, cmap="gray")
        self._ov = ax.imshow(np.zeros((self.H, self.W, 4), dtype=np.float32))
        ax.set_title("Trace each kernel; close the window when done")

        if self.selector_kind == "polygon":
            sel = PolygonSelector(ax, self._commit)
        else:
            sel = LassoSelector(ax, self._commit)

        plt.show()
        del sel  # keep a reference alive until the window closes
        print(f"Segmented {self.n} kernel(s).")
        return self.labels

    def mean_spectra(self):
        """(N, channels) mean spectrum per kernel, in label order (1..N)."""
        return np.stack([
            self.cube[self.labels == i].mean(axis=0)
            for i in range(1, self.n + 1)
        ]) if self.n else np.empty((0, self.cube.shape[2]))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", help="Path to a stitched uint16 cube .npy.")
    parser.add_argument("--band", type=int, default=None,
                        help="Band index to display (default: mean across bands).")
    parser.add_argument("--selector", choices=["lasso", "polygon"], default="lasso",
                        help="Freehand drag (lasso) or click vertices (polygon).")
    parser.add_argument("--out", default=None,
                        help="Output prefix (default: cube path without _cube.npy).")
    parser.add_argument("--spectra", action="store_true",
                        help="Also write per-kernel mean spectra.")
    args = parser.parse_args()

    cube_path = Path(args.cube)
    cube = np.load(cube_path)
    if cube.ndim != 3:
        sys.exit(f"Expected a 3D cube, got shape {cube.shape}")
    print(f"Loaded {cube_path.name}: shape {cube.shape}, dtype {cube.dtype}")

    seg = KernelSegmenter(cube, band=args.band, selector=args.selector)
    labels = seg.run()
    if seg.n == 0:
        print("No kernels traced; nothing written.")
        return

    prefix = Path(args.out) if args.out else Path(str(cube_path).removesuffix("_cube.npy"))
    mask_path = prefix.with_name(prefix.name + "_masks.npy")
    np.save(mask_path, labels)
    print(f"wrote {mask_path} ({seg.n} kernels)")

    if args.spectra:
        spectra = seg.mean_spectra()
        spectra_path = prefix.with_name(prefix.name + "_spectra.npy")
        np.save(spectra_path, spectra)
        print(f"wrote {spectra_path} (shape {spectra.shape})")


if __name__ == "__main__":
    main()
