"""
Sanity-check overlay: draw the two teflon-tape reference bands that
white_correction.py auto-detects, so we can visually confirm they're actually
sitting on the tape and not on something else (metal bars, wood, etc.).

Run:
    python3 white_exploration/inspect_tape_bands.py
    python3 white_exploration/inspect_tape_bands.py --darksub <cube.npy>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from white_correction import find_tape_bands, DEFAULT_DARKSUB  # noqa: E402


def intensity_preview(cube):
    mean = np.asarray(cube).mean(axis=2)
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--darksub", default=DEFAULT_DARKSUB,
                   help="Dark-subtracted stitched cube .npy.")
    p.add_argument("--out", default=None, help="Output PNG (default: alongside this script).")
    args = p.parse_args()

    cube = np.load(args.darksub, mmap_mode="r")
    print(f"Loaded {args.darksub}: shape {cube.shape}")
    gray = np.asarray(cube.mean(axis=2))
    bands = find_tape_bands(gray)
    for s, e in bands:
        print(f"  tape band [{s}:{e}] width={e - s} mean={gray[:, s:e].mean():.1f}")

    vis = cv2.cvtColor(intensity_preview(cube), cv2.COLOR_GRAY2BGR)
    H, W = gray.shape
    colors = [(0, 0, 255), (0, 255, 0)]
    labels = ["LEFT tape", "RIGHT tape"]
    for (s, e), color, label in zip(bands, colors, labels):
        cv2.rectangle(vis, (s, 0), (e, H - 1), color, 3)
        cv2.putText(vis, f"{label} [{s}:{e}]", (s, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    out = Path(args.out) if args.out else Path(__file__).parent / (
        Path(args.darksub).name.removesuffix("_cube.npy") + "_tape_bands_overlay.png")
    cv2.imwrite(str(out), vis)
    print(f"wrote {out} ({vis.shape[1]}x{vis.shape[0]})")


if __name__ == "__main__":
    main()
