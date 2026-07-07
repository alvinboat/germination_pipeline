"""
Contour-based teflon-tape strip finder, replacing the column-projection
assumption in white_correction.find_tape_bands (which implicitly assumes the
tape spans the full frame height -- confirmed wrong on grain_ref_exp_2500: the
tape blob's true boundingRect is (0,0,595,612) against a 640-tall frame, so a
full-height box picks up ~28 rows of wood at the bottom).

Approach: percentile-threshold the mean-across-bands greyscale, morphologically
clean up speckle (grid lines, marker edges), keep the largest contours, and use
each contour's own footprint (bounding box, minAreaRect, or the contour mask
itself) as the reference region -- so the shape adapts to whatever the tape's
true extent actually is instead of assuming it fills the frame.

Run:
    python3 white_exploration/find_tape_strips.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DARKSUB = "07012026/stitched/grain_ref_exp_2500_darksub_cube.npy"


def find_tape_contours(gray, pct=85.0, min_area_frac=0.02, open_k=9, close_k=25):
    """Return the two largest bright contours as (contour, bbox, minAreaRect).

    pct: percentile threshold for "bright". open_k despeckles small bright
    noise (grid lines, marker corners); close_k bridges small dark gaps
    *within* a tape blob (e.g. faint scan-line seams) so it stays one contour.
    min_area_frac: a contour must cover at least this fraction of the image
    to qualify -- the tape blobs are huge (>30% of the frame height x a few
    hundred px wide); this rejects checkerboards, markers, and dish glare.
    """
    H, W = gray.shape
    thr = np.percentile(gray, pct)
    mask = (gray > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * H * W
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    big.sort(key=cv2.contourArea, reverse=True)
    if len(big) < 2:
        sys.exit(f"found only {len(big)} contour(s) >= {min_area_frac*100:.0f}% of frame area; "
                 f"lower --pct or min_area_frac.")

    out = []
    for c in big[:2]:
        bbox = cv2.boundingRect(c)          # (x, y, w, h)
        rect = cv2.minAreaRect(c)           # (center, (w,h), angle)
        out.append((c, bbox, rect))
    out.sort(key=lambda t: t[1][0])         # left-to-right by bbox x
    return out, mask


def intensity_preview(cube):
    mean = np.asarray(cube).mean(axis=2)
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DARKSUB
    cube = np.load(src, mmap_mode="r")
    print(f"Loaded {src}: shape {cube.shape}")
    gray = np.asarray(cube.mean(axis=2)).astype(np.float32)
    H, W = gray.shape

    strips, mask = find_tape_contours(gray)
    vis = cv2.cvtColor(intensity_preview(cube), cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0)]
    labels = ["LEFT tape", "RIGHT tape"]

    for (c, bbox, rect), color, label in zip(strips, colors, labels):
        x, y, w, h = bbox
        area_full = w * h
        area_true = cv2.contourArea(c)
        wasted = 1 - area_true / area_full
        print(f"{label}: bbox=({x},{y},{w},{h})  minAreaRect angle={rect[2]:.2f}  "
              f"contour_area={area_true:.0f}  bbox_area={area_full}  "
              f"non-tape px in bbox if used as-is: {wasted*100:.1f}%")
        cv2.drawContours(vis, [c], -1, color, 3)                 # true footprint
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)      # bbox for comparison
        cv2.putText(vis, f"{label} bbox h={h}/{H}", (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

    out_dir = Path(__file__).parent
    name = Path(src).name.removesuffix("_cube.npy")
    cv2.imwrite(str(out_dir / f"{name}_tape_contours_overlay.png"), vis)
    cv2.imwrite(str(out_dir / f"{name}_tape_threshold_mask.png"), mask)
    print(f"wrote {out_dir}/{name}_tape_contours_overlay.png")
    print(f"wrote {out_dir}/{name}_tape_threshold_mask.png")


if __name__ == "__main__":
    main()
