"""
White-reference (flat-field) correction for a dark-subtracted reflectance cube,
using the two teflon-tape reference blobs captured in-scene rather than a
separate white-reference acquisition.

corrected[..., b] = darksub[..., b] / white_ref[b] * SATURATION   (per band b)

where white_ref[b] is the MEDIAN dark-subtracted tape value for band b, taken
from two bright blobs that flank the dish along the scan axis (teflon tape
strips mounted on the stage, imaged at the start/end of every scan). Teflon is
a true-white photometric standard, so these blobs stand in for a dedicated
white capture, which does not exist for reflectance-mode acquisitions in this
checkout (see correction.py). Median, not mean: robust to the handful of
edge/transition pixels any thresholded blob mask inevitably includes.

Blob extraction (find_tape_blobs)
-----------------------------------
Each tape strip is a rounded/irregular blob, not a clean rectangle, and does
not span the full frame height -- confirmed on grain_ref_exp_2500 via
cv2.boundingRect (0,0,595,612 against a 640-tall frame: the naive "full-height
column range" this module used to use included ~28 rows of wood at the
bottom). Fixed by pulling out the actual blob shape instead of assuming one:
percentile-threshold -> despeckle -> cv2.connectedComponentsWithStats -> keep
the top-2 components by area (see white_exploration/find_tape_blobs.py for the
exploration, including a documented Otsu attempt that failed: global Otsu
merges the tape blob together with the dish on this scene, since the dish's
bright grid walls and kernel highlights push enough area into the same
"foreground" class -- a fixed high percentile isolates just the two tape
strips because they're brighter than essentially everything else in the
scene, including the dish's own (kernel/grid-diluted) average).

Saturation
-----------
Per the user's call, no saturation check or skip logic at all: the reference
for every band is simply the median of that band's dark-subtracted tape-blob
pixels. This is deliberate, not an oversight -- median depends only on rank,
not magnitude, so it is inherently insensitive to a minority of clipped
pixels regardless of how many there are, as long as they're under half the
population (confirmed on grain_ref_exp_2500: bands with any raw saturation
topped out at 46.6% of their tape-blob pixels clipped, comfortably under
50%). An earlier version of this module skipped 95/224 bands on an "any
saturated pixel" rule left over from when the reference was a MEAN (which
really is that fragile) -- it was never revisited after the mean->median
switch, then a majority-fraction gate was tried as a middle ground and also
removed: simplicity won given the median's robustness already covers the
observed cases.

Pipeline position
------------------
Runs AFTER dark subtraction (correction.py) and BEFORE the geometric/scan-axis
stretch correction (generate_viable_reflectance.py), operating on the raw
(un-stretched) scan geometry -- consistent with the standard reflectance formula
(raw-dark)/(white-dark) being a purely radiometric step, kept separate from the
geometric resample.

Outputs (into 07012026/stitched/ by default):
    <name>_whitecorr_cube.npy       float32 cube, same shape as the darksub input
    <name>_whitecorr_meta.json      tape blob bboxes/pixel counts, per-band white_ref
    <name>_whitecorr_masks.npz      the two boolean tape-blob masks (for inspection/reuse)
    <name>_whitecorr_intensity.png  greyscale mean-across-bands preview

Run:
    python3 white_correction.py                              # default grain cube
    python3 white_correction.py --darksub <cube.npy>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SATURATION = 4095  # 12-bit ADC ceiling (Mono12Packed)

DEFAULT_DARKSUB = "07012026/stitched/grain_ref_exp_2500_darksub_cube.npy"


def _largest_component(mask255, open_k=None):
    """255-mask -> boolean mask of its single largest connected component, or None."""
    m = mask255
    if open_k:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def find_tape_blobs(gray, pct=85.0, open_k=9, close_k=25, min_area_frac=0.02, k=2, pad=30):
    """Top-k tape blobs -> list of full-frame bool masks, each at its TRUE extent.

    Two stages -- a single pass over-includes (Otsu, merges with the dish) or
    under-includes (a global percentile, clipped by the tape's own top-to-
    bottom brightness gradient/vignetting: confirmed on grain_ref_exp_2500,
    fading ~3018 DN to ~2114 DN before its real edge, a hard cliff to ~80 DN
    at the wood -- a global-percentile mask stops around row 522 when the
    tape actually continues to ~610):

    Stage 1 (locate): global-percentile threshold -> despeckle -> connected
    components -> keep the k largest by area. Safe against the dish merging
    in (the dish's own brightness average is pulled down by its darker
    grid/kernel pixels, so it doesn't compete with the tape at a high
    percentile) but clips each blob's dimmer edges.
    Stage 2 (refine): re-threshold with Otsu inside a small padded crop
    around each stage-1 bbox, where tape-vs-wood *is* locally bimodal (the
    dish isn't in a narrow column crop around one tape strip), and take that
    crop's largest component -- recovers the blob's true full extent.
    """
    H, W = gray.shape
    g8 = np.clip((gray - gray.min()) / (gray.max() - gray.min() + 1e-9) * 255,
                0, 255).astype(np.uint8)

    thr = np.percentile(g8, pct)
    mask = (g8 > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]                     # skip label 0 (background)
    min_area = min_area_frac * H * W
    candidates = [1 + i for i in range(len(areas)) if areas[i] >= min_area]
    if len(candidates) < k:
        sys.exit(f"stage 1 found only {len(candidates)} tape candidate(s) >= "
                 f"{min_area_frac*100:.0f}% of frame area (need {k}); try --tape-pct lower.")
    candidates.sort(key=lambda lab: stats[lab, cv2.CC_STAT_AREA], reverse=True)
    top = sorted(candidates[:k], key=lambda lab: stats[lab, cv2.CC_STAT_LEFT])  # left-to-right

    refined = []
    for lab in top:
        x, y, w, h, _ = stats[lab]
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        _, crop_mask = cv2.threshold(g8[:, x0:x1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local = _largest_component(crop_mask, open_k=5)
        if local is None:
            sys.exit(f"stage 2 refinement failed for tape blob at columns [{x0}:{x1}]")
        full = np.zeros((H, W), bool)
        full[:, x0:x1] = local
        refined.append(full)
    return refined


def compute_white_reference(darksub_cube, masks):
    """Per-band white reference: the median dark-subtracted tape-blob value.

    Returns white_ref (C,) float -- the median of both tape blobs' pixels
    pooled together, per band. No saturation check: see module docstring for
    why the median doesn't need one here.
    """
    combined = np.logical_or.reduce(masks)
    return np.median(np.asarray(darksub_cube)[combined], axis=0).astype(np.float64)


def apply_white_correction(darksub_cube, white_ref):
    """corrected[...,b] = darksub[...,b] / white_ref[b] * SATURATION.

    white_ref is cast to float32 before dividing -- dividing a float32 cube by
    a float64 array upcasts the (huge) result to float64 via numpy's type
    promotion, silently doubling memory and on-disk size for no precision
    that matters here (confirmed: an earlier version of this line did exactly
    that and produced a 9.8GB cube instead of the expected 4.9GB).
    """
    ref32 = white_ref.astype(np.float32)
    return darksub_cube.astype(np.float32) / ref32[None, None, :] * np.float32(SATURATION)


def intensity_preview(cube):
    mean = np.asarray(cube).mean(axis=2)
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--darksub", default=DEFAULT_DARKSUB,
                        help="Dark-subtracted stitched cube .npy (output of correction.py).")
    parser.add_argument("--out", default="07012026/stitched",
                        help="Output subfolder (created if missing).")
    parser.add_argument("--tape-pct", type=float, default=85.0,
                        help="Brightness percentile a tape blob must exceed.")
    args = parser.parse_args()

    darksub_cube = np.load(args.darksub, mmap_mode="r")
    print(f"darksub {args.darksub}: shape {darksub_cube.shape}")

    gray = np.asarray(darksub_cube.mean(axis=2)).astype(np.float32)
    masks = find_tape_blobs(gray, pct=args.tape_pct)
    bboxes = []
    for mask, label in zip(masks, ("LEFT", "RIGHT")):
        ys, xs = np.nonzero(mask)
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
        bboxes.append([x, y, w, h])
        print(f"{label} tape blob: {int(mask.sum())} px, bbox=({x},{y},{w},{h}) "
              f"row span [{y}:{y + h}] of {gray.shape[0]}")

    white_ref = compute_white_reference(darksub_cube, masks)
    print(f"white reference (median) computed for all {darksub_cube.shape[2]} bands.")

    corrected = apply_white_correction(darksub_cube, white_ref)

    name = Path(args.darksub).name.removesuffix("_cube.npy")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube_path = out_dir / f"{name.replace('_darksub', '')}_whitecorr_cube.npy"
    np.save(cube_path, corrected)
    print(f"wrote {cube_path} ({corrected.nbytes / 1e6:.0f} MB, shape {corrected.shape})")

    masks_path = out_dir / f"{name.replace('_darksub', '')}_whitecorr_masks.npz"
    np.savez_compressed(masks_path, left=masks[0], right=masks[1])
    print(f"wrote {masks_path}")

    meta = {
        "tape_blob_bboxes": {"left": bboxes[0], "right": bboxes[1]},
        "tape_blob_px_counts": [int(m.sum()) for m in masks],
        "white_ref": [round(float(v), 2) for v in white_ref],
        "white_ref_stat": "median",
        "saturation": SATURATION,
    }
    meta_path = out_dir / f"{name.replace('_darksub', '')}_whitecorr_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {meta_path}")

    png_path = out_dir / f"{name.replace('_darksub', '')}_whitecorr_intensity.png"
    Image.fromarray(intensity_preview(corrected), mode="L").save(png_path)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
