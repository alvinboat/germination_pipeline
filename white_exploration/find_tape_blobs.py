"""
Pull out the two teflon-tape blobs as connected components, following the
structural pattern in ~/Germination/generate_crops.py's get_cutout_mask()
(threshold -> cv2.connectedComponents -> take the component(s) by pixel
count) -- generalized to top-2 components (two tape strips) and done in TWO
STAGES (see below).

Threshold choice: Germination's get_cutout_mask uses Otsu, because its input
is a single tightly-cropped kernel cutout (one bright grain vs its own local
background -- a clean bimodal scene). Tried the same globally here first and
it does NOT hold up: global Otsu on this full, busier scene merges the tape
blob together with the dish (confirmed: bbox width 4966px, spanning clear
through the dish) -- the dish's bright grid walls/kernel highlights push
enough area into the "foreground" class, and enough of it stays 8-connected
after despeckling, that Otsu can't tell "the tape" from "everything bright".

Why two stages: a single fixed high percentile (e.g. 85th, over the WHOLE
image) avoids the dish-merging problem but then UNDER-includes the tape
instead -- the tape has a genuine top-to-bottom brightness gradient
(vignetting), fading from ~3018 DN to ~2114 DN before its real edge (a hard
cliff to ~80 DN at the wood) on grain_ref_exp_2500. A global percentile
calibrated against the whole image's histogram sits above the tape's dimmer
lower portion, cutting the detected blob off at row ~522 when the tape
actually continues to ~610. Locally, though, tape-vs-immediate-wood *is*
basically bimodal (the dish isn't in a narrow column crop around one tape
strip), so Otsu recovers the true extent (~610) there. Hence: stage 1 (global
percentile) safely LOCATES the two blobs without dish interference; stage 2
(local Otsu on a padded crop around each stage-1 bbox) RECOVERS each blob's
true full extent, including the dim/faded parts stage 1 clips.

Run:
    python3 white_exploration/find_tape_blobs.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DARKSUB = "07012026/stitched/grain_ref_exp_2500_darksub_cube.npy"


def _largest_component(mask01, open_k=None):
    """255-mask -> boolean mask of its single largest connected component, or None."""
    m = mask01
    if open_k:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def find_tape_blobs(gray, pct=85.0, open_k=9, close_k=25, min_area_frac=0.02, k=2, pad=30):
    """Top-k tape blobs -> list of full-frame bool masks, each at its TRUE extent.

    Stage 1 (locate): global-percentile threshold -> despeckle -> connected
    components -> keep the k largest by area. Safe against the dish (see
    module docstring) but clips each blob's dimmer edges.
    Stage 2 (refine): re-threshold (Otsu) within a small padded crop around
    each stage-1 bbox, where tape-vs-wood is locally bimodal, and take that
    crop's largest component -- recovers the blob's true full extent.
    """
    H, W = gray.shape
    g8 = np.clip((gray - gray.min()) / (gray.max() - gray.min() + 1e-9) * 255,
                0, 255).astype(np.uint8)

    # -- stage 1: locate --
    thr = np.percentile(g8, pct)
    mask = (g8 > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = min_area_frac * H * W
    candidates = [1 + i for i in range(len(areas)) if areas[i] >= min_area]
    if len(candidates) < k:
        sys.exit(f"stage 1 found only {len(candidates)} component(s) >= "
                 f"{min_area_frac*100:.0f}% of frame area (need {k}); lower --tape-pct.")
    candidates.sort(key=lambda lab: stats[lab, cv2.CC_STAT_AREA], reverse=True)
    top = sorted(candidates[:k], key=lambda lab: stats[lab, cv2.CC_STAT_LEFT])

    # -- stage 2: refine each blob's true extent locally --
    refined = []
    for lab in top:
        x, y, w, h, _ = stats[lab]
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        crop = g8[:, x0:x1]
        _, crop_mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local = _largest_component(crop_mask, open_k=5)
        if local is None:
            sys.exit(f"stage 2 refinement failed for blob at columns [{x0}:{x1}]")
        full = np.zeros((H, W), bool)
        full[:, x0:x1] = local
        refined.append(full)
    return refined


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

    blob_masks = find_tape_blobs(gray)
    vis = cv2.cvtColor(intensity_preview(cube), cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0)]
    labels_txt = ["LEFT", "RIGHT"]

    for mask, color, label in zip(blob_masks, colors, labels_txt):
        ys, xs = np.nonzero(mask)
        x, y, w, h = xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()
        print(f"{label} tape blob: {int(mask.sum())} px, bbox=({x},{y},{w},{h}) "
              f"row span [{y}:{y + h}] of {H}")
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 3)
        cv2.putText(vis, f"{label} tape ({int(mask.sum())}px)", (int(x), max(30, int(y) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)

    out_dir = Path(__file__).parent
    name = Path(src).name.removesuffix("_cube.npy")
    out_path = out_dir / f"{name}_tape_blobs_overlay.png"
    cv2.imwrite(str(out_path), vis)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
