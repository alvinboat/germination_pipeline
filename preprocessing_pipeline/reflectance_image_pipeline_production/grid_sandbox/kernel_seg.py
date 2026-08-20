"""
Floor-anchored, brightness-based kernel segmentation for white-corrected
REFLECTANCE wells.

Why not SAM (the previous kernel_sam.py, now retired)? Each well holds three
things and only their SEPARABILITY matters:

  * dark well floor  (broadband ~0 reflectance)
  * bright grain kernel
  * bright plastic  (grid wall + dish rim/clip intruding on edge cells)

Measured on 7222026_ref_t13:
  kernel vs floor   -- brightness ratio huge; spectral angle ~31deg  -> separable
  kernel vs plastic -- brightness EQUAL (ratio ~1.08); angle ~6.8deg -> NOT separable

So (a) kernel-vs-floor is a pure BRIGHTNESS problem -- and SAM, being magnitude-
invariant by design, throws away the one cue that works, then leaks into noisy
near-zero floor pixels it happens to match in shape (the "drips"); and (b) the
plastic is bright, spectrally kernel-like AND spatially fused to the kernel
(rides the top wall with no dark gap), so NO pixel cue or connectivity trick can
remove it -- only geometry can.

The dark floor is the one unambiguous region (walls/rim/kernel are all bright):

  MASK (complete kernel, never clipped):
  1. Otsu on in-well broadband brightness -> bright (kernel+plastic) vs dark.
  2. kernel = largest bright component -> fill holes. NEVER eroded, so thin tips
     and the germ survive (the original reason SAM was adopted). Brightness (not
     SAM) rejects the floor -- so the SAM floor "drips" cannot occur.

The bright plastic wall/rim CANNOT be separated from the kernel by any pixel cue
(equally bright, ~7deg apart spectrally, often spatially fused), so we do NOT try
to bound the mask geometrically -- a fixed inset / floor-hull bound silently
CLIPS kernels that ride the wall, biasing every spectrum. Instead we keep the
complete kernel and DETECT likely rim/wall contamination for review:

  FLAGS (diagnostics, never alter the mask):
  * floor = largest dark component (the one wall/rim-free region).
  * outside_floor_frac = mask fraction lying outside the floor's (dilated) convex
    hull -- a rim slash across a corner shows up as a large value; a clean kernel
    barely poking past its floor reads low.
  * border_touch_frac, floor_frac -- corroborating signals.
extract_cells thresholds these into a per-cell `rim_suspect` flag, surfaced in
the metadata and coloured orange in the overlay -- contamination is flagged, not
hidden, and never silently trimmed.

Entry point: segment(crop, well_mask, **params) -> KernelSeg.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class KernelSeg:
    mask: np.ndarray            # (H,W) bool: the kernel (complete, never clipped)
    floor: np.ndarray           # (H,W) bool: the dark well floor (diagnostic)
    thr: float                  # brightness threshold used
    found: bool                 # a kernel was segmented (vs empty/low-contrast well)
    floor_frac: float           # floor px / well px  (low -> rim ate the floor)
    border_touch_frac: float    # mask px on the well border ring (high -> rim/wall bleed)
    outside_floor_frac: float   # mask px outside the floor's hull (high -> rim/wall bleed)


def _largest(mask):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lbl == biggest


def _fill_holes(mask):
    """Fill interior holes (flood the background from a corner; unreached = hole)."""
    h, w = mask.shape
    ff = mask.astype(np.uint8).copy()
    scratch = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, scratch, (0, 0), 1)
    return mask | (ff == 0)


def _otsu_threshold(values):
    """Otsu split of a 1-D brightness sample, robustly scaled to [p1, p99]."""
    lo, hi = np.percentile(values, 1), np.percentile(values, 99)
    span = hi - lo
    if span < 1e-9:
        return float(hi)
    q = np.clip((values - lo) / span * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + (t / 255.0) * span)


def _floor_hull(floor, shape):
    """Convex hull of the floor as a filled mask (the well's wall-free interior)."""
    H, W = shape
    ys, xs = np.where(floor)
    if len(xs) < 10:
        return floor.copy()
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    filled = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool) | floor


def segment(crop, well_mask, min_contrast=1.6, border_px=4, hull_dilate=6):
    """Brightness segmentation of one cell -> complete kernel mask + rim flags.

    crop: (H,W,B) float; well_mask: (H,W) bool (the inset cell rectangle).
    Returns a KernelSeg. `min_contrast` is the bright-mode / floor-mode ratio
    below which the well is declared empty (guards a kernel-free cell). The mask
    is NEVER geometrically bounded/eroded; `hull_dilate` only sizes the floor
    hull used to MEASURE rim contamination (outside_floor_frac).
    """
    H, W = well_mask.shape
    empty = KernelSeg(np.zeros((H, W), bool), np.zeros((H, W), bool),
                      0.0, False, 0.0, 0.0, 0.0)
    if well_mask.sum() < 20:
        return empty

    b = crop.mean(-1).astype(np.float32)
    inw = b[well_mask]
    thr = _otsu_threshold(inw)
    bright = (b > thr) & well_mask
    dark = (~bright) & well_mask
    if bright.sum() < 20 or dark.sum() < 20:
        return empty

    # emptiness / contrast guard: the bright mode must clearly beat the floor
    if np.median(b[bright]) < min_contrast * (np.median(b[dark]) + 1e-9):
        return empty

    floor = _largest(dark)
    floor_frac = float(floor.sum()) / float(well_mask.sum())

    # complete kernel: brightest blob, holes filled, NEVER eroded (tips/germ kept)
    km = _largest(bright)
    km = _fill_holes(km)
    km = _largest(km) & well_mask
    if km.sum() < 20:
        return KernelSeg(np.zeros((H, W), bool), floor, thr, False, floor_frac, 0.0, 0.0)

    # rim/wall contamination diagnostics (do NOT alter the mask)
    er = cv2.erode(well_mask.astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * border_px + 1,) * 2))
    border = well_mask & (~er.astype(bool))
    border_touch = float((km & border).sum()) / float(km.sum())

    interior = _floor_hull(floor, (H, W))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hull_dilate + 1,) * 2)
    interior = cv2.dilate(interior.astype(np.uint8), k).astype(bool)
    outside_floor = float((km & ~interior).sum()) / float(km.sum())

    return KernelSeg(km, floor, thr, True, floor_frac, border_touch, outside_floor)
