"""Cube + cell quad + COCO mask -> one fixed-size kernel patch.

WHY A WARP AND NOT A CROP
Each cell arrives as four corners ordered in PLATE coordinates, so warping them
onto a fixed rectangle does four jobs at once: it removes the plate's rotation
(a few degrees, different every capture), it removes the dorsal/ventral mirror,
it puts reflectance and transmittance on a common grid, and it makes an 0 h
patch pixel-comparable with an 8 h one. An axis-aligned crop does none of those.

The COCO mask goes through the *same* homography, so `patch` and `mask` are
registered to each other by construction rather than by assumption.

FRAMES
The cube is (width, n_lines, bands) and the COCO mask is (width, n_lines) --
the annotation PNGs were generated in cube layout precisely so a mask indexes
the cube with no rotation. gridfit's cell corners live in the *working* frame,
(n_lines, width), where `working[r, c] = cube[width - 1 - c, r]`. Everything
below moves into the working frame to warp, and `verify_frame` checks that move
against gridfit's own reference implementation on real data before any run
writes a byte.
"""
import sys

import cv2
import numpy as np

import config
from . import coco

sys.path.insert(0, str(config.COLLECTION))
from gridfit import render  # noqa: E402


def verify_frame(cube, width):
    """The bbox-slice-and-transpose below must equal render.to_working exactly.

    It is three index tricks deep and a silent error would mirror every patch,
    which is precisely the failure the whole pipeline exists to prevent. So it
    is checked against the reference implementation on real data, once per run.
    """
    x0, x1, y0, y1 = 40, 90, 60, 120
    full = render.to_working(np.asarray(cube[:, :, config.BAND_LO], np.float32))
    mine = np.asarray(cube[width - x1:width - x0, y0:y1, config.BAND_LO], np.float32)[::-1].T
    if not np.array_equal(mine, full[y0:y1, x0:x1]):
        raise SystemExit("working-frame slice disagrees with render.to_working -- "
                         "patch orientation cannot be trusted, refusing to write")


def quad(cell, scale=None):
    """The cell's four corners, scaled about its centre. See config.CELL_SCALE."""
    s = config.CELL_SCALE if scale is None else scale
    ctr = np.asarray(cell["center"], float)
    return ctr + (np.asarray(cell["corners"], float) - ctr) * s


def bbox(corners, width, n_lines):
    """Clamped working-frame bounding box of a cell quad, or None if degenerate.

    Depends only on the quad and the capture's shape, both of which are known
    from cells.json -- so `build_dataset` can decide up front exactly how many
    rows it will write, and size its memmaps once.
    """
    q = np.asarray(corners, float)
    x0 = int(max(np.floor(q[:, 0].min()) - 2, 0))
    y0 = int(max(np.floor(q[:, 1].min()) - 2, 0))
    x1 = int(min(np.ceil(q[:, 0].max()) + 2, width))
    y1 = int(min(np.ceil(q[:, 1].max()) + 2, n_lines))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, x1, y0, y1


def absorbance(x, eps=config.ABSORBANCE_EPS):
    """Pseudo-absorbance, per pixel per band: A = -log10(x).

    The floor is not cosmetic. Corrected transmittance reaches -0.0005 on real
    captures and a reflectance voxel can land on exactly 0 after dark
    subtraction; both make log10 undefined. Values above 1.0 (2.8% of
    reflectance voxels, the known tape clipping) give small negative
    absorbances -- those are meaningful and are kept.

    NaN in, NaN out: an invalid voxel must stay invalid.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.maximum(x, eps))


def _warp_stack(img, M, size):
    """warpPerspective over an (h, w, C) stack -- cv2 takes at most 4 channels."""
    out = np.empty((size[1], size[0], img.shape[2]), np.float32)
    for i in range(0, img.shape[2], 4):
        chunk = img[:, :, i:i + 4]
        out[:, :, i:i + 4] = cv2.warpPerspective(
            chunk, M, size, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan).reshape(
                size[1], size[0], chunk.shape[2])
    return out


def cut(cube, sat, mask_cube, width, corners):
    """One kernel -> (patch, mask, stats) or (None, None, reason).

    patch  (PATCH_H, PATCH_W, N_BANDS) float32 pseudo-absorbance, NaN where invalid
    mask   (PATCH_H, PATCH_W) bool, the annotator's mask through the same warp

    Only the cell's bounding box is pulled off the memmap, so a 0.6 GB cube
    costs about 12 MB per kernel rather than a full transpose.
    """
    q = np.asarray(corners, float)
    box = bbox(q, width, cube.shape[1])
    if box is None:
        return None, None, "cell falls outside the frame"
    x0, x1, y0, y1 = box

    lo, hi = config.BAND_LO, config.BAND_HI
    # np.array, not np.asarray: the cube is already float32, so asarray hands
    # back a read-only view of the memmap and invalidating railed voxels below
    # would fail. The copy is the cell's bounding box only, ~12 MB.
    sub = np.array(cube[width - x1:width - x0, y0:y1, lo:hi], np.float32)
    work = sub[::-1].transpose(1, 0, 2)

    # Railed voxels are invalidated BEFORE the log: a clipped voxel carries no
    # information about how bright it really was, and -log10 of a clipped value
    # is a confident wrong number rather than an obviously missing one.
    sat_frac = 0.0
    if sat is not None:
        s = np.asarray(sat[width - x1:width - x0, y0:y1, lo:hi])[::-1].transpose(1, 0, 2)
        sat_frac = float(s.mean())
        work[s] = np.nan

    finite = np.isfinite(work)
    clipped = float((work[finite] <= config.ABSORBANCE_EPS).mean()) if finite.any() else 0.0
    work = absorbance(work)

    dst = np.float32([[0, 0], [config.PATCH_W, 0],
                      [config.PATCH_W, config.PATCH_H], [0, config.PATCH_H]])
    M = cv2.getPerspectiveTransform(np.float32(q - [x0, y0]), dst)
    patch = _warp_stack(work, M, (config.PATCH_W, config.PATCH_H))

    mwork = mask_cube[width - x1:width - x0, y0:y1][::-1].T
    mask = cv2.warpPerspective(
        mwork.astype(np.uint8), M, (config.PATCH_W, config.PATCH_H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    # Nearest-neighbour resampling can pinch a thin neck and leave a stray islet.
    # The source mask is one component by construction, so the warped one must be
    # too -- otherwise the patch mask disagrees with the mask it came from.
    mask, _, _ = coco.keep_largest_component(mask)

    stats = {
        "sat_frac": sat_frac,
        "clipped_frac": clipped,
        "nan_frac": float(np.isnan(patch).mean()),
        "patch_mask_px": int(mask.sum()),
    }
    return patch, mask, stats


def mask_mean_spectrum(patch, mask):
    """-> (N_BANDS,) float32 mean pseudo-absorbance over the kernel's pixels.

    nanmean over the masked pixels, so an invalid voxel costs that band one
    sample rather than poisoning the whole spectrum. A band with no valid pixel
    at all comes back NaN and is caught by verify_dataset.
    """
    if not mask.any():
        return np.full(patch.shape[2], np.nan, np.float32)
    with np.errstate(invalid="ignore"):
        sel = patch[mask]
        out = np.where(np.isfinite(sel).any(axis=0), np.nanmean(sel, axis=0), np.nan)
    return out.astype(np.float32)
