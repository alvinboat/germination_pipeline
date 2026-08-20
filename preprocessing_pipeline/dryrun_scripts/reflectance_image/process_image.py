"""
End-to-end reflectance correction: stitch -> dark frame -> dark subtraction
-> per-row white correction -> checkerboard geometry correction -> greyscale
intensity preview -> ArUco-anchored plate grid + per-cell masks. Stages 1-8.

Stage 1 -- stitching
---------------------
Unpacks per-line Mono12Packed .bin captures into a (width, n_lines, channels)
uint16 cube. Mirrors pipeline/stitch_grain.py; reuses the same hsi_save_load
codec (see loadstich/hsi_save_load.py -- the byte-packing math is subtle, do
not reimplement it here).

Stage 2 -- dark frame
----------------------
dark_frame() collapses the dark reference cube to one (width, channels) frame
by averaging across scan lines. The push-broom sensor's dark current/offset
is fixed per (spatial pixel, band), not per scene, so a single averaged frame
is the correction target for every line of the grain scan later.

Stage 3 -- dark subtraction
------------------------------
apply_dark_correction() subtracts the (width, channels) dark frame from every
one of the grain cube's lines (broadcast over the scan-line axis), clipping
negative results to 0. This is the (r0-D) half of the (r0-D)/(W-D) reflectance
formula; the white half (W-D) is added in stages 4-5.

Stage 4 -- tape-strip detection
----------------------------------
The two teflon tape strips are captured at the start/end of the scan, so
they show up as bright blobs near the two ends of the scan-line axis, not
at fixed width positions. Each blob is irregular and doesn't span the full
640px width, so detection recovers each blob's true per-row extent.

Stage 5 -- per-row white reference + ratio correction
----------------------------------------------------------
Builds a full (width, channels) map: white_ref[row, band] is the 75th
percentile of that row's dark-subtracted tape pixels (both blobs pooled).
Only rows holding a reliable share of tape are measured -- a blob's taper
rows keep a few part-background pixels that read dim, and trusting them
biases the white low and sends reflectance past 1. Those rows, and rows with
no coverage at all, are filled by per-band linear interpolation across width;
reflectance = darksub / white_ref, i.e. (r0-D)/(W-D).

Stage 6 -- checkerboard scan-axis geometry correction
------------------------------------------------------------
The push-broom scan axis (n_lines) is stretched relative to the optical
spatial axis (640px) by however much faster/slower the stage moved than
frame rate, and a checkerboard of known cell geometry gives the factor that
undoes it.

Everything is detected and measured at native resolution with the classic
cv2 detector plus sub-pixel refinement; every board in frame is measured
independently, judged against the lattice model, and only then pooled. The
reasoning behind each of those choices -- and why findChessboardCornersSB
and the old scan-precompression search are both deliberately absent -- is
recorded at the top of the stage 6 section.

Two things this stage cannot determine on its own, exposed as explicit
inputs rather than assumed away:
  --cell-size    the target's true physical cell, if it is not square. The
                 measurement is of the TARGET; if the printed cell is not
                 square the correction inherits that error exactly. Defaults
                 to square, i.e. a no-op.
  --plane-factor the kernels are not on the checkerboards' plane, so the
                 measured factor is wrong for the sample by a fixed ratio.
                 Defaults to DEFAULT_PLANE_FACTOR, a constant measured for
                 this rig -- read that comment before trusting it on another.
Both are echoed in the log on every run, including when they are no-ops.
Note they are two names for one multiplier: setting both double-counts.

Stage 7 -- intensity preview
-------------------------------
intensity_preview() collapses the final corrected cube to a single greyscale
image (mean across all 224 bands, stretched between the 1st and 99th
percentile), for a quick look at the corrected result without needing a
wavelength calibration. Percentiles, not min/max: the checkerboards' paper
backing out-reflects the teflon tape the cube is referenced to, so it sits
above 1.0 and would otherwise claim the whole white point.

Stage 8 -- plate grid + per-cell masks
--------------------------------------------
Fits the rigid 4x7 well plate to the corrected cube, anchored on the dish's two
ArUco fiducials (gridfit/, vendored from grid_sandbox), annotates the preview
PNG with the fitted lattice, and writes one boolean mask per kernel cell.

The point is addressability, not detection: because the lattice is placed by the
markers and the plate model is fixed, cell index N denotes the same physical well
on every capture whatever the plate's rotation -- so the same kernel can be
followed from day0 to day1 by index alone. The fit is gated (pass /
needs_review / fail) and a failed fit writes no masks rather than plausible
wrong ones.

Segmenting the kernel WITHIN a cell is a separate problem and is not attempted
here; these masks are the cell/well regions.

eBUS Player saves a session's .bin files flat into raw_image_bin/ (no
per-session subfolder). Run:
    python3 process_image.py day2_dish3_ref_2500

This stitches/corrects everything currently in raw_image_bin/, writes
corrected_file/day2_dish3_ref_2500.npy,
corrected_image/day2_dish3_ref_2500.png (with the grid overlay) and
corrected_cells/day2_dish3_ref_2500/cell_0..cell_21.npy + cells.json, then moves
those .bin files into raw_image_storage/day2_dish3_ref_2500/ -- leaving
raw_image_bin/ empty and ready for the next capture.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# hsi_save_load's own import path assumes a wider jarvis_gui package not
# present in this checkout, so import the local copy directly instead.
sys.path.insert(0, str(Path(__file__).resolve().parent / "loadstich"))
from hsi_save_load import load_hsi  # noqa: E402

# gridfit/ is a vendored, library-only copy of grid_sandbox's marker localiser +
# plate-grid fitter (see gridfit/fit_grid.py's header). Imported by path the same
# way loadstich is, rather than as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "gridfit"))
import fit_grid  # noqa: E402

WIDTH = 640       # spatial pixels per line
CHANNELS = 224    # spectral bands per line

DEFAULT_DARK = "raw_dark/dark_3k"  # same dark reference every session

# Scan-axis multiplier applied on top of the checkerboard measurement, because the
# checkerboards and the kernels are not on the same plane. THIS IS A CONSTANT OF
# THIS RIG, not a property of the algorithm.
#
# Measured, not assumed: pooling the kernel well-lattice pitch over the day0
# captures in BOTH plate orientations cancels the plate's own pitch ratio and
# leaves a residual scan/spatial scale of 0.9165, i.e. this factor. It
# cross-checks against the dish rim (which independently implies 1.096) to 0.5%,
# and it is a fixed number rather than a per-capture one -- unchanged across a 50%
# conveyor-speed difference and a 90deg plate rotation. Applying it brings the
# dish rim from 0.888-0.901 to 0.990-0.998 of round on all five captures.
#
# What it is NOT: an explanation. It corrects the sample plane empirically, but
# whether the underlying cause is the target sitting at a different height or the
# printed cells not being square is undetermined -- the two are indistinguishable
# from a single capture, and both produce exactly this constant, uniform,
# axis-aligned error. Settle it with calipers on a printed cell, or by rotating a
# board 90deg on the baseplate and seeing whether the measured ratio flips.
#
# Re-derive it if the camera height, the dish/insert stack, or the targets change.
# Pass --plane-factor 1.0 to disable it and get the raw board measurement.
DEFAULT_PLANE_FACTOR = 1.091
RAW_IMAGE_DIR = "raw_image"        # eBUS Player saves flat into here, no subfolder
STORAGE_DIR = "raw_image_storage"  # processed .bin files archived to <STORAGE_DIR>/<sample>/
CELLS_DIR = "corrected_cells"      # per-cell masks written to <CELLS_DIR>/<sample>/


# ---------------------------------------------------------------- stage 1 --
def load_line(raw, w=WIDTH, c=CHANNELS):
    """Unpack one raw Mono12Packed line into an oriented (w, c) uint16 frame.

    A corrupt/unparseable line becomes a zero frame instead of aborting the
    whole stitch (mirrors pipeline/stitch_grain.py:load_line).
    """
    try:
        line = load_hsi(raw)
        return line.reshape([c, w]).swapaxes(0, 1)[::-1]
    except Exception:
        return np.zeros((w, c), dtype=np.uint16)


def stitch(directory, w=WIDTH, c=CHANNELS):
    """Stitch a directory of per-line .bin captures into a (w, n_lines, c) cube."""
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise ValueError(f"no .bin files found in {directory}")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    bad = 0
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        frame = load_line(raw, w, c)
        if not frame.any():
            bad += 1
        cube[:, i, :] = frame
    print(f"  {directory.name}: {len(files)} lines stitched ({bad} blank/bad).")
    return cube


# ---------------------------------------------------------------- stage 2 --
def dark_frame(dark_cube):
    """Mean (width, channels) dark frame, averaged across the dark cube's lines."""
    return dark_cube.mean(axis=1).astype(np.float64)


# ---------------------------------------------------------------- stage 3 --
def apply_dark_correction(grain_cube, dark_frame):
    """clip(grain - dark_frame, 0), broadcasting dark_frame over every scan line.

    Subtracts/clips in-place into one float32 buffer rather than allocating
    several full-cube temporaries (each is ~5GB on an 8500-line cube).
    Returns (corrected (grain_cube.dtype), n_clipped).
    """
    diff = grain_cube.astype(np.float32)
    diff -= dark_frame[:, None, :].astype(np.float32)
    n_clipped = int((diff < 0).sum())
    corrected = np.clip(diff, 0, None, out=diff).astype(grain_cube.dtype)
    return corrected, n_clipped


# ---------------------------------------------------------------- stage 4 --
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

    Two-stage locate/refine: a single percentile or Otsu pass either merges
    the tape with the dish or clips the blob's dimmer edges.
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
        print(f"  warning: only {len(candidates)} tape candidate(s) >= "
              f"{min_area_frac*100:.0f}% of frame area found (wanted {k}); "
              f"continuing with what's available.")
    candidates.sort(key=lambda lab: stats[lab, cv2.CC_STAT_AREA], reverse=True)
    top = sorted(candidates[:k], key=lambda lab: stats[lab, cv2.CC_STAT_LEFT])  # left-to-right

    refined = []
    for lab in top:
        x, y, w, h, _ = stats[lab]
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        _, crop_mask = cv2.threshold(g8[:, x0:x1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local = _largest_component(crop_mask, open_k=5)
        if local is None:
            print(f"  warning: stage 2 refinement failed for tape blob at columns "
                  f"[{x0}:{x1}]; skipping it.")
            continue
        full = np.zeros((H, W), bool)
        full[:, x0:x1] = local
        refined.append(full)
    return refined


def tape_column_span(masks):
    """(c0, c1) inclusive scan-line bounds from the LEFT blob's leftmost
    column to the RIGHT blob's rightmost column, across both masks pooled.

    Everything outside is pre/post-scan padding, not tape/dish/kernel.
    Returns None if no tape blobs were found at all.
    """
    if not masks:
        return None
    combined = np.logical_or.reduce(masks)
    cols = np.nonzero(combined.any(axis=0))[0]
    return int(cols.min()), int(cols.max())


# ---------------------------------------------------------------- stage 5 --
def build_white_reference(darksub_cube, masks, pct=75.0, min_coverage_frac=0.75):
    """Per-row (width, channels) white reference from the tape blob masks.

    For each width row whose tape pixel count is reliable, white_ref[row, band]
    is the pct-th percentile of that row's dark-subtracted tape pixels (both
    masks pooled where a row falls in both). A row's count is reliable if it is
    at least `min_coverage_frac` of the median nonzero count.

    That reliability test is not cosmetic. A blob tapers off over several rows
    before its coverage reaches zero, and those boundary rows hold only a
    handful of pixels -- pixels which are themselves part tape, part background,
    so they read dim. Their percentile is both noisy and biased low, and taking
    it as known white makes reflectance = darksub/white_ref overshoot by however
    much the bias is. On the day0/day1 captures the left blob tapers across
    spatial rows 0-5 (111 px in row 0 against ~300 in the interior) and its
    white came out up to 30% low there, sending reflectance to 1.33 on a scene
    the rest of the frame put at 1.00 -- roughly 1500 px, enough to hijack any
    min-max stretch downstream (see intensity_preview).

    Demoting those rows also protects every row past them: np.interp
    flat-extrapolates beyond the last known row, so one bad boundary row would
    otherwise be inherited by the whole uncovered run beyond it instead of the
    stable interior plateau.

    Rows with no reliable coverage (including the demoted taper rows) are filled
    by per-band linear interpolation across the width axis from the nearest
    reliable rows, flat-extrapolated at either edge via np.interp's default clamp.

    Returns (white_ref (width, channels) float64, reliable_rows bool (width,)).
    """
    H, C = darksub_cube.shape[0], darksub_cube.shape[2]
    if masks:
        combined = np.logical_or.reduce(masks)  # (width, n_lines)
        counts = combined.sum(axis=1)           # (width,)
        covered = counts > 0
    else:
        counts = np.zeros(H, dtype=np.int64)
        covered = np.zeros(H, dtype=bool)
    if not covered.any():
        print("  warning: no tape coverage found in any width row; skipping white "
              "correction (white reference set to 1.0 everywhere).")
        return np.ones((H, C), dtype=np.float64), covered

    reliable = covered & (counts >= min_coverage_frac * np.median(counts[covered]))
    n_demoted = int((covered & ~reliable).sum())
    if n_demoted:
        print(f"  {n_demoted} width row(s) had tape coverage but below "
              f"{min_coverage_frac:g}x the median count (blob taper); interpolated "
              f"rather than trusted.")
    if not reliable.any():
        print("  warning: no width row met the tape coverage threshold; falling back to "
              "every covered row (white reference may read low at the blob edges).")
        reliable = covered

    white_ref = np.full((H, C), np.nan, dtype=np.float64)
    for r in np.nonzero(reliable)[0]:
        cols = combined[r]
        white_ref[r] = np.percentile(np.asarray(darksub_cube[r, cols, :]), pct, axis=0)

    known_rows = np.nonzero(reliable)[0]
    unknown_rows = np.nonzero(~reliable)[0]
    if unknown_rows.size:
        for b in range(C):
            white_ref[unknown_rows, b] = np.interp(unknown_rows, known_rows, white_ref[known_rows, b])

    return white_ref, reliable


def apply_white_correction(darksub_cube, white_ref):
    """reflectance[row,...,b] = darksub[row,...,b] / white_ref[row,b].

    Both sides are already dark-subtracted, so this is exactly (r0-D)/(W-D).
    Divides in-place (/=) rather than allocating a second full-cube temporary.
    """
    reflectance = darksub_cube.astype(np.float32)
    reflectance /= white_ref.astype(np.float32)[:, None, :]
    return reflectance


# ---------------------------------------------------------------- stage 6 --
# Corner detection is done with the CLASSIC cv2.findChessboardCorners plus
# cv2.cornerSubPix, and deliberately NOT with findChessboardCornersSB. Benchmarked
# against synthetically rendered boards of known cell size, SB's measured axis
# ratio is off by -7% at 2.1:1 anisotropy and -15% at 3.2:1, and its detection
# rate collapses above ~6:1; the classic detector plus sub-pixel refinement stays
# within 0.16% from 1:1 all the way to 14:1 and detected 6/6 at every step. The
# scan axis here is routinely oversampled 2-4x, which is squarely in the range
# where SB fails, so it is not used at any stage.
#
# For the same reason there is no scan-axis "pre-compression" search. That existed
# to make the board look square enough to detect; with the classic detector it is
# unnecessary, and resampling before measuring can only discard scan-axis
# information. Everything is detected and measured at native resolution.
CHECKERBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

# Acceptance thresholds, set from measured behaviour on the day0 captures: real
# boards fit the lattice model at rms 0.17-0.51px with 1.3-2.1deg of skew, while
# spurious "boards" the detectors find in the kernel well grid sit at rms 4-11px.
# The gap is nearly an order of magnitude, so these sit well clear of both ends.
FIT_RMS_WARN_PX = 0.6
FIT_RMS_REJECT_PX = 1.5
SKEW_WARN_DEG = 3.0
SKEW_REJECT_DEG = 10.0
BOARD_DISAGREE_WARN = 0.02      # fractional spread across boards worth flagging
FX_SANITY_RANGE = (0.02, 50.0)  # a correction outside this is a bug, not a measurement
TILE_WINDOWS = (160, 320, 640)  # fallback sweep: window sizes, each stepped at 50% overlap


def render_detection_gray(cube, band=None, lo_pct=0.5, hi_pct=99.5):
    """(spatial, scan) uint8 render of a cube, for corner detection.

    Mean across bands unless a single band is requested -- averaging 224 bands
    is the cheapest available SNR gain and corner localisation is noise-limited.

    Contrast is stretched on percentiles rather than min/max: a single saturated
    speck or dead pixel sets the min/max range and crushes the actual scene into
    a handful of grey levels, which is exactly what the detector's adaptive
    threshold cannot work with.
    """
    img = cube.mean(axis=2) if band is None else np.asarray(cube[:, :, band])
    img = np.asarray(img, dtype=np.float32)
    lo, hi = (float(v) for v in np.percentile(img, [lo_pct, hi_pct]))
    if not hi > lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _ideal_lattice(inner_size):
    """(N,2) (column, row) index of each inner corner, in cv2's return order.

    findChessboardCorners returns row-major for a (cols, rows) pattern, so the
    row index is the slow axis.
    """
    cols, rows = inner_size
    j, i = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.column_stack([i.ravel(), j.ravel()]).astype(np.float64)


def fit_lattice(corners, inner_size, cell_size=(1.0, 1.0)):
    """Measure the per-image-axis scale from one board's corners.

    The board is a rigid grid in the object plane, so the map from lattice index
    to image pixel is

        A = diag(s_scan, s_spatial) @ R(theta) @ diag(cell_w, cell_h)

    i.e. the physical cell, an unknown in-plane rotation, and the axis-aligned
    image sampling we are actually after. Fit A by least squares, divide out the
    cell to get B, and then

        B @ B.T = diag(s_scan**2, s_spatial**2)

    so the two scales are the ROW norms of B. This is the whole point: the row
    norms are rotation-invariant, whereas the corner-to-corner step lengths are
    the COLUMN norms of B and are not. The two agree only when the board happens
    to be axis-aligned in the image, so measuring steps silently biases the
    result toward isotropy on any rotated board -- at 2.2:1 anisotropy a 15deg
    rotation is already a ~10% error.

    B @ B.T being diagonal is also the model's own consistency check. The
    off-diagonal term is zero for ANY rotated rigid grid under axis-aligned
    scaling, so a non-zero "skew" means the premise is broken: tilt, perspective,
    or a target whose cells are parallelograms rather than rectangles.

    cell_size is the physical cell extent along the board's own two lattice axes
    (any unit). At the default (1, 1) the scales come out in pixels-per-cell and
    only their ratio is meaningful, which is all the correction needs.
    """
    corners = np.asarray(corners, dtype=np.float64)
    lattice = _ideal_lattice(inner_size)

    def _fit(lat):
        design = np.column_stack([lat, np.ones(len(lat))])
        coef, *_ = np.linalg.lstsq(design, corners, rcond=None)
        return coef[:2].T, np.linalg.norm(design @ coef - corners, axis=1)

    A, residual = _fit(lattice)
    # Canonicalise the labelling. The detector may start from any corner, so a
    # square board comes back in any of four rotations -- board 2 here reports
    # +0.85deg in one capture and +102deg in the next, same board. That is
    # harmless for the row norms, which are rotation-invariant, but cell_size is
    # indexed BY LATTICE AXIS, so without a fixed convention a non-square cell
    # would be applied to different physical directions on different boards.
    # Convention: lattice axis 0 is whichever board direction runs nearest the
    # image x (scan) axis, so cell_size is always (along-scan, along-spatial).
    if abs(A[0, 1]) > abs(A[0, 0]):
        lattice = lattice[:, ::-1].copy()
        A, residual = _fit(lattice)
    ambiguous = abs(abs(A[0, 1]) - abs(A[0, 0])) < 0.1 * abs(A[0, 0])
    if A[0, 0] < 0:
        # a 180deg relabelling: both lattice axes reversed, so each still runs
        # along the same physical direction and cell_size is unaffected. Undone
        # only so the reported rotation reads as ~0 rather than ~180 on a board
        # the detector happened to start from the opposite corner.
        lattice = -lattice
        A, residual = _fit(lattice)

    cell = np.asarray(cell_size, dtype=np.float64)
    if np.any(cell <= 0):
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")
    B = A / cell[None, :]
    G = B @ B.T
    px_scan, px_spatial = float(np.sqrt(G[0, 0])), float(np.sqrt(G[1, 1]))
    denom = px_scan * px_spatial
    skew = float(np.degrees(np.arcsin(np.clip(G[0, 1] / denom, -1.0, 1.0)))) if denom > 0 else float("inf")

    step = A @ A.T   # pixels per lattice step, before the cell is divided out
    return {
        "px_scan": px_scan,
        "px_spatial": px_spatial,
        "spacing_px": (float(np.sqrt(step[0, 0])), float(np.sqrt(step[1, 1]))),
        "theta_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
        "skew_deg": skew,
        "axis_ambiguous": bool(ambiguous),
        "rms_px": float(np.sqrt((residual ** 2).mean())),
        "max_resid_px": float(residual.max()),
        "centre": corners.mean(axis=0),
        "corners": corners,
    }


def _subpix_window(spacing_px):
    """cornerSubPix half-window, sized per axis from the real corner spacing.

    The window must stay clear of the neighbouring corners or it integrates
    their gradients as well. The two axes are sampled very differently here (the
    scan axis is oversampled several-fold), so a single fixed value is
    necessarily either wasteful on one axis or oversized on the other -- at
    ~15px spatial spacing the stock (11, 11) window is wider than the gap
    between corners.
    """
    return tuple(int(max(2, min(11, np.floor(s / 2.0) - 1))) for s in spacing_px)


def _detect_raw(gray, inner_size):
    """One board's inner corners via the classic detector, or None.

    Wrapped because OpenCV raises rather than returning False on degenerate
    input -- adaptiveThreshold rejects any frame only a few pixels across.
    """
    try:
        ok, corners = cv2.findChessboardCorners(gray, tuple(inner_size), CHECKERBOARD_FLAGS)
    except cv2.error:
        return None
    if not ok or corners is None or len(corners) != inner_size[0] * inner_size[1]:
        return None
    return corners.reshape(-1, 2).astype(np.float64)


def _measure(gray, corners_raw, inner_size, cell_size):
    """Refine raw corners sub-pixel and measure them. None if unusable.

    Refinement is checked, not trusted: cornerSubPix can walk a corner onto a
    neighbouring feature, and it reports no error when it does. A corner that
    moves further than its own search window has not been refined, it has been
    relocated, so the unrefined detection is kept instead.
    """
    prelim = fit_lattice(corners_raw, inner_size, cell_size)
    if not np.all(np.isfinite(prelim["spacing_px"])) or min(prelim["spacing_px"]) < 2.0:
        return None                      # collapsed grid: all corners on top of each other

    win = _subpix_window(prelim["spacing_px"])
    refined = cv2.cornerSubPix(gray, corners_raw.astype(np.float32).reshape(-1, 1, 2),
                               win, (-1, -1), SUBPIX_CRITERIA).reshape(-1, 2).astype(np.float64)
    moved = np.linalg.norm(refined - corners_raw, axis=1)
    if np.any(moved > np.hypot(*win)):
        board = prelim
        board["refined"] = False
    else:
        board = fit_lattice(refined, inner_size, cell_size)
        board["refined"] = True
    board["subpix_win"] = win
    return board


def _grow_hull(corners, margin):
    """Convex hull of the corners pushed `margin` px outward from its centroid."""
    hull = cv2.convexHull(np.asarray(corners, dtype=np.float32)).reshape(-1, 2)
    centre = hull.mean(axis=0)
    radial = hull - centre
    norm = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    return (hull + radial / norm * margin).astype(np.int32)


def _mask_board(gray, board):
    """Erase a measured board so the next pass finds a different one.

    Fills the corner convex hull grown by one cell -- the detected corners are
    the INNER ones, so the board's outer squares sit about a cell beyond them.
    A convex hull rather than a bounding box, because a box around a rotated
    board over-erases the corners of the frame region and can swallow a close
    neighbour. Filled with the frame median rather than 0 so the patch does not
    become a high-contrast rectangle with detectable corners of its own.
    """
    out = gray.copy()
    cv2.fillConvexPoly(out, _grow_hull(board["corners"], max(board["spacing_px"])),
                       int(np.median(gray)))
    return out


def _judge(board):
    """(accepted, reason) -- is this a real board, well enough measured to use?"""
    if board["rms_px"] > FIT_RMS_REJECT_PX:
        return False, f"lattice residual {board['rms_px']:.2f}px > {FIT_RMS_REJECT_PX}px"
    if abs(board["skew_deg"]) > SKEW_REJECT_DEG:
        return False, f"skew {board['skew_deg']:+.1f}deg > {SKEW_REJECT_DEG}deg"
    if min(board["px_scan"], board["px_spatial"]) <= 0:
        return False, "non-positive axis scale"
    return True, ""


def detect_boards(gray, inner_size, max_boards, cell_size=(1.0, 1.0), use_tiles=True):
    """Every checkerboard in the frame, each measured independently.

    Two passes, both at native resolution:

    1. Whole frame: detect -> measure -> judge -> erase -> repeat. One call only
       ever returns one board, so erasing is what makes the rest findable. This
       alone finds both targets on all the day0 captures.
    2. If pass 1 came up short, a deterministic tiled sweep at several window
       sizes with 50% overlap. Whole-frame detection is context-sensitive -- the
       adaptive threshold has to cope with the dish, the kernel grid and the tape
       strips at once -- and a board that is missed globally is usually found
       immediately once it is the dominant structure in its own window. The
       sweep is bounded and exhaustive rather than an adaptive search whose
       termination depends on a heuristic.

    Rejected candidates are reported, not silently dropped: a detection failing
    the lattice check is how a spurious grid announces itself.
    """
    boards, rejected = [], []

    def _seen(board, others):
        """Same physical board as one already recorded? Keyed on centroid, since a
        board re-found from an overlapping tile lands within a pixel or two."""
        return any(np.linalg.norm(np.asarray(o["centre"]) - board["centre"])
                   < max(board["spacing_px"]) for o in others)

    def register(board, source):
        if _seen(board, boards):
            return False
        ok, reason = _judge(board)
        if not ok:
            # dedupe rejects too: the tiled sweep revisits every location several
            # times over, and one bad candidate must not fill the log with copies
            if not _seen(board, (r[0] for r in rejected)):
                board["source"] = source
                rejected.append((board, source, reason))
            return False
        board["source"] = source
        boards.append(board)
        return True

    work = gray.copy()
    for _ in range(max_boards + 2):   # +2: room to erase rejects and keep looking
        raw = _detect_raw(work, inner_size)
        if raw is None:
            break
        board = _measure(gray, raw, inner_size, cell_size)
        if board is None:
            # a collapsed grid still has to be erased, or the next pass finds it
            # again and the loop stalls on it instead of reaching the real boards
            work = _mask_board(work, {"corners": raw, "spacing_px": (2.0, 2.0)})
            continue
        register(board, "whole-frame")
        work = _mask_board(work, board)
        if len(boards) >= max_boards:
            break

    if use_tiles and len(boards) < max_boards:
        h, w = gray.shape
        for win in TILE_WINDOWS:
            step = max(1, win // 2)
            for y0 in range(0, max(1, h - step), step):
                for x0 in range(0, max(1, w - step), step):
                    tile = gray[y0:min(h, y0 + win), x0:min(w, x0 + win)]
                    if min(tile.shape) < 40:
                        continue
                    raw = _detect_raw(tile, inner_size)
                    if raw is None:
                        continue
                    board = _measure(gray, raw + [x0, y0], inner_size, cell_size)
                    if board is not None:
                        register(board, f"tile{win}")
                if len(boards) >= max_boards:
                    break
            if len(boards) >= max_boards:
                break

    boards.sort(key=lambda b: (b["centre"][1], b["centre"][0]))   # top-to-bottom
    return boards, rejected


def reconcile_boards(boards, expected):
    """Pool per-board measurements into one (px_scan, px_spatial).

    Median per axis, which for the usual two boards is their mean, and which
    ignores a single outlier once there are three or more.

    The two axes are reported separately on purpose, because their spreads mean
    different things. px_scan is frame rate over stage speed: it is the same for
    everything in the capture regardless of where or how high it sits, so boards
    disagreeing on it points at the stage speed drifting mid-scan. px_spatial is
    the optical across-track scale and goes as 1/object-distance, so boards
    disagreeing on that are not coplanar -- and that spread is a lower bound on
    how wrong the correction is for anything off the plane it was measured on.
    """
    if not boards:
        return None, None
    px_scan = statistics.median(b["px_scan"] for b in boards)
    px_spatial = statistics.median(b["px_spatial"] for b in boards)

    if len(boards) < expected:
        print(f"  warning: expected {expected} checkerboard(s), accepted {len(boards)}; "
              f"reconciling across what was found.")
    for i, b in enumerate(boards, 1):
        flags = []
        if b["rms_px"] > FIT_RMS_WARN_PX:
            flags.append(f"HIGH RESIDUAL {b['rms_px']:.2f}px")
        if abs(b["skew_deg"]) > SKEW_WARN_DEG:
            flags.append(f"HIGH SKEW {b['skew_deg']:+.1f}deg")
        if not b["refined"]:
            flags.append("SUBPIX REJECTED")
        if b["axis_ambiguous"]:
            flags.append("BOARD NEAR 45deg -- cell-size axis assignment is ambiguous")
        print(f"  board {i}/{len(boards)} at ({b['centre'][0]:7.1f},{b['centre'][1]:6.1f}) "
              f"[{b['source']}, win{b['subpix_win']}]: scan {b['px_scan']:.3f} "
              f"spatial {b['px_spatial']:.3f} ratio {b['px_scan'] / b['px_spatial']:.4f} | "
              f"rot {b['theta_deg']:+.2f}deg skew {b['skew_deg']:+.2f}deg "
              f"rms {b['rms_px']:.3f}px max {b['max_resid_px']:.3f}px"
              + (("  << " + "; ".join(flags)) if flags else ""))

    if len(boards) > 1:
        def spread(key):
            v = [b[key] for b in boards]
            return (max(v) - min(v)) / statistics.median(v)
        s_scan, s_spatial = spread("px_scan"), spread("px_spatial")
        print(f"  reconciled {len(boards)} boards by median: px_scan spread {100 * s_scan:.1f}%, "
              f"px_spatial spread {100 * s_spatial:.1f}%")
        if s_spatial > BOARD_DISAGREE_WARN:
            print(f"    note: px_spatial is the across-track optical scale and goes as "
                  f"1/object-distance, so a {100 * s_spatial:.1f}% spread means the targets are "
                  f"not coplanar. The correction is only exact on the plane it is measured on; "
                  f"anything at another height keeps a residual stretch of at least that order.")
        if s_scan > BOARD_DISAGREE_WARN:
            print(f"    note: px_scan is frame rate / stage speed and is the same everywhere in "
                  f"the capture, so a {100 * s_scan:.1f}% spread points at the stage speed "
                  f"drifting during the scan rather than at target geometry.")
    return px_scan, px_spatial


def scan_scale_factor(px_scan, px_spatial, reference="spatial"):
    """(fx, fy) multipliers for the (scan, spatial) axes that equalise the two scales.

    spatial (default) rescales the scan axis onto the 640px optical axis; scan
    does the reverse. Only these two: resampling BOTH axes onto some common
    pitch, as a min/max option would, degrades whichever axis was already well
    sampled and cannot add information to the other.
    """
    if reference == "spatial":
        return px_spatial / px_scan, 1.0
    if reference == "scan":
        return 1.0, px_scan / px_spatial
    raise ValueError(f"unknown reference {reference!r}")


def _resample_axis(img, factor, axis):
    """Resample one axis (0 = scan/columns, 1 = spatial/rows). -> (out, achieved).

    INTER_AREA when shrinking, which integrates over the source footprint and so
    is the only choice that does not alias an oversampled axis. INTER_LINEAR when
    growing: INTER_CUBIC is sharper but overshoots at high-contrast edges, and on
    a reflectance cube overshoot means physically impossible negative values.

    cv2.resize handles at most 4 channels, so a 224-band cube goes band by band
    into a preallocated buffer rather than through a stack of full-cube temporaries.
    """
    h, w = img.shape[:2]
    if axis == 0:
        new = max(1, int(round(w * factor)))
        if new == w:
            return img, 1.0
        size, achieved = (new, h), new / w
    else:
        new = max(1, int(round(h * factor)))
        if new == h:
            return img, 1.0
        size, achieved = (w, new), new / h
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR

    if img.ndim == 3 and img.shape[2] > 4:
        out = np.empty((size[1], size[0], img.shape[2]), dtype=img.dtype)
        for b in range(img.shape[2]):
            out[:, :, b] = cv2.resize(img[:, :, b], size, interpolation=interp)
        return out, achieved
    return cv2.resize(img, size, interpolation=interp), achieved


def resample(img, fx, fy):
    """Resample a 2D or multi-band 3D array by per-axis factors. -> (out, (ax, ay)).

    One axis at a time, because the correct interpolation depends on whether that
    axis is shrinking or growing and the two axes need not agree. The achieved
    factors are returned separately: the output size is an integer number of
    pixels, so what actually gets applied is round(n*f)/n, not f.
    """
    out, ax = _resample_axis(img, fx, 0)
    out, ay = _resample_axis(out, fy, 1)
    return out, (ax, ay)


def draw_qc_overlay(gray, boards, rejected, path):
    """Write a QC image: what was accepted, what was rejected, and where.

    The failure this guards against is a confident number measured off the wrong
    structure -- the kernel well grid is regular enough that detectors do latch
    onto it. That is invisible in a log line and obvious in a picture.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for board, _src, _why in rejected:
        cv2.polylines(vis, [_grow_hull(board["corners"], 2)], True, (0, 0, 255), 2)
    for i, b in enumerate(boards, 1):
        cv2.polylines(vis, [_grow_hull(b["corners"], max(b["spacing_px"]))], True, (0, 200, 0), 2)
        for (x, y) in b["corners"]:
            cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 255, 255),
                           cv2.MARKER_CROSS, 9, 1)
        x, y = b["centre"]
        cv2.putText(vis, f"#{i} {b['px_scan']:.2f}/{b['px_spatial']:.2f}",
                    (int(x) - 60, max(12, int(y) - 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def verify_correction(gray, fx, inner_size, max_boards, cell_size):
    """Re-measure the boards after applying fx, and report what is left.

    Closed-loop check on the detection render rather than the corrected cube, so
    it costs one resample of a single 2D image. A correctly measured board should
    come back isotropic; anything else is either a bad measurement or -- as here
    -- a target that is not on the plane the correction was wanted for.
    """
    corrected, _ = _resample_axis(gray, fx, 0)
    # no tiled fallback here: this is a confirmation, not a measurement, and it
    # must not cost more than the measurement it is checking
    boards, _ = detect_boards(corrected, inner_size, max_boards, cell_size, use_tiles=False)
    if not boards:
        print("  verify: no board re-detected after correction (cannot confirm).")
        return None
    ratios = [b["px_scan"] / b["px_spatial"] for b in boards]
    for i, r in enumerate(ratios, 1):
        print(f"  verify: board {i} residual anisotropy {r:.4f} ({100 * (r - 1):+.1f}%)")
    return ratios


# ---------------------------------------------------------------- stage 7 --
def intensity_preview(cube, lo_pct=1.0, hi_pct=99.0):
    """Greyscale (height, width) uint8 preview: mean across all bands, percentile-clipped.

    Percentiles rather than min/max, for the same reason render_detection_gray
    uses them: the scene does not set the maximum. Reflectance here is measured
    against teflon tape, and the checkerboard targets are printed on paper that
    is a brighter reflector than the tape, so those pixels legitimately land
    above 1.0 -- as does any specular glint. A min-max stretch hands the whole
    white point to whichever of them is brightest and drags everything else down
    with it: on day1_dish0 a 0.25% population at 1.33 put the tape itself at
    190 DN instead of 255 and washed the entire frame out to mid-grey.

    Clipping at a percentile also keeps two captures roughly comparable, which a
    stretch anchored to each one's own extremes does not.
    """
    mean = np.asarray(cube.mean(axis=2))
    lo, hi = (float(v) for v in np.percentile(mean, [lo_pct, hi_pct]))
    norm = np.clip((mean - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- stage 8 --
# The plate grid is fitted from the two ArUco fiducials rather than from the
# wells' own appearance, which is what makes a well addressable rather than merely
# visible: cell index N is defined by the rigid plate model (4x7, fixed clipped
# corners, fixed marker cells, row-major), so it denotes the same physical well on
# every capture regardless of how the plate happens to sit in frame. Tracking a
# kernel across days is then just reading the same index.
#
# Only the CELL geometry is emitted. Segmenting the kernel inside each cell is a
# separate problem and is deliberately not attempted here.
def report_grid(res):
    """Log the fit: what the markers gave, what the lattice measured, how it scored."""
    ms = ", ".join(f"id{mid} score={m['score']:.2f} margin={m['margin']:.3f}"
                   for mid, m in sorted(res["markers"].items()))
    print(f"  markers: {res['n_markers']}/2 localized" + (f" ({ms})" if ms else ""))
    if res["frame"] is None:
        print(f"  grid FAIL: {'; '.join(res['reasons'])}")
        return
    f, d = res["frame"], res["diag"]
    rms_s = f"{d['rms']:.2f}px" if d.get("rms") is not None else "n/a"
    mres = " ".join(f"m{k}={v:.1f}px" for k, v in d.get("marker_resid", {}).items())
    print(f"  radon {res['angle']:+.1f}deg -> lattice angle {f['angle']:.1f}deg, "
          f"pitch px={f['px']:.2f} py={f['py']:.2f}")
    print(f"  grid {res['status'].upper()}: wells {d['well_inliers']}/{d['n_well']} inliers, "
          f"rms {rms_s}, marker cross-check {mres or 'n/a'}"
          + (f"  <- {'; '.join(res['reasons'])}" if res["reasons"] else ""))
    if d.get("rejected"):
        print(f"  wells dropped as outliers by the trimmed fit: {', '.join(d['rejected'])}")


def write_cell_masks(cells, out_dir, width, n_lines, inset_frac, grid_meta):
    """Write cell_0..cell_21.npy boolean masks plus a cells.json index.

    One file per kernel cell, each a (width, n_lines) boolean mask in the cube's
    own orientation so it indexes the saved cube directly: cube[mask] -> (px, bands).
    By default a mask is the full lattice cell, matching the quad drawn on the
    overlay. cells.json records the index -> (row, col) mapping, each cell's
    corners, and the fit diagnostics, so a mask is never left as an anonymous blob.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kcells, records = fit_grid.kernel_cells(cells), []
    for cell in kcells:
        mask = fit_grid.cell_mask(cell, width, n_lines, inset_frac)
        np.save(out_dir / f"cell_{cell['index']}.npy", mask)
        records.append({
            "index": cell["index"], "row": cell["row"], "col": cell["col"],
            "file": f"cell_{cell['index']}.npy",
            "center_px": cell["center_px"], "corners_px": cell["corners_px"],
            "mask_px": int(mask.sum()),
        })
    meta = {**grid_meta, "cell_inset_frac": inset_frac,
            "mask_shape": [int(width), int(n_lines)],
            "mask_orientation": "(width, n_lines) -- indexes the cube directly",
            "n_cells": len(records), "cells": records}
    with open(out_dir / "cells.json", "w") as f:
        json.dump(meta, f, indent=2)
    px = [r["mask_px"] for r in records]
    print(f"  wrote {len(records)} cell masks to {out_dir}/ "
          f"({min(px)}-{max(px)} px each) + cells.json")
    return records


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample",
                        help=f"Sample name, e.g. day2_dish3_ref_2500 -- used to name outputs "
                             f"(<sample>.npy / <sample>.png) and the archive folder "
                             f"({STORAGE_DIR}/<sample>/) the raw .bin files are moved into afterward.")
    parser.add_argument("--dark", default=DEFAULT_DARK,
                        help="Dark reference: raw capture dir or stitched cube .npy.")
    parser.add_argument("--grain", default=None,
                        help=f"Grain/sample scan: raw capture dir or stitched cube .npy. "
                             f"Defaults to {RAW_IMAGE_DIR}/ (eBUS Player's flat capture dir); "
                             f"overriding this skips the post-processing .bin archive step.")
    parser.add_argument("--out", default="corrected_file",
                        help="Output subfolder for .npy artifacts (created if missing).")
    parser.add_argument("--out-img", default="corrected_image",
                        help="Output subfolder for the intensity preview PNG (created if missing).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite <sample>.npy / <sample>.png if they already exist.")
    parser.add_argument("--tape-pct", type=float, default=85.0,
                        help="Brightness percentile a tape blob must exceed to be located.")
    parser.add_argument("--white-pct", type=float, default=75.0,
                        help="Percentile of each row's tape pixels used as its white reference.")
    parser.add_argument("--white-min-coverage", type=float, default=0.75,
                        help="A width row's tape sample is trusted only if it holds at least this "
                             "fraction of the median per-row count. Rows below it lie in a blob's "
                             "taper, where the few surviving pixels are part background and read "
                             "dim; they are interpolated from the interior rows instead. Pass 0 to "
                             "trust every covered row (the old behaviour).")
    parser.add_argument("--checkerboard", nargs=2, type=int, default=(3, 3),
                        metavar=("COLS", "ROWS"),
                        help="Inner-corner count (default 3 3 = a 4x4-square board).")
    parser.add_argument("--boards", type=int, default=2,
                        help="How many checkerboard targets are in frame. All of them are "
                             "measured and reconciled, rather than trusting whichever one the "
                             "detector finds first -- boards on different planes disagree.")
    parser.add_argument("--cell-size", nargs=2, type=float, default=(1.0, 1.0),
                        metavar=("W", "H"),
                        help="True physical cell extent along the board's own two lattice axes "
                             "(any consistent unit). Default 1 1 assumes square cells; only the "
                             "ratio matters. Set this if the printed target is measurably not "
                             "square -- the correction inherits the target's aspect error exactly.")
    parser.add_argument("--plane-factor", type=float, default=DEFAULT_PLANE_FACTOR,
                        help=f"Extra scan-axis multiplier applied after the board measurement, "
                             f"because the checkerboards are not on the same plane as the kernels. "
                             f"Defaults to {DEFAULT_PLANE_FACTOR}, measured from the day0 captures "
                             f"for this rig -- see DEFAULT_PLANE_FACTOR for how, and for when it "
                             f"stops being valid. Pass 1.0 to disable it and get the raw board "
                             f"measurement.")
    parser.add_argument("--reference", choices=["spatial", "scan"], default="spatial",
                        help="Which axis to trust when equalising the scan-axis stretch.")
    parser.add_argument("--qc", action="store_true",
                        help="Also write <sample>_checkerboard_qc.png, an overlay of which "
                             "checkerboards were detected, accepted and rejected. Off by default: "
                             "the reflectance run's deliverables are the cube and its preview, and "
                             "the acceptance/rejection decisions are already in the log.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the closed-loop re-measurement of the boards after correction.")
    parser.add_argument("--manual-scale", type=float, default=None,
                        help="Skip checkerboard detection; force this scan-axis (x) scale factor. "
                             "--plane-factor is NOT applied on top of a manual scale.")
    parser.add_argument("--no-grid", action="store_true",
                        help="Skip stage 8 entirely: no plate-grid fit, no grid overlay on the "
                             "preview PNG, no per-cell masks. The preview is then the plain "
                             "greyscale render.")
    parser.add_argument("--cells", default=CELLS_DIR,
                        help=f"Output root for per-cell masks; they land in "
                             f"<this>/<sample>/cell_N.npy (default {CELLS_DIR}).")
    parser.add_argument("--cell-inset", type=float, default=fit_grid.CELL_INSET_FRAC,
                        help=f"Shrink each cell mask toward its centre by this fraction of pitch. "
                             f"Default {fit_grid.CELL_INSET_FRAC:g} keeps the mask at the full "
                             f"lattice cell, corner to corner, so nothing is clipped; raise it to "
                             f"pull the mask inside the bright walls (0.16 is what "
                             f"grid_sandbox/extract_cells.py uses before segmenting). The overlay "
                             f"always outlines whatever this produces.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_img_dir = Path(args.out_img)
    corrected_path = out_dir / f"{args.sample}.npy"
    intensity_path = out_img_dir / f"{args.sample}.png"
    cells_dir = Path(args.cells) / args.sample
    existing = [p for p in (corrected_path, intensity_path, cells_dir) if p.exists()]
    if not args.force and existing:
        sys.exit(f"{', '.join(str(p) for p in existing)} already exist(s) -- "
                 f"pass --force to overwrite, or use a different sample name.")

    # archiving only applies to the default flat raw_image_bin/ capture dir, not a
    # custom --grain override (e.g. an already-stitched .npy elsewhere)
    archive_dir = Path(STORAGE_DIR) / args.sample if args.grain is None else None
    if archive_dir is not None and archive_dir.exists():
        sys.exit(f"{archive_dir} already exists -- pass a different sample name, or "
                 f"clear that folder if it's stale.")

    dark_path = Path(args.dark)
    print(f"Loading dark reference from {dark_path} ...")
    if dark_path.is_file() and dark_path.suffix == ".npy":
        dark_cube = np.load(dark_path)
    else:
        dark_cube = stitch(dark_path)
    print(f"  shape {dark_cube.shape}, dtype {dark_cube.dtype}")

    frame = dark_frame(dark_cube)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    grain_path = Path(args.grain) if args.grain else Path(RAW_IMAGE_DIR)
    print(f"Loading grain scan from {grain_path} ...")
    if grain_path.is_file() and grain_path.suffix == ".npy":
        grain_cube = np.load(grain_path)
    else:
        grain_cube = stitch(grain_path)
    print(f"  shape {grain_cube.shape}, dtype {grain_cube.dtype}")

    darksub, n_clipped = apply_dark_correction(grain_cube, frame)
    clip_frac = n_clipped / grain_cube.size
    print(f"Applied dark subtraction ({n_clipped} px clipped at 0, {100 * clip_frac:.3f}%).")

    del grain_cube  # ~2.5GB for this capture; nothing after this point needs the raw cube

    gray = np.asarray(darksub.mean(axis=2)).astype(np.float32)
    masks = find_tape_blobs(gray, pct=args.tape_pct)
    for mask, label_ in zip(masks, ("LEFT", "RIGHT")):
        rows = np.nonzero(mask.any(axis=1))[0]
        print(f"{label_} tape blob: {int(mask.sum())} px, "
              f"row span [{rows.min()}:{rows.max()}] of {gray.shape[0]}")

    span = tape_column_span(masks)
    n_lines = darksub.shape[1]
    if span is None:
        print(f"  warning: no tape blobs found at all; skipping scan-line crop "
              f"(keeping all {n_lines} lines).")
    else:
        c0, c1 = span
        print(f"Cropping scan-line axis to tape span [{c0}:{c1 + 1}] of {n_lines} "
              f"(dropping {c0} lines before, {n_lines - 1 - c1} after).")
        # .copy(): plain slicing returns a view into the full-size buffer, which would
        # keep the whole uncropped array resident in memory for the rest of the run --
        # defeating the point of cropping. .copy() actually releases the dropped region.
        darksub = darksub[:, c0:c1 + 1, :].copy()
        masks = [m[:, c0:c1 + 1].copy() for m in masks]

    white_ref, reliable = build_white_reference(darksub, masks, pct=args.white_pct,
                                                min_coverage_frac=args.white_min_coverage)
    n_interpolated = int((~reliable).sum())
    print(f"White reference: {reliable.sum()}/{reliable.size} width rows measured from a "
          f"reliable tape sample ({n_interpolated} interpolated).")

    reflectance = apply_white_correction(darksub, white_ref)
    del darksub  # superseded by reflectance; nothing after this needs it
    print(f"Reflectance: min {reflectance.min():.4f}, max {reflectance.max():.4f}, "
          f"mean {reflectance.mean():.4f}")

    inner = tuple(args.checkerboard)
    cell = tuple(args.cell_size)
    if args.manual_scale is not None:
        fx, fy = args.manual_scale, 1.0
        print(f"Manual scan-axis scale: x{fx:.5f} (checkerboard detection skipped; "
              f"--plane-factor not applied).")
    else:
        gray = render_detection_gray(reflectance)
        print(f"Checkerboard {inner[0]}x{inner[1]} inner corners, expecting {args.boards} "
              f"board(s), cell {cell[0]:g}x{cell[1]:g}:")
        boards, rejected = detect_boards(gray, inner, args.boards, cell)
        for board, source, why in rejected:
            print(f"  rejected candidate at ({board['centre'][0]:7.1f},{board['centre'][1]:6.1f}) "
                  f"[{source}]: {why}")
        if args.qc:
            qc_path = out_img_dir / f"{args.sample}_checkerboard_qc.png"
            draw_qc_overlay(gray, boards, rejected, qc_path)
            print(f"  wrote {qc_path}")

        px_scan, px_spatial = reconcile_boards(boards, args.boards)
        if px_scan is None:
            print(f"  warning: no checkerboard accepted; skipping scan-axis geometry correction "
                  f"(the cube is written UNCORRECTED).")
            fx, fy = 1.0, 1.0
        else:
            fx, fy = scan_scale_factor(px_scan, px_spatial, args.reference)
            print(f"Measured scan={px_scan:.3f} spatial={px_spatial:.3f} per cell; "
                  f"correction ({args.reference} ref): scan x{fx:.5f}, spatial x{fy:.5f}.")
            if not args.no_verify:
                verify_correction(gray, fx, inner, args.boards, cell)
            # Printed unconditionally, including when it is 1.0. A non-unity default
            # that only announced itself when overridden would be exactly the kind of
            # silent correction that made the original ellipse so hard to track down.
            origin = ("default for this rig" if args.plane_factor == DEFAULT_PLANE_FACTOR
                      else "user-supplied")
            if args.plane_factor == 1.0:
                print(f"Plane factor 1.0 ({origin}): raw board measurement kept, no "
                      f"sample-plane correction. scan x{fx:.5f}.")
            else:
                fx *= args.plane_factor
                print(f"Plane factor {args.plane_factor:g} ({origin}) applied for the "
                      f"target/sample plane offset: scan x{fx:.5f}.")
            lo, hi = FX_SANITY_RANGE
            if not lo <= fx <= hi:
                print(f"  warning: scan scale x{fx:.5f} is outside the plausible range "
                      f"[{lo}, {hi}]; treating as a bad measurement and skipping the correction.")
                fx, fy = 1.0, 1.0

    corrected, achieved = resample(reflectance, fx, fy)
    del reflectance  # superseded by corrected; nothing after this needs it
    if abs(achieved[0] - fx) > 1e-4 or abs(achieved[1] - fy) > 1e-4:
        print(f"  note: integer output size means the applied factors are "
              f"scan x{achieved[0]:.5f}, spatial x{achieved[1]:.5f}.")
    np.save(corrected_path, corrected)
    print(f"wrote {corrected_path} ({corrected.nbytes / 1e6:.0f} MB, shape {corrected.shape})")

    preview = intensity_preview(corrected)
    width, n_lines = corrected.shape[0], corrected.shape[1]

    grid = None
    if not args.no_grid:
        print(f"Plate grid ({fit_grid.N_ROWS}x{fit_grid.N_COLS} lattice, "
              f"{fit_grid.N_KERNEL_CELLS} kernel cells) from the ArUco fiducials:")
        # the fitter works in (n_lines, width); the preview is the cube's own
        # (width, n_lines), and cv2 needs the transpose materialised
        grid = fit_grid.fit(np.ascontiguousarray(preview.T))
        report_grid(grid)

    if grid is not None and grid["frame"] is not None:
        cv2.imwrite(str(intensity_path),
                    fit_grid.draw_grid_overlay(preview, grid["cells"], grid["markers"],
                                               grid["frame"], grid["diag"],
                                               grid["status"], grid["reasons"],
                                               inset_frac=args.cell_inset))
        note = f"; grid {grid['status']}"
    else:
        Image.fromarray(preview, mode="L").save(intensity_path)
        note = "" if grid is None else "; no overlay (grid failed)"
    print(f"wrote {intensity_path} ({n_lines}x{width}{note})")

    if grid is not None:
        if grid["frame"] is None or grid["status"] == "fail":
            print(f"  warning: no per-cell masks written -- the grid did not fit "
                  f"({'; '.join(grid['reasons'])}). Cell positions are not guessed at "
                  f"from a failed fit.")
        else:
            if grid["status"] != "pass":
                print(f"  warning: grid status is {grid['status']} "
                      f"({'; '.join(grid['reasons'])}); cell masks are written anyway, "
                      f"but check the overlay before trusting them.")
            write_cell_masks(grid["cells"], cells_dir, width, n_lines, args.cell_inset,
                             {"sample": args.sample, "cube": str(corrected_path),
                              "status": grid["status"], "reasons": grid["reasons"],
                              "n_markers": grid["n_markers"],
                              "angle": grid["frame"]["angle"],
                              "px": grid["frame"]["px"], "py": grid["frame"]["py"],
                              **{k: grid["diag"][k] for k in
                                 ("n_well", "well_inliers", "rms", "marker_resid", "rejected")}})

    if archive_dir is not None:
        bin_files = sorted(Path(RAW_IMAGE_DIR).glob("*.bin"))
        archive_dir.mkdir(parents=True)
        for f in bin_files:
            f.rename(archive_dir / f.name)
        print(f"Archived {len(bin_files)} .bin files from {RAW_IMAGE_DIR}/ to {archive_dir}/ "
              f"({RAW_IMAGE_DIR}/ now empty and ready for the next scan).")


if __name__ == "__main__":
    main()
