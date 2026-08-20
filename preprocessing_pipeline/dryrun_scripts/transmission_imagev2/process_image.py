"""End-to-end TRANSMISSION correction: stitch -> reference frames -> saturation
mask -> dark subtraction -> dedicated-white ratio -> checkerboard scan-axis
geometry correction -> masks, preview PNG, provenance.

The correction, in full, with no constant anywhere in it:

    T = (raw - D_long) / (W - D_short)

Transmittance is not reflectance with a different name, and the three places it
differs are the reason this is a separate script rather than a flag on the
reflectance pipeline:

  1. There is no in-scene white. Reflectance divides by teflon tape strips that
     sit in the same frame as the dish; in transmission nothing in the frame is
     an unobstructed beam, so the white reference is its OWN capture -- a static
     line scan of the bare light.

  2. The two darks are not interchangeable. The sample needs a long exposure to
     get signal through a kernel; that same exposure drives an unobstructed beam
     far past full scale, so the white is shot short. Each therefore has its own
     dark at its own exposure -- D_long for the sample, D_short for the white.

     What is NOT done is rescaling the white by the exposure ratio. That would
     make T an absolute transmittance, and it is available via --exposure-scale,
     but it is not the default: the ratio is a number the data cannot verify
     (the open beam is saturated everywhere it could be checked against), and
     applying a silent 47x multiplier to every voxel on the strength of a folder
     name is not a default. Without it, T is RELATIVE -- an unobstructed beam
     reads ~1 here only because the white happens to be exposed near full scale.

  3. Most of the frame is saturated, by design. Everything outside a kernel is
     open beam and pins at full scale; only the kernels carry usable signal.
     That is fine for the measurement and fatal for anything that reads the cube
     without knowing which pixels were clipped, so stage 3 records it.

Stage 0 -- capture metadata and the line-rate gate
--------------------------------------------------
eBUS names each line <index>_<hex timestamp>_w640_h224_pMono12Packed.bin, so the
median inter-line timestamp delta measures the line period of a capture without
opening a single file. Stage 7's scan-axis factor is frame-rate-over-stage-speed
and is only transferable from the checkerboard capture to the sample scan if
both ran at the same line rate, so that equality is checked and the geometry
correction is skipped (loudly) when it fails. See --ignore-line-rate.

Line period is deliberately NOT used to infer exposure. It equals the exposure
only when the camera is exposure-limited; a capture that free-runs slower than
its exposure requires (as the 2.1k white reference here does, 2.1ms of exposure
inside a 21ms period) breaks that identity, and reading the ratio off the
timestamps would silently rescale the whole cube by 10x. Exposures come from the
folder names or from --exposure-*, never from the clock.

Stage 1 -- stitching
--------------------
Unpacks per-line Mono12Packed .bin captures into a (width, n_lines, channels)
uint16 cube, reusing the hsi_save_load codec (the byte-packing math is subtle --
see loadstich/hsi_save_load.py, and do not reimplement it here).

Stage 2 -- reference frames
---------------------------
reference_frame() collapses a reference capture to one (width, channels) frame.
The push-broom sensor's dark current/offset and the lamp's spatial profile are
fixed per (spatial pixel, band) rather than per line, so one frame is the
correction target for every line of the sample scan.

The collapse is a MEDIAN, not a mean. A reference capture is a handful of lines
(27 for the long dark here) and a single bad one moves a mean by percent-level
-- the 100k dark in this rig has one line 7.7% below the rest, which a median
ignores and a mean bakes into every pixel of the output. Per-line spread is
reported so a reference that is drifting rather than merely noisy is visible.

Stage 3 -- saturation mask
--------------------------
Recorded from the RAW cube, before any arithmetic: once the dark is subtracted
and the white divided out, a clipped pixel is numerically indistinguishable from
a legitimately bright one. The mask travels with the cube (resampled alongside
it in stage 7, with any-source-pixel-saturated semantics) so downstream
consumers can drop clipped pixels instead of fitting to a plateau.

Stage 4 -- dark subtraction
---------------------------
Sample minus the long-exposure dark frame, broadcast over the scan-line axis.
Unlike the reflectance pipeline this does NOT clip negatives to zero: kernel
signal here is a few hundred counts above dark, and clipping rectifies the noise
floor into a positive bias exactly where the measurement is weakest. Negatives
are carried through as float32 and reported.

Stage 5 -- white reference, validity, exposure scaling
------------------------------------------------------
(W - D_short) from the dedicated white capture and its own matched dark, then
scaled by t_long/t_short (stage 0's note on where exposures come from applies).

A white reference is only usable where the lamp actually delivered light, and
this one does not everywhere: it collapses over the outer spatial rows and can
carry narrow occlusions (a wire, an edge) that the sample scan does not have.
Dividing there turns a saturated sample pixel into a transmittance of 10-50
rather than 1. Two complementary tests mark those (row, band) cells invalid:

  absolute  W-D below --min-white counts is not a measurement, it is noise.
  relative  W-D below --outlier-frac of a rolling spatial median flags a dip
            whose neighbours are healthy -- an occlusion in the white path.
            --white-window must be several times the widest occlusion expected
            or the dip drags down its own baseline and hides itself: the
            occlusion in this rig's white capture is ~28 spatial rows wide, and
            a 31-row window caught 6 cells of it against the 121-row window's
            several thousand.

Invalid cells are recorded in a mask and their transmittance is set to
--invalid-value, rather than being papered over with a white reference of 1.0.

Stage 6 -- transmittance
------------------------
T = darksub / white_scaled, float32. Statistics are reported over the valid and
unsaturated population, which is the only part of the frame that means anything.

Stage 7 -- checkerboard scan-axis geometry correction
-----------------------------------------------------
The scan axis is stretched relative to the 640px optical axis by however much
faster/slower the stage moved than the frame rate. In reflectance the board is
in the scene; here it is a dedicated capture at the same exposure and stage
speed, measured on its own cube and applied to the sample's.

The measurement machinery is the reflectance pipeline's: classic detector plus
checked sub-pixel refinement at native resolution, a rotation-invariant lattice
fit, every board in frame measured independently and judged before being pooled
by median, plus a QC overlay and a closed-loop re-measurement. The reasoning for
each of those choices is recorded at the top of the stage 7 section.

--plane-factor defaults to 1.0 here, unlike reflectance. The boards and the
kernels not being coplanar is a real effect and reflectance carries a measured
1.091 for it, but that constant belongs to that rig's dish/insert stack and does
NOT transfer -- on this one the raw board measurement already brings the dish
round in the corrected preview. That default rests on looking at the preview,
not on a number: see the note above verify_correction() for why there is no
automatic sample-plane check, and re-check it by eye if the rig changes.

Stage 8 -- outputs
------------------
  corrected_file/<sample>.npy         float32 transmittance cube
  corrected_file/<sample>_masks.npz   saturation + white-validity masks
  corrected_file/<sample>_meta.json   full provenance: every path, exposure,
                                      factor, fraction and warning of the run
  corrected_image/<sample>.png        percentile-stretched intensity preview
  corrected_image/<sample>_checkerboard_qc.png

Run (defaults expect this folder's layout):
    raw_image/                 sample scan, flat .bin as eBUS Player drops them
    raw_dark/image/<exp>/      dark at the SAMPLE's exposure
    raw_dark/white/<exp>/      dark at the WHITE's exposure
    raw_white/<exp>/           the dedicated white line scan
    raw_checkerboard/          board capture, same exposure and stage speed

    python3 process_image.py day1_dish0

Each reference directory may hold .bin files directly or exactly one
subdirectory; more than one and the run stops rather than guessing. Afterwards
raw_image/'s .bin files are moved to raw_image_storage/<sample>/, leaving
raw_image/ empty for the next capture.
"""

import argparse
import json
import os
import re
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

WIDTH = 640       # spatial pixels per line
CHANNELS = 224    # spectral bands per line
FULL_SCALE = 4095  # 12-bit sensor: a pixel at or above this is clipped

RAW_IMAGE_DIR = "raw_image"                 # eBUS Player saves flat into here
DEFAULT_DARK_IMAGE_DIR = "raw_dark/image"   # dark at the sample's exposure
DEFAULT_DARK_WHITE_DIR = "raw_dark/white"   # dark at the white capture's exposure
DEFAULT_WHITE_DIR = "raw_white"             # the dedicated white line scan
DEFAULT_CHECKERBOARD_DIR = "raw_checkerboard"
STORAGE_DIR = "raw_image_storage"           # processed .bin archived to <STORAGE_DIR>/<sample>/

# Scan-axis multiplier applied on top of the checkerboard measurement, for the
# board and the kernels not being on the same plane. 1.0 here: on this rig the
# raw board measurement already brings the dish visibly round in the corrected
# preview, so there is nothing left for this to absorb. That is an eyeball
# result, not a measured constant -- nothing in this script measures the sample
# plane (see the note above verify_correction()), so re-check the preview if the
# camera height or the dish/insert stack changes. Reflectance's 1.091 is that
# rig's constant and must not be carried over.
DEFAULT_PLANE_FACTOR = 1.0


# ---------------------------------------------------------------- stage 0 --
# eBUS names lines <index>_<hex timestamp>_w<width>_h<channels>_p<format>.bin.
LINE_NAME_RE = re.compile(r"^(\d+)_([0-9A-Fa-f]+)_w")
# Exposures are deliberately not parsed from folder names any more: the
# correction is the plain ratio with no constant, so nothing needs them, and
# guessing a 47x multiplier off a folder called "100k" was never something to
# do silently. Pass --exposure-scale if absolute transmittance is wanted.


def line_period(directory):
    """Median inter-line timestamp delta over a capture directory, or None.

    In the camera's own timestamp ticks -- the unit is not fixed across sessions
    and does not need to be, because this is only ever used to compare two
    captures from the same session (stage 0's line-rate gate). Returns
    (median, n_lines, min, max) so jitter is visible alongside the median.
    """
    stamps = []
    for name in sorted(os.listdir(directory)):
        m = LINE_NAME_RE.match(name)
        if m and name.endswith(".bin"):
            stamps.append(int(m.group(2), 16))
    if len(stamps) < 3:
        return None
    dt = np.diff(np.asarray(stamps, dtype=np.int64))
    return float(np.median(dt)), len(stamps), int(dt.min()), int(dt.max())


def resolve_capture(base, what):
    """A capture directory from `base`: itself if it holds .bin, else its lone subdir.

    Reference captures are shared across samples and named for their exposure
    (raw_white/2100/), not for the sample, so prefix matching would be wrong
    here. Exactly one candidate is required -- a second one means a stale folder
    from another session, and picking either silently is how a run gets
    corrected against the wrong reference.
    """
    base = Path(base)
    if base.is_file() and base.suffix == ".npy":
        return base                      # an already-stitched cube, used as-is
    if not base.is_dir():
        raise SystemExit(f"{base} does not exist (expected the {what}).")
    if any(base.glob("*.bin")):
        return base
    subdirs = sorted(p for p in base.iterdir() if p.is_dir() and any(p.glob("*.bin")))
    if not subdirs:
        raise SystemExit(f"no .bin files in {base} or any subdirectory (expected the {what}).")
    if len(subdirs) > 1:
        raise SystemExit(f"{base} holds {len(subdirs)} capture subdirectories "
                         f"({[p.name for p in subdirs]}) -- expected exactly one; pass the "
                         f"{what} explicitly, or clear the stale folder.")
    return subdirs[0]


# ---------------------------------------------------------------- stage 1 --
def load_line(raw, w=WIDTH, c=CHANNELS):
    """Unpack one raw Mono12Packed line into an oriented (w, c) uint16 frame.

    A corrupt/unparseable line becomes a zero frame instead of aborting the
    whole stitch.
    """
    try:
        line = load_hsi(raw)
        return line.reshape([c, w]).swapaxes(0, 1)[::-1]
    except Exception:
        return np.zeros((w, c), dtype=np.uint16)


def stitch(directory, w=WIDTH, c=CHANNELS):
    """Stitch a directory of per-line .bin captures into a (w, n_lines, c) cube.

    Returns (cube, blank_line_indices) -- the indices matter for the reference
    captures, where a handful of lines is the whole measurement (stage 2).
    """
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise ValueError(f"no .bin files found in {directory}")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    blank = []
    for i, fname in enumerate(files):
        frame = load_line(np.fromfile(directory / fname, dtype=np.uint8), w, c)
        if not frame.any():
            blank.append(i)
        cube[:, i, :] = frame
    print(f"  {directory.name}: {len(files)} lines stitched ({len(blank)} blank/bad).")
    return cube, blank


def load_cube(path, what):
    """Stitch a capture directory, or load an already-stitched .npy. -> (cube, blank)."""
    path = Path(path)
    print(f"Loading {what} from {path} ...")
    if path.is_file() and path.suffix == ".npy":
        cube, blank = np.load(path), []
    else:
        cube, blank = stitch(path)
    print(f"  shape {cube.shape}, dtype {cube.dtype}")
    return cube, blank


# ---------------------------------------------------------------- stage 2 --
def reference_frame(cube, name, blank=()):
    """Robust (width, channels) frame from a reference capture. -> (frame, stats).

    Median across the scan-line axis with blank lines excluded. See the stage 2
    note in the module docstring for why the median rather than the mean.
    """
    keep = np.setdiff1d(np.arange(cube.shape[1]), np.asarray(blank, dtype=int))
    if keep.size == 0:
        raise SystemExit(f"every line of the {name} reference is blank/unparseable.")
    usable = cube[:, keep, :]

    frame = np.median(usable, axis=1).astype(np.float64)
    per_line = usable.mean(axis=(0, 2))
    med = float(np.median(per_line))
    spread = float(per_line.max() - per_line.min()) / med if med else float("inf")
    stats = {"n_lines": int(cube.shape[1]), "n_used": int(keep.size),
             "n_blank": int(cube.shape[1] - keep.size),
             "per_line_mean_median": med, "per_line_spread_frac": spread,
             "frame_min": float(frame.min()), "frame_median": float(np.median(frame)),
             "frame_max": float(frame.max())}
    print(f"{name} reference: {keep.size}/{cube.shape[1]} lines used, "
          f"per-line mean {med:.1f} (spread {100 * spread:.2f}%), "
          f"frame {frame.min():.1f}..{frame.max():.1f} median {np.median(frame):.1f}")
    if spread > 0.02:
        print(f"  note: {100 * spread:.1f}% line-to-line spread -- the median collapse absorbs "
              f"a stray line, but a drifting reference would look the same here. Worth a look "
              f"if this grows.")
    return frame, stats


# ---------------------------------------------------------------- stage 3 --
def saturation_mask(cube, level=FULL_SCALE):
    """Bool (width, n_lines, channels): raw pixels at or above the clipping level."""
    mask = cube >= level
    frac = float(mask.mean())
    print(f"Saturation: {int(mask.sum())} of {mask.size} raw px at >= {level} "
          f"({100 * frac:.2f}%). Expected to be large in transmission -- everything "
          f"outside a kernel is open beam.")
    return mask, frac


# ---------------------------------------------------------------- stage 4 --
def apply_dark_correction(cube, dark):
    """cube - dark, broadcast over the scan-line axis, as float32. -> (darksub, n_neg).

    Negatives are kept (see the stage 4 note in the module docstring). Casts
    once and subtracts in place rather than allocating a second full-cube
    temporary.
    """
    diff = cube.astype(np.float32)
    diff -= dark[:, None, :].astype(np.float32)
    return diff, int((diff < 0).sum())


# ---------------------------------------------------------------- stage 5 --
def _rolling_median(profile, k):
    """Rolling median down axis 0 of a (rows, bands) array, edge-replicated."""
    k = max(3, int(k) | 1)                      # odd, so the window is centred
    pad = k // 2
    padded = np.pad(profile, ((pad, pad), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)
    return np.median(windows, axis=-1)


def build_white_reference(white_frame, dark_white_frame, min_white, outlier_frac, window,
                          min_band_frac):
    """(W - D) with a per-(row, band) validity mask. -> (white_ref, valid, stats).

    All three tests run on raw counts, before the exposure scaling, so
    --min-white is expressed in the white capture's own units. Invalid entries
    are set to 1.0 purely so the division cannot produce inf/nan; stage 6
    overwrites those positions with --invalid-value.

    The first two tests run DOWN the spatial axis and catch a beam that is dark
    at one edge or occluded partway across. The third runs ACROSS the spectral
    axis, and exists because the lamp dies at both ends of the range the way it
    dies at the edges of the field: on this rig W-D peaks at ~3470 counts near
    1200nm and falls to 246 at 900nm and 109 at 1700nm, 7% and 3% of peak.
    Nothing in the spatial tests notices -- 109 counts clears any sane absolute
    floor, and the rolling SPATIAL median at that band is uniformly low too, so
    the relative test sees a perfectly healthy row. The result is a divide by
    ~3% of the lamp at exactly the two bands that bound every spectrum, and a
    kernel that really spans 0.003-0.025 comes back spanning 0.003-0.085 with
    both endpoints 3x the largest real feature in between. Every per-region
    spectrum plot then autoscales to those two endpoints and the actual
    absorption structure is squashed into the bottom fifth of the axis -- the
    signal is intact and unreadable, which looks exactly like the signal being
    gone.
    """
    white_ref = white_frame - dark_white_frame

    valid_abs = white_ref >= min_white
    smooth = _rolling_median(white_ref, window)
    valid_rel = white_ref >= outlier_frac * smooth
    band_peak = white_ref.max(axis=1, keepdims=True)      # per spatial row, over bands
    valid_band = white_ref >= min_band_frac * band_peak
    valid = valid_abs & valid_rel & valid_band

    n = white_ref.size
    stats = {"min": float(white_ref.min()), "median": float(np.median(white_ref)),
             "max": float(white_ref.max()),
             "n_invalid": int((~valid).sum()), "n_invalid_absolute": int((~valid_abs).sum()),
             "n_invalid_relative": int((~valid_rel).sum()),
             "n_invalid_band": int((~valid_band).sum()),
             "invalid_frac": float((~valid).mean())}
    print(f"White reference (W-D): {white_ref.min():.1f}..{white_ref.max():.1f}, "
          f"median {np.median(white_ref):.1f}")
    print(f"  validity: {(~valid).sum()}/{n} (row, band) cells invalid "
          f"({100 * (~valid).mean():.2f}%) -- {(~valid_abs).sum()} below {min_white} counts, "
          f"{(valid_abs & ~valid_rel).sum()} more below {outlier_frac:g}x the rolling "
          f"spatial median (window {window}), {(valid_abs & valid_rel & ~valid_band).sum()} more "
          f"below {min_band_frac:g}x their row's spectral peak.")

    dead_rows = np.nonzero(~valid.any(axis=1))[0]
    weak_rows = np.nonzero((~valid).mean(axis=1) > 0.5)[0]
    if weak_rows.size:
        print(f"  spatial rows with no usable white in >50% of bands: "
              f"{_summarise_runs(weak_rows)} ({weak_rows.size} of {white_ref.shape[0]}). "
              f"Transmittance there is set to the invalid value, not divided by noise.")
    weak_bands = np.nonzero((~valid_band).mean(axis=0) > 0.5)[0]
    if weak_bands.size:
        print(f"  bands where the lamp is below {min_band_frac:g}x its peak: "
              f"{_summarise_runs(weak_bands)} ({weak_bands.size} of {white_ref.shape[1]}). "
              f"Dividing by these is what turns a spectrum's endpoints into spikes.")
    stats["dead_rows"] = [int(r) for r in dead_rows]
    stats["weak_rows"] = [int(r) for r in weak_rows]
    stats["weak_bands"] = [int(b) for b in weak_bands]

    white_ref = np.where(valid, white_ref, 1.0)
    return white_ref, valid, stats


def band_lamp_profile(white_ref, valid):
    """Per-band median of (W-D) over the rows that have a usable white reference.

    This is the lamp x sensor-QE envelope, and it is the ONLY difference between
    the two flat-field modes: spatial mode keeps it in the data, full mode
    divides it out. Saved alongside the cube so the two are interconvertible
    without re-running -- see the note on `band_scale` where the masks are
    written.
    """
    return np.array([np.median(white_ref[valid[:, b], b]) if valid[:, b].any() else 1.0
                     for b in range(white_ref.shape[1])])


def spatial_only_divisor(white_ref, valid):
    """White reference renormalised per band, so only its SPATIAL shape divides out.

    Full transmittance divides by (W-D) outright, which removes the lamp's
    spectral shape along with its spatial one. That is correct and it is what
    makes the output an absolute, cross-session-comparable transmittance -- but
    it also means the divisor is tiny at the bands where the lamp is dead, which
    is where the endpoint spikes come from.

    This divides instead by (W-D) / median_over_rows(W-D), which is ~1 at every
    band by construction. Kernels at different points across the beam are still
    equalised -- that is the same spatial correction, pixel for pixel -- but the
    lamp's spectral envelope stays in the data, so a corrected spectrum keeps
    the shape of the raw one and nothing is ever divided by 3% of the lamp.

    The two modes differ by exactly one per-band constant, so neither can be
    "more correct" about any single spectrum's structure. What differs is
    comparability: full transmittance survives a change of lamp or exposure,
    this does not. Output is in counts at the sample's exposure, and the
    exposure scale does not apply -- it cancels in the renormalisation.
    """
    lamp = band_lamp_profile(white_ref, valid)
    return white_ref / np.maximum(lamp, 1e-9)[None, :]


def _summarise_runs(indices):
    """[0,1,2,7,8] -> '0-2, 7-8', for readable row-range reporting."""
    if len(indices) == 0:
        return "none"
    runs, start, prev = [], int(indices[0]), int(indices[0])
    for i in map(int, indices[1:]):
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


# ---------------------------------------------------------------- stage 6 --
def apply_white_correction(darksub, white_ref, valid, invalid_value):
    """darksub / white_ref, in place, with invalid (row, band) cells overwritten.

    white_ref is already exposure-scaled by the caller, so this is exactly
    (raw - D_long) / ((W - D_short) * t_long/t_short).
    """
    darksub /= white_ref.astype(np.float32)[:, None, :]
    if not valid.all():
        rows, bands = np.nonzero(~valid)
        darksub[rows, :, bands] = invalid_value
    return darksub


# ---------------------------------------------------------------- stage 7 --
# Corner detection is done with the CLASSIC cv2.findChessboardCorners plus
# cv2.cornerSubPix, and deliberately NOT with findChessboardCornersSB.
# Benchmarked against synthetically rendered boards of known cell size, SB's
# measured axis ratio is off by -7% at 2.1:1 anisotropy and -15% at 3.2:1, and
# its detection rate collapses above ~6:1; the classic detector plus sub-pixel
# refinement stays within 0.16% from 1:1 all the way to 14:1. The scan axis here
# is routinely oversampled, which is squarely where SB fails.
#
# For the same reason there is no scan-axis "pre-compression" search (which the
# earlier transmission pipeline used). It existed to make the board look square
# enough to detect; with the classic detector it is unnecessary, and resampling
# before measuring can only discard scan-axis information. Everything is
# detected and measured at native resolution.
CHECKERBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

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
    threshold cannot work with. In transmission that is not a corner case --
    a third of the frame is at full scale.
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
    result toward isotropy on any rotated board.

    B @ B.T being diagonal is also the model's own consistency check. The
    off-diagonal term is zero for ANY rotated rigid grid under axis-aligned
    scaling, so a non-zero "skew" means the premise is broken: tilt, perspective,
    or a target whose cells are parallelograms rather than rectangles.
    """
    corners = np.asarray(corners, dtype=np.float64)
    lattice = _ideal_lattice(inner_size)

    def _fit(lat):
        design = np.column_stack([lat, np.ones(len(lat))])
        coef, *_ = np.linalg.lstsq(design, corners, rcond=None)
        return coef[:2].T, np.linalg.norm(design @ coef - corners, axis=1)

    A, residual = _fit(lattice)
    # Canonicalise the labelling. The detector may start from any corner, so a
    # square board comes back in any of four rotations. That is harmless for the
    # row norms, which are rotation-invariant, but cell_size is indexed BY
    # LATTICE AXIS, so without a fixed convention a non-square cell would be
    # applied to different physical directions on different boards. Convention:
    # lattice axis 0 is whichever board direction runs nearest the image x
    # (scan) axis, so cell_size is always (along-scan, along-spatial).
    if abs(A[0, 1]) > abs(A[0, 0]):
        lattice = lattice[:, ::-1].copy()
        A, residual = _fit(lattice)
    ambiguous = abs(abs(A[0, 1]) - abs(A[0, 0])) < 0.1 * abs(A[0, 0])
    if A[0, 0] < 0:
        # a 180deg relabelling: both lattice axes reversed, so each still runs
        # along the same physical direction and cell_size is unaffected. Undone
        # only so the reported rotation reads as ~0 rather than ~180.
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
    their gradients as well. The two axes are sampled very differently here, so
    a single fixed value is necessarily either wasteful on one axis or oversized
    on the other -- at ~15px spatial spacing the stock (11, 11) window is wider
    than the gap between corners.
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
    board over-erases and can swallow a close neighbour. Filled with the frame
    median rather than 0 so the patch does not become a high-contrast rectangle
    with detectable corners of its own.
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
       ever returns one board, so erasing is what makes the rest findable.
    2. If pass 1 came up short, a deterministic tiled sweep at several window
       sizes with 50% overlap. Whole-frame detection is context-sensitive, and a
       board missed globally is usually found immediately once it is the
       dominant structure in its own window. The sweep is bounded and exhaustive
       rather than an adaptive search whose termination depends on a heuristic.

    inner_size must be the target's true inner-corner count. Ask for a smaller
    pattern and the detector happily returns SUB-WINDOWS of the real board --
    each fits the lattice model perfectly, because a sub-window of a rigid grid
    is a rigid grid, so nothing is rejected and nothing looks wrong. On this rig
    that mattered: a 3x3 request returned two sub-windows of the 20x20 target
    plus the two small boards fixed to the tray, which sit on a different plane,
    and pooling all four biased px_spatial enough to leave +2.3% residual
    anisotropy. Asking for the true 19x19 finds the one real board and the
    residual drops to 0.0%.

    Rejected candidates are reported, not silently dropped: a detection failing
    the lattice check is how a spurious grid announces itself.
    """
    boards, rejected = [], []

    def _seen(board, others):
        """Same patch as one already recorded? Keyed on centroid, since a patch
        re-found from an overlapping tile lands within a pixel or two."""
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

    Median per axis, which ignores a single outlier once there are three or more.

    The two axes are reported separately on purpose, because their spreads mean
    different things. px_scan is frame rate over stage speed: it is the same for
    everything in the capture regardless of where or how high it sits, so
    boards disagreeing on it points at the stage speed drifting mid-scan.
    px_spatial is the optical across-track scale and goes as 1/object-distance,
    so boards disagreeing on that are not coplanar -- and that spread is a lower
    bound on how wrong the correction is for anything off the plane it was
    measured on.
    """
    if not boards:
        return None, None
    px_scan = statistics.median(b["px_scan"] for b in boards)
    px_spatial = statistics.median(b["px_spatial"] for b in boards)

    if len(boards) < expected:
        print(f"  warning: expected {expected} board(s), accepted {len(boards)}; "
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
                  f"1/object-distance, so a {100 * s_spatial:.1f}% spread means the boards are "
                  f"not coplanar. The correction is only exact on the plane it is measured on.")
        if s_scan > BOARD_DISAGREE_WARN:
            print(f"    note: px_scan is frame rate / stage speed and is the same everywhere in "
                  f"the capture, so a {100 * s_scan:.1f}% spread points at the stage speed "
                  f"drifting during the scan rather than at target geometry.")
    return px_scan, px_spatial


def scan_scale_factor(px_scan, px_spatial, reference="spatial"):
    """(fx, fy) multipliers for the (scan, spatial) axes that equalise the two scales.

    spatial (default) rescales the scan axis onto the 640px optical axis; scan
    does the reverse. Only these two: resampling BOTH axes onto some common
    pitch degrades whichever axis was already well sampled and cannot add
    information to the other.
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
    growing: INTER_CUBIC is sharper but overshoots at high-contrast edges, and
    the transmission frame is nothing but high-contrast edges between saturated
    open beam and kernel shadow.

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


def resample_mask(mask, fx, fy):
    """Resample a bool cube with any-source-pixel-set semantics.

    A saturation mask must not be interpolated: a pixel whose value is a blend
    of a clipped and an unclipped source pixel is itself untrustworthy, so any
    overlap marks the output. Done in float32 per band and thresholded above
    zero, which is exactly that rule under INTER_AREA's box average.
    """
    out, _ = resample(mask.astype(np.float32), fx, fy)
    return out > 0.0


def draw_qc_overlay(gray, boards, rejected, path):
    """Write a QC image: what was accepted, what was rejected, and where.

    The failure this guards against is a confident number measured off the wrong
    structure. That is invisible in a log line and obvious in a picture.
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
    come back isotropic; anything else is either a bad measurement or a target
    that is not on the plane the correction was wanted for.
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


# There is deliberately no automatic sample-plane roundness check here, though
# the dish being round makes one look easy. Two candidates were built and both
# measure the wrong thing on this data: segmenting the saturation mask returns
# the convex hull of the WELL CLUSTER, whose aspect is the well layout's (1.16
# here) and has no reason to be 1.0; and cv2.aruco finds neither plate marker on
# the corrected transmittance render. Separating the dish body from the tray
# needs a real dish detector, which is its own task rather than a QC line here.
# What this stage does check -- verify_correction() -- closes the loop on the
# BOARD measurement; the board-to-sample plane offset is left to --plane-factor,
# whose default was set by inspecting the corrected preview.


# ---------------------------------------------------------------- stage 8 --
def intensity_preview(cube, sat_mask=None, lo_pct=1.0, hi_pct=99.0, open_frac=0.5):
    """Greyscale (n_lines, width) uint8 preview: mean across bands, stretched to
    the OPEN BEAM rather than to a percentile.

    A percentile white point is wrong for a transmission frame, and badly so.
    The brightest thing in the scene is the open beam, which is clipped and so
    reads a small absolute transmittance (~0.05 here); the largest VALUES in the
    frame are instead the few rows where the white reference is weak and the
    ratio blows up. Stretching on the 99th percentile therefore sets white from
    those edge rows and renders the actual open beam at ~39% grey, with the
    kernels crushed into near-black below it -- the scene looks far too dark and
    nothing is wrong with the data.

    The open beam is the one level in the frame with a known physical value: it
    is unobstructed, so its true transmittance is 1.0 and every real measurement
    sits below it. Anchoring white to the median of the saturated pixels puts
    the scene where it belongs and clips only the weak-white rows, which are
    genuinely "at least fully open" and are masked anyway.

    Falls back to the percentile stretch when nothing is saturated, i.e. when
    there is no open beam to anchor to.
    """
    # nan-aware throughout: invalid cells are nan by default, and every pixel
    # has some (the lamp is dead in the outermost bands), so a plain mean would
    # return nan for the entire frame.
    mean = np.asarray(np.nanmean(cube, axis=2)).T
    lo = float(np.nanpercentile(mean, lo_pct))
    hi = None
    if sat_mask is not None:
        open_px = (sat_mask.mean(axis=2).T > open_frac) & np.isfinite(mean)
        if open_px.any():
            hi = float(np.nanmedian(mean[open_px]))
    if hi is None or not np.isfinite(hi) or not hi > lo:
        hi = float(np.nanpercentile(mean, hi_pct))
    mean = np.nan_to_num(mean, nan=lo)   # render "no measurement" as the floor
    norm = np.clip((mean - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


def _json_safe(obj):
    """numpy scalars/arrays -> plain Python, so json.dump can write the metadata."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample",
                        help=f"Sample name, e.g. day1_dish0 -- names the outputs "
                             f"(<sample>.npy / .png / _masks.npz / _meta.json) and the archive "
                             f"folder ({STORAGE_DIR}/<sample>/) the raw .bin files move into.")

    src = parser.add_argument_group("inputs")
    src.add_argument("--image", default=None,
                     help=f"Sample scan: raw capture dir or stitched cube .npy. Defaults to "
                          f"{RAW_IMAGE_DIR}/ (eBUS Player's flat capture dir); overriding this "
                          f"skips the post-processing .bin archive step.")
    src.add_argument("--dark-image", default=DEFAULT_DARK_IMAGE_DIR,
                     help=f"Dark reference at the SAMPLE's exposure (default "
                          f"{DEFAULT_DARK_IMAGE_DIR}/). Not interchangeable with --dark-white.")
    src.add_argument("--dark-white", default=DEFAULT_DARK_WHITE_DIR,
                     help=f"Dark reference at the WHITE capture's exposure (default "
                          f"{DEFAULT_DARK_WHITE_DIR}/).")
    src.add_argument("--white", default=DEFAULT_WHITE_DIR,
                     help=f"Dedicated white capture -- a static line scan of the bare light "
                          f"(default {DEFAULT_WHITE_DIR}/).")
    src.add_argument("--checkerboard-dir", default=DEFAULT_CHECKERBOARD_DIR,
                     help=f"Dedicated checkerboard capture, taken at the same exposure and "
                          f"stage speed as the sample scan (default {DEFAULT_CHECKERBOARD_DIR}/).")

    out = parser.add_argument_group("outputs")
    out.add_argument("--out", default="corrected_file",
                     help="Output subfolder for .npy/.npz/.json artifacts (created if missing).")
    out.add_argument("--out-img", default="corrected_image",
                     help="Output subfolder for the preview and QC PNGs (created if missing).")
    out.add_argument("--force", action="store_true",
                     help="Overwrite this sample's outputs if they already exist.")

    rad = parser.add_argument_group("radiometry")
    rad.add_argument("--exposure-scale", type=float, default=None,
                     help="Multiply the white reference by this before dividing, i.e. the "
                          "image/white exposure ratio, turning the output into ABSOLUTE "
                          "transmittance. Off by default: the correction is the plain ratio "
                          "T = (raw - D_long) / (W - D_short) with no constant.")
    rad.add_argument("--sat-level", type=int, default=FULL_SCALE,
                     help=f"Raw count at or above which a pixel is recorded as clipped "
                          f"(default {FULL_SCALE}, the 12-bit full scale).")
    rad.add_argument("--min-white", type=float, default=50.0,
                     help="Absolute floor, in raw counts, below which a (W-D) entry is not a "
                          "measurement and the (row, band) cell is marked invalid.")
    rad.add_argument("--outlier-frac", type=float, default=0.5,
                     help="Relative floor: a (W-D) entry below this fraction of the rolling "
                          "spatial median is marked invalid -- catches a narrow occlusion in the "
                          "white path whose neighbouring rows are healthy.")
    rad.add_argument("--min-band-frac", type=float, default=0.10,
                     help="Spectral floor: a (row, band) cell whose (W-D) is below this fraction "
                          "of that row's peak across bands is marked invalid. The lamp dies at "
                          "both ends of the range (3%% of peak at 1700nm on this rig) and neither "
                          "spatial test notices, so without this every region spectrum gets two "
                          "endpoint spikes several times larger than any real feature and "
                          "autoscales the real structure into the floor.")
    rad.add_argument("--flatfield", choices=["full", "spatial"], default="full",
                     help="full (default): T = (raw - D_long) / (W - D_short), the plain ratio. "
                          "Divides out the lamp's spectral shape along with its spatial one, so "
                          "each spectrum is the sample's own -- at the cost of amplifying the bands "
                          "where the lamp is weakest (see --min-band-frac). spatial: divide by "
                          "(W-D) renormalised per band instead, which equalises illumination "
                          "between kernels identically but keeps the lamp envelope, so spectra "
                          "keep the shape of the raw ones. The two differ by one per-band constant "
                          "(saved as band_scale in the masks .npz), so neither changes any "
                          "spectrum's structure and you can convert between them after the fact.")
    rad.add_argument("--mask-saturated", action="store_true",
                     help="Also write --invalid-value at every saturated voxel. A clipped pixel "
                          "is a lower bound, not a measurement; this stops one being averaged "
                          "into a region spectrum silently.")
    rad.add_argument("--white-window", type=int, default=121,
                     help="Rolling-median window, in spatial rows, for the relative test. Must be "
                          "several times the widest occlusion expected, or the dip drags its own "
                          "baseline down and escapes the test.")
    rad.add_argument("--invalid-value", type=float, default=float("nan"),
                     help="Transmittance written where the white reference is unusable. Defaults "
                          "to nan, because 0.0 is a perfectly legitimate transmittance (an opaque "
                          "sample) and is therefore indistinguishable from a real measurement: a "
                          "region mean over a masked band comes back quietly wrong instead of "
                          "loudly nan, and a spectrum plot shows the curve diving to zero at the "
                          "band edges as though that were the sample. Pass 0 for the old "
                          "behaviour if something downstream cannot take nan (numpy's nanmean "
                          "family generally can).")

    geo = parser.add_argument_group("geometry")
    geo.add_argument("--checkerboard", nargs=2, type=int, default=(19, 19),
                     metavar=("COLS", "ROWS"),
                     help="Inner-corner count of the board (default 19 19 = the 20x20-square "
                          "transmission target). Must be the target's TRUE size -- see --boards.")
    geo.add_argument("--boards", type=int, default=1,
                     help="How many checkerboard targets are in frame (default 1: transmission "
                          "uses a single dedicated board). Raising this on a frame that holds one "
                          "board makes the detector pool unrelated structure into the median.")
    geo.add_argument("--cell-size", nargs=2, type=float, default=(1.0, 1.0),
                     metavar=("W", "H"),
                     help="True physical cell extent along the board's own two lattice axes (any "
                          "consistent unit). Default 1 1 assumes square cells; only the ratio "
                          "matters. The measurement is of the TARGET -- if the printed cell is "
                          "not square the correction inherits that error exactly.")
    geo.add_argument("--plane-factor", type=float, default=DEFAULT_PLANE_FACTOR,
                     help=f"Extra scan-axis multiplier applied after the board measurement, for "
                          f"the board and the kernels not being coplanar. Default "
                          f"{DEFAULT_PLANE_FACTOR}; this script does not measure it (see "
                          f"DEFAULT_PLANE_FACTOR), so changing it means checking the corrected "
                          f"preview for a round dish yourself.")
    geo.add_argument("--reference", choices=["spatial", "scan"], default="spatial",
                     help="Which axis to trust when equalising the scan-axis stretch.")
    geo.add_argument("--manual-scale", type=float, default=None,
                     help="Skip checkerboard detection; force this scan-axis (x) scale factor. "
                          "--plane-factor is NOT applied on top of a manual scale.")
    geo.add_argument("--no-geometry", action="store_true",
                     help="Skip the scan-axis geometry correction entirely.")
    geo.add_argument("--ignore-line-rate", action="store_true",
                     help="Apply the checkerboard factor even when its line period does not match "
                          "the sample scan's. The factor is frame-rate-over-stage-speed, so this "
                          "is only correct if you know why the periods differ.")
    geo.add_argument("--line-rate-tol", type=float, default=0.05,
                     help="Fractional line-period mismatch tolerated between the checkerboard "
                          "capture and the sample scan (default 0.05).")

    qc = parser.add_argument_group("QC")
    qc.add_argument("--no-qc", action="store_true",
                    help="Skip the checkerboard detection overlay PNG.")
    qc.add_argument("--no-verify", action="store_true",
                    help="Skip the closed-loop re-measurement of the boards after correction.")

    args = parser.parse_args()
    meta = {"sample": args.sample, "args": {k: v for k, v in vars(args).items()},
            "warnings": []}

    def warn(msg):
        print(f"  warning: {msg}")
        meta["warnings"].append(msg)

    out_dir, out_img_dir = Path(args.out), Path(args.out_img)
    corrected_path = out_dir / f"{args.sample}.npy"
    masks_path = out_dir / f"{args.sample}_masks.npz"
    meta_path = out_dir / f"{args.sample}_meta.json"
    preview_path = out_img_dir / f"{args.sample}.png"
    existing = [p for p in (corrected_path, masks_path, meta_path, preview_path) if p.exists()]
    if existing and not args.force:
        sys.exit(f"{', '.join(str(p) for p in existing)} already exist(s) -- pass --force to "
                 f"overwrite, or use a different sample name.")

    # archiving only applies to the default flat raw_image/ capture dir, not a
    # custom --image override (e.g. an already-stitched .npy elsewhere)
    archive_dir = Path(STORAGE_DIR) / args.sample if args.image is None else None
    if archive_dir is not None and archive_dir.exists():
        sys.exit(f"{archive_dir} already exists -- pass a different sample name, or clear that "
                 f"folder if it's stale.")

    # -------------------------------------------------------------- stage 0 --
    image_path = Path(args.image) if args.image else Path(RAW_IMAGE_DIR)
    dark_image_path = resolve_capture(args.dark_image, "dark-for-image reference")
    dark_white_path = resolve_capture(args.dark_white, "dark-for-white reference")
    white_path = resolve_capture(args.white, "white reference")
    # Resolved the same way as the other references (dir of .bin, or its lone
    # subdir), but only when it is going to be read: --no-geometry/--manual-scale
    # are the way to run without a board capture at all, and neither should trip
    # over an empty raw_checkerboard/.
    geometry_wanted = not args.no_geometry and args.manual_scale is None
    board_path = (resolve_capture(args.checkerboard_dir, "checkerboard capture")
                  if geometry_wanted else Path(args.checkerboard_dir))

    print("Capture metadata:")
    periods = {}
    for label, path in [("image", image_path), ("dark-image", dark_image_path),
                        ("dark-white", dark_white_path), ("white", white_path),
                        ("checkerboard", board_path)]:
        if not path.is_dir():
            continue
        p = line_period(path)
        periods[label] = p[0] if p else None
        if p:
            print(f"  {label:13s} {path}  {p[1]} lines, line period {p[0]:.0f} ticks "
                  f"(min {p[2]}, max {p[3]})")
        else:
            print(f"  {label:13s} {path}  (no parseable timestamps)")
    meta["line_periods"] = periods

    # No exposure constant. The correction is the plain ratio
    #     T = (raw - D_long) / (W - D_short)
    # and nothing rescales the white onto the sample's exposure. That makes T a
    # RELATIVE transmittance: the white capture here is exposed to near full
    # scale, so an unobstructed (clipped) beam lands close to 1.0 and the scene
    # falls below it -- convenient, but it is a property of how the white was
    # exposed, not a physical normalisation. Change either exposure and the
    # whole cube rescales with nothing recording why, so --exposure-scale is
    # kept for when absolute transmittance is wanted.
    exposure_scale = args.exposure_scale if args.exposure_scale is not None else 1.0
    if args.exposure_scale is None:
        print("Correction: T = (raw - D_long) / (W - D_short), no exposure constant. Output is "
              "RELATIVE transmittance -- an unobstructed beam reads ~1 only because the white "
              "was exposed near full scale, and the scale shifts if either exposure changes.")
    else:
        print(f"Correction: T = (raw - D_long) / ((W - D_short) x {exposure_scale:g}) -- the white "
              f"is scaled onto the sample's exposure, giving absolute transmittance.")
    meta["exposure"] = {"scale": exposure_scale}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------ stages 1 and 2 --
    dark_image_cube, blank = load_cube(dark_image_path, "dark-for-image reference")
    dark_image_frame, meta["dark_image"] = reference_frame(dark_image_cube, "Dark-for-image", blank)
    del dark_image_cube

    dark_white_cube, blank = load_cube(dark_white_path, "dark-for-white reference")
    dark_white_frame, meta["dark_white"] = reference_frame(dark_white_cube, "Dark-for-white", blank)
    del dark_white_cube

    white_cube, blank = load_cube(white_path, "white reference")
    white_frame, meta["white"] = reference_frame(white_cube, "White", blank)
    del white_cube

    image_cube, blank = load_cube(image_path, "sample scan")
    meta["image"] = {"shape": list(image_cube.shape), "n_blank_lines": len(blank)}
    if blank:
        warn(f"{len(blank)} blank/unparseable line(s) in the sample scan at indices "
             f"{_summarise_runs(np.asarray(blank))}.")

    # -------------------------------------------------------------- stage 3 --
    sat_mask, sat_frac = saturation_mask(image_cube, args.sat_level)
    meta["saturation"] = {"level": args.sat_level, "frac": sat_frac}

    # Per band, and restricted to the pixels that clip in SOME bands but not
    # all. A whole-frame per-band average says nothing here: the open beam is
    # tens of times over range at every wavelength, so it clips everywhere and
    # buries the population that matters. A pixel that clips only in part of
    # the spectrum is one with a sample in the path, clipping where that sample
    # is most transparent -- exactly the bands a kernel spectrum is read from,
    # and exactly the ones where a plateau at the rail is indistinguishable
    # from a broad peak unless the mask is consulted.
    band_frac = sat_mask.mean(axis=2)
    partial = (band_frac > 0.01) & (band_frac < 0.99)
    meta["saturation"]["partial_frac"] = float(partial.mean())
    if partial.sum() > 100:
        sat_by_band = sat_mask[partial].mean(axis=0)
        meta["saturation"]["by_band_partial"] = [float(v) for v in sat_by_band]
        hot = np.nonzero(sat_by_band > 0.5)[0]
        print(f"  {100 * partial.mean():.1f}% of pixels clip in only PART of the spectrum -- "
              f"these carry a sample. Among them, clipping peaks at "
              f"{100 * sat_by_band.max():.0f}% (band {int(np.argmax(sat_by_band))}).")
        if hot.size:
            print(f"  bands where >50% of them are clipped: {_summarise_runs(hot)} "
                  f"({hot.size} of {CHANNELS}). No white/dark correction recovers a clipped "
                  f"pixel -- a spectrum over these bands reads the sensor rail as a peak unless "
                  f"it excludes them using the saturated mask in the .npz.")

    # -------------------------------------------------------------- stage 4 --
    darksub, n_neg = apply_dark_correction(image_cube, dark_image_frame)
    del image_cube  # nothing after this point needs the raw cube
    print(f"Dark subtraction: {n_neg} px below zero ({100 * n_neg / darksub.size:.3f}%), "
          f"kept as negatives rather than clipped.")
    meta["dark_subtraction"] = {"n_negative": n_neg, "negative_frac": n_neg / darksub.size}

    # -------------------------------------------------------------- stage 5 --
    white_ref, valid, meta["white_reference"] = build_white_reference(
        white_frame, dark_white_frame, args.min_white, args.outlier_frac, args.white_window,
        args.min_band_frac)
    # Saved regardless of mode: it is the lamp envelope, i.e. the entire
    # difference between the two, so keeping it makes them interconvertible.
    band_scale = band_lamp_profile(white_ref, valid) * exposure_scale
    if args.flatfield == "spatial":
        white_ref = spatial_only_divisor(white_ref, valid)
        print(f"Flat field: SPATIAL only -- dividing by (W-D) renormalised per band, so the "
              f"lamp's spectral envelope stays in the data and a corrected spectrum keeps the "
              f"shape of the raw one. Output is counts at the sample's exposure, NOT absolute "
              f"transmittance, and the exposure scale does not apply (it cancels).")
    else:
        # Scaled after the validity fallback so the literal 1.0 written into invalid
        # cells lands in the same unit space as everything else.
        white_ref = white_ref * exposure_scale
    meta["flatfield"] = args.flatfield

    # -------------------------------------------------------------- stage 6 --
    trans = apply_white_correction(darksub, white_ref, valid, args.invalid_value)
    del darksub  # apply_white_correction works in place; this is the same buffer
    good = valid[:, None, :] & ~sat_mask
    if good.any():
        sample = trans[good]
        lo, med, hi = (float(v) for v in np.percentile(sample, [1, 50, 99]))
        label = "Transmittance" if args.flatfield == "full" else "Flat-fielded signal (counts)"
        print(f"{label} over valid, unsaturated px ({100 * good.mean():.1f}% of the cube): "
              f"p1 {lo:.4f}, median {med:.4f}, p99 {hi:.4f}, max {float(sample.max()):.4f}")
        meta["transmittance"] = {"valid_unsaturated_frac": float(good.mean()),
                                 "p1": lo, "median": med, "p99": hi,
                                 "max": float(sample.max()), "min": float(sample.min()),
                                 "frac_above_1": float((sample > 1.0).mean())}
        # Only meaningful against an ABSOLUTE transmittance scale. Spatial mode
        # emits counts, and without an exposure constant the ceiling is
        # (full_scale - D_long)/(W - D_short), which exceeds 1 wherever the beam
        # is dim -- in both cases "above 1" is expected and says nothing.
        absolute = args.flatfield == "full" and exposure_scale != 1.0
        if absolute and (sample > 1.0).mean() > 0.01:
            warn(f"{100 * (sample > 1.0).mean():.1f}% of valid, unsaturated pixels have "
                 f"transmittance above 1. Above 1 is unphysical for an unobstructed-beam "
                 f"reference; suspect the white reference or the exposure scale.")
        del sample
    else:
        warn("no valid, unsaturated pixels at all -- check the white reference and --sat-level.")
        meta["transmittance"] = {"valid_unsaturated_frac": 0.0}
    del good

    # Why the absolute numbers above are small, stated rather than left to be
    # rediscovered: the open beam is clipped, so it reports a transmittance far
    # below the 1.0 it physically has, and every real measurement sits under
    # that ceiling. Sampled on a 4x4x4 stride -- this is a log line, not a
    # measurement, and the full extraction is a quarter-GB copy.
    # full mode only: the whole point of the number is that an unobstructed beam
    # is 1.0 by definition, which is true of transmittance and meaningless of the
    # counts that spatial mode emits.
    if sat_mask.any() and args.flatfield == "full":
        stride = (slice(None, None, 4),) * 3
        sub, sub_valid = trans[stride], valid[::4, ::4]
        # valid-only: an invalidated cell keeps its saturation flag but its
        # transmittance is the invalid value (nan by default), which would take
        # the median with it.
        sub_mask = sat_mask[stride] & sub_valid[:, None, :]
        if sub_mask.any():
            ceiling = float(np.nanmedian(sub[sub_mask]))
            meta["transmittance"]["saturated_ceiling"] = ceiling
            if ceiling > 0:
                print(f"Open-beam ceiling: saturated px read T={ceiling:.4f} where the true value "
                      f"is 1.0, i.e. the unobstructed beam is ~{1 / ceiling:.0f}x over full scale "
                      f"at this exposure. Everything unsaturated is below that ceiling -- which is "
                      f"why absolute transmittance here is small, and is not a scaling error.")
            # The ceiling is 3355 counts over the local white reference, so it
            # tracks 1/beam-profile and is nowhere near constant across the
            # width. Reported because it looks like an artifact and is not: at
            # the dim edge of the beam the sensor is barely over-range and the
            # open beam reads its true ~1.0, while mid-beam it is tens of times
            # over and pins near zero. That is the whole reason the preview
            # cannot show both ends of the frame on one stretch.
            per_row = []
            for i in range(sub.shape[0]):
                if sub_mask[i].sum() > 50:
                    c = float(np.nanmedian(sub[i][sub_mask[i]]))
                    if np.isfinite(c) and c > 0:
                        per_row.append(c)
            if len(per_row) > 2:
                c_lo, c_hi = min(per_row), max(per_row)
                meta["transmittance"]["ceiling_row_range"] = [c_lo, c_hi]
                print(f"  that ceiling is not uniform: it runs {c_lo:.4f}..{c_hi:.4f} across the "
                      f"spatial axis ({c_hi / c_lo:.0f}x), tracking 1/(white reference). Rows at "
                      f"the dim edge of the beam therefore read near the true 1.0 and clip to "
                      f"white in the preview -- correct measurement, not a blown-out artifact.")
        del sub, sub_mask

    # Applied only now, after the ceiling above has been read off the saturated
    # population -- overwriting them first would leave nothing to measure it from.
    if args.mask_saturated:
        trans[sat_mask] = args.invalid_value
        print(f"Saturated voxels overwritten with {args.invalid_value} (--mask-saturated): a "
              f"clipped pixel is a lower bound, not a measurement, and this makes it impossible "
              f"to average one into a spectrum by accident.")

    # -------------------------------------------------------------- stage 7 --
    fx, fy = 1.0, 1.0
    geo_meta = {"applied": False}
    if args.no_geometry:
        print("Skipping scan-axis geometry correction (--no-geometry).")
    elif args.manual_scale is not None:
        fx, fy = args.manual_scale, 1.0
        geo_meta = {"applied": True, "source": "manual", "fx": fx, "fy": fy}
        print(f"Manual scan-axis scale: x{fx:.5f} (checkerboard detection skipped; "
              f"--plane-factor not applied).")
    else:
        p_img, p_board = periods.get("image"), periods.get("checkerboard")
        mismatch = None
        if p_img and p_board:
            mismatch = abs(p_board / p_img - 1.0)
            print(f"Line-rate gate: checkerboard {p_board:.0f} ticks vs sample {p_img:.0f} ticks "
                  f"({100 * mismatch:.1f}% apart, tolerance {100 * args.line_rate_tol:.0f}%).")
        geo_meta["line_period_mismatch"] = mismatch
        if mismatch is not None and mismatch > args.line_rate_tol and not args.ignore_line_rate:
            warn(f"the checkerboard capture's line period differs from the sample scan's by "
                 f"{100 * mismatch:.1f}%. The scan-axis factor is frame-rate-over-stage-speed and "
                 f"does not transfer between line rates, so the geometry correction is SKIPPED "
                 f"and the cube is written with its native scan-axis stretch. Pass "
                 f"--ignore-line-rate to apply it anyway, or --manual-scale to force a factor.")
        else:
            board_cube, _ = load_cube(board_path, "checkerboard capture")
            gray = render_detection_gray(board_cube)
            del board_cube
            inner, cell = tuple(args.checkerboard), tuple(args.cell_size)
            print(f"Checkerboard {inner[0]}x{inner[1]} inner corners, detecting up to "
                  f"{args.boards} board(s), cell {cell[0]:g}x{cell[1]:g}:")
            boards, rejected = detect_boards(gray, inner, args.boards, cell)
            for board, source, why in rejected:
                print(f"  rejected candidate at ({board['centre'][0]:7.1f},"
                      f"{board['centre'][1]:6.1f}) [{source}]: {why}")
            if not args.no_qc:
                qc_path = out_img_dir / f"{args.sample}_checkerboard_qc.png"
                draw_qc_overlay(gray, boards, rejected, qc_path)
                print(f"  wrote {qc_path}")

            px_scan, px_spatial = reconcile_boards(boards, args.boards)
            geo_meta["boards"] = [{k: b[k] for k in
                                    ("px_scan", "px_spatial", "theta_deg", "skew_deg", "rms_px",
                                     "max_resid_px", "refined", "source")} for b in boards]
            geo_meta["n_rejected"] = len(rejected)
            if px_scan is None:
                warn("no checkerboard accepted; skipping the scan-axis geometry correction "
                     "(the cube is written with its native scan-axis stretch).")
            else:
                fx, fy = scan_scale_factor(px_scan, px_spatial, args.reference)
                print(f"Measured scan={px_scan:.3f} spatial={px_spatial:.3f} px per cell; "
                      f"correction ({args.reference} ref): scan x{fx:.5f}, spatial x{fy:.5f}.")
                if not args.no_verify:
                    geo_meta["verify_anisotropy"] = verify_correction(
                        gray, fx, inner, args.boards, cell)
                # Printed unconditionally, including at 1.0: a non-unity default that
                # only announced itself when overridden is exactly the kind of silent
                # correction that is impossible to track down later.
                if args.plane_factor == 1.0:
                    print(f"Plane factor 1.0: raw board measurement kept, no sample-plane "
                          f"correction. scan x{fx:.5f}.")
                else:
                    fx *= args.plane_factor
                    print(f"Plane factor {args.plane_factor:g} applied for the target/sample "
                          f"plane offset: scan x{fx:.5f}.")
                lo_f, hi_f = FX_SANITY_RANGE
                if not lo_f <= fx <= hi_f:
                    warn(f"scan scale x{fx:.5f} is outside the plausible range "
                         f"[{lo_f}, {hi_f}]; treating it as a bad measurement and skipping "
                         f"the correction.")
                    fx, fy = 1.0, 1.0
                else:
                    geo_meta.update({"applied": True, "source": "checkerboard",
                                     "px_scan": px_scan, "px_spatial": px_spatial,
                                     "plane_factor": args.plane_factor, "fx": fx, "fy": fy})

    if fx != 1.0 or fy != 1.0:
        trans, achieved = resample(trans, fx, fy)
        sat_mask = resample_mask(sat_mask, fx, fy)
        geo_meta["achieved"] = list(achieved)
        if abs(achieved[0] - fx) > 1e-4 or abs(achieved[1] - fy) > 1e-4:
            print(f"  note: integer output size means the applied factors are "
                  f"scan x{achieved[0]:.5f}, spatial x{achieved[1]:.5f}.")
    meta["geometry"] = geo_meta

    # -------------------------------------------------------------- stage 8 --
    np.save(corrected_path, trans)
    if args.flatfield != "full":
        units = "float32 flat-fielded counts at the sample's exposure"
    elif exposure_scale != 1.0:
        units = f"float32 absolute transmittance (white x {exposure_scale:g})"
    else:
        units = "float32 relative transmittance, (raw - D_long)/(W - D_short), no constant"
    print(f"wrote {corrected_path} ({trans.nbytes / 1e6:.0f} MB, shape {trans.shape}, {units})")

    # band_scale is the lamp x QE envelope (times the exposure scale) and is the
    # ONLY thing separating the two flat-field modes, so it is stored in both:
    #     spatial cube / band_scale  ==  the full-transmittance cube
    #     full cube    * band_scale  ==  the spatial cube
    # which makes the choice of mode reversible after the fact rather than a
    # 20-minute re-run.
    np.savez_compressed(masks_path, saturated=sat_mask, white_valid=valid,
                        band_scale=band_scale)
    print(f"wrote {masks_path} (saturated: {sat_mask.shape} bool per-pixel; "
          f"white_valid: {valid.shape} bool per (spatial row, band); "
          f"band_scale: ({len(band_scale)},) -- divide a spatial-mode cube by it to get "
          f"absolute transmittance, multiply a full-mode cube by it to go back)")

    Image.fromarray(intensity_preview(trans, sat_mask), mode="L").save(preview_path)
    print(f"wrote {preview_path} ({trans.shape[1]}x{trans.shape[0]})")

    if archive_dir is not None:
        bin_files = sorted(Path(RAW_IMAGE_DIR).glob("*.bin"))
        archive_dir.mkdir(parents=True)
        for f in bin_files:
            f.rename(archive_dir / f.name)
        print(f"Archived {len(bin_files)} .bin files from {RAW_IMAGE_DIR}/ to {archive_dir}/ "
              f"({RAW_IMAGE_DIR}/ now empty and ready for the next scan).")
        meta["archived"] = {"n_files": len(bin_files), "to": str(archive_dir)}

    meta["outputs"] = {"cube": str(corrected_path), "masks": str(masks_path),
                       "preview": str(preview_path), "meta": str(meta_path)}
    meta["inputs"] = {"image": str(image_path), "dark_image": str(dark_image_path),
                      "dark_white": str(dark_white_path), "white": str(white_path),
                      "checkerboard": str(board_path)}
    with open(meta_path, "w") as fh:
        json.dump(_json_safe(meta), fh, indent=2)
    print(f"wrote {meta_path} ({len(meta['warnings'])} warning(s) recorded)")


if __name__ == "__main__":
    main()
