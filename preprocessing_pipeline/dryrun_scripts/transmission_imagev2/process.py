"""Transmission correction: dark -> white -> geometry -> cube + previews.

The correction is one line:

    T = (raw - D_long) / (W - D_short) * (t_short / t_long)

t_short/t_long is required, not optional. The sample and the white are shot at
different exposures (60 ms and 2.1 ms here), so without that factor the
numerator and denominator are in different units and T comes out ~29x too
large -- which is how the previous script ended up reporting 15% of its pixels
above a transmittance of 1.0. The factor is a single scalar: it sets the units
and cannot distort the spatial flat field, the spectral shape, or one kernel
relative to another.

Everything else in the correction is per (spatial pixel, band). D_long,
D_short and W are each a (640, 224) frame -- the median over that reference
capture's lines -- and they broadcast over the scan axis only. Every pixel gets
its own dark and its own gain; the medians printed in the log are whole-frame
summaries for reading, and nothing divides by them.

Exposures come from the reference folder names (`60k` -> 60000 us, `2100` ->
2100 us). The sample's exposure is read from its DARK folder, since a matched
dark is at the sample's exposure by definition. If either cannot be determined
the run stops rather than assuming 1.0.

THE CHECK THAT MATTERS -- open beam reads 1.0
---------------------------------------------
A light path with nothing in it must come out at T = 1. That is the only
end-to-end test of dark, white and exposure factor together, and it is
available at 60 ms: bands 222-223 sit at the dead end of the lamp, so the bare
beam there stays inside the sensor's range instead of railing. At 100 ms it was
not available at any wavelength -- the beam was 30-47x over full scale
everywhere, so anything called open beam was reading 4095 and its "spectrum"
was the shape of 1/white. Measuring that and calling it a ceiling is how the
previous script convinced itself it had worked.

The check runs only where a band clips almost nowhere, because a clipped beam
reads 4095 no matter how bright it really was: T becomes a lower bound and
falls off smoothly as clipping worsens (measured on this capture: 1.07 at 0%
clipped, 1.00 at 0.9%, 0.88 at 7%, 0.31 at 28%). Ungated, the check would fail
for a reason that has nothing to do with the correction.

What this still does NOT check, deliberately:

  * lamp drift along the scan axis -- would need a column of open beam present
    in every line, and there isn't one (the frame edges are holder, ~700
    counts, not beam). Reporting a per-line mean instead would measure scene
    structure and label it drift.
  * per-band clipping of the kernels -- the useful version needs to know where
    the kernels are. Segmentation-free proxies are dominated by open beam,
    which clips at almost every band, and flag the whole range regardless of
    the kernels. Instead the per-voxel saturation mask is written out and
    rendered as its own PNG, which shows at a glance whether the kernels are
    inside range.

Bad-pixel fill
--------------
A (spatial row, band) cell with no usable white is NaN for every line of the
scan, so it punches a hole through every spectrum taken from that row and makes
a plain region average return NaN for the whole band. Isolated defects are
therefore interpolated from the 8 neighbouring cells at (row +/- 1, band +/- 1),
per scan position. The wide dead stripe is NOT filled -- a cell in the middle of
19 contiguous dead rows has no valid neighbour, and averaging across the stripe
would invent data. `white_valid` in the masks file still records what was
measured; `interpolated` records what was filled.

What it does check, and will say out loud:
  * open beam reads 1.0, at every band where that is measurable
  * T > 1 over valid unsaturated voxels (always, never gated)
  * how much statistical noise the white gain carries -- this is what stripes
    along the scan axis, since a per-(spatial pixel, band) gain error is
    identical for all lines
  * saturation fraction, hot pixels, dropped bands, dead white rows
  * sample vs checkerboard line count, since the scan-axis factor only
    transfers between captures at the same stage speed

Layout (matches transmission_image/; all overridable):
    storage_image/<sample>.npy   stitched sample cube, or storage_raw/<sample>/ of .bin
    dark/<exp>/                  dark at the SAMPLE's exposure   (dark/60k)
    dark/<exp>/                  dark at the WHITE's exposure    (dark/2100)
    white/<exp>/                 dedicated white line scan       (white/2100)
    checkerboard/                board capture, same exposure and stage speed

    python3 process.py day3_dish0_trans

Outputs land in corrected_file/ (cube, masks, meta) and corrected_image/
(previews). Source captures are never moved or deleted.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# hsi_save_load assumes a wider jarvis_gui package that is not in this
# checkout, so import the local copy directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "loadstich"))
from hsi_save_load import load_hsi  # noqa: E402

WIDTH = 640         # spatial pixels per line
CHANNELS = 224      # spectral bands per line
FULL_SCALE = 4095   # 12-bit sensor; at or above this the pixel is clipped

# A (spatial row, band) cell of the white is usable as a gain only if it has
# real signal: at least MIN_WHITE_ABS counts, and at least MIN_WHITE_FRAC of
# what a typical row delivers at that band. The second test is what catches the
# dead stripe in this rig's white capture (~40 counts where a live row reads
# thousands) without needing a window size to tune.
MIN_WHITE_ABS = 100.0
MIN_WHITE_FRAC = 0.15

# Bands where the lamp is below this fraction of its peak are dropped to NaN.
# This is the single most important knob for whether a spectrum looks sane.
#
# The lamp only usefully illuminates ~930-1620 nm. At 1700 nm it delivers 3% of
# its peak and at 900 nm about 7%. Transmittance divides by the lamp, so out
# there the correction divides a nearly-constant floor by a number heading for
# zero, and every spectrum turns up sharply at both ends. That rise is
# arithmetic, not signal -- it appears no matter how good the dark and white
# are, and it is absent from a raw-counts plot only because raw counts never
# divide by the lamp.
#
# 0.50 keeps bands 8-200 (929-1617 nm) on this rig, which is where the lamp
# actually works. Measured on a kernel region, that takes the reported spread
# from 11.9x (at 0.10, both ends blowing up) to 6.2x of real structure.
# Lower it to recover range at the cost of unusable ends; the log prints which
# bands each setting keeps.
MIN_BAND_FRAC = 0.5

# A dark pixel this many times the dark's median is hot, not dark current.
HOT_DARK_FACTOR = 3.0

# Isolated (row, band) defects are interpolated from their 8 neighbours in the
# (row, band) plane, but only if at least this many of the 8 are valid. Half is
# the cut that separates the two populations actually present in this rig: the
# scattered single-pixel defects have 4-8 valid neighbours, while every cell
# with fewer sits inside the ~19-row dead stripe, where the neighbours are dead
# too and there is nothing to interpolate from.
FILL_MIN_NEIGHBOURS = 4

# Open-beam check. Only bands clipping at or below OPEN_BEAM_MAX_CLIP of the
# frame are trusted (see the module docstring). "Open beam" is the brightest
# OPEN_BEAM_TOP_PCT of the scene at that band -- scene-independent, and it
# agrees with the scan lead-in/lead-out lines to within 0.02 
OPEN_BEAM_MAX_CLIP = 0.01
OPEN_BEAM_TOP_PCT = 90.0
OPEN_BEAM_RANGE = (0.80, 1.25)

# Preview stretch. Percentiles are taken over pixels that clip in at most
# PREVIEW_CLIP_TOL of the kept bands -- i.e. the part of the frame that carries
# real signal. Stretching on the whole frame instead lets the clipped open beam
# (~31% here, T up to 0.96) set the white point and flattens the kernels.
PREVIEW_FLOOR = 1e-4          # T is clipped to this before the log preview
PREVIEW_CLIP_TOL = 0.02
PREVIEW_LO_PCT = 0.5
PREVIEW_HI_PCT = 99.5

DEFAULT_IMAGE_DIR = "storage_image"
DEFAULT_IMAGE_RAW_DIR = "storage_raw"
DEFAULT_DARK_IMAGE_DIR = "dark/60k"
DEFAULT_DARK_WHITE_DIR = "dark/2100"
DEFAULT_WHITE_DIR = "white/2100"
DEFAULT_CHECKERBOARD_DIR = "checkerboard"

LINE_NAME_RE = re.compile(r"^(\d+)_([0-9A-Fa-f]+)_w")
EXPOSURE_RE = re.compile(r"^(\d+)([kK])?$")

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print(f"  WARNING: {msg}")


# --------------------------------------------------------------- captures --
def resolve_capture(base, what):
    """A capture from `base`: an .npy, a dir of .bin, or a dir holding one of those.

    Exactly one candidate is required -- a second one usually means a stale
    folder from another session, and picking either silently corrects the run
    against the wrong reference.
    """
    base = Path(base)
    if base.is_file():
        if base.suffix != ".npy":
            raise SystemExit(f"{base} is not an .npy (expected the {what}).")
        return base
    if not base.is_dir():
        raise SystemExit(f"{base} does not exist (expected the {what}).")
    if any(base.glob("*.bin")):
        return base
    cands = sorted([p for p in base.iterdir() if p.suffix == ".npy"]
                   + [p for p in base.iterdir() if p.is_dir() and any(p.glob("*.bin"))])
    if not cands:
        raise SystemExit(f"no .bin or .npy in {base} or below (expected the {what}).")
    if len(cands) > 1:
        raise SystemExit(f"{base} holds {len(cands)} candidates "
                         f"({[p.name for p in cands]}) -- expected one. Pass the "
                         f"{what} explicitly or clear the stale one.")
    return cands[0]


def resolve_sample(explicit, sample):
    """The sample scan: --image if given, else storage_image/<sample>.npy, else storage_raw/<sample>/."""
    if explicit:
        return resolve_capture(explicit, "sample scan")
    npy = Path(DEFAULT_IMAGE_DIR) / f"{sample}.npy"
    if npy.is_file():
        return npy
    raw = Path(DEFAULT_IMAGE_RAW_DIR) / sample
    if raw.is_dir() and any(raw.glob("*.bin")):
        return raw
    raise SystemExit(f"no sample scan for '{sample}': looked for {npy} and "
                     f"{raw}/*.bin. Pass --image explicitly.")

def tfunc():
    pass

def exposure_us(name):
    """Folder name -> exposure in microseconds. '60k' -> 60000, '2100' -> 2100."""
    m = EXPOSURE_RE.match(str(name))
    if not m:
        return None
    return float(m.group(1)) * (1000.0 if m.group(2) else 1.0)


def line_period(path):
    """Median inter-line timestamp delta, in the camera's own ticks, or None.

    Only ever used to compare two captures from the same session, so the tick
    unit does not matter. Returns None for an already-stitched .npy, which has
    no per-line timestamps left.
    """
    path = Path(path)
    if not path.is_dir():
        return None
    stamps = []
    for name in sorted(os.listdir(path)):
        m = LINE_NAME_RE.match(name)
        if m and name.endswith(".bin"):
            stamps.append(int(m.group(2), 16))
    if len(stamps) < 3:
        return None
    dt = np.diff(np.asarray(stamps, dtype=np.int64))
    return float(np.median(dt)), len(stamps)


# --------------------------------------------------------------- stitching --
def stitch(directory):
    """Stitch a directory of per-line .bin captures into (WIDTH, n_lines, CHANNELS)."""
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise SystemExit(f"no .bin files in {directory}")
    cube = np.empty((WIDTH, len(files), CHANNELS), dtype=np.uint16)
    blank = []
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        try:
            frame = load_hsi(raw).reshape([CHANNELS, WIDTH]).swapaxes(0, 1)[::-1]
        except Exception:
            frame = np.zeros((WIDTH, CHANNELS), dtype=np.uint16)
        if not frame.any():
            blank.append(i)
        cube[:, i, :] = frame
    return cube, blank


def load_cube(path, what):
    """Stitch a capture dir, or load an already-stitched .npy."""
    path = Path(path)
    print(f"Loading {what}: {path}")
    if path.is_file() and path.suffix == ".npy":
        cube, blank = np.load(path), []
    else:
        cube, blank = stitch(path)
    if cube.shape[0] != WIDTH or cube.shape[2] != CHANNELS:
        raise SystemExit(f"{path} is {cube.shape}, expected ({WIDTH}, n_lines, {CHANNELS}).")
    print(f"  {cube.shape[1]} lines, shape {cube.shape}"
          + (f", {len(blank)} blank" if blank else ""))
    if blank:
        warn(f"{len(blank)} blank/unparseable line(s) in the {what}.")
    return cube, blank


def reference_frame(cube, name, blank=()):
    """Median over the scan axis -> one (WIDTH, CHANNELS) frame, plus stats.

    Median, not mean: a reference capture is a few dozen lines and one bad line
    moves a mean by percent-level. Both darks here have line 0 sitting ~11%
    below the rest, which a median ignores.
    """
    keep = np.setdiff1d(np.arange(cube.shape[1]), np.asarray(blank, dtype=int))
    if keep.size == 0:
        raise SystemExit(f"every line of the {name} reference is blank.")
    frame = np.median(cube[:, keep, :], axis=1).astype(np.float64)
    stats = {"n_lines": int(cube.shape[1]), "n_used": int(keep.size),
             "min": float(frame.min()), "median": float(np.median(frame)),
             "max": float(frame.max())}
    print(f"  {name}: {keep.size} lines -> frame {frame.min():.0f}"
          f"..{frame.max():.0f}, median {np.median(frame):.0f} "
          f"(a (640, 224) frame -- the median is only a summary)")
    return frame, stats


# ------------------------------------------------------------------ masks --
def _runs(indices):
    """[0,1,2,7,8] -> '0-2, 7-8'."""
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


def hot_pixel_mask(dark_frame):
    """(WIDTH, CHANNELS) bool: dark pixels far above the dark's own median.

    These are sensor defects, not dark current. They survive dark subtraction as
    near-zero or negative outliers and are conspicuous in any spectrum.
    """
    thr = HOT_DARK_FACTOR * float(np.median(dark_frame))
    mask = dark_frame > thr
    print(f"  hot pixels: {int(mask.sum())} of {mask.size} above {thr:.0f} counts")
    return mask, thr


def white_reference(wref, hot, min_band_frac=MIN_BAND_FRAC):
    """Validity masks for a (W - D_short) gain frame.

    Two masks come back. `cell_valid` marks cells with real white signal and is
    what the open-beam check uses, since the bands that make that check possible
    are exactly the weak-lamp ones dropped from the output. `valid` is
    `cell_valid` with those bands removed, and is what the output cube uses.

    The returned gain has invalid cells set to 1.0 so the division cannot
    produce inf; the caller overwrites those positions with NaN afterwards.
    """
    row_typical = np.median(wref, axis=0, keepdims=True)      # per band, over rows
    cell_valid = (wref >= MIN_WHITE_ABS) & (wref >= MIN_WHITE_FRAC * row_typical) & ~hot

    # Bands where the lamp is essentially dead everywhere it is usable.
    band_level = np.array([np.median(wref[cell_valid[:, b], b]) if cell_valid[:, b].any()
                           else 0.0 for b in range(wref.shape[1])])
    band_kept = band_level >= min_band_frac * band_level.max()
    valid = cell_valid & band_kept[None, :]

    dropped = np.nonzero(~band_kept)[0]
    dead_rows = np.nonzero(~valid[:, band_kept].any(axis=1))[0]
    print(f"  white (W-D): {wref.min():.0f}..{wref.max():.0f}, median {np.median(wref):.0f}")
    print(f"  peak lamp at band {int(np.argmax(band_level))} ({band_level.max():.0f} counts)")
    print(f"  bands dropped (lamp below {min_band_frac:g}x peak): "
          f"{_runs(dropped)} ({dropped.size} of {CHANNELS})")
    print(f"  spatial rows with no usable white in any kept band: "
          f"{_runs(dead_rows)} ({dead_rows.size} of {WIDTH})")
    print(f"  invalid (row, band) cells: {int((~valid).sum())} "
          f"({100 * (~valid).mean():.2f}%) -> NaN in the output")

    stats = {"min": float(wref.min()), "median": float(np.median(wref)),
             "max": float(wref.max()), "invalid_frac": float((~valid).mean()),
             "peak_band": int(np.argmax(band_level)),
             "bands_dropped": [int(b) for b in dropped],
             "n_bands_kept": int(band_kept.sum()),
             "min_band_frac": float(min_band_frac),
             "bands_kept_range": [int(np.nonzero(band_kept)[0].min()),
                                  int(np.nonzero(band_kept)[0].max())],
             "dead_rows": [int(r) for r in dead_rows]}
    return np.where(valid, wref, 1.0), valid, cell_valid, band_kept, band_level, stats


def white_gain_noise(white_cube, wref, valid):
    """Relative statistical noise in the white gain, as a percentage.

    Each spatial pixel carries its own gain, and an error in that gain is
    identical for every line of the scan -- so it renders as a stripe along the
    scan axis. This is the amplitude to expect. Scene-independent by
    construction; it never touches the sample.
    """
    n = white_cube.shape[1]
    per_line_std = white_cube.astype(np.float32).std(axis=1)
    rel = (per_line_std / np.sqrt(n)) / np.maximum(wref, 1.0)
    good = rel[valid]
    med, p99 = 100 * float(np.median(good)), 100 * float(np.percentile(good, 99))
    print(f"  white gain noise over {n} lines: median {med:.3f}%, p99 {p99:.3f}% "
          f"-> expected scan-axis striping amplitude")
    return {"median_pct": med, "p99_pct": p99, "n_lines": int(n)}


# --------------------------------------------------------------- the check --
def open_beam_check(cube, dark_long, wref_raw, cell_valid, sat_mask, k):
    """T of the brightest part of the scene, at every band that barely clips.

    The one end-to-end test of dark, white and exposure factor together: a light
    path with nothing in it must read T = 1. Uses the unmasked (W - D_short) and
    `cell_valid` rather than the output's `valid`, because the bands where the
    bare beam stays in range are the weak-lamp ones the output drops.

    Gated on per-band clipping -- see the module docstring for why an ungated
    version reads 0.3 and means nothing.
    """
    clip = sat_mask.mean(axis=(0, 1))
    cands = [b for b in np.nonzero(clip <= OPEN_BEAM_MAX_CLIP)[0]
             if cell_valid[:, b].sum() >= 32]
    if not cands:
        best = int(np.argmin(clip))
        warn(f"open-beam check unavailable: the least-clipped band ({best}) still clips "
             f"{100 * clip[best]:.1f}% of the frame, over the {100 * OPEN_BEAM_MAX_CLIP:g}% "
             f"limit. A railed beam reads {FULL_SCALE} regardless of its true level, so T "
             f"there is a lower bound, not a test. Shorten the exposure to enable this.")
        return {"available": False, "min_clip_frac": float(clip[best]),
                "min_clip_band": best}

    per_band = {}
    for b in cands:
        rows = np.nonzero(cell_valid[:, b])[0]
        raw = cube[rows, :, b].astype(np.float64)
        t = (raw - dark_long[rows, b][:, None]) / wref_raw[rows, b][:, None] * k
        bright = raw >= np.percentile(raw, OPEN_BEAM_TOP_PCT)
        per_band[int(b)] = float(np.median(t[bright]))

    vals = np.array(list(per_band.values()))
    med = float(np.median(vals))
    print(f"  open beam should read T = 1.0. Measured on the brightest "
          f"{100 - OPEN_BEAM_TOP_PCT:g}% of the scene, at the {len(cands)} band(s) "
          f"clipping under {100 * OPEN_BEAM_MAX_CLIP:g}%:")
    for b, v in sorted(per_band.items()):
        print(f"    band {b:3d} ({100 * clip[b]:.2f}% clipped): T = {v:.3f}")
    lo, hi = OPEN_BEAM_RANGE
    if not (lo <= med <= hi):
        warn(f"open beam reads T = {med:.3f}, outside {lo}-{hi}. Dark, white or the "
             f"exposure factor k={k:.6g} is wrong -- the whole cube is off by about "
             f"this factor.")
    else:
        print(f"  -> median {med:.3f}, inside {lo}-{hi}. Dark, white and k all check out.")
    return {"available": True, "median": med, "per_band": per_band,
            "bands_used": [int(b) for b in cands]}


# ------------------------------------------------------------- correction --
def transmittance(cube, dark_long, wref, valid, k):
    """T = (raw - D_long) / (W - D_short) * k, on one float32 buffer.

    Negatives are kept. Kernel signal is a few hundred to a few thousand counts
    above dark, and clipping at zero would rectify the noise floor into a
    positive bias exactly where the measurement is weakest.
    """
    out = cube.astype(np.float32)
    out -= dark_long[:, None, :].astype(np.float32)
    out /= wref.astype(np.float32)[:, None, :]
    out *= np.float32(k)
    rows, bands = np.nonzero(~valid)
    out[rows, :, bands] = np.nan
    return out


def fill_bad_cells(cube, valid, band_kept):
    """Interpolate isolated (spatial row, band) defects from their 8 neighbours.

    A defect here is not one voxel. An invalid white cell kills that (row, band)
    for every line of the scan, which is why it shows up as a missing point in
    every spectrum drawn from that row -- and why a region average over it
    returns NaN for the whole band. The defect is fixed in the (row, band)
    plane, so the fill is taken there: the 8 cells at (row +/- 1, band +/- 1),
    averaged independently at each scan position. Row +/- 1 is an adjacent
    pixel and band +/- 1 is ~3.6 nm away, so both are close enough to stand in.

    Only cells with at least FILL_MIN_NEIGHBOURS valid neighbours are filled,
    and only inside the kept bands. That confines the fill to genuine isolated
    defects; the wide dead stripe is left as NaN because interpolating across 19
    contiguous dead rows would be inventing data, not recovering it. Dropped
    bands are left alone for the same reason -- the lamp is dead there.

    Sources are always original valid cells, never other fills, so the result
    does not depend on the order cells are visited.
    """
    n_rows, n_bands = valid.shape
    filled = np.zeros_like(valid)
    jobs = []
    for r, b in zip(*np.nonzero(~valid & band_kept[None, :])):
        src = [(r + dr, b + db) for dr in (-1, 0, 1) for db in (-1, 0, 1)
               if (dr or db) and 0 <= r + dr < n_rows and 0 <= b + db < n_bands
               and valid[r + dr, b + db]]
        if len(src) >= FILL_MIN_NEIGHBOURS:
            jobs.append((r, b, src))
    for r, b, src in jobs:
        acc = np.zeros(cube.shape[1], dtype=np.float32)
        for nr, nb in src:
            acc += cube[nr, :, nb]
        cube[r, :, b] = acc / len(src)
        filled[r, b] = True

    left = int((~valid & band_kept[None, :]).sum()) - int(filled.sum())
    print(f"  interpolated {int(filled.sum())} isolated (row, band) defect(s) from "
          f"their 8 neighbours (>= {FILL_MIN_NEIGHBOURS} valid required)")
    print(f"    rows affected: {_runs(np.unique(np.nonzero(filled)[0]))}")
    print(f"    left as NaN: {left} cell(s) with too few valid neighbours -- the dead "
          f"stripe, where there is nothing to interpolate from")
    return filled, {"n_filled": int(filled.sum()), "n_left_nan": left,
                    "min_neighbours": FILL_MIN_NEIGHBOURS,
                    "rows": [int(r) for r in np.unique(np.nonzero(filled)[0])]}


# --------------------------------------------------------------- geometry --
def render_gray(cube, dark_frame, band):
    """One percentile-stretched uint8 band image, for corner detection."""
    plane = cube[:, :, band].astype(np.float32) - dark_frame[:, band][:, None]
    lo, hi = np.percentile(plane, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((plane - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def measure_scan_scale(gray, inner):
    """Scan-axis compression factor from a checkerboard capture, or None.

    The push-broom scan axis is stretched relative to the 640 px optical axis by
    however much faster or slower the stage moved than the frame rate. A board of
    square cells measures that directly: fx = spatial_spacing / scan_spacing.

    Classic findChessboardCorners plus cornerSubPix, deliberately not
    findChessboardCornersSB -- SB's measured axis ratio degrades badly on an
    oversampled axis, which is exactly this case.
    """
    found, corners = cv2.findChessboardCorners(
        gray, inner, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))

    # gray is indexed [spatial, scan], so cv2's (x, y) is (scan, spatial).
    pts = corners.reshape(inner[1], inner[0], 2)
    steps = [np.diff(pts, axis=0).reshape(-1, 2), np.diff(pts, axis=1).reshape(-1, 2)]
    spacing = [float(np.linalg.norm(s, axis=1).mean()) for s in steps]
    # Whichever lattice axis steps mostly along cv2-x is the scan axis. Using
    # vector norms for the spacing makes this tolerant of small board rotation.
    along_scan = 0 if abs(steps[0].mean(axis=0)[0]) > abs(steps[1].mean(axis=0)[0]) else 1
    px_scan, px_spatial = spacing[along_scan], spacing[1 - along_scan]
    if not (px_scan > 1.0 and px_spatial > 1.0):
        return None
    return {"px_scan": px_scan, "px_spatial": px_spatial, "fx": px_spatial / px_scan}


def resample_scan(arr, new_n):
    """Rescale the scan axis of a (spatial, n, bands) float array, band by band.

    Only the scan axis is touched, so a whole-(row, band) NaN line stays NaN and
    no NaN bleeds into a valid neighbour.
    """
    out = np.empty((arr.shape[0], new_n, arr.shape[2]), dtype=np.float32)
    for b in range(arr.shape[2]):
        out[:, :, b] = cv2.resize(arr[:, :, b], (new_n, arr.shape[0]),
                                  interpolation=cv2.INTER_AREA)
    return out


def resample_mask(mask, new_n):
    """Rescale a bool (spatial, n, bands) mask; any contributing source pixel wins."""
    out = np.empty((mask.shape[0], new_n, mask.shape[2]), dtype=bool)
    for b in range(mask.shape[2]):
        r = cv2.resize(mask[:, :, b].astype(np.float32), (new_n, mask.shape[0]),
                       interpolation=cv2.INTER_AREA)
        out[:, :, b] = r > 0
    return out


# ---------------------------------------------------------------- outputs --
def band_mean(cube, sat_mask, band_kept):
    """Mean transmittance across the kept bands. -> (mean_plane, clipped_frac).

    Accumulated band by band rather than np.nanmean over the whole cube, which
    would allocate a second copy of a ~900 MB array.

    Each band contributes equally: T is already normalised by the white, so a
    plain mean is a mean transmittance and not a lamp-weighted one. Voxels that
    are NaN (invalid white, dropped band) are skipped, so a pixel valid in only
    some bands still gets a mean over the ones it has. A pixel with nothing
    valid stays NaN.
    """
    acc = np.zeros(cube.shape[:2], dtype=np.float64)
    cnt = np.zeros(cube.shape[:2], dtype=np.int32)
    sat = np.zeros(cube.shape[:2], dtype=np.int32)
    bands = np.nonzero(band_kept)[0]
    for b in bands:
        plane = cube[:, :, b]
        m = np.isfinite(plane)
        acc[m] += plane[m]
        cnt[m] += 1
        sat += sat_mask[:, :, b]
    out = np.full(cube.shape[:2], np.nan, dtype=np.float32)
    good = cnt > 0
    out[good] = (acc[good] / cnt[good]).astype(np.float32)
    return out, (sat / max(len(bands), 1)).astype(np.float32)


def preview_png(plane, clipped_frac, path, note):
    """Log-scaled PNG of a 2D plane. Bright = more light through, NaN black.

    Log, not linear: the sample spans T ~ 0.002-0.07 against a clipped-beam
    ceiling near 1.0, so a linear stretch spends its whole range on nothing.

    The stretch percentiles come from pixels that barely clip, which is the part
    of the frame that carries real signal. Anchoring on the whole frame instead
    lets the railed open beam set the white point, and the sample collapses into
    the bottom few grey levels.
    """
    finite = np.isfinite(plane)
    if not finite.any():
        raise SystemExit("preview plane is entirely invalid.")
    clean = finite & (clipped_frac <= PREVIEW_CLIP_TOL)
    pop = clean if clean.sum() >= 1000 else finite
    v = np.log10(np.clip(plane, PREVIEW_FLOOR, None))
    lo = float(np.percentile(v[pop], PREVIEW_LO_PCT))
    hi = float(np.percentile(v[pop], PREVIEW_HI_PCT))
    if hi <= lo:
        hi = lo + 1.0
    img = np.clip((v - lo) / (hi - lo) * 255.0, 0, 255)
    img[~finite] = 0
    cv2.imwrite(str(path), cv2.rotate(img.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE))
    print(f"  preview: {path} ({note}, stretched T {10 ** lo:.5f}..{10 ** hi:.5f} over "
          f"the {100 * pop.mean():.0f}% of pixels clipping under "
          f"{100 * PREVIEW_CLIP_TOL:g}% of bands)")
    return {"lo_T": 10 ** lo, "hi_T": 10 ** hi, "stretch_pop_frac": float(pop.mean())}


def saturation_png(clipped_frac, finite, path):
    """Per-pixel fraction of kept bands that clipped, as a colour map.

    The companion the greyscale preview needs: a clipped pixel renders bright
    there and is indistinguishable from a genuinely transparent one, because a
    railed reading carries no information about how bright it really was. Here
    black means every band is in range and yellow means every band is railed, so
    it is directly readable as "is my sample inside range". Cyan is invalid
    white (NaN) -- cyan is absent from the colour map, so it cannot be misread
    as a value.
    """
    cm = cv2.applyColorMap((np.clip(clipped_frac, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    cm[~finite] = (255, 255, 0)   # BGR cyan
    cv2.imwrite(str(path), cv2.rotate(cm, cv2.ROTATE_90_CLOCKWISE))
    clean = float((clipped_frac[finite] == 0).mean()) if finite.any() else 0.0
    print(f"  saturation map: {path} (black = no band clipped, yellow = all clipped, "
          f"cyan = invalid; {100 * clean:.1f}% of valid pixels fully in range)")
    return {"fully_in_range_frac": clean}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample", help="output name, and the default input name "
                                  "(storage_image/<sample>.npy), e.g. day3_dish0_trans")
    p.add_argument("--image", default=None,
                   help=f"sample scan (default {DEFAULT_IMAGE_DIR}/<sample>.npy, "
                        f"else {DEFAULT_IMAGE_RAW_DIR}/<sample>/)")
    p.add_argument("--dark-image", default=DEFAULT_DARK_IMAGE_DIR,
                   help="dark at the SAMPLE's exposure; its folder name sets t_long")
    p.add_argument("--dark-white", default=DEFAULT_DARK_WHITE_DIR,
                   help="dark at the WHITE's exposure")
    p.add_argument("--white", default=DEFAULT_WHITE_DIR,
                   help="dedicated white scan; its folder name sets t_short")
    p.add_argument("--min-band-frac", type=float, default=MIN_BAND_FRAC,
                   help="drop bands where the lamp is below this fraction of its peak "
                        f"(default {MIN_BAND_FRAC}: keeps ~929-1617 nm, the range this "
                        "lamp actually illuminates)")
    p.add_argument("--checkerboard", default=DEFAULT_CHECKERBOARD_DIR)
    p.add_argument("--checkerboard-size", type=int, nargs=2, default=(19, 19),
                   metavar=("COLS", "ROWS"), help="inner corner count")
    p.add_argument("--exposure-long", type=float, default=None,
                   help="sample exposure in ms, overriding the dark folder name")
    p.add_argument("--exposure-short", type=float, default=None,
                   help="white exposure in ms, overriding the white folder name")
    p.add_argument("--manual-scale", type=float, default=None,
                   help="scan-axis factor, skipping checkerboard detection")
    p.add_argument("--no-geometry", action="store_true")
    p.add_argument("--out", default="corrected_file")
    p.add_argument("--out-img", default="corrected_image")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = p.parse_args()

    out_dir, out_img_dir = Path(args.out), Path(args.out_img)
    cube_path = out_dir / f"{args.sample}.npy"
    if cube_path.exists() and not args.force:
        raise SystemExit(f"{cube_path} exists; pass --force to overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    meta = {"sample": args.sample, "args": vars(args)}

    image_path = resolve_sample(args.image, args.sample)
    dark_long_path = resolve_capture(args.dark_image, "dark at the sample's exposure")
    dark_short_path = resolve_capture(args.dark_white, "dark at the white's exposure")
    white_path = resolve_capture(args.white, "white reference")
    want_geometry = not args.no_geometry and args.manual_scale is None
    board_path = (resolve_capture(args.checkerboard, "checkerboard capture")
                  if want_geometry else Path(args.checkerboard))

    # ---------------------------------------------------------- exposures --
    print("Exposures:")
    t_long = (args.exposure_long * 1000.0 if args.exposure_long is not None
              else exposure_us(dark_long_path.name))
    t_short = (args.exposure_short * 1000.0 if args.exposure_short is not None
               else exposure_us(white_path.name))
    if t_long is None:
        raise SystemExit(f"cannot read the sample's exposure from '{dark_long_path.name}' "
                         f"-- rename it (e.g. 60k) or pass --exposure-long.")
    if t_short is None:
        raise SystemExit(f"cannot read the white's exposure from '{white_path.name}' "
                         f"-- rename it (e.g. 2100) or pass --exposure-short.")
    t_dark_short = exposure_us(dark_short_path.name)
    if t_dark_short is not None and abs(t_dark_short - t_short) > 1e-6:
        warn(f"the white's dark is at {t_dark_short:.0f} us but the white is at "
             f"{t_short:.0f} us -- they should match.")
    k = t_short / t_long
    print(f"  sample {t_long:.0f} us, white {t_short:.0f} us -> k = t_short/t_long "
          f"= {k:.6g} (1/{1 / k:.2f})")
    meta["exposure"] = {"t_long_us": t_long, "t_short_us": t_short, "k": k}

    # --------------------------------------------------------- references --
    print("References:")
    cube, blank = load_cube(dark_long_path, "dark at the sample's exposure")
    dark_long, meta["dark_long"] = reference_frame(cube, "dark_long", blank)
    del cube
    cube, blank = load_cube(dark_short_path, "dark at the white's exposure")
    dark_short, meta["dark_short"] = reference_frame(cube, "dark_short", blank)
    del cube
    white_cube, blank = load_cube(white_path, "white reference")
    white_frame, meta["white"] = reference_frame(white_cube, "white", blank)

    hot, hot_thr = hot_pixel_mask(dark_long)
    meta["hot_pixels"] = {"n": int(hot.sum()), "threshold": hot_thr}
    wref_raw = white_frame - dark_short          # W - D_short, unmasked
    (wref, valid, cell_valid, band_kept, band_level,
     meta["white_reference"]) = white_reference(wref_raw, hot, args.min_band_frac)
    meta["white_gain_noise"] = white_gain_noise(white_cube, wref, valid)
    del white_cube

    # ------------------------------------------------------------- sample --
    print("Sample:")
    image_cube, blank = load_cube(image_path, "sample scan")
    meta["image"] = {"shape": list(image_cube.shape), "n_blank_lines": len(blank)}

    sat_mask = image_cube >= FULL_SCALE
    sat_frac = float(sat_mask.mean())
    print(f"  saturated: {100 * sat_frac:.2f}% of voxels at >= {FULL_SCALE}. Expected to be "
          f"large -- the open beam beside and between the wells is still over range.")
    meta["saturation"] = {"frac": sat_frac}
    # Deliberately no per-band clipping breakdown here. The number that matters
    # is how much KERNEL data clips, and that needs to know where the kernels
    # are. Every segmentation-free proxy is dominated by open beam, which clips
    # almost everywhere, and reports nearly every band as compromised
    # regardless of the kernels. The saturation PNG shows it spatially instead,
    # and the per-voxel mask is written out for when segmentation exists.

    print("Checks:")
    meta["open_beam"] = open_beam_check(image_cube, dark_long, wref_raw,
                                        cell_valid, sat_mask, k)

    trans = transmittance(image_cube, dark_long, wref, valid, k)
    del image_cube
    filled, meta["fill"] = fill_bad_cells(trans, valid, band_kept)

    # ------------------------------------------------------------ geometry --
    fx = 1.0
    geo = {"applied": False}
    if args.manual_scale is not None:
        fx, geo = args.manual_scale, {"applied": True, "source": "manual"}
        print(f"Geometry: manual scan-axis factor {fx:.5f}")
    elif args.no_geometry:
        print("Geometry: skipped (--no-geometry)")
    else:
        print("Geometry:")
        board_cube, _ = load_cube(board_path, "checkerboard capture")
        band = int(np.argmax(band_level))
        result = measure_scan_scale(
            render_gray(board_cube, dark_long, band), tuple(args.checkerboard_size))
        n_board = board_cube.shape[1]
        del board_cube
        if result is None:
            warn(f"no {args.checkerboard_size[0]}x{args.checkerboard_size[1]} board found in "
                 f"{board_path} -- scan axis left uncorrected. Pass --manual-scale.")
            geo = {"applied": False, "source": "checkerboard", "detected": False}
        else:
            fx = result["fx"]
            geo = {"applied": True, "source": "checkerboard", "detected": True, **result}
            print(f"  board cell {result['px_spatial']:.2f} px spatial x "
                  f"{result['px_scan']:.2f} px scan -> fx = {fx:.5f}")
            # The factor is frame-rate over stage-speed. Line rate is checkable
            # from the timestamps, when both captures still have them; stage
            # speed is not, and a capture that swept the same object in a very
            # different number of lines did not run at the same speed. Neither
            # test can run on an already-stitched .npy, hence the line-count
            # fallback below.
            per_board, per_image = line_period(board_path), line_period(image_path)
            if per_board and per_image:
                geo["line_period_board"] = per_board[0]
                geo["line_period_image"] = per_image[0]
                if abs(per_board[0] - per_image[0]) > 0.05 * per_board[0]:
                    warn(f"line period differs: board {per_board[0]:.0f} ticks vs sample "
                         f"{per_image[0]:.0f}. The factor does not transfer.")
            n_image = int(meta["image"]["shape"][1])
            geo["n_lines_board"], geo["n_lines_image"] = int(n_board), n_image
            if abs(n_image - n_board) > 0.10 * n_board:
                # Fewer lines over the same travel means the stage moved further
                # per line, so the scan axis is less oversampled and needs less
                # compression: px_scan scales by n_image/n_board, so fx scales by
                # its inverse. Only valid if both swept the same travel, hence a
                # suggestion to check against the preview rather than an auto-fix.
                warn(f"sample is {n_image} lines but the board is {n_board} "
                     f"({n_image / n_board:.2f}x). If both swept the same travel then the "
                     f"stage speed differed and fx={fx:.4f} is wrong for this sample by about "
                     f"that ratio -- the dish will not come out round. Check the preview, and "
                     f"if it is off try --manual-scale {fx * n_board / n_image:.4f}.")
            else:
                print(f"  sample {n_image} lines vs board {n_board} "
                      f"({n_image / n_board:.3f}x) -- same stage speed, factor transfers")

    if geo.get("applied"):
        new_n = max(1, int(round(trans.shape[1] * fx)))
        print(f"  scan axis {trans.shape[1]} -> {new_n} px")
        trans = resample_scan(trans, new_n)
        sat_mask = resample_mask(sat_mask, new_n)
        geo["fx"] = fx
    meta["geometry"] = geo

    # ------------------------------------------------------------- checks --
    print("Checks:")
    good = valid[:, None, :] & ~sat_mask
    if good.any():
        sample = trans[good]
        sample = sample[np.isfinite(sample)]
        lo, med, hi = (float(v) for v in np.percentile(sample, [1, 50, 99]))
        n_above = int((sample > 1.0).sum())
        frac_above = n_above / sample.size
        print(f"  transmittance over valid unsaturated voxels "
              f"({100 * good.mean():.1f}% of the cube): p1 {lo:.5f}, median {med:.5f}, "
              f"p99 {hi:.5f}, max {float(sample.max()):.5f}")
        meta["transmittance"] = {"valid_unsaturated_frac": float(good.mean()),
                                 "p1": lo, "median": med, "p99": hi,
                                 "max": float(sample.max()), "min": float(sample.min()),
                                 "frac_above_1": frac_above}
        # Always on. T > 1 means more light through the sample than through the
        # bare beam, which is not possible.
        if frac_above > 0.001:
            warn(f"{100 * frac_above:.2f}% of valid unsaturated voxels have T > 1, which is "
                 f"unphysical. Suspect the exposure factor k={k:.6g} or the white reference.")
        else:
            print(f"  T > 1: {n_above} voxels ({100 * frac_above:.3f}%) -- physically OK")
        del sample
    else:
        warn("no valid unsaturated voxels at all -- check the white reference.")
        meta["transmittance"] = {"valid_unsaturated_frac": 0.0}
    del good

    # ------------------------------------------------------------ outputs --
    print("Writing:")
    np.save(cube_path, trans)
    print(f"  cube: {cube_path} {trans.shape} float32")
    masks_path = out_dir / f"{args.sample}_masks.npz"
    # white_valid stays the record of what was MEASURED -- it is not widened by
    # the fill. `interpolated` says which cells hold a neighbour average instead,
    # so anything downstream can exclude them.
    np.savez_compressed(masks_path, saturated=sat_mask, white_valid=valid,
                        band_kept=band_kept, hot_pixels=hot, band_level=band_level,
                        interpolated=filled)
    print(f"  masks: {masks_path} (saturated, white_valid, band_kept, hot_pixels, "
          f"interpolated)")

    mean_plane, clipped_frac = band_mean(trans, sat_mask, band_kept)
    finite = np.isfinite(mean_plane)
    meta["preview"] = {"kind": "mean over kept bands", "n_bands": int(band_kept.sum())}
    meta["preview"].update(preview_png(
        mean_plane, clipped_frac, out_img_dir / f"{args.sample}.png",
        f"mean of {int(band_kept.sum())} bands"))
    meta["preview"].update(saturation_png(
        clipped_frac, finite, out_img_dir / f"{args.sample}_saturation.png"))

    meta["warnings"] = WARNINGS
    meta["inputs"] = {"image": image_path, "dark_long": dark_long_path,
                      "dark_short": dark_short_path, "white": white_path,
                      "checkerboard": board_path if want_geometry else None}
    meta_path = out_dir / f"{args.sample}_meta.json"
    meta_path.write_text(json.dumps(_json_safe(meta), indent=2))
    print(f"  meta: {meta_path}")
    print(f"\n{len(WARNINGS)} warning(s)."
          + ("" if WARNINGS else " Nothing flagged."))


if __name__ == "__main__":
    main()
