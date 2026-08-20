"""TRANSMITTANCE correction. The whole thing is one line:

    T = (raw - D_long) / (W - D_short) * (t_short / t_long)

t_short/t_long is required, not optional. The sample and the white are shot at
different exposures (60 ms and 2.1 ms on this rig), so without that factor the
numerator and denominator are in different units and T comes out ~29x too large
-- which is how an earlier version ended up reporting 15% of its pixels above a
transmittance of 1.0. The factor is a single scalar: it sets the units and
cannot distort the spatial flat field, the spectral shape, or one kernel
relative to another.

Everything else is per (spatial pixel, band). D_long, D_short and W are each a
(640, 224) frame -- the median over that reference capture's lines -- and they
broadcast over the scan axis only. Every pixel gets its own dark and its own
gain; the medians in the log are whole-frame summaries for reading, and nothing
divides by them.

Exposures come from the calibration folder names ("60k" -> 60000 us, "2100" ->
2100 us). The sample's exposure is read from its DARK folder, since a matched
dark is at the sample's exposure by definition. If either cannot be determined
the run stops rather than assuming 1.0.

THE CHECK THAT MATTERS -- open beam reads 1.0
---------------------------------------------
A light path with nothing in it must come out at T = 1. That is the only
end-to-end test of dark, white and exposure factor together, and it is
available at 60 ms: the bands at the dead end of the lamp keep the bare beam
inside the sensor's range instead of railing. At 100 ms it was not available at
any wavelength -- the beam was 30-47x over full scale everywhere, so anything
called open beam was reading 4095 and its "spectrum" was the shape of 1/white.

The check runs only where a band clips almost nowhere, because a clipped beam
reads 4095 no matter how bright it really was: T becomes a lower bound and falls
off smoothly as clipping worsens (measured: 1.07 at 0% clipped, 1.00 at 0.9%,
0.88 at 7%, 0.31 at 28%). Ungated, the check would fail for a reason that has
nothing to do with the correction.

Bad-pixel fill
--------------
A (spatial row, band) cell with no usable white is NaN for every line of the
scan, so it punches a hole through every spectrum taken from that row and makes
a plain region average return NaN for the whole band. Isolated defects are
interpolated from the 8 neighbouring cells at (row +/- 1, band +/- 1), per scan
position. The wide dead stripe is NOT filled -- a cell in the middle of 19
contiguous dead rows has no valid neighbour, and averaging across the stripe
would invent data. `white_valid` in the masks file records what was measured;
`interpolated` records what was filled.

What it checks and will say out loud:
  * open beam reads 1.0, at every band where that is measurable
  * T > 1 over valid unsaturated voxels (always, never gated)
  * how much statistical noise the white gain carries -- this is what stripes
    along the scan axis, since a per-(spatial pixel, band) gain error is
    identical for all lines
  * saturation fraction, hot pixels, dropped bands, dead white rows
  * sample vs checkerboard line count, since the scan-axis factor only transfers
    between captures at the same stage speed
"""

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from common import (CHANNELS, FULL_SCALE, WIDTH, cached_npz, capture_key,  # noqa: E402
                    combine_keys, describe, json_safe, load_cube, measure_geometry,
                    reference_frame, resample, resample_mask, resolve_calibration,
                    runs, warn)

CALIBRATION = Path(__file__).resolve().parent / "calibration"
CACHE = CALIBRATION / ".cache"

# A (spatial row, band) cell of the white is usable as a gain only if it has
# real signal: at least MIN_WHITE_ABS counts, and at least MIN_WHITE_FRAC of
# what a typical row delivers at that band. The second test is what catches the
# dead stripe in this rig's white capture (~40 counts where a live row reads
# thousands) without needing a window size to tune.
MIN_WHITE_ABS = 100.0
MIN_WHITE_FRAC = 0.15

# A dark pixel this many times the dark's median is hot, not dark current.
HOT_DARK_FACTOR = 3.0

# Isolated (row, band) defects are interpolated from their 8 neighbours in the
# (row, band) plane, but only if at least this many of the 8 are valid. Half is
# the cut that separates the two populations actually present in this rig: the
# scattered single-pixel defects have 4-8 valid neighbours, while every cell with
# fewer sits inside the ~19-row dead stripe, where the neighbours are dead too
# and there is nothing to interpolate from.
FILL_MIN_NEIGHBOURS = 4

# Open-beam check. Only bands clipping at or below OPEN_BEAM_MAX_CLIP of the
# frame are trusted. "Open beam" is the brightest OPEN_BEAM_TOP_PCT of the scene
# at that band -- scene-independent, and it agrees with the scan lead-in/lead-out
# lines to within 0.02.
OPEN_BEAM_MAX_CLIP = 0.01
OPEN_BEAM_TOP_PCT = 90.0
OPEN_BEAM_RANGE = (0.80, 1.25)

# Preview stretch. Percentiles are taken over pixels that clip in at most
# PREVIEW_CLIP_TOL of the kept bands -- i.e. the part of the frame that carries
# real signal. Stretching on the whole frame instead lets the clipped open beam
# (~31% of it) set the white point and flattens the kernels.
PREVIEW_FLOOR = 1e-4          # T is clipped to this before the log preview
PREVIEW_CLIP_TOL = 0.02
PREVIEW_LO_PCT = 0.5
PREVIEW_HI_PCT = 99.5

EXPOSURE_RE = re.compile(r"^(\d+)([kK])?$")
LINE_NAME_RE = re.compile(r"^(\d+)_([0-9A-Fa-f]+)_w")


def exposure_us(name):
    """Folder name -> exposure in microseconds. '60k' -> 60000, '2100' -> 2100."""
    m = EXPOSURE_RE.match(str(name))
    return float(m.group(1)) * (1000.0 if m.group(2) else 1.0) if m else None


def line_period(path):
    """Median inter-line timestamp delta, in the camera's own ticks, or None.

    Only ever used to compare two captures from the same session, so the tick
    unit does not matter. None for an already-stitched .npy, which has no
    per-line timestamps left.
    """
    path = Path(path)
    if not path.is_dir():
        return None
    stamps = []
    for name in sorted(p.name for p in path.iterdir()):
        m = LINE_NAME_RE.match(name)
        if m and name.endswith(".bin"):
            stamps.append(int(m.group(2), 16))
    if len(stamps) < 3:
        return None
    dt = np.diff(np.asarray(stamps, dtype=np.int64))
    return float(np.median(dt)), len(stamps)


# ------------------------------------------------------------------ masks --
def hot_pixel_mask(dark_frame):
    """(WIDTH, CHANNELS) bool: dark pixels far above the dark's own median.

    Sensor defects, not dark current. They survive dark subtraction as near-zero
    or negative outliers and are conspicuous in any spectrum.
    """
    thr = HOT_DARK_FACTOR * float(np.median(dark_frame))
    mask = dark_frame > thr
    print(f"  hot pixels: {int(mask.sum())} of {mask.size} above {thr:.0f} counts")
    return mask, thr


def white_reference(wref, hot, min_band_frac):
    """Validity masks for a (W - D_short) gain frame.

    Two masks come back. `cell_valid` marks cells with real white signal and is
    what the open-beam check uses, since the bands that make that check possible
    are exactly the weak-lamp ones dropped from the output. `valid` is
    `cell_valid` with those bands removed, and is what the output cube uses.

    The returned gain has invalid cells set to 1.0 so the division cannot produce
    inf; the caller overwrites those positions with NaN afterwards.
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
    print(f"  bands dropped (lamp below {min_band_frac:g}x peak): {runs(dropped)} "
          f"({dropped.size} of {CHANNELS})")
    print(f"  spatial rows with no usable white in any kept band: {runs(dead_rows)} "
          f"({dead_rows.size} of {WIDTH})")
    print(f"  invalid (row, band) cells: {int((~valid).sum())} "
          f"({100 * (~valid).mean():.2f}%) -> NaN in the output")

    stats = {"min": float(wref.min()), "median": float(np.median(wref)),
             "max": float(wref.max()), "invalid_frac": float((~valid).mean()),
             "peak_band": int(np.argmax(band_level)),
             "bands_dropped": [int(b) for b in dropped],
             "n_bands_kept": int(band_kept.sum()), "min_band_frac": float(min_band_frac),
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


# ------------------------------------------------------------- the check --
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
             f"{100 * clip[best]:.1f}% of the frame, over the "
             f"{100 * OPEN_BEAM_MAX_CLIP:g}% limit. A railed beam reads {FULL_SCALE} "
             f"regardless of its true level, so T there is a lower bound, not a test. "
             f"Shorten the exposure to enable this.")
        return {"available": False, "min_clip_frac": float(clip[best]), "min_clip_band": best}

    per_band = {}
    for b in cands:
        rows = np.nonzero(cell_valid[:, b])[0]
        raw = cube[rows, :, b].astype(np.float64)
        t = (raw - dark_long[rows, b][:, None]) / wref_raw[rows, b][:, None] * k
        bright = raw >= np.percentile(raw, OPEN_BEAM_TOP_PCT)
        per_band[int(b)] = float(np.median(t[bright]))

    med = float(np.median(np.array(list(per_band.values()))))
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


# ---------------------------------------------------------------- correct --
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
    returns NaN for the whole band. The defect is fixed in the (row, band) plane,
    so the fill is taken there: the 8 cells at (row +/- 1, band +/- 1), averaged
    independently at each scan position. Row +/- 1 is an adjacent pixel and
    band +/- 1 is ~3.6 nm away, so both are close enough to stand in.

    Only cells with at least FILL_MIN_NEIGHBOURS valid neighbours are filled, and
    only inside the kept bands. That confines the fill to genuine isolated
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
    print(f"  interpolated {int(filled.sum())} isolated (row, band) defect(s) from their "
          f"8 neighbours (>= {FILL_MIN_NEIGHBOURS} valid required)")
    print(f"    rows affected: {runs(np.unique(np.nonzero(filled)[0]))}")
    print(f"    left as NaN: {left} cell(s) with too few valid neighbours -- the dead "
          f"stripe, where there is nothing to interpolate from")
    return filled, {"n_filled": int(filled.sum()), "n_left_nan": left,
                    "min_neighbours": FILL_MIN_NEIGHBOURS,
                    "rows": [int(r) for r in np.unique(np.nonzero(filled)[0])]}


# --------------------------------------------------------------- previews --
def band_mean(cube, sat_mask, band_kept):
    """Mean transmittance across the kept bands. -> (mean_plane, clipped_frac).

    Accumulated band by band rather than np.nanmean over the whole cube, which
    would allocate a second copy of a ~2 GB array.

    Each band contributes equally: T is already normalised by the white, so a
    plain mean is a mean transmittance and not a lamp-weighted one. Voxels that
    are NaN (invalid white, dropped band) are skipped, so a pixel valid in only
    some bands still gets a mean over the ones it has. A pixel with nothing valid
    stays NaN.
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
    of the frame carrying real signal. Anchoring on the whole frame instead lets
    the railed open beam set the white point, and the sample collapses into the
    bottom few grey levels.
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
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
    it is directly readable as "is my sample inside range". Cyan is invalid white
    (NaN) -- cyan is absent from the colour map, so it cannot be misread as a
    value.
    """
    cm = cv2.applyColorMap((np.clip(clipped_frac, 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    cm[~finite] = (255, 255, 0)   # BGR cyan
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.rotate(cm, cv2.ROTATE_90_CLOCKWISE))
    clean = float((clipped_frac[finite] == 0).mean()) if finite.any() else 0.0
    print(f"  saturation map: {path} (black = no band clipped, yellow = all clipped, "
          f"cyan = invalid; {100 * clean:.1f}% of valid pixels fully in range)")
    return {"fully_in_range_frac": clean}


# ------------------------------------------------------------ calibration --
def load_calibration(day, cfg):
    """Resolve, measure (or reuse) and report every calibration input. -> dict.

    The dark/white frames are pure functions of their capture directories, so
    they are cached on disk keyed on those directories' contents. A dish is one
    run of process.py; re-deriving the same three frames from ~230 line files a
    hundred times a day is waste, and a changed calibration folder changes the
    key so a stale frame can never be used silently.
    """
    dark_long_path, ds = resolve_calibration(CALIBRATION, "dark_sample", day,
                                            "dark at the sample's exposure")
    dark_short_path, dw = resolve_calibration(CALIBRATION, "dark_white", day,
                                             "dark at the white's exposure")
    white_path, ws = resolve_calibration(CALIBRATION, "white", day, "white reference")
    print("Calibration:")
    for label, path, scope in (("dark_sample", dark_long_path, ds),
                               ("dark_white", dark_short_path, dw),
                               ("white", white_path, ws)):
        print(f"  {label:11s} [{scope}] {describe(path)}")

    t_long = (cfg.TRANS_EXPOSURE_LONG_MS * 1000.0 if cfg.TRANS_EXPOSURE_LONG_MS is not None
              else exposure_us(dark_long_path.name))
    t_short = (cfg.TRANS_EXPOSURE_SHORT_MS * 1000.0
               if cfg.TRANS_EXPOSURE_SHORT_MS is not None else exposure_us(white_path.name))
    if t_long is None:
        raise SystemExit(f"cannot read the sample's exposure from '{dark_long_path.name}' "
                         f"-- rename that folder (e.g. 60k) or set "
                         f"TRANS_EXPOSURE_LONG_MS in config.py.")
    if t_short is None:
        raise SystemExit(f"cannot read the white's exposure from '{white_path.name}' -- "
                         f"rename that folder (e.g. 2100) or set TRANS_EXPOSURE_SHORT_MS "
                         f"in config.py.")
    t_dark_short = exposure_us(dark_short_path.name)
    if t_dark_short is not None and abs(t_dark_short - t_short) > 1e-6:
        warn(f"the white's dark is at {t_dark_short:.0f} us but the white is at "
             f"{t_short:.0f} us -- they should match.")
    k = t_short / t_long
    print(f"  exposures: sample {t_long:.0f} us, white {t_short:.0f} us "
          f"-> k = t_short/t_long = {k:.6g} (1/{1 / k:.2f})")

    key = combine_keys(capture_key(dark_long_path), capture_key(dark_short_path),
                       capture_key(white_path), cfg.TRANS_MIN_BAND_FRAC)

    def compute():
        cube, blank = load_cube(dark_long_path, "dark at the sample's exposure",
                               cfg.STITCH_WORKERS)
        dark_long, s_dl = reference_frame(cube, "dark_long", blank)
        del cube
        cube, blank = load_cube(dark_short_path, "dark at the white's exposure",
                               cfg.STITCH_WORKERS)
        dark_short, s_ds = reference_frame(cube, "dark_short", blank)
        del cube
        white_cube, blank = load_cube(white_path, "white reference", cfg.STITCH_WORKERS)
        white_frame, s_w = reference_frame(white_cube, "white", blank)

        hot, hot_thr = hot_pixel_mask(dark_long)
        wref_raw = white_frame - dark_short          # W - D_short, unmasked
        wref, valid, cell_valid, band_kept, band_level, s_wr = white_reference(
            wref_raw, hot, cfg.TRANS_MIN_BAND_FRAC)
        noise = white_gain_noise(white_cube, wref, valid)
        return {"dark_long": dark_long, "dark_short": dark_short, "wref_raw": wref_raw,
                "wref": wref, "valid": valid, "cell_valid": cell_valid,
                "band_kept": band_kept, "band_level": band_level, "hot": hot,
                "_scalars": {"dark_long": s_dl, "dark_short": s_ds, "white": s_w,
                             "white_reference": s_wr, "white_gain_noise": noise,
                             "hot_pixels": {"n": int(hot.sum()), "threshold": hot_thr}}}

    cal, hit = cached_npz(CACHE, "refs", key, compute)
    if hit:
        s = cal["_scalars"]
        print(f"  frames reused from cache (same calibration captures): "
              f"{s['white_reference']['n_bands_kept']} bands kept "
              f"{s['white_reference']['bands_kept_range']}, "
              f"{100 * s['white_reference']['invalid_frac']:.2f}% invalid cells, "
              f"{s['hot_pixels']['n']} hot pixels, white gain noise median "
              f"{s['white_gain_noise']['median_pct']:.3f}%")
    cal.update({"k": k, "t_long": t_long, "t_short": t_short,
                "paths": {"dark_sample": str(dark_long_path),
                          "dark_white": str(dark_short_path), "white": str(white_path)},
                "scopes": {"dark_sample": ds, "dark_white": dw, "white": ws},
                "cache_hit": hit})
    return cal


def load_geometry(day, cfg, cal, n_lines_image, image_path):
    """The scan-axis factor: configured, or measured from the checkerboard capture.

    Cached the same way the reference frames are. The board capture is a
    dedicated one shared by every dish, so measuring it is a property of the
    calibration set, not of the dish -- once per calibration set rather than 100
    times a day. The QC overlay is written into the cache directory alongside it
    for the same reason: one board capture, one overlay.
    """
    if cfg.TRANS_SCAN_SCALE is not None:
        fx = float(cfg.TRANS_SCAN_SCALE)
        print(f"Geometry: fixed scan-axis factor {fx:.5f} from config.py "
              f"(TRANS_SCAN_SCALE); checkerboard detection skipped.")
        return fx, 1.0, {"applied": True, "source": "config", "fx": fx, "fy": 1.0}

    board_path, scope = resolve_calibration(CALIBRATION, "checkerboard", day,
                                           "checkerboard capture")
    print(f"Geometry: [{scope}] {describe(board_path)}")
    band = int(np.argmax(cal["band_level"]))
    key = combine_keys(capture_key(board_path),
                       capture_key(Path(cal["paths"]["dark_sample"])),
                       cfg.TRANS_CHECKERBOARD_INNER, cfg.TRANS_CELL_SIZE,
                       cfg.TRANS_PLANE_FACTOR, band)

    def compute():
        board_cube, _ = load_cube(board_path, "checkerboard capture", cfg.STITCH_WORKERS)
        n_board = int(board_cube.shape[1])
        gray = common.render_detection_gray(board_cube, band=band, dark=cal["dark_long"])
        del board_cube
        print(f"    checkerboard {cfg.TRANS_CHECKERBOARD_INNER[0]}x"
              f"{cfg.TRANS_CHECKERBOARD_INNER[1]} inner corners on band {band} "
              f"(lamp peak), expecting {cfg.TRANS_N_BOARDS} board(s):")
        # Named with the geometry cache tag+key so it is pruned along with the
        # entry it belongs to when the calibration set changes.
        qc = (CACHE / f"geometry_{key}_qc.png") if cfg.SAVE_CHECKERBOARD_QC else None
        fx, fy, meta = measure_geometry(
            gray, tuple(cfg.TRANS_CHECKERBOARD_INNER), cfg.TRANS_N_BOARDS,
            tuple(cfg.TRANS_CELL_SIZE), cfg.TRANS_PLANE_FACTOR, "spatial", qc_path=qc)
        meta.update({"n_lines_board": n_board, "band": band, "qc": str(qc) if qc else None,
                     "line_period_board": (line_period(board_path) or [None])[0]})
        return {"fx": np.float64(fx), "fy": np.float64(fy), "_scalars": meta}

    geo_cache, hit = cached_npz(CACHE, "geometry", key, compute)
    fx, fy = float(geo_cache["fx"]), float(geo_cache["fy"])
    meta = dict(geo_cache.get("_scalars") or {})
    meta["cache_hit"] = hit
    if hit:
        print(f"    reused from cache: fx = {fx:.5f} (band {meta.get('band')}, "
              f"{meta.get('n_boards_found')} board(s), "
              f"{meta.get('n_lines_board')} board lines)"
              + (f", QC {meta['qc']}" if meta.get("qc") else ""))

    # The factor is frame-rate over stage-speed. Line rate is checkable from the
    # timestamps when both captures still have them; stage speed is not, and a
    # capture that swept the same object in a very different number of lines did
    # not run at the same speed. Hence the line-count fallback.
    p_board, p_image = meta.get("line_period_board"), line_period(image_path)
    if p_board and p_image:
        meta["line_period_image"] = p_image[0]
        if abs(p_board - p_image[0]) > 0.05 * p_board:
            warn(f"line period differs: board {p_board:.0f} ticks vs sample "
                 f"{p_image[0]:.0f}. The scan-axis factor does not transfer between "
                 f"line rates.")
    n_board = meta.get("n_lines_board")
    if n_board and abs(n_lines_image - n_board) > 0.10 * n_board:
        # Fewer lines over the same travel means the stage moved further per
        # line, so the scan axis is less oversampled and needs less compression:
        # px_scan scales by n_image/n_board, so fx scales by its inverse. Only
        # valid if both swept the same travel, hence a suggestion to check the
        # preview rather than an auto-fix.
        warn(f"sample is {n_lines_image} lines but the checkerboard capture is "
             f"{n_board} ({n_lines_image / n_board:.2f}x). If both swept the same "
             f"travel then the stage speed differed and fx={fx:.4f} is wrong for this "
             f"sample by about that ratio -- the dish will not come out round. Check the "
             f"preview; if it is off, set TRANS_SCAN_SCALE = "
             f"{fx * n_board / n_lines_image:.4f} in config.py, or re-shoot the "
             f"checkerboard at the collection stage speed.")
    elif n_board:
        print(f"    sample {n_lines_image} lines vs board {n_board} "
              f"({n_lines_image / n_board:.3f}x) -- same stage speed, factor transfers")
    return fx, fy, meta


# -------------------------------------------------------------------- run --
def process(cube, blank, day, cfg, image_path, preview_path, sat_preview_path,
            _qc_path):
    """Correct one transmittance capture. -> (trans, masks, meta).

    `_qc_path` is unused: the transmission board is a dedicated capture shared by
    every dish, so its QC overlay belongs with the calibration set (see
    load_geometry) rather than being rewritten identically per dish.
    """
    meta = {"mode": "transmittance"}
    cal = load_calibration(day, cfg)
    k, valid, cell_valid = cal["k"], cal["valid"], cal["cell_valid"]
    band_kept, wref, wref_raw = cal["band_kept"], cal["wref"], cal["wref_raw"]
    dark_long = cal["dark_long"]
    meta["calibration"] = {"paths": cal["paths"], "scopes": cal["scopes"],
                           "cache_hit": cal["cache_hit"], **cal["_scalars"]}
    meta["exposure"] = {"t_long_us": cal["t_long"], "t_short_us": cal["t_short"], "k": k}

    print("Sample:")
    meta["image"] = {"shape": list(cube.shape), "n_blank_lines": len(blank)}
    sat_mask = cube >= FULL_SCALE
    sat_frac = float(sat_mask.mean())
    print(f"  saturated: {100 * sat_frac:.2f}% of voxels at >= {FULL_SCALE}. Expected to "
          f"be large -- the open beam beside and between the wells is still over range.")
    meta["saturation"] = {"frac": sat_frac}
    # Deliberately no per-band clipping breakdown. The number that matters is how
    # much KERNEL data clips, and that needs to know where the kernels are. Every
    # segmentation-free proxy is dominated by open beam, which clips almost
    # everywhere, and reports nearly every band as compromised regardless of the
    # kernels. The saturation PNG shows it spatially instead, and the per-voxel
    # mask is written out for when segmentation exists.

    print("Checks:")
    meta["open_beam"] = open_beam_check(cube, dark_long, wref_raw, cell_valid, sat_mask, k)

    trans = transmittance(cube, dark_long, wref, valid, k)
    del cube
    filled, meta["fill"] = fill_bad_cells(trans, valid, band_kept)

    fx, fy, geo = load_geometry(day, cfg, cal, trans.shape[1], image_path)
    if geo.get("applied") and (fx != 1.0 or fy != 1.0):
        print(f"    scan axis {trans.shape[1]} -> "
              f"{max(1, int(round(trans.shape[1] * fx)))} px")
        trans, achieved = resample(trans, fx, fy)
        sat_mask = resample_mask(sat_mask, fx, fy)
        geo["achieved"] = list(achieved)
        if abs(achieved[0] - fx) > 1e-4 or abs(achieved[1] - fy) > 1e-4:
            print(f"    note: integer output size means the applied factors are "
                  f"scan x{achieved[0]:.5f}, spatial x{achieved[1]:.5f}")
    meta["geometry"] = geo

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
            warn(f"{100 * frac_above:.2f}% of valid unsaturated voxels have T > 1, which "
                 f"is unphysical. Suspect the exposure factor k={k:.6g} or the white "
                 f"reference.")
        else:
            print(f"  T > 1: {n_above} voxels ({100 * frac_above:.3f}%) -- physically OK")
        del sample
    else:
        warn("no valid unsaturated voxels at all -- check the white reference.")
        meta["transmittance"] = {"valid_unsaturated_frac": 0.0}
    del good

    mean_plane, clipped_frac = band_mean(trans, sat_mask, band_kept)
    finite = np.isfinite(mean_plane)
    meta["preview"] = {"kind": "mean over kept bands", "n_bands": int(band_kept.sum())}
    meta["preview"].update(preview_png(mean_plane, clipped_frac, preview_path,
                                       f"mean of {int(band_kept.sum())} bands"))
    # Off by default so preview/ stays one PNG per capture. Nothing is lost: the
    # per-voxel saturation mask is written to capture_masks.npz either way, and
    # the fraction fully in range is recorded here.
    finite_clean = float((clipped_frac[finite] == 0).mean()) if finite.any() else 0.0
    meta["preview"]["fully_in_range_frac"] = finite_clean
    if getattr(cfg, "SAVE_SATURATION_MAP", False):
        saturation_png(clipped_frac, finite, sat_preview_path)
    else:
        print(f"  saturation: {100 * finite_clean:.1f}% of valid pixels have no clipped "
              f"band (map not written; SAVE_SATURATION_MAP is off)")

    masks = {"saturated": sat_mask, "white_valid": valid, "band_kept": band_kept,
             "hot_pixels": cal["hot"], "band_level": cal["band_level"],
             "interpolated": filled}
    return trans, masks, json_safe(meta)
