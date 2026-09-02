"""REFLECTANCE correction: dark subtraction -> per-row tape white -> scan-axis
geometry -> greyscale preview.

    reflectance = (r0 - D) / (W - D)

There is no dedicated white capture in this mode. The white reference is the two
teflon tape strips that sit in the frame with the dish, so both halves of that
ratio come out of the same capture and (W - D) is measured per spatial row
rather than assumed flat.

Stage 1 -- dark frame
---------------------
The mean (width, channels) frame over the dark capture's lines. The push-broom
sensor's dark current/offset is fixed per (spatial pixel, band), not per scene,
so one averaged frame is the correction target for every line of the scan.

Mean, not median, here: this is what the dry-run reflectance pipeline was
validated on, and switching the collapse would change every corrected value by a
small amount for no measured gain. (Transmittance uses a median, where a single
stray line in a 27-line dark moved the result by percent-level.)

Stage 2 -- dark subtraction
---------------------------
clip(r0 - D, 0), broadcast over the scan-line axis. Unlike transmittance this
DOES clip negatives: the reflectance scene is bright everywhere, so a negative
here is noise about zero rather than the weak signal a clip would bias.

Stage 3 -- tape-strip detection
-------------------------------
The two tape strips are captured at the start and end of the scan, so they are
bright blobs near the two ends of the scan-line axis, not at fixed width
positions. Each blob is irregular and does not span the full 640 px, so
detection recovers each blob's true per-row extent: a two-stage locate/refine,
because a single percentile or Otsu pass either merges the tape with the dish or
clips the blob's dimmer edges.

Everything outside the two blobs' column span is pre/post-scan padding -- not
tape, dish or kernel -- and is cropped, which is also what keeps the corrected
cube down from ~1.9 GB to ~0.5 GB.

Stage 4 -- per-row white reference
----------------------------------
white_ref[row, band] is the 75th percentile of that row's dark-subtracted tape
pixels, both blobs pooled. Only rows holding a reliable share of tape are
measured: a blob tapers off over several rows before its coverage reaches zero,
and those boundary rows hold a handful of pixels that are themselves part tape,
part background, so they read dim. Their percentile is noisy AND biased low, and
taking it as known white makes reflectance overshoot by however much the bias is
-- on the dry-run captures the left blob tapered across rows 0-5 (111 px in row 0
against ~300 in the interior) and its white came out up to 30% low there, sending
reflectance to 1.33 on a scene the rest of the frame put at 1.00. Roughly 1500
px, enough to hijack any min-max stretch downstream.

Demoting those rows also protects every row past them: np.interp
flat-extrapolates beyond the last known row, so one bad boundary row would
otherwise be inherited by the whole uncovered run beyond it instead of the stable
interior plateau.

Stage 5 -- scan-axis geometry
-----------------------------
The push-broom scan axis is stretched relative to the 640 px optical axis by
however much faster or slower the stage moved than the frame rate, and the two
in-scene checkerboards of known cell geometry give the factor that undoes it.
Both boards are measured independently, judged against the lattice model, and
only then pooled -- boards on different planes disagree, and that disagreement is
a lower bound on the correction's error off the plane it was measured on.

Two things this cannot determine on its own are explicit config values rather
than assumptions: REFL_CELL_SIZE (the target's true cell, if not square) and
REFL_PLANE_FACTOR (the kernels not being on the boards' plane). Both are echoed
in the log on every run, including when they are no-ops. Note they are two names
for one multiplier: setting both double-counts.

Stage 6 -- preview
------------------
The corrected cube collapsed to greyscale (mean across all 224 bands, stretched
between the 1st and 99th percentile). Percentiles, not min/max: the
checkerboards' paper backing out-reflects the teflon tape the cube is referenced
to, so it sits above 1.0 and would otherwise claim the whole white point.
Clipping at a percentile also keeps two captures roughly comparable, which a
stretch anchored to each one's own extremes does not.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402
from common import (FULL_SCALE, cached_npz, capture_key, combine_keys,  # noqa: E402
                    describe, json_safe, load_cube, measure_geometry, reference_frame,
                    resample, resolve_calibration, warn)

CALIBRATION = Path(__file__).resolve().parent / "calibration"
CACHE = CALIBRATION / ".cache"


# ------------------------------------------------------------------ stage 2 --
def apply_dark_correction(cube, dark_frame):
    """clip(cube - dark_frame, 0), broadcasting over every scan line.

    Subtracts and clips in-place into one float32 buffer rather than allocating
    several full-cube temporaries (each is ~1.9 GB on a 3,300-line capture).
    -> (corrected (cube.dtype), n_clipped)
    """
    diff = cube.astype(np.float32)
    diff -= dark_frame[:, None, :].astype(np.float32)
    n_clipped = int((diff < 0).sum())
    corrected = np.clip(diff, 0, None, out=diff).astype(cube.dtype)
    return corrected, n_clipped


# ------------------------------------------------------------------ stage 3 --
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

    Two-stage locate/refine: a single percentile or Otsu pass either merges the
    tape with the dish or clips the blob's dimmer edges.
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
        warn(f"only {len(candidates)} tape candidate(s) >= {min_area_frac * 100:.0f}% of "
             f"frame area found (wanted {k}); continuing with what's available. The white "
             f"reference is interpolated across every row a blob does not cover.")
    candidates.sort(key=lambda lab: stats[lab, cv2.CC_STAT_AREA], reverse=True)
    top = sorted(candidates[:k], key=lambda lab: stats[lab, cv2.CC_STAT_LEFT])  # L-to-R

    refined = []
    for lab in top:
        x, y, w, h, _ = stats[lab]
        x0, x1 = max(0, x - pad), min(W, x + w + pad)
        _, crop_mask = cv2.threshold(g8[:, x0:x1], 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local = _largest_component(crop_mask, open_k=5)
        if local is None:
            warn(f"refinement failed for the tape blob at columns [{x0}:{x1}]; skipping it.")
            continue
        full = np.zeros((H, W), bool)
        full[:, x0:x1] = local
        refined.append(full)
    return refined


def tape_saturation(sat_mask, masks):
    """Split raw clipping into tape and scene, and say what each one means.

    A whole-frame clipping percentage is the wrong number here, because the two
    places a reflectance capture can clip fail in opposite directions:

      inside the tape   the tape IS the white reference. A clipped tape pixel
                        reads 4095 when the real signal was higher, so (W - D)
                        comes out LOW and reflectance = darksub/white_ref is
                        biased HIGH at exactly those bands -- a systematic,
                        band-shaped error in every kernel in the frame, not a
                        few bad pixels. Nothing downstream can detect or undo
                        it, because a railed reading carries no record of how
                        bright it really was.
      outside the tape  the sample and dish. Those pixels are lower bounds, so
                        their reflectance is a lower bound too. Bad, but local.

    Measured on the dry-run capture this pipeline was validated against: 27.9%
    of tape voxels clip, over bands 19-130, peaking at 76% -- so that capture's
    reflectance is over-estimated across half the spectrum. Both are reported
    every run, because the fix (a shorter exposure) has to happen at capture
    time and is unrecoverable afterwards.
    """
    out = {}
    if not masks:
        return out
    tape = np.logical_or.reduce(masks)                 # (width, n_lines)
    tape_sat = sat_mask[tape]                          # (n_tape_px, bands)
    scene_frac = float(sat_mask[~tape].mean())
    tape_frac = float(tape_sat.mean())
    per_band = tape_sat.mean(axis=0)
    hot = np.nonzero(per_band > 0.01)[0]
    out = {"tape_frac": tape_frac, "scene_frac": scene_frac,
           "tape_bands_clipping": [int(b) for b in hot],
           "tape_peak_band": int(np.argmax(per_band)),
           "tape_peak_frac": float(per_band.max())}
    print(f"    of which: {100 * tape_frac:.2f}% of TAPE voxels (the white reference) "
          f"and {100 * scene_frac:.2f}% of the rest of the frame")
    if tape_frac > 0.01:
        warn(f"{100 * tape_frac:.1f}% of the teflon tape's voxels are clipped at "
             f"{FULL_SCALE}, over bands {common.runs(hot)} (peak "
             f"{100 * per_band.max():.0f}% at band {out['tape_peak_band']}). The tape is "
             f"the white reference, so a clipped tape pixel makes (W-D) read low and "
             f"reflectance read HIGH at those bands, for every kernel in the frame. This "
             f"is not recoverable after capture -- shorten the exposure.")
    if scene_frac > 0.01:
        warn(f"{100 * scene_frac:.1f}% of the non-tape frame is clipped at {FULL_SCALE}; "
             f"those pixels are lower bounds, not measurements.")
    return out


def tape_column_span(masks):
    """(c0, c1) inclusive scan-line bounds across both blobs pooled.

    From the left blob's leftmost column to the right blob's rightmost. Everything
    outside is pre/post-scan padding, not tape/dish/kernel. None if no blobs.
    """
    if not masks:
        return None
    combined = np.logical_or.reduce(masks)
    cols = np.nonzero(combined.any(axis=0))[0]
    return int(cols.min()), int(cols.max())


# ------------------------------------------------------------------ stage 4 --
def build_white_reference(darksub_cube, masks, pct=75.0, min_coverage_frac=0.75):
    """Per-row (width, channels) white reference from the tape blob masks.

    See the module docstring's stage 4 for why the coverage test is not cosmetic.
    Rows with no reliable coverage (including the demoted taper rows) are filled
    by per-band linear interpolation across the width axis from the nearest
    reliable rows, flat-extrapolated at either edge via np.interp's default clamp.

    -> (white_ref (width, channels) float64, reliable bool (width,))
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
        warn("no tape coverage found in any width row; skipping white correction (white "
             "reference set to 1.0 everywhere, so the cube is NOT reflectance-normalised).")
        return np.ones((H, C), dtype=np.float64), covered

    reliable = covered & (counts >= min_coverage_frac * np.median(counts[covered]))
    n_demoted = int((covered & ~reliable).sum())
    if n_demoted:
        print(f"  {n_demoted} width row(s) had tape coverage but below "
              f"{min_coverage_frac:g}x the median count (blob taper); interpolated rather "
              f"than trusted.")
    if not reliable.any():
        warn("no width row met the tape coverage threshold; falling back to every covered "
             "row (the white reference may read low at the blob edges).")
        reliable = covered

    white_ref = np.full((H, C), np.nan, dtype=np.float64)
    for r in np.nonzero(reliable)[0]:
        cols = combined[r]
        white_ref[r] = np.percentile(np.asarray(darksub_cube[r, cols, :]), pct, axis=0)

    known_rows = np.nonzero(reliable)[0]
    unknown_rows = np.nonzero(~reliable)[0]
    if unknown_rows.size:
        for b in range(C):
            white_ref[unknown_rows, b] = np.interp(unknown_rows, known_rows,
                                                   white_ref[known_rows, b])
    return white_ref, reliable


def apply_white_correction(darksub_cube, white_ref):
    """reflectance[row, ..., b] = darksub[row, ..., b] / white_ref[row, b].

    Both sides are already dark-subtracted, so this is exactly (r0-D)/(W-D).
    Divides in-place rather than allocating a second full-cube temporary.
    """
    reflectance = darksub_cube.astype(np.float32)
    reflectance /= white_ref.astype(np.float32)[:, None, :]
    return reflectance


# ------------------------------------------------------------------ stage 6 --
def intensity_preview(cube, lo_pct=1.0, hi_pct=99.0):
    """Greyscale (width, n_lines) uint8 preview: band mean, percentile-clipped.

    Percentiles rather than min/max: the scene does not set the maximum.
    Reflectance here is measured against teflon tape, and the checkerboard
    targets are printed on paper that is a brighter reflector than the tape, so
    those pixels legitimately land above 1.0 -- as does any specular glint. A
    min-max stretch hands the whole white point to whichever of them is
    brightest and drags everything else down with it: on a dry-run capture a
    0.25% population at 1.33 put the tape itself at 190 DN instead of 255 and
    washed the entire frame out to mid-grey.
    """
    mean = np.asarray(cube.mean(axis=2))
    lo, hi = (float(v) for v in np.percentile(mean, [lo_pct, hi_pct]))
    norm = np.clip((mean - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8), {"lo": lo, "hi": hi}


# ------------------------------------------------------------- calibration --
def load_calibration(day, cfg):
    """Resolve, measure (or reuse) and report the dark frame. -> dict.

    Cached on the dark capture directory's contents, so the 168-line dark is
    collapsed once per calibration set rather than once per dish.
    """
    dark_path, scope = resolve_calibration(CALIBRATION, "dark", day,
                                          "dark at the capture exposure")
    print("Calibration:")
    print(f"  dark        [{scope}] {describe(dark_path)}")

    key = combine_keys(capture_key(dark_path), "mean")

    def compute():
        cube, blank = load_cube(dark_path, "dark reference", cfg.STITCH_WORKERS)
        frame, stats = reference_frame(cube, "dark", blank, how="mean")
        return {"dark": frame, "_scalars": {"dark": stats}}

    cal, hit = cached_npz(CACHE, "refs", key, compute)
    if hit:
        s = cal["_scalars"]["dark"]
        print(f"  frame reused from cache (same dark capture): {s['n_used']} lines "
              f"({s['collapse']}), median {s['median']:.0f}")
    cal.update({"paths": {"dark": str(dark_path)}, "scopes": {"dark": scope},
                "cache_hit": hit})
    return cal


# -------------------------------------------------------------------- run --
def process(cube, blank, day, cfg, image_path, preview_path, _sat_preview_path, qc_path):
    """Correct one reflectance capture. -> (reflectance, None, meta).

    No masks file: unlike transmittance, nothing here is NaN and there is no
    band-drop or white-validity mask to carry. The white reference's per-row
    reliability and the raw saturation fraction go into meta.json instead.
    """
    meta = {"mode": "reflectance"}
    cal = load_calibration(day, cfg)
    meta["calibration"] = {"paths": cal["paths"], "scopes": cal["scopes"],
                           "cache_hit": cal["cache_hit"], **cal["_scalars"]}

    print("Sample:")
    meta["image"] = {"shape": list(cube.shape), "n_blank_lines": len(blank)}
    # Kept until the tape blobs are known, because WHERE the clipping is decides
    # what it means. Held as a bool cube (~0.5 GB on a 3,300-line capture) and
    # dropped as soon as the tape statistics below are read off it.
    sat_mask = cube >= FULL_SCALE
    sat_frac = float(sat_mask.mean())
    meta["saturation"] = {"frac": sat_frac}
    print(f"  saturated: {100 * sat_frac:.3f}% of raw voxels at >= {FULL_SCALE}")

    darksub, n_clipped = apply_dark_correction(cube, cal["dark"])
    del cube  # ~0.9 GB; nothing after this point needs the raw cube
    print(f"  dark subtraction: {n_clipped} px clipped at 0 "
          f"({100 * n_clipped / darksub.size:.3f}%)")
    meta["dark_subtraction"] = {"n_clipped": n_clipped,
                               "clipped_frac": n_clipped / darksub.size}

    gray = np.asarray(darksub.mean(axis=2)).astype(np.float32)
    masks = find_tape_blobs(gray, pct=cfg.REFL_TAPE_PCT)
    tape_meta = []
    for mask, label in zip(masks, ("LEFT", "RIGHT")):
        rows = np.nonzero(mask.any(axis=1))[0]
        print(f"  {label} tape blob: {int(mask.sum())} px, row span "
              f"[{rows.min()}:{rows.max()}] of {gray.shape[0]}")
        tape_meta.append({"side": label, "px": int(mask.sum()),
                          "row_span": [int(rows.min()), int(rows.max())]})
    meta["tape_blobs"] = tape_meta
    meta["saturation"].update(tape_saturation(sat_mask, masks))
    del sat_mask

    span = tape_column_span(masks)
    n_lines = darksub.shape[1]
    if span is None:
        warn(f"no tape blobs found at all; skipping the scan-line crop (keeping all "
             f"{n_lines} lines) and the white correction below.")
        meta["crop"] = None
    else:
        c0, c1 = span
        print(f"  cropping scan axis to the tape span [{c0}:{c1 + 1}] of {n_lines} "
              f"(dropping {c0} lines before, {n_lines - 1 - c1} after)")
        # .copy(): plain slicing returns a view into the full-size buffer, which
        # would keep the whole uncropped array resident for the rest of the run --
        # defeating the point of cropping. .copy() actually releases the dropped
        # region.
        darksub = darksub[:, c0:c1 + 1, :].copy()
        masks = [m[:, c0:c1 + 1].copy() for m in masks]
        meta["crop"] = {"c0": c0, "c1": c1, "n_lines_in": n_lines,
                        "n_lines_out": int(darksub.shape[1])}

    white_ref, reliable = build_white_reference(
        darksub, masks, pct=cfg.REFL_WHITE_PCT,
        min_coverage_frac=cfg.REFL_WHITE_MIN_COVERAGE)
    n_interp = int((~reliable).sum())
    print(f"  white reference: {reliable.sum()}/{reliable.size} width rows measured from "
          f"a reliable tape sample ({n_interp} interpolated)")
    meta["white_reference"] = {"n_reliable_rows": int(reliable.sum()),
                              "n_interpolated_rows": n_interp,
                              "pct": cfg.REFL_WHITE_PCT,
                              "min_coverage_frac": cfg.REFL_WHITE_MIN_COVERAGE}

    reflectance = apply_white_correction(darksub, white_ref)
    del darksub  # superseded by reflectance
    print(f"  reflectance: min {reflectance.min():.4f}, max {reflectance.max():.4f}, "
          f"mean {reflectance.mean():.4f}")
    meta["reflectance"] = {"min": float(reflectance.min()), "max": float(reflectance.max()),
                          "mean": float(reflectance.mean())}

    # -------------------------------------------------------------- geometry --
    inner, cell = tuple(cfg.REFL_CHECKERBOARD_INNER), tuple(cfg.REFL_CELL_SIZE)
    if cfg.REFL_SCAN_SCALE is not None:
        fx, fy = float(cfg.REFL_SCAN_SCALE), 1.0
        geo = {"applied": True, "source": "config", "fx": fx, "fy": fy}
        print(f"Geometry: fixed scan-axis factor {fx:.5f} from config.py "
              f"(REFL_SCAN_SCALE); checkerboard detection skipped.")
    else:
        # Measured per capture, not cached: the boards are IN the scene, so each
        # capture carries its own measurement -- and a board that stops being
        # found is a real change in the scene worth hearing about per dish.
        print(f"Geometry: in-scene checkerboards, {inner[0]}x{inner[1]} inner corners, "
              f"expecting {cfg.REFL_N_BOARDS} board(s), cell {cell[0]:g}x{cell[1]:g}:")
        gray = common.render_detection_gray(reflectance)
        fx, fy, geo = measure_geometry(
            gray, inner, cfg.REFL_N_BOARDS, cell, cfg.REFL_PLANE_FACTOR, "spatial",
            qc_path=qc_path if cfg.SAVE_CHECKERBOARD_QC else None)

    if geo.get("applied") and (fx != 1.0 or fy != 1.0):
        print(f"    scan axis {reflectance.shape[1]} -> "
              f"{max(1, int(round(reflectance.shape[1] * fx)))} px")
        reflectance, achieved = resample(reflectance, fx, fy)
        geo["achieved"] = list(achieved)
        if abs(achieved[0] - fx) > 1e-4 or abs(achieved[1] - fy) > 1e-4:
            print(f"    note: integer output size means the applied factors are "
                  f"scan x{achieved[0]:.5f}, spatial x{achieved[1]:.5f}")
    meta["geometry"] = geo

    preview, stretch = intensity_preview(reflectance)
    Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(preview_path)
    print(f"  preview: {preview_path} ({preview.shape[1]}x{preview.shape[0]}, stretched "
          f"reflectance {stretch['lo']:.4f}..{stretch['hi']:.4f})")
    meta["preview"] = {"kind": "mean over all bands, percentile stretch", **stretch}

    return reflectance, None, json_safe(meta)
