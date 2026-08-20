"""
End-to-end reflectance correction: stitch -> dark frame -> dark subtraction
-> per-row white correction -> checkerboard geometry correction -> greyscale
intensity preview. Currently implements stages 1-7.

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
Rows with no coverage are filled by per-band linear interpolation across
width; reflectance = darksub / white_ref, i.e. (r0-D)/(W-D).

Stage 6 -- checkerboard scan-axis geometry correction
------------------------------------------------------------
The push-broom scan axis (n_lines) is stretched relative to the optical
spatial axis (640px) by however much faster/slower the stage moved than
frame rate. A checkerboard's known-square cells give the rescale factor;
detection runs on several scan-precompressed copies and takes the median.

Stage 7 -- intensity preview
-------------------------------
intensity_preview() collapses the final corrected cube to a single greyscale
image (mean across all 224 bands, min-max normalised to 0-255), for a quick
look at the corrected result without needing a wavelength calibration.

eBUS Player saves a session's .bin files flat into raw_image_bin/ (no
per-session subfolder). Run:
    python3 process_image.py day2_dish3_ref_2500

This stitches/corrects everything currently in raw_image_bin/, writes
corrected_file_npy/day2_dish3_ref_2500.npy and
corrected_image/day2_dish3_ref_2500.png, then moves those .bin files into
raw_image_bin_storage/day2_dish3_ref_2500/ -- leaving raw_image_bin/ empty
and ready for the next capture.
"""

import argparse
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

WIDTH = 640       # spatial pixels per line
CHANNELS = 224    # spectral bands per line

DEFAULT_DARK = "raw_dark/dark_day2_ref_2500"  # same dark reference every session
RAW_IMAGE_DIR = "raw_image"        # eBUS Player saves flat into here, no subfolder
STORAGE_DIR = "raw_image_storage"  # processed .bin files archived to <STORAGE_DIR>/<sample>/


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
def build_white_reference(darksub_cube, masks, pct=75.0):
    """Per-row (width, channels) white reference from the tape blob masks.

    white_ref[row, band] is the pct-th percentile of that row's tape pixels;
    rows with no coverage are filled by per-band linear interpolation.
    Returns (white_ref (width, channels) float64, covered_rows bool (width,)).
    """
    H, C = darksub_cube.shape[0], darksub_cube.shape[2]
    if masks:
        combined = np.logical_or.reduce(masks)  # (width, n_lines)
        covered = combined.any(axis=1)          # (width,)
    else:
        covered = np.zeros(H, dtype=bool)
    if not covered.any():
        print("  warning: no tape coverage found in any width row; skipping white "
              "correction (white reference set to 1.0 everywhere).")
        return np.ones((H, C), dtype=np.float64), covered

    white_ref = np.full((H, C), np.nan, dtype=np.float64)
    for r in np.nonzero(covered)[0]:
        cols = combined[r]
        white_ref[r] = np.percentile(np.asarray(darksub_cube[r, cols, :]), pct, axis=0)

    known_rows = np.nonzero(covered)[0]
    unknown_rows = np.nonzero(~covered)[0]
    if unknown_rows.size:
        for b in range(C):
            white_ref[unknown_rows, b] = np.interp(unknown_rows, known_rows, white_ref[known_rows, b])

    return white_ref, covered


def apply_white_correction(darksub_cube, white_ref):
    """reflectance[row,...,b] = darksub[row,...,b] / white_ref[row,b].

    Both sides are already dark-subtracted, so this is exactly (r0-D)/(W-D).
    Divides in-place (/=) rather than allocating a second full-cube temporary.
    """
    reflectance = darksub_cube.astype(np.float32)
    reflectance /= white_ref.astype(np.float32)[:, None, :]
    return reflectance


# ---------------------------------------------------------------- stage 6 --
def render_gray8(cube, band=None):
    """(width, n_lines) uint8 greyscale for checkerboard detection.

    band=None -> mean across bands, else that single band. Min-max normalised
    so detection isn't thrown off by reflectance's ~0-1.3 range.
    """
    img = cube.mean(axis=2) if band is None else np.asarray(cube[:, :, band]).astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img, dtype=np.float32)
    return (norm * 255.0).round().astype(np.uint8)


def detect_checkerboard(gray, inner_size):
    """Find `inner_size` = (cols, rows) inner corners in an 8-bit greyscale image.

    Tries the classic detector (sub-pixel refined) then the newer SB detector,
    each on the image, a 2x upscale, and an inverted copy. Returns (corners
    (N,2) float in the ORIGINAL image, method label) or (None, None).
    """
    cols, rows = inner_size
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    def variants():
        yield "classic", gray, 1.0
        yield "classic-inv", cv2.bitwise_not(gray), 1.0
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        yield "classic-2x", up, 2.0
        yield "classic-2x-inv", cv2.bitwise_not(up), 2.0

    for label_, img, scale in variants():
        ok, corners = cv2.findChessboardCorners(img, (cols, rows), classic_flags)
        if ok:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(-1, 2) / scale, label_

    for label_, img, scale in variants():
        try:
            ok, corners = cv2.findChessboardCornersSB(
                img, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        except cv2.error:
            ok = False
        if ok:
            return corners.reshape(-1, 2) / scale, label_.replace("classic", "SB")

    return None, None


def square_spacing(corners, inner_size):
    """Median px-per-square along the scan (x) and spatial (y) image axes."""
    cols, rows = inner_size
    grid = corners.reshape(rows, cols, 2)
    step_c = np.diff(grid, axis=1)
    step_r = np.diff(grid, axis=0)
    len_c = np.linalg.norm(step_c, axis=2).mean()
    len_r = np.linalg.norm(step_r, axis=2).mean()
    horiz_is_cols = np.abs(step_c[..., 0]).mean() >= np.abs(step_r[..., 0]).mean()
    if horiz_is_cols:
        return float(len_c), float(len_r)   # px_scan, px_spatial
    return float(len_r), float(len_c)


def _find_plateau(hits, rel_tol=0.06, min_run=3):
    """Longest run of consecutive (by k) hits whose px_scan_raw agrees within
    rel_tol of the run's running median -- a real detection re-measured at
    several compressions lands on (near) the same raw value every time, while
    a spurious one (moire/aliasing artifacts, noise the detector mistook for
    a corner grid) jumps around. None if the longest run is shorter than
    min_run, i.e. nothing in this sweep was self-consistent enough to trust.
    """
    best_run, run = [], []
    for h in hits:
        if run:
            med = statistics.median(x[1] for x in run)
            if med and abs(h[1] - med) / med <= rel_tol:
                run.append(h)
            else:
                if len(run) > len(best_run):
                    best_run = run
                run = [h]
        else:
            run = [h]
    if len(run) > len(best_run):
        best_run = run
    return best_run if len(best_run) >= min_run else None


def measure_scan_scale(gray, inner_size, seed_width=800, growth=1.6, max_rounds=10):
    """Robustly measure (px_scan_raw, px_spatial) via scan-precompressed checkerboard detection.

    How much the scan axis needs to be precompressed for reliable corner
    detection depends on how oversampled it is, i.e. on the conveyor speed at
    capture time -- a slow scan (many raw lines per square) needs heavy
    compression, a fast one (few raw lines per square) needs almost none, and
    that speed isn't known in advance. So rather than a fixed list of
    precompressed widths to try (which only covers whatever speed range it
    was picked against), this starts from one seed width and, each round it
    fails to find a self-consistent detection (see _find_plateau), widens the
    search a further factor of `growth` smaller AND larger -- until either a
    plateau turns up or both ends have hit the physically possible extremes
    (a handful of raw pixels per square on one side, native resolution on the
    other), at which point it's genuinely undetected rather than unsearched.

    Returns (px_scan_raw, px_spatial, k_used, corners_pc, method,
    spread_frac), or all-None if nothing self-consistent was found.
    """
    H, W = gray.shape
    hits = []
    seen_k = set()

    def try_width(target_w):
        k = min(1.0, target_w / W)
        k_key = round(k, 4)
        if k_key in seen_k:
            return
        seen_k.add(k_key)
        new_w = max(1, int(round(W * k)))
        g = gray if new_w == W else cv2.resize(gray, (new_w, H), interpolation=cv2.INTER_AREA)
        corners, method = detect_checkerboard(g, inner_size)
        if corners is None:
            return
        px_scan_pc, px_spatial = square_spacing(corners, inner_size)
        hits.append((k, px_scan_pc / k, px_spatial, corners, method))

    try_width(seed_width)
    lo = hi = seed_width
    for _ in range(max_rounds):
        hits.sort(key=lambda h: h[0])
        if _find_plateau(hits) is not None or (lo < 5 and hi > W):
            break
        lo, hi = lo / growth, hi * growth
        try_width(lo)
        try_width(hi)

    hits.sort(key=lambda h: h[0])
    if not hits:
        return None, None, None, None, None, None

    plateau = _find_plateau(hits)
    if plateau is None:
        print(f"  scan-scale: search widened to [{lo:.0f}, {hi:.0f}]px "
              f"({len(hits)} detections total), none self-consistent -- treating as undetected.")
        return None, None, None, None, None, None

    px_scan_raw = statistics.median(h[1] for h in plateau)
    k, _, px_spatial, corners, method = min(plateau, key=lambda h: abs(h[1] - px_scan_raw))
    spread = max(h[1] for h in plateau) - min(h[1] for h in plateau)
    spread_frac = spread / px_scan_raw if px_scan_raw else float("inf")
    print(f"  scan-scale: {len(plateau)}/{len(hits)} pre-compressed detections agree ({method}), "
          f"search reached [{lo:.0f}, {hi:.0f}]px: px_scan_raw={px_scan_raw:.2f} "
          f"(plateau spread {spread:.2f}, {100 * spread_frac:.1f}% of value)")
    return px_scan_raw, px_spatial, k, corners, method, spread_frac


def scan_scale_factor(px_scan, px_spatial, reference="spatial"):
    """Factor to multiply the scan-axis length by so squares become square.

    reference: spatial (trust the 640px optical axis, default), scan (trust
    the scan axis instead), min/max (resample both toward the finer/coarser
    common pitch). Returns (fx, fy) multipliers for the (x=scan, y=spatial) axes.
    """
    if reference == "spatial":
        return px_spatial / px_scan, 1.0
    if reference == "scan":
        return 1.0, px_scan / px_spatial
    if reference in ("min", "max"):
        target = min(px_scan, px_spatial) if reference == "min" else max(px_scan, px_spatial)
        return target / px_scan, target / px_spatial
    raise ValueError(f"unknown reference {reference!r}")


def resample(img, fx, fy):
    """Resample a 2D or multi-band 3D array by per-axis factors (x=cols, y=rows).

    cv2.resize only accepts <=4 channels per call, so a hyperspectral cube
    (224 bands) is resampled band-by-band.
    """
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * fx)))
    new_h = max(1, int(round(h * fy)))
    interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
    if img.ndim == 3 and img.shape[2] > 4:
        out = np.empty((new_h, new_w, img.shape[2]), dtype=img.dtype)
        for b in range(img.shape[2]):
            out[:, :, b] = cv2.resize(img[:, :, b], (new_w, new_h), interpolation=interp)
        return out
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


# ---------------------------------------------------------------- stage 7 --
def intensity_preview(cube):
    """Greyscale (height, width) uint8 preview: mean across all bands, min-max normalised."""
    mean = np.asarray(cube.mean(axis=2))
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo) if hi > lo else np.zeros_like(mean)
    return (norm * 255.0).round().astype(np.uint8)


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
    parser.add_argument("--checkerboard", nargs=2, type=int, default=(3, 3),
                        metavar=("COLS", "ROWS"),
                        help="Inner-corner count (default 3 3 = a 4x4-square board).")
    parser.add_argument("--reference", choices=["spatial", "scan", "min", "max"],
                        default="spatial",
                        help="Which axis to trust when equalising the scan-axis stretch.")
    parser.add_argument("--manual-scale", type=float, default=None,
                        help="Skip checkerboard detection; force this scan-axis (x) scale factor.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_img_dir = Path(args.out_img)
    corrected_path = out_dir / f"{args.sample}.npy"
    intensity_path = out_img_dir / f"{args.sample}.png"
    if not args.force and (corrected_path.exists() or intensity_path.exists()):
        sys.exit(f"{corrected_path} and/or {intensity_path} already exist -- "
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

    white_ref, covered = build_white_reference(darksub, masks, pct=args.white_pct)
    n_uncovered = int((~covered).sum())
    print(f"White reference: {covered.sum()}/{covered.size} width rows had direct tape "
          f"coverage ({n_uncovered} interpolated).")

    reflectance = apply_white_correction(darksub, white_ref)
    del darksub  # superseded by reflectance; nothing after this needs it
    print(f"Reflectance: min {reflectance.min():.4f}, max {reflectance.max():.4f}, "
          f"mean {reflectance.mean():.4f}")

    if args.manual_scale is not None:
        fx, fy = args.manual_scale, 1.0
        print(f"Manual scan-axis scale: x{fx:.4f} (checkerboard detection skipped).")
    else:
        gray8 = render_gray8(reflectance)
        px_scan, px_spatial, k, corners, method, spread_frac = measure_scan_scale(
            gray8, tuple(args.checkerboard))
        if px_scan is None:
            print(f"  warning: checkerboard {args.checkerboard[0]}x{args.checkerboard[1]} not "
                  f"detected at any pre-compression; skipping scan-axis geometry correction.")
            fx, fy = 1.0, 1.0
        else:
            fx, fy = scan_scale_factor(px_scan, px_spatial, args.reference)
            print(f"Detected via {method} (pre-compress k={k}): raw px/square scan={px_scan:.1f} "
                  f"spatial={px_spatial:.1f}; correction ({args.reference} ref): "
                  f"scan x{fx:.5f}, spatial x{fy:.5f}.")

    corrected = resample(reflectance, fx, fy)
    del reflectance  # superseded by corrected; nothing after this needs it
    np.save(corrected_path, corrected)
    print(f"wrote {corrected_path} ({corrected.nbytes / 1e6:.0f} MB, shape {corrected.shape})")

    Image.fromarray(intensity_preview(corrected), mode="L").save(intensity_path)
    print(f"wrote {intensity_path} ({corrected.shape[1]}x{corrected.shape[0]})")

    if archive_dir is not None:
        bin_files = sorted(Path(RAW_IMAGE_DIR).glob("*.bin"))
        archive_dir.mkdir(parents=True)
        for f in bin_files:
            f.rename(archive_dir / f.name)
        print(f"Archived {len(bin_files)} .bin files from {RAW_IMAGE_DIR}/ to {archive_dir}/ "
              f"({RAW_IMAGE_DIR}/ now empty and ready for the next scan).")


if __name__ == "__main__":
    main()
