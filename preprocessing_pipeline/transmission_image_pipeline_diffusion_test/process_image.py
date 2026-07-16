"""
End-to-end reflectance correction: stitch -> dark frame -> dark subtraction
-> white-capture ratio correction -> checkerboard geometry correction ->
greyscale intensity preview. Currently implements stages 1-6.

Stage 1 -- stitching
---------------------
Unpacks per-line Mono12Packed .bin captures into a (width, n_lines, channels)
uint16 cube. Mirrors pipeline/stitch_grain.py; reuses the same hsi_save_load
codec (see loadstich/hsi_save_load.py -- the byte-packing math is subtle, do
not reimplement it here).

Stage 2 -- dark frame
----------------------
mean_frame() collapses the dark reference cube to one (width, channels) frame
by averaging across scan lines. The push-broom sensor's dark current/offset
is fixed per (spatial pixel, band), not per scene, so a single averaged frame
is the correction target for every line of the grain scan later.

Stage 3 -- dark subtraction
------------------------------
apply_dark_correction() subtracts the (width, channels) dark frame from every
one of the grain cube's lines (broadcast over the scan-line axis), clipping
negative results to 0. This is the (r0-D) half of the (r0-D)/(W-D) reflectance
formula; the white half (W-D) is added in stage 4.

Stage 4 -- dedicated white capture + ratio correction
----------------------------------------------------------
Unlike reflectance mode, transmittance mode has no in-scene white target, so
each sample is paired with its own dedicated white capture: a line scan of
the light with nothing in the way, in the same raw format as the dark
reference (see raw_white/<sample>/, paired by sample name). mean_frame()
collapses it to one (width, channels) frame the same way the dark reference
is collapsed, then the dark frame is subtracted from it (clipped to 0) to
get (W-D). reflectance = darksub / (W-D) -- both sides are already
dark-subtracted, so this is exactly (r0-D)/(W-D).

Stage 5 -- checkerboard scan-axis geometry correction
------------------------------------------------------------
Unlike reflectance mode's in-scene checkerboard (ported from
pipeline/generate_viable_reflectance.py), transmittance mode has no board in
the same capture as the dish -- it's scanned as its own dedicated session
(same stage speed/optics setup as the dish scan) under
raw_checkerboard_image/, stitched separately, and its measured scan-axis
scale factor is applied to the dish/grain cube instead of being measured on
that cube directly.

The push-broom scan axis (n_lines) is stretched relative to the optical
spatial axis (640 px) by however much faster/slower the stage moved than the
frame rate -- here ~8500 lines for a plate that's only ~600 px across. The
checkerboard's known-square cells give the px-per-square on each axis needed
to recover that rescale, which then gets applied to the dish cube to make
its geometry square again. The raw board scan is too stretched for the
board to detect directly, so detection runs on several scan-precompressed
copies and the scale is recovered as the (robust, self-checking) median
across every copy that detected it.

Stage 6 -- intensity preview
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
DEFAULT_WHITE_DIR = "raw_white"    # per-sample white capture: <DEFAULT_WHITE_DIR>/<sample>/
DEFAULT_CHECKERBOARD = "raw_checkerboard"   # dedicated board capture, same stage/optics setup
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


def stitch(directory, w=WIDTH, c=CHANNELS, max_bad_frac=0.01):
    """Stitch a directory of per-line .bin captures into a (w, n_lines, c) cube.

    Raises if more than `max_bad_frac` of lines failed to parse/were blank --
    a handful of flaky lines is normal noise, but silently averaging in a
    large fraction of zero-filled frames would quietly bias every downstream
    stage (dark frame, white reference) without any error.

    Also raises if the leading `<index>_` in the filenames has any gap or
    duplicate -- a dropped capture file would otherwise be invisible: the
    cube is sized from however many files are present, so a missing line
    just silently shifts every subsequent line by one instead of erroring.
    """
    directory = Path(directory)
    files = sorted(f for f in os.listdir(directory) if f.endswith(".bin"))
    if not files:
        raise ValueError(f"no .bin files found in {directory}")

    indices = [int(f.split("_", 1)[0]) for f in files]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{directory}: duplicate frame indices in filenames -- capture is corrupt.")
    gaps = [(a, b) for a, b in zip(indices, indices[1:]) if b - a != 1]
    if gaps:
        raise ValueError(
            f"{directory}: {len(gaps)} gap(s) in frame index sequence (e.g. {gaps[0]}) -- "
            f"one or more scan lines are missing; every line after a gap would silently shift "
            f"out of alignment with the true scan geometry if stitched as-is.")

    cube = np.empty((w, len(files), c), dtype=np.uint16)
    bad = 0
    for i, fname in enumerate(files):
        raw = np.fromfile(directory / fname, dtype=np.uint8)
        frame = load_line(raw, w, c)
        if not frame.any():
            bad += 1
        cube[:, i, :] = frame
    print(f"  {directory.name}: {len(files)} lines stitched ({bad} blank/bad).")

    bad_frac = bad / len(files)
    if bad_frac > max_bad_frac:
        raise ValueError(
            f"{directory}: {bad}/{len(files)} lines ({100 * bad_frac:.1f}%) were blank/unparseable "
            f"-- exceeds the {100 * max_bad_frac:.0f}% integrity threshold; investigate the capture "
            f"(e.g. a loose cable or truncated files) before trusting this data.")
    return cube


# ---------------------------------------------------------------- stage 2 --
def mean_frame(cube):
    """Mean (width, channels) frame, averaged across a reference cube's lines.

    Used for both the dark reference and (stage 4) the dedicated white
    capture -- the push-broom sensor's per-(pixel, band) response is fixed
    per scene, not per line, so a single averaged frame is the correction
    target for every line of the grain scan later.
    """
    return cube.mean(axis=1).astype(np.float64)


# ---------------------------------------------------------------- stage 3 --
def apply_dark_correction(grain_cube, dark_frame):
    """clip(grain - dark_frame, 0), broadcasting dark_frame over every scan line.

    Cast back to grain_cube's dtype (uint16) to match the raw cube's size on
    disk -- the float32 headroom is only needed transiently for the
    subtraction itself. Casts once and subtracts/clips in-place into that
    same buffer (rather than each producing a fresh full-cube temporary), and
    returns the clipped-px count computed from this same diff instead of the
    caller redoing the subtraction from scratch -- on an 8500-line cube each
    full-cube float32 temporary is ~5GB, so this keeps peak memory to roughly
    one such temporary instead of three-plus.

    Returns (corrected (grain_cube.dtype), n_clipped).
    """
    diff = grain_cube.astype(np.float32)
    diff -= dark_frame[:, None, :].astype(np.float32)
    n_clipped = int((diff < 0).sum())
    corrected = np.clip(diff, 0, None, out=diff).astype(grain_cube.dtype)
    return corrected, n_clipped


# ---------------------------------------------------------------- stage 4 --
def apply_white_correction(darksub_cube, white_frame):
    """reflectance[...,b] = darksub[...,b] / white_frame[row,b].

    Both sides are already dark-subtracted, so this is exactly the
    (r0-D)/(W-D) formula -- white_frame stands in for (W-D), broadcast over
    every scan line the same way apply_dark_correction broadcasts the dark
    frame. Divides in-place (/=) into the float32 cast rather than allocating
    a second full-cube temporary for the division result.
    """
    reflectance = darksub_cube.astype(np.float32)
    reflectance /= white_frame.astype(np.float32)[:, None, :]
    return reflectance


# ---------------------------------------------------------------- stage 5 --
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
    each on the image, a 2x upscale, and an inverted copy -- push-broom
    targets are often low-contrast and blurred along the scan axis. Returns
    (corners (N,2) float in the ORIGINAL image, method label) or (None, None).
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


def measure_scan_scale(gray, inner_size, ks=(0.12, 0.13, 0.14, 0.15, 0.16)):
    """Robustly measure (px_scan_raw, px_spatial) via scan-precompressed checkerboard detection.

    Detecting on several precompression factors k and taking the median keeps
    this self-checking (the per-k spread should be tiny -- a large spread
    means the detections disagree, most likely because `inner_size` doesn't
    match the true board and cv2 is locking onto arbitrary, differently
    located sub-windows of a larger grid rather than the true corners).
    Returns (px_scan_raw, px_spatial, k_used, corners_pc, method, spread_frac)
    or all-None if no k let the board detect. spread_frac is the per-k spread
    in s relative to the chosen median s -- the caller should gate on this
    rather than trust any single result blindly.
    """
    H, W = gray.shape
    hits = []
    for k in ks:
        g = cv2.resize(gray, (max(1, int(round(W * k))), H), interpolation=cv2.INTER_AREA)
        corners, method = detect_checkerboard(g, inner_size)
        if corners is None:
            continue
        px_scan_pc, px_spatial = square_spacing(corners, inner_size)
        hits.append((px_spatial * k / px_scan_pc, px_scan_pc / k, px_spatial, k, corners, method))
    if not hits:
        return None, None, None, None, None, None
    hits.sort(key=lambda h: h[0])
    # statistics.median (not hits[len(hits)//2], which is biased toward the
    # upper-middle value for an even hit count) of the s values, then the hit
    # closest to it -- keeps a real, self-consistent (px_scan, px_spatial,
    # corners, method) tuple rather than an average of unrelated detections.
    median_s = statistics.median(h[0] for h in hits)
    s, px_scan_raw, px_spatial, k, corners, method = min(hits, key=lambda h: abs(h[0] - median_s))
    spread = hits[-1][0] - hits[0][0]
    spread_frac = spread / median_s if median_s else float("inf")
    print(f"  scan-scale from {len(hits)} pre-compressed detections ({method}): "
          f"s={s:.5f} (per-k spread {spread:.5f}, {100 * spread_frac:.1f}% of median)")
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


# ---------------------------------------------------------------- stage 6 --
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
    parser.add_argument("--white", default=None,
                        help=f"White reference for this sample: raw capture dir or stitched cube "
                             f".npy -- a dedicated line scan of the light, same format as --dark. "
                             f"Defaults to {DEFAULT_WHITE_DIR}/<sample>/ (paired to the grain scan "
                             f"by sample name).")
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
    parser.add_argument("--max-clip-frac", type=float, default=0.02,
                        help="Abort if more than this fraction of pixels go negative during dark "
                             "subtraction (before clipping), for either the grain scan or the white "
                             "reference -- a high fraction usually means the dark reference doesn't "
                             "match that capture's exposure/gain.")
    parser.add_argument("--max-scale-spread-frac", type=float, default=0.03,
                        help="Abort if the per-precompression-level checkerboard scale detections "
                             "disagree by more than this fraction of their median -- a large spread "
                             "means detections aren't landing on the same true corners (e.g. wrong "
                             "--checkerboard inner-corner count), so the scale factor can't be trusted.")
    parser.add_argument("--checkerboard", nargs=2, type=int, default=(17, 17),
                        metavar=("COLS", "ROWS"),
                        help="Inner-corner count (default 17 17 = the dedicated 18x18-square "
                             "transmission checkerboard target in raw_checkerboard/).")
    parser.add_argument("--checkerboard-dir", default=DEFAULT_CHECKERBOARD,
                        help="Dedicated checkerboard capture used to measure the scan-axis "
                             "rescale (stitched separately, then applied to the dish/grain "
                             "cube). Raw capture dir or stitched cube .npy.")
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

    frame = mean_frame(dark_cube)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    grain_path = Path(args.grain) if args.grain else Path(RAW_IMAGE_DIR)
    print(f"Loading grain scan from {grain_path} ...")
    if grain_path.is_file() and grain_path.suffix == ".npy":
        grain_cube = np.load(grain_path)
    else:
        grain_cube = stitch(grain_path)
    print(f"  shape {grain_cube.shape}, dtype {grain_cube.dtype}")

    if grain_cube.shape[0] != frame.shape[0] or grain_cube.shape[2] != frame.shape[1]:
        sys.exit(f"grain/dark shape mismatch: grain {grain_cube.shape} vs dark frame {frame.shape}")

    darksub, n_clipped = apply_dark_correction(grain_cube, frame)
    clip_frac = n_clipped / grain_cube.size
    print(f"Applied dark subtraction ({n_clipped} px clipped at 0, {100 * clip_frac:.3f}%).")
    if clip_frac > args.max_clip_frac:
        sys.exit(f"{100 * clip_frac:.1f}% of pixels went negative during dark subtraction -- "
                 f"exceeds the {100 * args.max_clip_frac:.0f}% integrity threshold; check that "
                 f"--dark matches this capture's exposure/gain before trusting this data.")

    del grain_cube  # ~2.5GB for this capture; nothing after this point needs the raw cube

    white_path = Path(args.white) if args.white else Path(DEFAULT_WHITE_DIR) / args.sample
    print(f"Loading white reference from {white_path} ...")
    if white_path.is_file() and white_path.suffix == ".npy":
        white_cube = np.load(white_path)
    else:
        white_cube = stitch(white_path)
    print(f"  shape {white_cube.shape}, dtype {white_cube.dtype}")

    white_frame_raw = mean_frame(white_cube)
    del white_cube
    if white_frame_raw.shape != frame.shape:
        sys.exit(f"white/dark shape mismatch: white {white_frame_raw.shape} vs "
                 f"dark frame {frame.shape}")

    white_diff = white_frame_raw - frame
    n_clipped_white = int((white_diff < 0).sum())
    clip_frac_white = n_clipped_white / white_diff.size
    print(f"Applied dark subtraction to white reference ({n_clipped_white} px clipped at 0, "
          f"{100 * clip_frac_white:.3f}%).")
    if clip_frac_white > args.max_clip_frac:
        sys.exit(f"{100 * clip_frac_white:.1f}% of white-reference pixels went negative during "
                 f"dark subtraction -- exceeds the {100 * args.max_clip_frac:.0f}% integrity "
                 f"threshold; check that --dark matches --white's exposure/gain before trusting "
                 f"this data.")
    white_frame = np.clip(white_diff, 0, None)

    n_bad_ref = int((white_frame <= 0).sum())
    if n_bad_ref:
        sys.exit(f"white reference has {n_bad_ref}/{white_frame.size} non-positive entries -- "
                 f"dividing by these would produce inf/nan reflectance; check that --white points "
                 f"at a valid light-only capture before retrying.")

    reflectance = apply_white_correction(darksub, white_frame)
    del darksub  # superseded by reflectance; nothing after this needs it
    print(f"Reflectance: min {reflectance.min():.4f}, max {reflectance.max():.4f}, "
          f"mean {reflectance.mean():.4f}")

    n_nonfinite = int((~np.isfinite(reflectance)).sum())
    if n_nonfinite:
        sys.exit(f"reflectance has {n_nonfinite}/{reflectance.size} non-finite (inf/nan) values "
                 f"-- aborting rather than writing corrupt data to disk.")

    if args.manual_scale is not None:
        fx, fy = args.manual_scale, 1.0
        print(f"Manual scan-axis scale: x{fx:.4f} (checkerboard detection skipped).")
    else:
        board_path = Path(args.checkerboard_dir)
        print(f"Loading checkerboard capture from {board_path} ...")
        if board_path.is_file() and board_path.suffix == ".npy":
            board_cube = np.load(board_path)
        else:
            board_cube = stitch(board_path)
        print(f"  shape {board_cube.shape}, dtype {board_cube.dtype}")

        board_gray8 = render_gray8(board_cube)
        px_scan, px_spatial, k, corners, method, spread_frac = measure_scan_scale(
            board_gray8, tuple(args.checkerboard))
        if px_scan is None:
            sys.exit(f"Checkerboard {args.checkerboard[0]}x{args.checkerboard[1]} not detected in "
                     f"{board_path} at any pre-compression; re-run with --manual-scale FLOAT to "
                     f"force the correction.")
        if spread_frac > args.max_scale_spread_frac:
            sys.exit(f"Checkerboard scale detections disagree by {100 * spread_frac:.1f}% across "
                     f"pre-compression levels (threshold {100 * args.max_scale_spread_frac:.0f}%) -- "
                     f"likely {args.checkerboard[0]}x{args.checkerboard[1]} doesn't match the true "
                     f"board in {board_path} (cv2 can lock onto an arbitrary sub-window of a larger "
                     f"grid). Verify --checkerboard against the physical board, or force with "
                     f"--manual-scale FLOAT.")
        fx, fy = scan_scale_factor(px_scan, px_spatial, args.reference)
        print(f"Detected via {method} (pre-compress k={k}) in {board_path.name}: raw px/square "
              f"scan={px_scan:.1f} spatial={px_spatial:.1f}; correction ({args.reference} ref): "
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
