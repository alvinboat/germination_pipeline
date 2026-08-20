"""
End-to-end reflectance correction: stitch -> dark frame -> dark subtraction
-> white-capture ratio correction (opt-in, see stage 4) -> checkerboard
geometry correction (opt-in, see stage 5) -> greyscale intensity preview.
Currently implements stages 1-6.

Stage 1 -- stitching
---------------------
Unpacks per-line Mono12Packed .bin captures into a (width, n_lines, channels)
uint16 cube. Mirrors pipeline/stitch_grain.py; reuses the same hsi_save_load
codec (see loadstich/hsi_save_load.py -- the byte-packing math is subtle, do
not reimplement it here).

Stage 2 -- dark frames (one per signal)
------------------------------------------
Each sample is captured as two pairs: the grain/image scan with its own dark
exposure, and the dedicated white capture with its own dark exposure. These
two darks are not interchangeable -- raw_dark/raws/<sample>*/ is the dark
reference for the grain scan, raw_dark/whites/<sample>*/ is the dark
reference for the white capture (see find_sample_dir()). mean_frame()
collapses each to one (width, channels) frame -- the push-broom sensor's
dark current/offset is fixed per (spatial pixel, band), not per scene, so a
single averaged frame is the correction target for every line of its paired
signal.

Stage 3 -- dark subtraction
------------------------------
apply_dark_correction() subtracts the (width, channels) dark-for-image frame
from every one of the grain cube's lines (broadcast over the scan-line axis),
clipping negative results to 0. This is the (r0-D) half of the (r0-D)/(W-D)
reflectance formula; the white half (W-D) is computed the same way in stage
4, using the dark-for-white frame instead.

Stage 4 -- dedicated white capture + ratio correction (OFF by default)
----------------------------------------------------------------------
Disabled unless --white-correct is passed. The dedicated white capture
(raw_white/<sample>*/) currently on hand is a straight, unobstructed beam
shot -- it shows the light source's own raw emission profile (sharply peaked
in the middle of the 640px spatial axis, ~0 at the edges), not the profile
the grain scan actually sees, since the grain scan's light passes through
the dish/tray first, which scatters and evens the light out before it hits
the sensor (confirmed: the grain scan's own dark-subtracted spatial profile
is flat, ~1000-2200 across the full width, nothing like the white capture's
~20-2700 spike). Dividing by this white reference doesn't cancel real
unevenness, it manufactures fake banding -- so it's off until a white
reference captured through the same optical path (e.g. an empty tray/dish,
no seeds) is available. See find_sample_dir(), parse_exposure(), and
apply_white_correction() -- kept for when that reference exists.

When enabled: mean_frame() collapses the white capture to one (width,
channels) frame the same way the dark references are collapsed, then the
dark-for-white frame (not the dark-for-image frame) is subtracted from it
(clipped to 0) to get (W-D). The white capture is taken at a much lower
exposure than the grain scan -- the grain scan needs a high exposure to see
through the kernel, but the same exposure blows out an unobstructed white
capture. Raw ADC counts scale ~linearly with exposure time, so (W-D)
measured at the white capture's exposure is scaled up by
(image_exposure / white_exposure) before use, putting it on the same
footing as the grain scan's dark-subtracted counts. Both exposures are
parsed from the matched dark folders' _<n>/_<n>k suffix (dark-for-image
shares the grain scan's exposure, dark-for-white shares the white capture's,
per stage 2).

reflectance = darksub / ((W-D) * exposure_scale) -- both sides are already
dark-subtracted and now on the same exposure footing, so this is exactly
(r0-D)/(W-D) once W-D is referred to the grain scan's exposure.

By default (--white-correct not passed), this stage is skipped entirely --
neither raw_dark/whites/<sample>*/ nor raw_white/<sample>*/ are read, and
the corrected output is just the dark-subtracted grain scan (stage 3's
result) carried through to stage 6.

Stage 5 -- checkerboard scan-axis geometry correction (OFF by default)
------------------------------------------------------------------------
Disabled unless --geometry-correct or --manual-scale is passed -- kept in
the codebase for later use, not currently part of the default run. The
scan-axis stretch factor this stage measures is only valid for a grain scan
captured at the same stage speed AND the same exposure/line-rate as the
checkerboard capture it's measured from (exposure affects line rate, which
affects the stretch, exactly like stage speed does). The four lighting
setups in use need different (high) exposures to see through the kernel,
which isn't necessarily compatible with a single shared checkerboard capture
(itself liable to saturate at those exposures) -- so this stage is disabled
until that's sorted out, rather than silently applying a wrong-for-this-scan
factor.

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

eBUS Player saves a session's .bin files flat into raw_image/ (no
per-session subfolder). Run:
    python3 process_image.py slit_diffuse

`sample` is a prefix, not an exact folder name -- it's matched against
raw_white/, raw_dark/raws/, and raw_dark/whites/ to find each one's single
subfolder starting with that prefix (e.g. raw_white/slit_diffuse_white_150/,
raw_dark/raws/slit_diffuse_dark_25k/, raw_dark/whites/slit_diffuse_dark_150/)
-- exactly one match is required in each, or the run aborts rather than
guessing.

This stitches/corrects everything currently in raw_image/, writes
corrected_file/slit_diffuse.npy and corrected_image/slit_diffuse.png, then
moves those .bin files into raw_image_storage/slit_diffuse/ -- leaving
raw_image/ empty and ready for the next capture.
"""

import argparse
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

DEFAULT_DARK_RAWS_DIR = "raw_dark/raws"      # dark-for-image reference: <...>/<sample>*/
DEFAULT_DARK_WHITES_DIR = "raw_dark/whites"  # dark-for-white reference: <...>/<sample>*/
DEFAULT_WHITE_DIR = "raw_white"    # per-sample white capture: <DEFAULT_WHITE_DIR>/<sample>*/
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


def find_sample_dir(base_dir, sample):
    """The single subdirectory of base_dir whose name is/starts with sample.

    Folders are named <sample>_<suffix>_<exp>/ (e.g. slit_diffuse_white_150),
    not the bare sample name, so an exact-name lookup won't work -- this
    matches by prefix instead, and requires exactly one hit so a stale or
    leftover folder from another session can't silently get picked up.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise SystemExit(f"{base} does not exist.")
    matches = sorted(p for p in base.iterdir()
                      if p.is_dir() and (p.name == sample or p.name.startswith(sample + "_")))
    if not matches:
        raise SystemExit(f"no folder matching '{sample}' found in {base} -- "
                          f"expected exactly one (e.g. {sample}_<suffix>_<exp>/).")
    if len(matches) > 1:
        raise SystemExit(f"multiple folders matching '{sample}' found in {base}: "
                          f"{[m.name for m in matches]} -- expected exactly one; "
                          f"use a more specific sample name, or clear the stale folder.")
    return matches[0]


EXPOSURE_SUFFIX_RE = re.compile(r"_(\d+)(k)?$")


def parse_exposure(name):
    """Exposure time encoded in a capture folder's trailing _<n>/_<n>k suffix.

    Matches this repo's <sample>_<suffix>_<exp> folder-naming convention (e.g.
    paper_dark_125k -> 125000, paper_dark_1250 -> 1250). Returns None if name
    doesn't end in a parseable exposure token.
    """
    m = EXPOSURE_SUFFIX_RE.search(name)
    if not m:
        return None
    value = int(m.group(1))
    return value * 1000 if m.group(2) else value


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
def intensity_preview(cube, lo_pct=1.0, hi_pct=99.0):
    """Greyscale (height, width) uint8 preview: mean across all bands, percentile-clip normalised.

    A handful of bad-white-reference pixels (see the white-reference fallback
    above) produce reflectance values orders of magnitude above the rest of
    the scene -- a true min-max stretch spends nearly the whole 0-255 range
    on those outliers and crushes everything else to near-black. Clipping to
    the 1st/99th percentile instead keeps the stretch anchored to the actual
    scene content.

    nan-aware, because transmission_imagev2 writes nan where it has no usable
    white reference and every pixel of such a cube carries some (the lamp is
    dead in the outermost bands). A plain mean propagates that to the whole
    frame, the percentiles follow, and the uint8 cast turns nan into 0 -- a
    silently, solidly black preview. On a cube with no nan these behave
    identically to the plain versions.
    """
    mean = np.asarray(np.nanmean(cube, axis=2))
    if not np.isfinite(mean).any():
        return np.zeros(mean.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(mean, [lo_pct, hi_pct])
    norm = np.clip((mean - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(mean)
    return (np.nan_to_num(norm, nan=0.0) * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample",
                        help=f"Sample name, e.g. slit_diffuse -- matched as a prefix against "
                             f"{DEFAULT_WHITE_DIR}/, {DEFAULT_DARK_RAWS_DIR}/, and "
                             f"{DEFAULT_DARK_WHITES_DIR}/ to find each one's single matching "
                             f"folder, and used verbatim to name outputs (<sample>.npy / "
                             f"<sample>.png) and the archive folder ({STORAGE_DIR}/<sample>/) the "
                             f"raw .bin files are moved into afterward.")
    parser.add_argument("--dark-raw", default=None,
                        help=f"Dark reference for the grain scan: raw capture dir or stitched cube "
                             f".npy. Defaults to the single {DEFAULT_DARK_RAWS_DIR}/<sample>*/ "
                             f"folder matching the sample argument.")
    parser.add_argument("--dark-white", default=None,
                        help=f"Dark reference for the white capture -- a separate capture from "
                             f"--dark-raw, since the grain scan and white capture are taken at "
                             f"different exposures. Raw capture dir or stitched cube .npy. Defaults "
                             f"to the single {DEFAULT_DARK_WHITES_DIR}/<sample>*/ folder matching "
                             f"the sample argument.")
    parser.add_argument("--white", default=None,
                        help=f"White reference for this sample: raw capture dir or stitched cube "
                             f".npy -- a dedicated line scan of the light. Defaults to the single "
                             f"{DEFAULT_WHITE_DIR}/<sample>*/ folder matching the sample argument.")
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
    parser.add_argument("--white-correct", action="store_true",
                        help="Apply the dedicated white capture ratio correction (stage 4). Off by "
                             "default: the white capture on hand is a straight unobstructed beam, not "
                             "shot through the dish/tray the grain scan's light actually passes "
                             "through, so dividing by it manufactures banding rather than correcting "
                             "real unevenness. Pass this once a white reference captured through the "
                             "same optical path (e.g. an empty tray) is available.")
    parser.add_argument("--exposure-scale", type=float, default=None,
                        help="Manual image_exposure/white_exposure ratio to scale the "
                             "dark-subtracted white reference by, skipping auto-detection. "
                             "Defaults to parsing both exposures from the matched dark "
                             "folders' _<n>/_<n>k suffix (see parse_exposure()); pass this "
                             "if those folder names don't encode the exposure. Only used with "
                             "--white-correct.")
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
                        help="Skip checkerboard detection; force this scan-axis (x) scale factor "
                             "(implies geometry correction runs, without needing --geometry-correct).")
    parser.add_argument("--geometry-correct", action="store_true",
                        help="Apply the checkerboard scan-axis geometry correction (stage 5). Off "
                             "by default: the checkerboard's measured scale factor is only valid if "
                             "it was captured at the same stage speed AND exposure/line-rate as this "
                             "scan, which isn't guaranteed yet across the different lighting/exposure "
                             "setups in use -- pass this flag once that's sorted out, or use "
                             "--manual-scale to force a factor regardless.")
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

    dark_raw_path = Path(args.dark_raw) if args.dark_raw else find_sample_dir(DEFAULT_DARK_RAWS_DIR, args.sample)
    print(f"Loading dark-for-image reference from {dark_raw_path} ...")
    if dark_raw_path.is_file() and dark_raw_path.suffix == ".npy":
        dark_raw_cube = np.load(dark_raw_path)
    else:
        dark_raw_cube = stitch(dark_raw_path)
    print(f"  shape {dark_raw_cube.shape}, dtype {dark_raw_cube.dtype}")

    dark_raw_frame = mean_frame(dark_raw_cube)

    if args.white_correct:
        dark_white_path = Path(args.dark_white) if args.dark_white else find_sample_dir(DEFAULT_DARK_WHITES_DIR, args.sample)
        print(f"Loading dark-for-white reference from {dark_white_path} ...")
        if dark_white_path.is_file() and dark_white_path.suffix == ".npy":
            dark_white_cube = np.load(dark_white_path)
        else:
            dark_white_cube = stitch(dark_white_path)
        print(f"  shape {dark_white_cube.shape}, dtype {dark_white_cube.dtype}")

        dark_white_frame = mean_frame(dark_white_cube)

        if args.exposure_scale is not None:
            exposure_scale = args.exposure_scale
            print(f"Using manual exposure scale x{exposure_scale:.4f} (auto-detection skipped).")
        else:
            image_exposure = parse_exposure(dark_raw_path.stem)
            white_exposure = parse_exposure(dark_white_path.stem)
            if image_exposure is None or white_exposure is None:
                sys.exit(f"Could not parse exposure from {dark_raw_path.name!r} and/or "
                          f"{dark_white_path.name!r} (expected a trailing _<n> or _<n>k, e.g. "
                          f"_125k) -- pass --exposure-scale to set the image/white exposure "
                          f"ratio manually.")
            exposure_scale = image_exposure / white_exposure
            print(f"Exposure scale: image {image_exposure} / white {white_exposure} = "
                  f"x{exposure_scale:.4f} (white reference will be scaled up to match the "
                  f"grain scan's higher exposure).")
    else:
        print("Skipping white-capture ratio correction (stage 4) -- off by default until a white "
              "reference captured through the same optical path (e.g. an empty tray) is available. "
              "Pass --white-correct to use the current straight-beam white capture anyway.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    grain_path = Path(args.grain) if args.grain else Path(RAW_IMAGE_DIR)
    print(f"Loading grain scan from {grain_path} ...")
    if grain_path.is_file() and grain_path.suffix == ".npy":
        grain_cube = np.load(grain_path)
    else:
        grain_cube = stitch(grain_path)
    print(f"  shape {grain_cube.shape}, dtype {grain_cube.dtype}")

    darksub, n_clipped = apply_dark_correction(grain_cube, dark_raw_frame)
    clip_frac = n_clipped / grain_cube.size
    print(f"Applied dark subtraction ({n_clipped} px clipped at 0, {100 * clip_frac:.3f}%).")

    del grain_cube  # ~2.5GB for this capture; nothing after this point needs the raw cube

    if args.white_correct:
        white_path = Path(args.white) if args.white else find_sample_dir(DEFAULT_WHITE_DIR, args.sample)
        print(f"Loading white reference from {white_path} ...")
        if white_path.is_file() and white_path.suffix == ".npy":
            white_cube = np.load(white_path)
        else:
            white_cube = stitch(white_path)
        print(f"  shape {white_cube.shape}, dtype {white_cube.dtype}")

        white_frame_raw = mean_frame(white_cube)
        del white_cube

        white_diff = white_frame_raw - dark_white_frame
        n_clipped_white = int((white_diff < 0).sum())
        clip_frac_white = n_clipped_white / white_diff.size
        print(f"Applied dark subtraction to white reference ({n_clipped_white} px clipped at 0, "
              f"{100 * clip_frac_white:.3f}%).")
        white_frame = np.clip(white_diff, 0, None)

        n_bad_ref = int((white_frame <= 0).sum())
        if n_bad_ref:
            print(f"  warning: white reference has {n_bad_ref}/{white_frame.size} non-positive entries "
                  f"(dividing by these would produce inf/nan) -- treating those (pixel, band) positions "
                  f"as uncorrected (white reference set to 1.0 there) rather than aborting.")
            white_frame[white_frame <= 0] = 1.0

        # Scale after the fallback so the literal 1.0 above lands in the same
        # (image-exposure-referred) unit space as every other entry, instead of
        # being ~exposure_scale too small relative to them.
        white_frame *= exposure_scale

        reflectance = apply_white_correction(darksub, white_frame)
        del darksub  # superseded by reflectance; nothing after this needs it
        print(f"Reflectance: min {reflectance.min():.4f}, max {reflectance.max():.4f}, "
              f"mean {reflectance.mean():.4f}")
    else:
        reflectance = darksub
        print(f"Dark-subtracted signal (no white correction): min {reflectance.min()}, "
              f"max {reflectance.max()}, mean {reflectance.mean():.4f}")

    if args.manual_scale is not None:
        fx, fy = args.manual_scale, 1.0
        print(f"Manual scan-axis scale: x{fx:.4f} (checkerboard detection skipped).")
    elif not args.geometry_correct:
        fx, fy = 1.0, 1.0
        print("Skipping scan-axis geometry correction (checkerboard) -- off by default until "
              "stage speed/exposure are confirmed consistent between the checkerboard capture and "
              "this scan. Pass --geometry-correct to run detection, or --manual-scale to force a "
              "factor.")
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
            print(f"  warning: checkerboard {args.checkerboard[0]}x{args.checkerboard[1]} not "
                  f"detected in {board_path} at any pre-compression; skipping scan-axis geometry "
                  f"correction.")
            fx, fy = 1.0, 1.0
        else:
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
