"""
Produce a geometrically *viable* reflectance image: stitch the reflectance
capture into a cube, then use a checkerboard target to undo the push-broom
scan-axis stretch so the pixels are physically square.

Why this is needed
------------------
A Specim push-broom camera builds the image one line at a time as the stage
moves. The spatial axis (640 px across the slit) is fixed by the optics, but the
scan axis is however many lines were captured — its pixel pitch depends entirely
on stage speed, so the stitched image is stretched (here ~8500 lines for a plate
that is only ~600 px tall). A checkerboard of known-square cells lets us measure
the px-per-square on each axis and rescale the scan axis so a physical square
looks square again.

This is *anisotropic rescale* only (a per-axis resample). It fixes the stretch;
it does not correct perspective or skew. See --help for a perspective variant if
you need it later.

Pipeline
--------
1. Load the cube: either stitch a capture directory (reuses stitch_grain.stitch)
   or load an existing stitched *_cube.npy directly.
2. Render an 8-bit greyscale (mean across bands, or one --band).
3. Detect the checkerboard (default 3x3 inner corners = a 4x4-square board).
4. Measure px-per-square on the scan and spatial axes; rescale the scan axis so
   both match (squares become square). The spatial axis is trusted by default.
5. Write PNGs for inspection: the raw intensity, a corner-overlay, and the
   corrected image. Optionally rescale and save the full cube too.

Coordinate convention (matches grid_overlay.py / analyze_kernels.py):
    image x = horizontal = scan-line axis   (the stretched one)
    image y = vertical   = spatial px axis   (640, trusted)

Run:
    python3 generate_viable_reflectance.py 07012026/grain_ref_exp_2500
    python3 generate_viable_reflectance.py 07012026/stitched/grain_ref_exp_2500_cube.npy
    python3 generate_viable_reflectance.py <cube> --checkerboard 3 3 --save-cube
    python3 generate_viable_reflectance.py <dir> --manual-scale 0.14   # detection off/failed
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("This script needs OpenCV: pip install opencv-python")

# stitch_grain lives next to this file and already handles the Mono12Packed
# unpack + orientation + the loadstich import path gotcha, so reuse it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_grain import stitch  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_cube(src):
    """Return a (spatial, n_lines, channels) uint16 cube.

    `src` may be an existing stitched *_cube.npy (loaded directly) or a capture
    directory of per-line .bin files (stitched on the fly).
    """
    src = Path(src)
    if src.is_file() and src.suffix == ".npy":
        cube = np.load(src)
        if cube.ndim != 3:
            sys.exit(f"Expected a 3D cube, got shape {cube.shape}")
        print(f"Loaded cube {src.name}: shape {cube.shape}, dtype {cube.dtype}")
        return cube
    if src.is_dir():
        print(f"Stitching capture directory {src} ...")
        return stitch(src)
    sys.exit(f"{src} is neither a *_cube.npy file nor a capture directory")


def render_gray8(cube, band=None):
    """(spatial, n_lines) uint8, min-max normalised. band=None -> mean across bands."""
    img = cube.mean(axis=2) if band is None else cube[:, :, band].astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    norm = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img, dtype=np.float32)
    return (norm * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Checkerboard detection
# ---------------------------------------------------------------------------
def detect_checkerboard(gray, inner_size):
    """Find `inner_size` = (cols, rows) inner corners in an 8-bit greyscale image.

    Tries the classic detector (sub-pixel refined) then the newer SB detector,
    each on the image, a 2x upscale, and an inverted copy — push-broom targets
    are often low-contrast and blurred along the scan axis. Returns an
    (N, 2) float array of corner (x, y) coords in the ORIGINAL image, plus a
    label describing which method won, or (None, None) if nothing was found.
    """
    cols, rows = inner_size
    classic_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                     | cv2.CALIB_CB_NORMALIZE_IMAGE)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    def variants():
        yield "classic", gray, 1.0
        yield "classic-inv", cv2.bitwise_not(gray), 1.0
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        yield "classic-2x", up, 2.0
        yield "classic-2x-inv", cv2.bitwise_not(up), 2.0

    # Classic detector (with sub-pixel refinement).
    for label, img, scale in variants():
        ok, corners = cv2.findChessboardCorners(img, (cols, rows), classic_flags)
        if ok:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(-1, 2) / scale, label

    # SB detector (more robust to blur; no separate sub-pixel step needed).
    for label, img, scale in variants():
        try:
            ok, corners = cv2.findChessboardCornersSB(
                img, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        except cv2.error:
            ok = False
        if ok:
            return corners.reshape(-1, 2) / scale, label.replace("classic", "SB")

    return None, None


def square_spacing(corners, inner_size):
    """Median px-per-square along the scan (x) and spatial (y) image axes.

    corners is (cols*rows, 2) in OpenCV row-major order. We measure the edge
    length between neighbouring corners in each grid direction (true length, so
    a slightly rotated board is fine) and assign each grid direction to whichever
    image axis it mostly runs along. Returns (px_scan, px_spatial).
    """
    cols, rows = inner_size
    grid = corners.reshape(rows, cols, 2)

    step_c = np.diff(grid, axis=1)          # neighbours along the "cols" direction
    step_r = np.diff(grid, axis=0)          # neighbours along the "rows" direction
    len_c = np.linalg.norm(step_c, axis=2).mean()
    len_r = np.linalg.norm(step_r, axis=2).mean()
    # Which grid direction runs horizontally (bigger mean |dx|) -> that's the scan axis.
    horiz_is_cols = np.abs(step_c[..., 0]).mean() >= np.abs(step_r[..., 0]).mean()
    if horiz_is_cols:
        return float(len_c), float(len_r)   # px_scan, px_spatial
    return float(len_r), float(len_c)


def measure_scan_scale(gray, inner_size, ks=(0.12, 0.13, 0.14, 0.15, 0.16)):
    """Robustly measure (px_scan, px_spatial) in RAW pixels via the checkerboard.

    Why this is not just detect_checkerboard(gray): a push-broom raw scan is
    hugely stretched along-track (here ~6.7x), so each checkerboard square is
    smeared into a wide blur and cv2's detector fails on the raw greyscale
    outright. So we detect on several scan-PRECOMPRESSED copies (factor k): in a
    k-compressed image a square's scan size is k x its raw size, so the raw
    scan-square size is (px_scan_pc / k) while px_spatial is unchanged. The
    resulting scan scale s = px_spatial * k / px_scan_pc is invariant to k, so we
    take the median over every k that detected — robust and self-checking (the
    per-k spread should be tiny).

    Returns (px_scan_raw, px_spatial, k_used, corners_pc) or (None,...) if no
    pre-compression let the board detect. corners_pc are in the k_used image, for
    drawing an inspection overlay.
    """
    H, W = gray.shape
    hits = []
    for k in ks:
        g = cv2.resize(gray, (int(round(W * k)), H), interpolation=cv2.INTER_AREA)
        corners, method = detect_checkerboard(g, inner_size)
        if corners is None:
            continue
        px_scan_pc, px_spatial = square_spacing(corners, inner_size)
        hits.append((px_spatial * k / px_scan_pc, px_scan_pc / k, px_spatial, k,
                     corners, method))
    if not hits:
        return None, None, None, None, None
    hits.sort(key=lambda h: h[0])
    s, px_scan_raw, px_spatial, k, corners, method = hits[len(hits) // 2]
    spread = hits[-1][0] - hits[0][0]
    print(f"  scan-scale from {len(hits)} pre-compressed detections ({method}): "
          f"s={s:.5f}  (per-k spread {spread:.5f})")
    return px_scan_raw, px_spatial, k, corners, method


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------
def scan_scale_factor(px_scan, px_spatial, reference="spatial"):
    """Factor to multiply the scan-axis length by so squares become square.

    reference:
        spatial -> trust the optical (640 px) axis, compress/expand scan to match
                   (recommended: scan pitch is the unreliable, stage-speed one)
        scan    -> trust the scan axis instead (rescales the spatial axis)
        min/max -> resample both toward the finer/coarser common pitch
    Returns (fx, fy) multipliers for the (x=scan, y=spatial) axes.
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
    """Resample a 2D or 3D (multi-band) array by per-axis factors (x=cols, y=rows).

    cv2.resize only accepts <=4 channels per call, so a hyperspectral cube
    (224 bands) is resampled band-by-band; 2D and <=4-band arrays go straight
    through.
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


def draw_corners(gray, corners, inner_size):
    """BGR image with the detected checkerboard corners drawn, for inspection."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cols, rows = inner_size
    cv2.drawChessboardCorners(vis, (cols, rows),
                              corners.reshape(-1, 1, 2).astype(np.float32), True)
    return vis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", nargs="?", default="07012026/grain_ref_exp_2500",
                        help="Capture directory to stitch, or an existing *_cube.npy.")
    parser.add_argument("--out", default="07012026/stitched",
                        help="Output folder for PNGs / cube (created if missing).")
    parser.add_argument("--name", default=None,
                        help="Output basename (default: derived from src).")
    parser.add_argument("--checkerboard", nargs=2, type=int, default=(3, 3),
                        metavar=("COLS", "ROWS"),
                        help="Inner-corner count (default 3 3 = a 4x4-square board).")
    parser.add_argument("--reference", choices=["spatial", "scan", "min", "max"],
                        default="spatial",
                        help="Which axis to trust when equalising (default: spatial).")
    parser.add_argument("--band", type=int, default=None,
                        help="Band index for the greyscale (default: mean across bands).")
    parser.add_argument("--manual-scale", type=float, default=None,
                        help="Skip detection; force this scan-axis (x) scale factor. "
                             "Useful on captures where the board can't be detected.")
    parser.add_argument("--save-cube", action="store_true",
                        help="Also rescale and save the full uint16 cube as "
                             "<name>_corrected_cube.npy.")
    parser.add_argument("--dpi", type=int, default=150, help="(reserved) unused for cv2 PNGs.")
    args = parser.parse_args()

    inner_size = tuple(args.checkerboard)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(str(args.src).removesuffix("_cube.npy")).name

    cube = load_cube(args.src)
    gray = render_gray8(cube, args.band)
    H, W = gray.shape
    print(f"Greyscale {W}x{H} (scan x spatial).")

    raw_png = out_dir / f"{name}_raw_intensity.png"
    cv2.imwrite(str(raw_png), gray)
    print(f"wrote {raw_png}")

    # --- determine the scan-axis scale factor -------------------------------
    if args.manual_scale is not None:
        fx, fy = args.manual_scale, 1.0
        print(f"Manual scale: scan axis x{fx:.4f} (detection skipped).")
    else:
        # Detect on scan-precompressed copies: the raw scan is too stretched for
        # the board to detect directly (see measure_scan_scale).
        px_scan, px_spatial, k, corners, method = measure_scan_scale(gray, inner_size)
        if px_scan is None:
            print(f"Checkerboard {inner_size[0]}x{inner_size[1]} NOT detected at "
                  f"any pre-compression.\n"
                  f"  Re-run with --manual-scale FLOAT to force the correction, or\n"
                  f"  --checkerboard COLS ROWS if the board size differs.")
            sys.exit(2)
        fx, fy = scan_scale_factor(px_scan, px_spatial, args.reference)
        # Draw the detected corners on the k-compressed image used for detection.
        gk = cv2.resize(gray, (int(round(gray.shape[1] * k)), gray.shape[0]),
                        interpolation=cv2.INTER_AREA)
        overlay = draw_corners(gk, corners, inner_size)
        overlay_png = out_dir / f"{name}_checkerboard.png"
        cv2.imwrite(str(overlay_png), overlay)
        print(f"Detected via {method} (pre-compress k={k}): raw px/square "
              f"scan={px_scan:.1f} spatial={px_spatial:.1f} "
              f"(raw aspect {px_scan / px_spatial:.1f}x stretched).")
        print(f"wrote {overlay_png}")
        print(f"Correction ({args.reference} reference): scan x{fx:.5f}, spatial x{fy:.5f}.")

    # --- apply the correction ----------------------------------------------
    corrected = resample(gray, fx, fy)
    corr_png = out_dir / f"{name}_corrected.png"
    cv2.imwrite(str(corr_png), corrected)
    print(f"wrote {corr_png} ({corrected.shape[1]}x{corrected.shape[0]})")

    if args.save_cube:
        # cv2.resize handles multi-channel arrays (channels <= 512); 224 is fine.
        corrected_cube = resample(cube, fx, fy)
        cube_path = out_dir / f"{name}_corrected_cube.npy"
        np.save(cube_path, corrected_cube)
        print(f"wrote {cube_path} (shape {corrected_cube.shape}, "
              f"{corrected_cube.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

