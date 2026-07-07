"""
Reflectance-mode bin -> pseudo image, WITH the checkerboard-detect + scan-axis
scale correction already established in the parent repo's
generate_viable_reflectance.py (ported here, trimmed to just the correction
this visualizer needs -- see that file for the full geometric-correction CLI
with perspective/save-cube options).

Why the correction: a push-broom raw scan is heavily over-sampled along the
scan axis (stage speed vs frame rate), so the stitched image reads visibly
stretched horizontally. A checkerboard target in the scene gives a
known-square reference to measure and undo that stretch. The raw scan is too
stretched for the checkerboard to detect directly, so detection runs on
several scan-precompressed copies and the scale is recovered as invariant
across them (see measure_scan_scale).

Run:
    python3 bin_to_image_ref.py <capture_dir> [out_dir]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from stitching import CHANNELS, WIDTH, pseudo_image, stitch

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "pseudo_image"

CHECKERBOARD_INNER_SIZE = (3, 3)   # (cols, rows) inner corners = a 4x4-square board


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

    for label, img, scale in variants():
        ok, corners = cv2.findChessboardCorners(img, (cols, rows), classic_flags)
        if ok:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(-1, 2) / scale, label

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
    this self-checking (the per-k spread should be tiny). Returns
    (px_scan_raw, px_spatial) or (None, None) if no k let the board detect.
    """
    H, W = gray.shape
    hits = []
    for k in ks:
        g = cv2.resize(gray, (max(1, int(round(W * k))), H), interpolation=cv2.INTER_AREA)
        corners, method = detect_checkerboard(g, inner_size)
        if corners is None:
            continue
        px_scan_pc, px_spatial = square_spacing(corners, inner_size)
        hits.append((px_spatial * k / px_scan_pc, px_scan_pc / k, px_spatial, k, method))
    if not hits:
        return None, None
    hits.sort(key=lambda h: h[0])
    s, px_scan_raw, px_spatial, k, method = hits[len(hits) // 2]
    spread = hits[-1][0] - hits[0][0]
    print(f"  scan-scale from {len(hits)} pre-compressed detections ({method}): "
          f"s={s:.5f} (per-k spread {spread:.5f})")
    return px_scan_raw, px_spatial


def resample(img, fx, fy):
    """Resample a 2D array by per-axis factors (x=cols, y=rows)."""
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(round(w * fx))), max(1, int(round(h * fy)))
    interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def convert_ref(capture_dir, out_dir=OUT_DIR, w=WIDTH, c=CHANNELS,
                inner_size=CHECKERBOARD_INNER_SIZE):
    """Stitch, apply the checkerboard scan-scale correction, and write the pseudo image."""
    cube = stitch(capture_dir, w, c)
    gray = pseudo_image(cube)

    px_scan, px_spatial = measure_scan_scale(gray, inner_size)
    if px_scan is None:
        print("  checkerboard not detected at any pre-compression; "
              "writing UNCORRECTED pseudo image")
        img = gray
    else:
        fx, fy = px_spatial / px_scan, 1.0   # trust the spatial (optical) axis
        img = resample(gray, fx, fy)
        print(f"  scan-axis corrected x{fx:.5f} "
              f"(raw px/square: scan={px_scan:.1f} spatial={px_spatial:.1f})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(capture_dir).name}.png"
    Image.fromarray(img, mode="L").save(out_path)
    print(f"  wrote {out_path} ({img.shape[1]}x{img.shape[0]})")
    return out_path


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 bin_to_image_ref.py <capture_dir> [out_dir]")
    capture_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    print(f"Converting {Path(capture_dir).name} (reflectance) ...")
    convert_ref(capture_dir, out_dir)


if __name__ == "__main__":
    main()
