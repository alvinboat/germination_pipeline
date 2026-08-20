"""
Renders a cube as the usual greyscale intensity image with every pixel that has
ANY saturated band tinted light red, so clipped regions can be spotted at a
glance before a region's spectra get trusted.

Everything comes out of the one .npy: the mask is `value >= --level` (default
4095, the 12-bit sensor's clipping point) tested in every band, the picture is
the band-mean of the same cube, and the two are composited into a single PNG at
native resolution.

Saturation is a raw-sensor property, so a raw/stitched counts cube
(storage_image/<name>_cube.npy, or anything straight off the stitcher) is the
input that answers the question honestly -- it still holds the clipped 4095s. A
corrected cube has been dark-subtracted and divided by white, which moves those
values somewhere else entirely; the script will still run on one, but it says so
and reports what it actually thresholded rather than pretending the number means
clipping. Pass --level to threshold corrected units directly (e.g. --level 1.0
on transmittance).

Usage:
    python3 saturation_overlay.py ../storage_image/day1_dish0_cube.npy
    python3 saturation_overlay.py ../storage_image/day1_dish0_cube.npy --flat --save-mask
    python3 saturation_overlay.py <corrected.npy> --level 1.0
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

FULL_SCALE = 4095       # 12-bit sensor: a pixel at or above this is clipped
TINT_RGB = (255, 120, 120)  # light red
BLOCK = 128             # scan lines per pass, so a 450 MB cube never doubles in RAM


def intensity_preview(cube, lo_pct=1.0, hi_pct=99.0):
    """Greyscale (width, n_lines) uint8 preview: mean across all bands, percentile-clip normalised.

    Copied verbatim from transmission_image_pipeline_diffusion_test/process_image.py
    to keep this script standalone -- it is the same normalisation that pipeline
    writes corrected_image/*.png with, so this overlay lines up pixel-for-pixel
    with those previews. If the pipeline's version ever changes, this one has to
    be re-copied by hand.

    A handful of bad-white-reference pixels produce reflectance values orders of
    magnitude above the rest of the scene -- a true min-max stretch spends nearly
    the whole 0-255 range on those outliers and crushes everything else to
    near-black. Clipping to the 1st/99th percentile instead keeps the stretch
    anchored to the actual scene content.

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


def saturation_mask(cube, level):
    """(width, n_lines) bool 'any band at or above level', per-band counts, cube max.

    Walks the scan-line axis in blocks: the >= test on a whole cube is a bool
    array the size of the cube, which is pointless when all that's wanted is the
    any-band reduction. nan-safe -- a nan compares False, so an unusable voxel is
    never reported as clipped.
    """
    width, n_lines, n_bands = cube.shape
    mask = np.zeros((width, n_lines), dtype=bool)
    per_band = np.zeros(n_bands, dtype=np.int64)
    peak = -np.inf
    for s in range(0, n_lines, BLOCK):
        block = np.asarray(cube[:, s:s + BLOCK, :])
        sat = block >= level
        mask[:, s:s + BLOCK] = sat.any(axis=2)
        per_band += sat.sum(axis=(0, 1))
        if np.isfinite(block).any():
            peak = max(peak, float(np.nanmax(block)))
    return mask, per_band, peak


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cube", help="Cube .npy (width, n_lines, channels) -- raw counts or corrected.")
    parser.add_argument("--level", type=float, default=None,
                        help=f"Value at or above which a band counts as saturated, in the cube's own "
                             f"units (default {FULL_SCALE}, the 12-bit sensor's full scale). Give this "
                             f"in corrected units when running on a corrected cube.")
    parser.add_argument("--min-tint", type=float, default=0.75, metavar="F",
                        help="How light the red stays where the image underneath is black, 0-1 "
                             "(default 0.75). Flagged pixels are the light red scaled between this "
                             "and full brightness by their own intensity, so they still show "
                             "structure; raise it towards 1 for a flatter, louder mask.")
    parser.add_argument("--flat", action="store_true",
                        help="Paint every flagged pixel the same flat light red, discarding the "
                             "structure inside the saturated area.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: <cube>_saturated.png next to this script).")
    parser.add_argument("--save-mask", action="store_true",
                        help="Also write the (width, n_lines) bool mask as <cube>_saturated_mask.npy.")
    args = parser.parse_args()
    if not 0.0 <= args.min_tint <= 1.0:
        raise SystemExit(f"--min-tint must be within 0-1, got {args.min_tint}")

    path = Path(args.cube).resolve()
    cube = np.load(path, mmap_mode="r")
    if cube.ndim != 3:
        raise SystemExit(f"{path.name} has shape {cube.shape}; expected a 3-D (width, n_lines, channels) cube.")
    width, n_lines, n_bands = cube.shape
    print(f"Loaded {path}: shape {cube.shape}, dtype {cube.dtype}")

    level = args.level if args.level is not None else FULL_SCALE
    mask, per_band, peak = saturation_mask(cube, level)
    how = "given on the command line" if args.level is not None else "12-bit full scale"
    print(f"Flagging any band >= {level:g} ({how}). Cube's own maximum: {peak:g}")
    if not np.issubdtype(cube.dtype, np.integer) and args.level is None:
        # warn, don't abort: this is still a legitimate picture of "where does this
        # cube reach 4095", it just isn't a picture of sensor clipping any more
        print(f"  warning: {path.name} is {cube.dtype}, i.e. already corrected -- dark subtraction "
              f"and the white divide have moved the sensor's {FULL_SCALE} clip elsewhere, so this "
              f"threshold no longer means saturation. Run it on the raw stitched counts cube, or "
              f"pass --level in the corrected cube's own units.")

    n_sat = int(mask.sum())
    n_px = mask.size
    print(f"Saturated pixels (any band clipped): {n_sat} of {n_px} ({100 * n_sat / n_px:.2f}%)")
    print(f"Clipped voxels: {int(per_band.sum())} of {n_px * n_bands} "
          f"({100 * per_band.sum() / (n_px * n_bands):.2f}%)")
    if per_band.any():
        worst = np.argsort(per_band)[::-1][:5]
        print("Worst bands (band: % of pixels clipped in that band): "
              + ", ".join(f"{b}: {100 * per_band[b] / n_px:.1f}%" for b in worst))
        clean = int((per_band == 0).sum())
        print(f"Bands with no clipping at all: {clean} of {n_bands}")

    with warnings.catch_warnings():
        # intensity_preview is deliberately nan-aware (a corrected cube carries nan
        # wherever the white reference was unusable); its "Mean of empty slice" on an
        # all-nan pixel is the documented path, not a problem worth a line of stderr.
        warnings.simplefilter("ignore", RuntimeWarning)
        gray = intensity_preview(cube)  # (width, n_lines) uint8
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    tint = np.array(TINT_RGB, dtype=np.float32)
    if args.flat:
        rgb[mask] = tint
    else:
        # Scale the light red by the pixel's own intensity instead of alpha-blending
        # it over the greyscale: on a corrected transmission cube the saturated area
        # is nearly black, and a blend there comes out dark maroon rather than light
        # red. Flooring the scale keeps it unmistakably light red while a bright
        # saturated kernel still reads as a kernel.
        shade = args.min_tint + (1.0 - args.min_tint) * (gray[mask].astype(np.float32) / 255.0)
        rgb[mask] = shade[:, None] * tint
    img = rgb.round().clip(0, 255).astype(np.uint8).transpose(1, 0, 2)  # -> (n_lines, width, 3)

    out = Path(args.out) if args.out else Path(__file__).resolve().parent / f"{path.stem}_saturated.png"
    Image.fromarray(img).save(out)
    print(f"wrote {out}  ({width} x {n_lines} px, 1 image px = 1 cube px)")

    if args.save_mask:
        mask_out = Path(__file__).resolve().parent / f"{path.stem}_saturated_mask.npy"
        np.save(mask_out, mask)
        print(f"wrote {mask_out}")


if __name__ == "__main__":
    main()
