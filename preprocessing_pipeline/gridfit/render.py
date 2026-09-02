"""Turn a corrected capture into the 2-D images the grid fit works on, and cache
them so a re-run costs milliseconds instead of half a gigabyte of disk reads.

THE WORKING FRAME
-----------------
A cube is stored (width, n_lines, bands) -- spatial axis first, scan axis
second. Every image in gridfit is instead

    working = rot90_clockwise(plane)        shape (n_lines, width)

so rows run down the scan and columns run across the sensor, the dish reads
upright, and cv2's (x, y) is (column, row) as usual. That is the same
orientation the transmittance preview PNG already uses. (The reflectance preview
writes the un-rotated cube layout instead; the two modes disagree today. gridfit
normalises both to the rotated frame so a dorsal reflectance overlay and a
ventral transmittance overlay of the same dish can be laid side by side.)

`to_cube_mask` converts a working-frame mask back to cube layout, which is the
only thing that has to survive the convention: a cell mask must index the cube.

CACHE
-----
Renders go to <cache>/<mode>/day<D>/dish<N>/<side>/{gray,clip}.png with a
key.json recording the source cube's size and mtime. Any change to the cube
changes the key and the render is recomputed; the cache is only ever a speed-up
and can be deleted at any time. A render is ~0.7 MB against a ~0.6 GB cube, and
reading the cube is what makes a full pass over the collection slow.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# Fraction of kept bands a pixel must clip in before it counts as an open-beam
# well. Empty plate area rails in essentially every band at the sample exposure,
# kernels sit well inside range, so anything from ~0.2 to ~0.8 separates them.
WELL_CLIP_FRAC = 0.5

REFL_STRETCH_PCT = (1.0, 99.0)      # matches reflectance_scripts.intensity_preview
TRANS_STRETCH_PCT = (0.5, 99.5)
TRANS_CLIP_TOL = 0.02               # pixels this clean set the transmittance stretch


# ------------------------------------------------------------------- frames --
def to_working(plane):
    """(width, n_lines) cube-layout plane -> (n_lines, width) working frame."""
    return cv2.rotate(plane, cv2.ROTATE_90_CLOCKWISE)


def to_cube_mask(mask):
    """(n_lines, width) working-frame mask -> (width, n_lines), indexing the cube."""
    return cv2.rotate(mask.astype(np.uint8), cv2.ROTATE_90_COUNTERCLOCKWISE).astype(bool)



# ------------------------------------------------------------------ renders --
def _stretch(v, pop, lo_pct, hi_pct):
    lo, hi = (float(t) for t in np.percentile(v[pop], [lo_pct, hi_pct]))
    if hi <= lo:
        hi = lo + 1e-9
    out = np.nan_to_num((v - lo) / (hi - lo) * 255.0, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _reflectance(cube_path):
    """Band-mean, percentile-stretched. Bright plate, dark wells, dark markers.

    Accumulated band by band off a memory-mapped cube: cube.mean(axis=2) on a
    0.5 GB float32 array materialises the whole thing.
    """
    cube = np.load(cube_path, mmap_mode="r")
    acc = np.zeros(cube.shape[:2], np.float64)
    for b in range(cube.shape[2]):
        acc += cube[:, :, b]
    mean = (acc / cube.shape[2]).astype(np.float32)
    return to_working(_stretch(mean, np.ones(mean.shape, bool), *REFL_STRETCH_PCT)), None


def _transmittance(cube_path, masks_path):
    """Log-stretched mean over kept bands, plus the per-pixel clipped fraction.

    Log, and stretched on the pixels that barely clip, for the same reason the
    pipeline's own preview does it: the railed open beam would otherwise take the
    whole white point and flatten the sample into the bottom few grey levels.

    The clip map is the transmittance well detector. An empty well passes the
    full beam -- far over the sensor's range at the sample exposure -- so it
    rails in nearly every band while a kernel does not, which hands over 22
    cleanly separated well blobs with the kernel silhouette inside each.
    """
    cube = np.load(cube_path, mmap_mode="r")
    z = np.load(masks_path)
    bands = np.nonzero(z["band_kept"])[0]
    sat = z["saturated"]
    acc = np.zeros(cube.shape[:2], np.float64)
    cnt = np.zeros(cube.shape[:2], np.int32)
    clipped = np.zeros(cube.shape[:2], np.int32)
    for b in bands:
        pl = cube[:, :, b]
        ok = np.isfinite(pl)
        acc[ok] += pl[ok]
        cnt[ok] += 1
        clipped += sat[:, :, b]
    plane = np.full(cube.shape[:2], np.nan, np.float32)
    good = cnt > 0
    plane[good] = acc[good] / cnt[good]
    clip = (clipped / max(len(bands), 1)).astype(np.float32)

    finite = np.isfinite(plane)
    if not finite.any():
        raise ValueError(f"{cube_path}: every pixel is invalid")
    v = np.log10(np.clip(plane, 1e-4, None))
    clean = finite & (clip <= TRANS_CLIP_TOL)
    pop = clean if clean.sum() > 1000 else finite
    gray = _stretch(v, pop, *TRANS_STRETCH_PCT)
    gray[~finite] = 0
    return to_working(gray), to_working(clip)


# -------------------------------------------------------------------- cache --
def _key(capture_dir):
    parts = {}
    for name in ("capture.npy", "capture_masks.npz"):
        p = capture_dir / name
        if p.exists():
            st = p.stat()
            parts[name] = [st.st_size, int(st.st_mtime)]
    return parts


def load(capture_dir, mode, cache_dir=None):
    """(gray8, clip|None) for one capture, from cache when the cube is unchanged.

    clip is a float32 fraction-of-bands-clipped map for transmittance and None
    for reflectance, which has no saturation mask and does not need one.
    """
    capture_dir = Path(capture_dir)
    cube_path = capture_dir / "capture.npy"
    if not cube_path.exists():
        raise FileNotFoundError(f"no capture.npy in {capture_dir}")

    slot = None
    if cache_dir is not None:
        slot = Path(cache_dir)
        want = _key(capture_dir)
        kf = slot / "key.json"
        if kf.exists() and (slot / "gray.png").exists():
            try:
                if json.loads(kf.read_text()) == want:
                    gray = cv2.imread(str(slot / "gray.png"), cv2.IMREAD_GRAYSCALE)
                    cp = slot / "clip.png"
                    clip = None
                    if cp.exists():
                        clip = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    if gray is not None:
                        return gray, clip
            except (ValueError, OSError):
                pass    # a corrupt cache entry is just a cache miss

    if mode == "reflectance":
        gray, clip = _reflectance(cube_path)
    else:
        gray, clip = _transmittance(cube_path, capture_dir / "capture_masks.npz")

    if slot is not None:
        slot.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(slot / "gray.png"), gray)
        if clip is not None:
            cv2.imwrite(str(slot / "clip.png"),
                        np.clip(clip * 255.0, 0, 255).astype(np.uint8))
        (slot / "key.json").write_text(json.dumps(_key(capture_dir)))
    return gray, clip
