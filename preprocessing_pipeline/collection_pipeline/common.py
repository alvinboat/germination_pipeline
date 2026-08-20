"""Shared machinery for both capture modes: stitching, raw archiving,
calibration resolution/caching, checkerboard geometry, resampling.

Nothing in here is mode-specific. The two corrections live in
transmittance_scripts/correct.py and reflectance_scripts/correct.py, and the
per-capture driver is process.py.

Three things here are new relative to the dry-run scripts, all for the volume
this collection implies (~100 captures/day for five days):

  * stitch() reads the capture directory on a thread pool and, in the SAME
    pass, writes the archived capture.bin. The dry-run scripts read the raw
    bytes once to stitch and then walked the directory again to move ~3,300
    files; this reads each line exactly once and writes one file. Measured on a
    real 3,276-line capture: 9.1s -> 2.1s, byte-identical cube.

  * calibration frames (dark/white medians) and the checkerboard scan-axis
    factor are cached on disk, keyed on the contents of the capture directory
    that produced them. A dish is one run of process.py, and re-deriving the
    same dark frame from 168 .bin files 100 times a day is pure waste. Any
    change to the calibration directory (new file, changed size, new mtime)
    changes the key, so a stale cache cannot be used silently -- and the
    resolved path plus its cache status is printed on every run.

  * the geometry measurement is ONE implementation, used by both modes. The
    dry-run transmission script measured the board with mean corner-to-corner
    step lengths, which are the column norms of the lattice map and are not
    rotation-invariant; the reflectance script fits the lattice properly and
    takes row norms, which are. On an axis-aligned board they agree, on a
    rotated one they do not, so the correct one is used for both. See
    fit_lattice's docstring for the algebra.
"""

import hashlib
import json
import os
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# hsi_save_load's own import path assumes a wider jarvis_gui package that is
# not in this checkout, so import the local copy directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "loadstich"))
from hsi_save_load import load_hsi  # noqa: E402

WIDTH = 640          # spatial pixels per line
CHANNELS = 224       # spectral bands per line
FULL_SCALE = 4095    # 12-bit sensor; at or above this a pixel is clipped
# One Mono12Packed line: 640 px x 224 bands x 12 bits / 8.
LINE_BYTES = WIDTH * CHANNELS * 3 // 2

ROOT = Path(__file__).resolve().parent

WARNINGS = []


def warn(msg):
    """Record a warning and print it. Collected into the run's meta.json."""
    WARNINGS.append(msg)
    print(f"  WARNING: {msg}")


def reset_warnings():
    WARNINGS.clear()


def runs(indices):
    """[0,1,2,7,8] -> '0-2, 7-8', for readable index-range reporting."""
    indices = list(indices)
    if not indices:
        return "none"
    out, start, prev = [], int(indices[0]), int(indices[0])
    for i in map(int, indices[1:]):
        if i != prev + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def json_safe(obj):
    """numpy scalars/arrays and Paths -> plain Python, so json.dump works."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


# ------------------------------------------------------------------- disk --
def free_bytes(path):
    """Free bytes on the filesystem holding `path` (nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def check_space(n_lines, raw_bytes, targets, min_free_gb):
    """Warn before a run that cannot fit its own outputs.

    The corrected cube is the big one: (640, n_lines, 224) float32 is 4 bytes a
    voxel, i.e. ~1.9 GB for a 3,300-line scan before any scan-axis compression.
    That is an upper bound (both modes shrink the scan axis, and reflectance
    also crops to the tape span), which is the right side to err on here.
    """
    need = raw_bytes + WIDTH * n_lines * CHANNELS * 4
    # The raw archive and the corrected cube may sit on different filesystems
    # (RAW_BINARIES_ROOT can point at another disk), so check each distinct one
    # and report the tightest. Deduped by device, or a single disk would be
    # counted twice and look half as free as it is.
    per_device = {}
    for t in targets:
        per_device.setdefault(_mount_of(t).stat().st_dev, t)
    worst = min(free_bytes(t) for t in per_device.values())
    gb = 1024 ** 3
    print(f"  disk: need up to {need / gb:.1f} GB (raw {raw_bytes / gb:.2f} GB + "
          f"cube <= {(need - raw_bytes) / gb:.2f} GB), {worst / gb:.1f} GB free")
    if worst < need + min_free_gb * gb:
        warn(f"only {worst / gb:.1f} GB free, and this capture may need "
             f"{need / gb:.1f} GB plus the {min_free_gb} GB reserve. Free space or move "
             f"raw_binaries/ to another disk (RAW_BINARIES_ROOT in config.py) before "
             f"this fills up mid-collection.")
    return {"need_bytes": int(need), "free_bytes": int(worst)}


def _mount_of(path):
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    return p


# --------------------------------------------------------------- captures --
def line_files(directory):
    """Sorted .bin line filenames in a capture directory.

    Lexical sort, which is also scan order: eBUS Player names every line with a
    zero-padded index prefix (00000000_..., 00000001_...).
    """
    return sorted(f for f in os.listdir(directory) if f.endswith(".bin"))


def resolve_capture(base, what):
    """A capture from `base`: an .npy, a dir of .bin, or a dir holding one of those.

    Exactly one candidate is required. A second one usually means a stale folder
    from another session, and picking either silently corrects the run against
    the wrong reference.
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
                         f"({[p.name for p in cands]}) -- expected one. Clear the stale "
                         f"one, or point config.py at the {what} explicitly.")
    return cands[0]


def resolve_calibration(cal_root, name, day, what):
    """Resolve one calibration input, preferring a day-specific override.

    Looks for <cal_root>/day<DAY>/<name> first and falls back to
    <cal_root>/<name>. That way a dark or white re-shot on day 3 only has to be
    dropped into calibration/day3/ to take effect for day 3, without disturbing
    the days already processed against the shared set.
    """
    day_specific = Path(cal_root) / f"day{day}" / name
    shared = Path(cal_root) / name
    if day_specific.exists():
        return resolve_capture(day_specific, what), "day-specific"
    if shared.exists():
        return resolve_capture(shared, what), "shared"
    raise SystemExit(f"no {what}: looked for {day_specific} and {shared}.")


def describe(path):
    """'<path> (N .bin lines, newest 2026-08-04 11:22)' -- provenance for the log.

    Printed for every calibration input on every run. A silently stale dark is
    the failure mode that is invisible in the output and ruins a day of data.
    """
    path = Path(path)
    if path.is_file():
        st = path.stat()
        return (f"{path} ({st.st_size / 1e6:.0f} MB, "
                f"{_stamp(st.st_mtime)})")
    files = line_files(path)
    if not files:
        return f"{path} (empty)"
    newest = max((path / f).stat().st_mtime for f in files)
    return f"{path} ({len(files)} .bin lines, newest {_stamp(newest)})"


def _stamp(epoch):
    import datetime
    return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def capture_key(path):
    """Content key for a capture directory or .npy: name, sizes and mtimes.

    Any edit to the calibration input -- a file added, replaced, or re-shot --
    changes this, so a cached frame derived from it can never be reused for
    different data.
    """
    path = Path(path)
    h = hashlib.sha1()
    h.update(str(path.resolve()).encode())
    if path.is_file():
        st = path.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    else:
        for f in line_files(path):
            st = (path / f).stat()
            h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def combine_keys(*parts):
    """One short cache key from several capture keys and settings.

    Every input that changes the cached result has to be in here, or a settings
    change would silently reuse frames derived under the old one.
    """
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def cached_npz(cache_dir, tag, key, compute):
    """Disk-cache the arrays/scalars `compute()` returns, keyed on `key`.

    compute() must return a dict of numpy arrays / json-able scalars. Returns
    (dict, hit) so the caller can say in the log whether the frames were
    measured or reused.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag}_{key}.npz"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            meta_path = path.with_suffix(".json")
            data["_scalars"] = (json.loads(meta_path.read_text())
                                if meta_path.is_file() else {})
            return data, True
        except Exception:
            path.unlink(missing_ok=True)   # corrupt cache entry: re-measure
    data = compute()
    scalars = data.pop("_scalars", None)
    np.savez_compressed(path, **data)
    if scalars is not None:
        path.with_suffix(".json").write_text(json.dumps(json_safe(scalars), indent=2))
        data["_scalars"] = scalars
    # A calibration folder that changes leaves its old entries behind; they are
    # never read again, so drop them rather than growing the cache forever.
    for stale in cache_dir.glob(f"{tag}_*"):
        if not stale.name.startswith(f"{tag}_{key}"):
            stale.unlink(missing_ok=True)
    return data, False


# --------------------------------------------------------------- stitching --
def stitch(directory, workers=4, archive_path=None):
    """Stitch a capture directory into (WIDTH, n_lines, CHANNELS) uint16.

    Reads each line file exactly once. When `archive_path` is given the raw
    bytes are buffered and written there as a single concatenated capture.bin,
    in scan order, so the archive costs no extra read of the ~700 MB capture.

    Threaded because the work is a 215 KB read plus a bit-shuffle per line, and
    numpy releases the GIL for both. Four workers is ~4x sequential on this
    box; more does not help (the unpack is memory-bandwidth bound).

    A line that will not parse becomes a zero frame and is reported, rather
    than aborting a scan that is otherwise fine.
    -> (cube, blank_indices, info)
    """
    directory = Path(directory)
    files = line_files(directory)
    if not files:
        raise SystemExit(f"no .bin files in {directory}")

    sizes = [(directory / f).stat().st_size for f in files]
    odd = [i for i, s in enumerate(sizes) if s != LINE_BYTES]
    if odd:
        warn(f"{len(odd)} line file(s) are not {LINE_BYTES} bytes (indices "
             f"{runs(odd[:40])}{', ...' if len(odd) > 40 else ''}). A short line cannot "
             f"be a full 640x224 frame; those become blank lines below.")

    n = len(files)
    cube = np.empty((WIDTH, n, CHANNELS), dtype=np.uint16)
    # One flat buffer holding the whole capture's raw bytes, so the archive is a
    # single sequential write instead of ~3,300 file copies. Only allocated when
    # every line is the expected length, which is the normal case.
    raw_buf = (np.empty((n, LINE_BYTES), dtype=np.uint8)
               if archive_path is not None and not odd else None)
    blank = []

    def one(i):
        raw = np.fromfile(directory / files[i], dtype=np.uint8)
        if raw_buf is not None:
            raw_buf[i] = raw
        try:
            frame = load_hsi(raw).reshape([CHANNELS, WIDTH]).swapaxes(0, 1)[::-1]
        except Exception:
            frame = np.zeros((WIDTH, CHANNELS), dtype=np.uint16)
        if not frame.any():
            return i
        cube[:, i, :] = frame
        return None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for bad in ex.map(one, range(n), chunksize=16):
            if bad is not None:
                blank.append(bad)
    for i in blank:
        cube[:, i, :] = 0

    info = {"n_lines": n, "raw_bytes": int(sum(sizes)), "n_blank": len(blank),
            "odd_size_lines": len(odd)}
    if archive_path is not None:
        info["archive"] = archive_capture(archive_path, directory, files, sizes, raw_buf)
    return cube, sorted(blank), info


def archive_capture(archive_path, directory, files, sizes, raw_buf):
    """Write the capture's raw bytes as one capture.bin, plus a capture.json index.

    The bytes are the camera's own Mono12Packed lines, concatenated in scan
    order and otherwise untouched -- capture.bin is byte-for-byte the
    concatenation of the per-line files it replaces, so nothing is lost by not
    keeping ~3,300 separate ones. capture.json records the line count, the
    frame geometry and the original filenames (which carry eBUS's per-line
    timestamps, the only thing concatenation would otherwise discard).

    load_capture_bin() reads it back into a cube.
    """
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_path.with_suffix(".bin.part")
    if raw_buf is not None:
        raw_buf.tofile(tmp)
    else:
        # Ragged fallback: some line was not LINE_BYTES, so there is no uniform
        # buffer. Concatenate straight from disk and record the real sizes.
        with open(tmp, "wb") as out:
            for f in files:
                with open(directory / f, "rb") as src:
                    shutil.copyfileobj(src, out, length=1 << 22)
    tmp.replace(archive_path)      # atomic: no half-written capture.bin on a crash

    index = {"n_lines": len(files), "width": WIDTH, "channels": CHANNELS,
             "line_bytes": LINE_BYTES, "pixel_format": "Mono12Packed",
             "layout": "lines concatenated in scan order; see load_capture_bin()",
             "uniform_line_size": bool(all(s == LINE_BYTES for s in sizes)),
             "total_bytes": int(sum(sizes)), "source_files": files}
    if not index["uniform_line_size"]:
        index["line_sizes"] = [int(s) for s in sizes]
    json_path = archive_path.with_name(archive_path.stem + ".json")
    json_path.write_text(json.dumps(index, indent=2))
    print(f"  raw archive: {archive_path} ({sum(sizes) / 1e6:.0f} MB, {len(files)} lines) "
          f"+ {json_path.name}")
    return {"path": str(archive_path), "bytes": int(sum(sizes)), "n_lines": len(files)}


def load_capture_bin(path):
    """Read an archived capture.bin back into a (WIDTH, n_lines, CHANNELS) cube.

    The inverse of archive_capture(), so an archived capture can be reprocessed
    without the original per-line files. Uniform-line captures only; a ragged
    one records its per-line sizes in capture.json and is a rarity worth
    handling by hand.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % LINE_BYTES:
        raise SystemExit(f"{path} is {raw.size} bytes, not a whole number of "
                         f"{LINE_BYTES}-byte lines -- see {path.stem}.json.")
    n = raw.size // LINE_BYTES
    raw = raw.reshape(n, LINE_BYTES)
    cube = np.empty((WIDTH, n, CHANNELS), dtype=np.uint16)
    for i in range(n):
        cube[:, i, :] = load_hsi(raw[i]).reshape([CHANNELS, WIDTH]).swapaxes(0, 1)[::-1]
    return cube


def load_cube(path, what, workers=4):
    """Stitch a capture dir, or load an already-stitched .npy. -> (cube, blank)."""
    path = Path(path)
    print(f"  loading {what}: {path.name}")
    if path.is_file() and path.suffix == ".npy":
        cube, blank = np.load(path), []
    else:
        cube, blank, _ = stitch(path, workers=workers)
    if cube.shape[0] != WIDTH or cube.shape[2] != CHANNELS:
        raise SystemExit(f"{path} is {cube.shape}, expected ({WIDTH}, n_lines, {CHANNELS}).")
    if blank:
        warn(f"{len(blank)} blank/unparseable line(s) in the {what}.")
    return cube, blank


def reference_frame(cube, name, blank=(), how="median"):
    """Collapse a reference capture to one (WIDTH, CHANNELS) frame.

    Median by default, not mean: a reference capture is a few dozen to a few
    hundred lines and one bad line moves a mean by percent-level. Both
    transmission darks in the dry-run rig have line 0 sitting ~11% below the
    rest, which a median ignores and a mean bakes into every output pixel.

    The reflectance pipeline was built and validated on a MEAN dark frame, so
    it passes how="mean" to stay numerically identical to the dry run.
    """
    keep = np.setdiff1d(np.arange(cube.shape[1]), np.asarray(blank, dtype=int))
    if keep.size == 0:
        raise SystemExit(f"every line of the {name} reference is blank.")
    usable = cube[:, keep, :]
    frame = (np.median(usable, axis=1) if how == "median"
             else usable.mean(axis=1)).astype(np.float64)
    per_line = usable.mean(axis=(0, 2))
    med = float(np.median(per_line))
    spread = float(per_line.max() - per_line.min()) / med if med else float("inf")
    stats = {"n_lines": int(cube.shape[1]), "n_used": int(keep.size), "collapse": how,
             "per_line_mean_median": med, "per_line_spread_frac": spread,
             "min": float(frame.min()), "median": float(np.median(frame)),
             "max": float(frame.max())}
    print(f"  {name}: {keep.size} lines ({how}) -> frame {frame.min():.0f}.."
          f"{frame.max():.0f}, median {np.median(frame):.0f}")
    return frame, stats


# --------------------------------------------------------------- geometry --
# Corner detection is the CLASSIC cv2.findChessboardCorners plus cornerSubPix,
# deliberately NOT findChessboardCornersSB. Benchmarked against synthetically
# rendered boards of known cell size, SB's measured axis ratio is off by -7% at
# 2.1:1 anisotropy and -15% at 3.2:1, and its detection rate collapses above
# ~6:1; the classic detector plus sub-pixel refinement stays within 0.16% from
# 1:1 all the way to 14:1. The scan axis here is routinely oversampled 2-4x,
# squarely where SB fails.
#
# For the same reason there is no scan-axis "pre-compression" search. That
# existed to make the board look square enough to detect; with the classic
# detector it is unnecessary, and resampling before measuring can only discard
# scan-axis information. Everything is detected and measured at native resolution.
CHECKERBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

# Acceptance thresholds, set from measured behaviour on the dry-run captures:
# real boards fit the lattice model at rms 0.17-0.51px with 1.3-2.1deg of skew,
# while spurious "boards" the detector finds in the kernel well grid sit at rms
# 4-11px. The gap is nearly an order of magnitude.
FIT_RMS_WARN_PX = 0.6
FIT_RMS_REJECT_PX = 1.5
SKEW_WARN_DEG = 3.0
SKEW_REJECT_DEG = 10.0
BOARD_DISAGREE_WARN = 0.02      # fractional spread across boards worth flagging
FX_SANITY_RANGE = (0.02, 50.0)  # a correction outside this is a bug, not a measurement
TILE_WINDOWS = (160, 320, 640)  # fallback sweep: window sizes, each stepped at 50% overlap


def render_detection_gray(cube, band=None, dark=None, lo_pct=0.5, hi_pct=99.5):
    """(spatial, scan) uint8 render of a cube, for corner detection.

    Mean across bands unless a single band is requested -- averaging 224 bands
    is the cheapest available SNR gain and corner localisation is noise-limited.
    Transmission passes a single band (the lamp's peak) and its dark frame,
    because most of its frame is saturated open beam and a band-mean render
    washes the board out.

    Contrast is stretched on percentiles rather than min/max: a single saturated
    speck or dead pixel sets the min/max range and crushes the scene into a
    handful of grey levels, which is exactly what the detector's adaptive
    threshold cannot work with.
    """
    if band is None:
        img = np.asarray(cube.mean(axis=2), dtype=np.float32)
    else:
        img = np.asarray(cube[:, :, band], dtype=np.float32)
        if dark is not None:
            img = img - np.asarray(dark[:, band], dtype=np.float32)[:, None]
    lo, hi = (float(v) for v in np.percentile(img, [lo_pct, hi_pct]))
    if not hi > lo:
        return np.zeros(img.shape, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _ideal_lattice(inner_size):
    """(N,2) (column, row) index of each inner corner, in cv2's return order.

    findChessboardCorners returns row-major for a (cols, rows) pattern, so the
    row index is the slow axis.
    """
    cols, rows = inner_size
    j, i = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.column_stack([i.ravel(), j.ravel()]).astype(np.float64)


def fit_lattice(corners, inner_size, cell_size=(1.0, 1.0)):
    """Measure the per-image-axis scale from one board's corners.

    The board is a rigid grid in the object plane, so the map from lattice index
    to image pixel is

        A = diag(s_scan, s_spatial) @ R(theta) @ diag(cell_w, cell_h)

    i.e. the physical cell, an unknown in-plane rotation, and the axis-aligned
    image sampling we are actually after. Fit A by least squares, divide out the
    cell to get B, and then

        B @ B.T = diag(s_scan**2, s_spatial**2)

    so the two scales are the ROW norms of B. This is the whole point: the row
    norms are rotation-invariant, whereas the corner-to-corner step lengths are
    the COLUMN norms of B and are not. The two agree only when the board happens
    to be axis-aligned in the image, so measuring steps silently biases the
    result toward isotropy on any rotated board -- at 2.2:1 anisotropy a 15deg
    rotation is already a ~10% error.

    B @ B.T being diagonal is also the model's own consistency check. The
    off-diagonal term is zero for ANY rotated rigid grid under axis-aligned
    scaling, so a non-zero "skew" means the premise is broken: tilt,
    perspective, or a target whose cells are parallelograms not rectangles.

    cell_size is the physical cell extent along the board's own two lattice axes
    (any unit). At the default (1, 1) the scales come out in pixels-per-cell and
    only their ratio is meaningful, which is all the correction needs.
    """
    corners = np.asarray(corners, dtype=np.float64)
    lattice = _ideal_lattice(inner_size)

    def _fit(lat):
        design = np.column_stack([lat, np.ones(len(lat))])
        coef, *_ = np.linalg.lstsq(design, corners, rcond=None)
        return coef[:2].T, np.linalg.norm(design @ coef - corners, axis=1)

    A, residual = _fit(lattice)
    # Canonicalise the labelling. The detector may start from any corner, so a
    # square board comes back in any of four rotations. That is harmless for the
    # row norms, which are rotation-invariant, but cell_size is indexed BY
    # LATTICE AXIS, so without a fixed convention a non-square cell would be
    # applied to different physical directions on different boards. Convention:
    # lattice axis 0 is whichever board direction runs nearest the image x
    # (scan) axis, so cell_size is always (along-scan, along-spatial).
    if abs(A[0, 1]) > abs(A[0, 0]):
        lattice = lattice[:, ::-1].copy()
        A, residual = _fit(lattice)
    ambiguous = abs(abs(A[0, 1]) - abs(A[0, 0])) < 0.1 * abs(A[0, 0])
    if A[0, 0] < 0:
        # a 180deg relabelling: both lattice axes reversed, so each still runs
        # along the same physical direction and cell_size is unaffected. Undone
        # only so the reported rotation reads as ~0 rather than ~180.
        lattice = -lattice
        A, residual = _fit(lattice)

    cell = np.asarray(cell_size, dtype=np.float64)
    if np.any(cell <= 0):
        raise ValueError(f"cell_size must be positive, got {cell_size!r}")
    B = A / cell[None, :]
    G = B @ B.T
    px_scan, px_spatial = float(np.sqrt(G[0, 0])), float(np.sqrt(G[1, 1]))
    denom = px_scan * px_spatial
    skew = (float(np.degrees(np.arcsin(np.clip(G[0, 1] / denom, -1.0, 1.0))))
            if denom > 0 else float("inf"))

    step = A @ A.T   # pixels per lattice step, before the cell is divided out
    return {"px_scan": px_scan, "px_spatial": px_spatial,
            "spacing_px": (float(np.sqrt(step[0, 0])), float(np.sqrt(step[1, 1]))),
            "theta_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))),
            "skew_deg": skew, "axis_ambiguous": bool(ambiguous),
            "rms_px": float(np.sqrt((residual ** 2).mean())),
            "max_resid_px": float(residual.max()),
            "centre": corners.mean(axis=0), "corners": corners}


def _subpix_window(spacing_px):
    """cornerSubPix half-window, sized per axis from the real corner spacing.

    The window must stay clear of the neighbouring corners or it integrates
    their gradients too. The two axes are sampled very differently here (the
    scan axis is oversampled several-fold), so a single fixed value is
    necessarily either wasteful on one axis or oversized on the other -- at
    ~15px spatial spacing the stock (11, 11) window is wider than the gap
    between corners.
    """
    return tuple(int(max(2, min(11, np.floor(s / 2.0) - 1))) for s in spacing_px)


def _detect_raw(gray, inner_size):
    """One board's inner corners via the classic detector, or None.

    Wrapped because OpenCV raises rather than returning False on degenerate
    input -- adaptiveThreshold rejects any frame only a few pixels across.
    """
    try:
        ok, corners = cv2.findChessboardCorners(gray, tuple(inner_size), CHECKERBOARD_FLAGS)
    except cv2.error:
        return None
    if not ok or corners is None or len(corners) != inner_size[0] * inner_size[1]:
        return None
    return corners.reshape(-1, 2).astype(np.float64)


def _measure(gray, corners_raw, inner_size, cell_size):
    """Refine raw corners sub-pixel and measure them. None if unusable.

    Refinement is checked, not trusted: cornerSubPix can walk a corner onto a
    neighbouring feature and reports no error when it does. A corner that moves
    further than its own search window has not been refined, it has been
    relocated, so the unrefined detection is kept instead.
    """
    prelim = fit_lattice(corners_raw, inner_size, cell_size)
    if not np.all(np.isfinite(prelim["spacing_px"])) or min(prelim["spacing_px"]) < 2.0:
        return None                      # collapsed grid: all corners on top of each other

    win = _subpix_window(prelim["spacing_px"])
    refined = cv2.cornerSubPix(gray, corners_raw.astype(np.float32).reshape(-1, 1, 2),
                               win, (-1, -1), SUBPIX_CRITERIA).reshape(-1, 2).astype(np.float64)
    moved = np.linalg.norm(refined - corners_raw, axis=1)
    if np.any(moved > np.hypot(*win)):
        board = prelim
        board["refined"] = False
    else:
        board = fit_lattice(refined, inner_size, cell_size)
        board["refined"] = True
    board["subpix_win"] = win
    return board


def _grow_hull(corners, margin):
    """Convex hull of the corners pushed `margin` px outward from its centroid."""
    hull = cv2.convexHull(np.asarray(corners, dtype=np.float32)).reshape(-1, 2)
    centre = hull.mean(axis=0)
    radial = hull - centre
    norm = np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    return (hull + radial / norm * margin).astype(np.int32)


def _mask_board(gray, board):
    """Erase a measured board so the next pass finds a different one.

    Fills the corner convex hull grown by one cell -- the detected corners are
    the INNER ones, so the board's outer squares sit about a cell beyond them.
    A convex hull rather than a bounding box, because a box around a rotated
    board over-erases and can swallow a close neighbour. Filled with the frame
    median rather than 0, so the patch does not become a high-contrast rectangle
    with detectable corners of its own.
    """
    out = gray.copy()
    cv2.fillConvexPoly(out, _grow_hull(board["corners"], max(board["spacing_px"])),
                       int(np.median(gray)))
    return out


def _judge(board):
    """(accepted, reason) -- is this a real board, well enough measured to use?"""
    if board["rms_px"] > FIT_RMS_REJECT_PX:
        return False, f"lattice residual {board['rms_px']:.2f}px > {FIT_RMS_REJECT_PX}px"
    if abs(board["skew_deg"]) > SKEW_REJECT_DEG:
        return False, f"skew {board['skew_deg']:+.1f}deg > {SKEW_REJECT_DEG}deg"
    if min(board["px_scan"], board["px_spatial"]) <= 0:
        return False, "non-positive axis scale"
    return True, ""


def detect_boards(gray, inner_size, max_boards, cell_size=(1.0, 1.0), use_tiles=True):
    """Every checkerboard in the frame, each measured independently.

    Two passes, both at native resolution:

    1. Whole frame: detect -> measure -> judge -> erase -> repeat. One call only
       ever returns one board, so erasing is what makes the rest findable. This
       alone finds both reflectance targets on every dry-run capture.
    2. If pass 1 came up short, a deterministic tiled sweep at several window
       sizes with 50% overlap. Whole-frame detection is context-sensitive -- the
       adaptive threshold has to cope with the dish, the wells and the tape
       strips at once -- and a board missed globally is usually found at once
       when it is the dominant structure in its own window. The sweep is bounded
       and exhaustive rather than an adaptive search whose termination depends
       on a heuristic.

    inner_size must be the target's TRUE inner-corner count. Ask for a smaller
    pattern and the detector happily returns SUB-WINDOWS of the real board --
    each fits the lattice model perfectly, because a sub-window of a rigid grid
    is a rigid grid, so nothing is rejected and nothing looks wrong.

    Rejected candidates are reported, not silently dropped: a detection failing
    the lattice check is how a spurious grid announces itself.
    """
    boards, rejected = [], []

    def _seen(board, others):
        """Same physical board as one already recorded? Keyed on centroid, since
        a board re-found from an overlapping tile lands within a pixel or two."""
        return any(np.linalg.norm(np.asarray(o["centre"]) - board["centre"])
                   < max(board["spacing_px"]) for o in others)

    def register(board, source):
        if _seen(board, boards):
            return False
        ok, reason = _judge(board)
        if not ok:
            # dedupe rejects too: the tiled sweep revisits every location several
            # times over, and one bad candidate must not fill the log with copies
            if not _seen(board, (r[0] for r in rejected)):
                board["source"] = source
                rejected.append((board, source, reason))
            return False
        board["source"] = source
        boards.append(board)
        return True

    work = gray.copy()
    for _ in range(max_boards + 2):   # +2: room to erase rejects and keep looking
        raw = _detect_raw(work, inner_size)
        if raw is None:
            break
        board = _measure(gray, raw, inner_size, cell_size)
        if board is None:
            # a collapsed grid still has to be erased, or the next pass finds it
            # again and the loop stalls on it instead of reaching the real boards
            work = _mask_board(work, {"corners": raw, "spacing_px": (2.0, 2.0)})
            continue
        register(board, "whole-frame")
        work = _mask_board(work, board)
        if len(boards) >= max_boards:
            break

    if use_tiles and len(boards) < max_boards:
        h, w = gray.shape
        for win in TILE_WINDOWS:
            step = max(1, win // 2)
            for y0 in range(0, max(1, h - step), step):
                for x0 in range(0, max(1, w - step), step):
                    tile = gray[y0:min(h, y0 + win), x0:min(w, x0 + win)]
                    if min(tile.shape) < 40:
                        continue
                    raw = _detect_raw(tile, inner_size)
                    if raw is None:
                        continue
                    board = _measure(gray, raw + [x0, y0], inner_size, cell_size)
                    if board is not None:
                        register(board, f"tile{win}")
                if len(boards) >= max_boards:
                    break
            if len(boards) >= max_boards:
                break

    boards.sort(key=lambda b: (b["centre"][1], b["centre"][0]))   # top-to-bottom
    return boards, rejected


def reconcile_boards(boards, expected):
    """Pool per-board measurements into one (px_scan, px_spatial).

    Median per axis, which for two boards is their mean and which ignores a
    single outlier once there are three or more.

    The two axes are reported separately on purpose, because their spreads mean
    different things. px_scan is frame rate over stage speed: it is the same for
    everything in the capture regardless of where or how high it sits, so boards
    disagreeing on it points at the stage speed drifting mid-scan. px_spatial is
    the optical across-track scale and goes as 1/object-distance, so boards
    disagreeing on that are not coplanar -- and that spread is a lower bound on
    how wrong the correction is for anything off the plane it was measured on.
    """
    if not boards:
        return None, None
    px_scan = statistics.median(b["px_scan"] for b in boards)
    px_spatial = statistics.median(b["px_spatial"] for b in boards)

    if len(boards) < expected:
        warn(f"expected {expected} checkerboard(s), accepted {len(boards)}; reconciling "
             f"across what was found.")
    for i, b in enumerate(boards, 1):
        flags = []
        if b["rms_px"] > FIT_RMS_WARN_PX:
            flags.append(f"HIGH RESIDUAL {b['rms_px']:.2f}px")
        if abs(b["skew_deg"]) > SKEW_WARN_DEG:
            flags.append(f"HIGH SKEW {b['skew_deg']:+.1f}deg")
        if not b["refined"]:
            flags.append("SUBPIX REJECTED")
        if b["axis_ambiguous"]:
            flags.append("BOARD NEAR 45deg -- cell-size axis assignment is ambiguous")
        print(f"    board {i}/{len(boards)} at ({b['centre'][0]:7.1f},{b['centre'][1]:6.1f}) "
              f"[{b['source']}, win{b['subpix_win']}]: scan {b['px_scan']:.3f} "
              f"spatial {b['px_spatial']:.3f} ratio {b['px_scan'] / b['px_spatial']:.4f} | "
              f"rot {b['theta_deg']:+.2f}deg skew {b['skew_deg']:+.2f}deg "
              f"rms {b['rms_px']:.3f}px max {b['max_resid_px']:.3f}px"
              + (("  << " + "; ".join(flags)) if flags else ""))

    if len(boards) > 1:
        def spread(key):
            v = [b[key] for b in boards]
            return (max(v) - min(v)) / statistics.median(v)
        s_scan, s_spatial = spread("px_scan"), spread("px_spatial")
        print(f"    reconciled {len(boards)} boards by median: px_scan spread "
              f"{100 * s_scan:.1f}%, px_spatial spread {100 * s_spatial:.1f}%")
        if s_spatial > BOARD_DISAGREE_WARN:
            print(f"      note: px_spatial is the across-track optical scale and goes as "
                  f"1/object-distance, so a {100 * s_spatial:.1f}% spread means the targets "
                  f"are not coplanar. The correction is only exact on the plane it is "
                  f"measured on; anything at another height keeps a residual stretch of at "
                  f"least that order.")
        if s_scan > BOARD_DISAGREE_WARN:
            print(f"      note: px_scan is frame rate / stage speed and is the same "
                  f"everywhere in the capture, so a {100 * s_scan:.1f}% spread points at "
                  f"the stage speed drifting during the scan rather than at target geometry.")
    return px_scan, px_spatial


def scan_scale_factor(px_scan, px_spatial, reference="spatial"):
    """(fx, fy) multipliers for the (scan, spatial) axes that equalise the scales.

    spatial (default) rescales the scan axis onto the 640px optical axis; scan
    does the reverse. Only these two: resampling BOTH axes onto some common
    pitch degrades whichever axis was already well sampled and cannot add
    information to the other.
    """
    if reference == "spatial":
        return px_spatial / px_scan, 1.0
    if reference == "scan":
        return 1.0, px_scan / px_spatial
    raise ValueError(f"unknown reference {reference!r}")


def _resample_axis(img, factor, axis):
    """Resample one axis (0 = scan/columns, 1 = spatial/rows). -> (out, achieved).

    INTER_AREA when shrinking, which integrates over the source footprint and so
    is the only choice that does not alias an oversampled axis. INTER_LINEAR when
    growing: INTER_CUBIC is sharper but overshoots at high-contrast edges, and
    overshoot on a reflectance cube means physically impossible negative values.

    cv2.resize handles at most 4 channels, so a 224-band cube goes band by band
    into a preallocated buffer rather than through full-cube temporaries.
    """
    h, w = img.shape[:2]
    if axis == 0:
        new = max(1, int(round(w * factor)))
        if new == w:
            return img, 1.0
        size, achieved = (new, h), new / w
    else:
        new = max(1, int(round(h * factor)))
        if new == h:
            return img, 1.0
        size, achieved = (w, new), new / h
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR

    if img.ndim == 3 and img.shape[2] > 4:
        out = np.empty((size[1], size[0], img.shape[2]), dtype=img.dtype)
        for b in range(img.shape[2]):
            out[:, :, b] = cv2.resize(img[:, :, b], size, interpolation=interp)
        return out, achieved
    return cv2.resize(img, size, interpolation=interp), achieved


def resample(img, fx, fy):
    """Resample a 2D or multi-band 3D array by per-axis factors. -> (out, (ax, ay)).

    One axis at a time, because the correct interpolation depends on whether that
    axis is shrinking or growing and the two axes need not agree. The achieved
    factors are returned separately: the output size is an integer number of
    pixels, so what actually gets applied is round(n*f)/n, not f.

    Only the scan axis is touched when fy == 1.0, which is what keeps a
    whole-(row, band) NaN plane in the transmittance cube NaN instead of
    bleeding it into valid spatial neighbours.
    """
    out, ax = _resample_axis(img, fx, 0)
    out, ay = _resample_axis(out, fy, 1)
    return out, (ax, ay)


def resample_mask(mask, fx, fy):
    """Resample a bool cube with any-source-pixel-set semantics.

    A saturation mask must not be interpolated: a pixel whose value blends a
    clipped and an unclipped source pixel is itself untrustworthy, so any
    overlap marks the output. Done band by band and thresholded above zero,
    which is exactly that rule under INTER_AREA's box average -- band by band
    rather than on the whole cube because a float32 copy of a bool cube this
    size is nearly a gigabyte.
    """
    def one(b):
        r, _ = _resample_axis(mask[:, :, b].astype(np.float32), fx, 0)
        r, _ = _resample_axis(r, fy, 1)
        return r > 0.0

    first = one(0)
    out = np.empty(first.shape + (mask.shape[2],), dtype=bool)
    out[:, :, 0] = first
    for b in range(1, mask.shape[2]):
        out[:, :, b] = one(b)
    return out


def draw_qc_overlay(gray, boards, rejected, path):
    """Write a QC image: what was accepted, what was rejected, and where.

    The failure this guards against is a confident number measured off the wrong
    structure -- the kernel well grid is regular enough that detectors do latch
    onto it. That is invisible in a log line and obvious in a picture.
    """
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for board, _src, _why in rejected:
        cv2.polylines(vis, [_grow_hull(board["corners"], 2)], True, (0, 0, 255), 2)
    for i, b in enumerate(boards, 1):
        cv2.polylines(vis, [_grow_hull(b["corners"], max(b["spacing_px"]))], True,
                      (0, 200, 0), 2)
        for (x, y) in b["corners"]:
            cv2.drawMarker(vis, (int(round(x)), int(round(y))), (0, 255, 255),
                           cv2.MARKER_CROSS, 9, 1)
        x, y = b["centre"]
        cv2.putText(vis, f"#{i} {b['px_scan']:.2f}/{b['px_spatial']:.2f}",
                    (int(x) - 60, max(12, int(y) - 18)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 200, 0), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)
    print(f"    checkerboard QC: {path}")


def verify_correction(gray, fx, inner_size, max_boards, cell_size):
    """Re-measure the boards after applying fx, and report what is left.

    Closed-loop check on the detection render rather than the corrected cube, so
    it costs one resample of a single 2D image. A correctly measured board should
    come back isotropic; anything else is either a bad measurement or a target
    that is not on the plane the correction was wanted for.
    """
    corrected, _ = _resample_axis(gray, fx, 0)
    # no tiled fallback here: this is a confirmation, not a measurement, and it
    # must not cost more than the measurement it is checking
    boards, _ = detect_boards(corrected, inner_size, max_boards, cell_size, use_tiles=False)
    if not boards:
        print("    verify: no board re-detected after correction (cannot confirm).")
        return None
    ratios = [b["px_scan"] / b["px_spatial"] for b in boards]
    for i, r in enumerate(ratios, 1):
        print(f"    verify: board {i} residual anisotropy {r:.4f} ({100 * (r - 1):+.1f}%)")
    return ratios


def measure_geometry(gray, inner_size, n_boards, cell_size, plane_factor, reference,
                     qc_path=None, verify=True):
    """Full geometry measurement on a detection render. -> (fx, fy, meta).

    Wraps detect -> judge -> reconcile -> factor -> plane factor -> sanity, which
    both modes do identically once they have produced their render. Returns
    (1.0, 1.0) with a warning when nothing usable is found, so a failed
    measurement writes an uncorrected cube rather than a wrongly corrected one.
    """
    boards, rejected = detect_boards(gray, inner_size, n_boards, cell_size)
    for board, source, why in rejected:
        print(f"    rejected candidate at ({board['centre'][0]:7.1f},"
              f"{board['centre'][1]:6.1f}) [{source}]: {why}")
    if qc_path is not None:
        draw_qc_overlay(gray, boards, rejected, qc_path)

    meta = {"n_boards_found": len(boards), "n_rejected": len(rejected),
            "boards": [{k: b[k] for k in ("px_scan", "px_spatial", "theta_deg", "skew_deg",
                                          "rms_px", "max_resid_px", "refined", "source")}
                       for b in boards]}
    px_scan, px_spatial = reconcile_boards(boards, n_boards)
    if px_scan is None:
        warn("no checkerboard accepted; skipping the scan-axis geometry correction "
             "(the cube is written with its native scan-axis stretch).")
        meta["applied"] = False
        return 1.0, 1.0, meta

    fx, fy = scan_scale_factor(px_scan, px_spatial, reference)
    print(f"    measured scan={px_scan:.3f} spatial={px_spatial:.3f} px per cell; "
          f"correction ({reference} ref): scan x{fx:.5f}, spatial x{fy:.5f}")
    if verify:
        meta["verify_anisotropy"] = verify_correction(gray, fx, inner_size, n_boards,
                                                      cell_size)
    # Printed unconditionally, including at 1.0. A non-unity default that only
    # announced itself when overridden is exactly the kind of silent correction
    # that is impossible to track down later.
    if plane_factor == 1.0:
        print(f"    plane factor 1.0: raw board measurement kept, no sample-plane "
              f"correction. scan x{fx:.5f}")
    else:
        fx *= plane_factor
        print(f"    plane factor {plane_factor:g} applied for the target/sample plane "
              f"offset: scan x{fx:.5f}")
    lo, hi = FX_SANITY_RANGE
    if not lo <= fx <= hi:
        warn(f"scan scale x{fx:.5f} is outside the plausible range [{lo}, {hi}]; "
             f"treating it as a bad measurement and skipping the correction.")
        meta["applied"] = False
        return 1.0, 1.0, meta

    meta.update({"applied": True, "source": "checkerboard", "px_scan": px_scan,
                 "px_spatial": px_spatial, "plane_factor": plane_factor,
                 "fx": fx, "fy": fy, "reference": reference})
    return fx, fy, meta
