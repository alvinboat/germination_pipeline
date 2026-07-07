"""
Detect the petri-dish cutout grid on a geometry-corrected REFLECTANCE HSI cube
and place one labelled cell per kernel.

Reimplemented from BarleyGermination/analysis/barley/grid/{step1,step3,step5}.py,
adapted for this dataset:

  * Input is a Specim HSI *cube* (mean-across-bands greyscale here), not an RGB
    jpeg. Later this can move to a pseudo-RGB render.
  * Dish detection uses a local-texture map + RANSAC circle fit, because the
    reference's "largest bright blob" fails here: the metal stage bars flanking
    the dish are just as bright as the dish, so an Otsu blob bleeds into them.
    The dish interior (grid + kernels) is high-texture while the smooth frame and
    bars are not, so the dish is the largest contiguous textured blob and its rim
    is a clean circle to fit.
  * BACKLIT polarity: the kernels are clipped in the grid with a light source
    below, so an empty cell aperture reads BRIGHT (opposite the reference, where a
    cutout interior sees the dark table). The cutout mask is therefore the bright
    apertures inside the dish, not the dark ones.

The rest follows the reference: ArUco markers (DICT_4X4_50, printed inverted) give
a rotation seed; the cutout mask is rotated flat and projected onto x/y to reveal
the column/row pitch; a clean pitch+phase lattice is regenerated so cells clipped
by the rim or hidden under a marker are still placed; every cell is rotated back
and flagged usable / clipped / marker-occupied; finally cells are re-labelled in a
marker-locked frame so the labels survive dish rotation across the 5-day sequence.

Outputs (into 07012026/stitched/ by default):
    <name>_grid_overlay.png  — dish circle + every usable cell drawn and labelled
    <name>_grid_cells.json   — every cell's corners, center, flags, marker-frame label

Run:
    python3 detect_grid.py                         # default corrected cube
    python3 detect_grid.py <corrected_cube.npy>
    python3 detect_grid.py <cube> --debug          # also dump dish/mask stages
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("This script needs OpenCV: pip install opencv-python")

DEFAULT_CUBE = "07012026/stitched/grain_ref_exp_2500_corrected_cube.npy"

ARUCO_DICT = cv2.aruco.DICT_4X4_50          # confirmed dictionary; markers are inverted

# ── tuning constants ─────────────────────────────────────────────────────────
MARKER_ADAPTIVE_BLOCK = 25  # px, local-threshold window (~0.44x the ~57px marker side)
MARKER_ADAPTIVE_C     = 5   # constant subtracted from the local mean
DISH_RADIUS_FRAC   = 1   # analyse just inside the rim so it doesn't leak in
DARK_PCT           = 50     # cutout mask = pixels darker than this in-dish pct
MORPH_KERNEL       = 7      # opening kernel (px) to despeckle the cutout mask
BAND_PEAK_FRAC     = 0.2   # a row/col band must reach this frac of the peak projection
MIN_BAND_FRAC      = 0.05   # ignore bands thinner than this frac of the dish radius
RIM_TOLERANCE_FRAC = 0.5   # how far a corner may graze the rim before "clipped"
# Slack on the dish radius for "is this cell in the dish". 1.05 gave the outer-corner
# lattice positions (e.g. R+3C+3, a rectangular grid's corners being closest to a round
# dish's rim) only ~2px margin in a perfectly still capture -- any realistic day-to-day
# dish placement jitter (~9px dish-center std, see stress_test_grid.py) then dropped
# them from the lattice ENTIRELY instead of just flagging them clipped, which is a real
# "lost kernel position" failure, not the harmless clipped<->usable flicker other cells
# see. 1.12 was picked empirically: 30/30 stress-test trials now keep every baseline
# kernel position, and the real capture still detects the same 28 cells / 22 usable.
CELL_INCLUSION_RADIUS_FRAC = 1.12


# ── loading ──────────────────────────────────────────────────────────────────
def load_gray(src):
    """Return an 8-bit mean-across-bands greyscale from a cube .npy (or dir)."""
    src = Path(src)
    if src.is_file() and src.suffix == ".npy":
        cube = np.load(src, mmap_mode="r")
        g = np.asarray(cube).mean(axis=2).astype(np.float32)
    else:
        sys.exit(f"{src} is not a *_cube.npy")
    lo, hi = np.percentile(g, (1, 99))
    return (np.clip((g - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def render_band_gray8(cube, band):
    """8-bit percentile-normalised render of a single spectral band (cf. load_gray)."""
    g = np.asarray(cube[:, :, band]).astype(np.float32)
    lo, hi = np.percentile(g, (1, 99))
    return (np.clip((g - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


# ── markers ──────────────────────────────────────────────────────────────────
def detect_markers(gray8):
    """{id: (4,2) corners TL,TR,BR,BL} for every ArUco marker; {} if none.

    Markers are 3D-printed inverted, so detectInvertedMarker must be enabled.
    The two markers on this dish don't reliably both surface from one render:
    plain/CLAHE/2x-upscale catch whichever marker has enough LOCAL contrast, but
    a global brightness/contrast shift (illumination drift between scans) can
    knock the weaker one out entirely — global Otsu binarization loses it too,
    since its contrast against the busy dish background is low in the global
    histogram even when its own light/dark modules are locally crisp. A local
    adaptive threshold (window ~ a fraction of the marker's own size) preserves
    that local module contrast regardless of where the marker sits in the global
    brightness range, and empirically recovers both markers in every stress-test
    trial where the plain/CLAHE variants only found one. We UNION detections
    across all these cheap preprocessings, mapping every result back to
    original-image coordinates. First detection of an id wins.
    """
    params = cv2.aruco.DetectorParameters()
    params.detectInvertedMarker = True
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray8)
    adaptive = cv2.adaptiveThreshold(gray8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, MARKER_ADAPTIVE_BLOCK,
                                     MARKER_ADAPTIVE_C)
    variants = [(gray8, 1.0), (clahe, 1.0), (adaptive, 1.0)]
    for base in (gray8, clahe):
        variants.append((cv2.resize(base, None, fx=2, fy=2,
                                    interpolation=cv2.INTER_CUBIC), 2.0))
    found = {}
    for img, scale in variants:
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        for i, c in zip(ids.ravel(), corners):
            found.setdefault(int(i), c.reshape(4, 2) / scale)
    return found


def grid_rotation_deg(markers):
    """Grid tilt (deg) from every marker edge via a length-weighted circular mean.

    Each square marker gives four readings of the same tilt (top/bottom along x,
    left/right 90° off, folded onto x). Returns 0.0 if no markers were found.
    """
    if not markers:
        return 0.0
    sin_sum = cos_sum = 0.0
    for c in markers.values():
        c0, c1, c2, c3 = c                       # TL, TR, BR, BL
        for e in (c1 - c0, c2 - c3,
                  (c3 - c0)[::-1] * [1, -1],
                  (c2 - c1)[::-1] * [1, -1]):
            ang = math.atan2(e[1], e[0])
            length = math.hypot(e[0], e[1])
            sin_sum += length * math.sin(ang)
            cos_sum += length * math.cos(ang)
    return math.degrees(math.atan2(sin_sum, cos_sum))


# ── dish ─────────────────────────────────────────────────────────────────────
def _lsq_circle(P):
    """Least-squares circle through points P (N,2). Returns (cx, cy, r)."""
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P[:, 0] ** 2 + P[:, 1] ** 2), rcond=None)
    cx, cy = sol[0], sol[1]
    return cx, cy, math.sqrt(sol[2] + cx * cx + cy * cy)


def _rim_points(blur, cx, cy, rlo, rhi, n_rays=540, min_drop=8.0):
    """One dish-edge point per ray: the strongest bright->dark drop in [rlo,rhi].

    The white plastic disk is bright; immediately outside it is the dark groove
    against the wood. Going radially outward from a seed centre, the disk edge is
    the sharpest fall in intensity — found per ray and kept only if the drop is
    real (> min_drop). This works at every angle (the groove rings the whole disk)
    regardless of whether the plastic/metal is brighter, which is why it beats a
    Hough edge fit here.
    """
    H, W = blur.shape
    P = []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        dx, dy = math.cos(th), math.sin(th)
        rr = np.arange(rlo, rhi)
        xs = (cx + rr * dx).astype(int)
        ys = (cy + rr * dy).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if ok.sum() < 20:
            continue
        prof = cv2.GaussianBlur(blur[ys[ok], xs[ok]].reshape(-1, 1), (1, 9), 0).ravel()
        d = np.diff(prof)
        i = int(np.argmin(d))
        if -d[i] > min_drop:
            P.append((cx + rr[ok][i] * dx, cy + rr[ok][i] * dy))
    return np.array(P)


def _rim_score(blur, cx, cy, r, n_rays=360, band=8.0, min_drop=8.0):
    """How dish-rim-like the exact circle (cx,cy,r) is: (coverage, mean_drop).

    Casts rays only in a narrow [r-band, r+band] window (not a wide search) and
    checks, per ray, for a real bright->dark drop right at that radius. coverage
    is the fraction of the full 360 degrees that shows a real drop -- the dish's
    groove rings the WHOLE disk, so a true dish edge scores near 1.0, while a
    partial/interrupted ring (e.g. a cutout broken up by ear-tab slots, or an
    off-centre candidate that only grazes the real edge on one side) scores
    much lower. mean_drop is the average strength of those drops, used as a
    tiebreaker.
    """
    H, W = blur.shape
    rlo, rhi = max(0, r - band), r + band
    hits = 0
    drops = []
    for th in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        dx, dy = math.cos(th), math.sin(th)
        rr = np.arange(rlo, rhi)
        xs = (cx + rr * dx).astype(int)
        ys = (cy + rr * dy).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if ok.sum() < 4:
            continue
        prof = blur[ys[ok], xs[ok]]
        d = np.diff(prof)
        if len(d) == 0:
            continue
        drop = float(-d.min())
        if drop > min_drop:
            hits += 1
            drops.append(drop)
    coverage = hits / n_rays
    return coverage, (float(np.mean(drops)) if drops else 0.0)


def find_dish(gray8, iters=4, inlier_px=6.0, n_candidates=8, confident_coverage=0.5):
    """Locate the white petri dish as (cx, cy, r) by fitting its rim edge.

    The scene has near-concentric circular edges (white-dish rim, groove, wood
    outer edge, ear-tab arcs) and, being a reflectance capture, white plastic /
    metal / wood barely differ spectrally, so neither Hough nor a spectral
    threshold isolates the disk reliably on its own. We combine them: Hough
    proposes several candidate circles (cheap, but it can't tell the dish rim
    from a similarly-circular cutout rim), and each candidate is graded by
    _rim_score -- the same bright->dark boundary trace the old Hough-only
    approach lacked, now used to ARBITRATE between candidates instead of just
    seeding one blind guess.

    That arbitration is only trusted when it's actually confident (coverage
    over `confident_coverage`, i.e. a real, mostly-unbroken ring was found) --
    then, and only then, do we restrict the least-squares refinement to a
    narrow window around that ring, which is what buys the extra robustness
    against locking onto a *different* nearby ring. A low-confidence or
    missing Hough hit (e.g. a dim/low-contrast capture where no candidate's
    edge clears the drop threshold) falls back to exactly the original
    wide-window, image-centre-seeded search -- tested and known-stable --
    rather than trusting a shaky seed with a narrow window, which is worse
    than the original on both counts (verified via stress_test_grid.py: an
    earlier version of this that always narrowed the window regressed 21/30
    -> 19/30 baseline-exact trials, with two runs locking onto a wrong,
    much-larger ring).
    """
    H, W = gray8.shape
    blur = cv2.GaussianBlur(gray8, (7, 7), 0).astype(np.float32)

    # Multiple Hough candidates: minDist is a fraction of the expected radius
    # (not the old W, which forced exactly one result) so distinct nearby
    # circles aren't merged away before they can be scored.
    hough = cv2.HoughCircles(cv2.GaussianBlur(gray8, (9, 9), 0), cv2.HOUGH_GRADIENT,
                             dp=1.2, minDist=int(0.15 * H), param1=100, param2=60,
                             minRadius=int(0.435 * H), maxRadius=int(0.527 * H))

    confident_seed = None
    if hough is not None:
        best = None
        for hx, hy, hr in hough[0][:n_candidates]:
            score = _rim_score(blur, float(hx), float(hy), float(hr))
            if best is None or score > best[0]:
                best = (score, float(hx), float(hy), float(hr))
        if best is not None and best[0][0] >= confident_coverage:
            confident_seed = best[1:]

    windows = [(int(0.34 * H), int(0.56 * H))]                    # original, always available
    if confident_seed is not None:
        seed_r = confident_seed[2]
        windows.insert(0, (max(1, int(seed_r - 0.08 * H)), int(seed_r + 0.08 * H)))

    for wi, (rlo, rhi) in enumerate(windows):
        cx, cy, r = confident_seed if confident_seed is not None else (W / 2.0, H / 2.0, 0.46 * H)
        ok = True
        for _ in range(iters):
            P = _rim_points(blur, cx, cy, rlo, rhi)
            if len(P) < 20:
                ok = False
                break
            cx, cy, r = _lsq_circle(P)
            for _ in range(3):                   # robust reweight: drop outliers
                keep = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < inlier_px
                if keep.sum() < 20:
                    break
                P = P[keep]
                cx, cy, r = _lsq_circle(P)
        if ok:
            return int(round(cx)), int(round(cy)), int(round(r))

    print("  dish rim fit failed; using centred fallback")
    return W // 2, H // 2, int(0.46 * H)


def select_discriminative_bands(cube, cx, cy, r, k=5, min_gap=5,
                                dish_ann=(0.90, 0.99), wood_ann=(1.03, 1.25),
                                sat_frac=0.95):
    """Top-k spectral bands where the dish is most distinguishable from wood.

    We have the full spectral cube, not just the mean-across-bands render the
    rim fit normally uses -- and the two materials don't separate equally well
    in every band. Samples a dish annulus (just inside a rough rim estimate)
    and a wood annulus (just outside it, in-frame pixels only), ranks every
    band by Fisher separability ((dish-wood)^2 / (var_dish+var_wood)) rather
    than raw |diff|: on grain_ref_exp_2500, bands ~86-90 have the 2nd-largest
    raw difference but saturate in the dish annulus (clipped at 4095), which
    inflates their apparent contrast without being usable -- Fisher, plus an
    explicit saturation exclusion, correctly ranks the true best band (~158,
    diff 931 vs the mean-image's 601) above them. A minimum index gap between
    picks avoids choosing several literally-adjacent (highly correlated)
    bands, so a later majority vote has some real independence against a
    band-local sensor defect instead of just re-measuring one signal k times.

    Polarity matters: _rim_points only looks for a bright->dark drop going
    outward (white plastic -> dark groove), so a band where the dish reads
    DARKER than the wood is actively wrong for it, not just less useful --
    confirmed on grain_ref_exp_2500, where band 207 has strong Fisher
    separability (dish darker than wood there) and, picked alongside the
    right-polarity bands, lands find_dish on a completely different circle
    (92px away) instead of merely being noisier. Fisher is squared and can't
    see this, so polarity is filtered explicitly before ranking.
    """
    H, W, C = cube.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - cx, yy - cy)
    dish_mask = (dist > dish_ann[0] * r) & (dist < dish_ann[1] * r)
    wood_mask = (dist > wood_ann[0] * r) & (dist < wood_ann[1] * r)
    if dish_mask.sum() < 100 or wood_mask.sum() < 100:
        return []

    dish_px = np.asarray(cube)[dish_mask]
    wood_px = np.asarray(cube)[wood_mask]
    dish_med = np.median(dish_px, axis=0).astype(np.float64)
    wood_med = np.median(wood_px, axis=0).astype(np.float64)
    dish_std = dish_px.std(axis=0)
    wood_std = wood_px.std(axis=0)
    dish_max = dish_px.max(axis=0)

    sat = float(np.iinfo(cube.dtype).max) if np.issubdtype(cube.dtype, np.integer) \
        else float(dish_px.max())
    fisher = (dish_med - wood_med) ** 2 / (dish_std ** 2 + wood_std ** 2 + 1e-6)
    fisher[dish_max > sat_frac * sat] = -1.0     # exclude near-saturated bands outright
    fisher[dish_med <= wood_med] = -1.0          # exclude wrong-polarity bands (dish must be brighter)

    picked = []
    for b in np.argsort(fisher)[::-1]:
        if fisher[b] <= 0:
            break
        if all(abs(int(b) - p) >= min_gap for p in picked):
            picked.append(int(b))
        if len(picked) == k:
            break
    return picked


def consensus_dish(cube, bands, **find_dish_kwargs):
    """Run find_dish independently on several bands; return their median agreement.

    Returns (cx, cy, r, spread_px, per_band_results). spread_px is the max
    centre distance from the median among the per-band results -- a cheap
    confidence signal: tight agreement means every band is converging on the
    same physical rim rather than each drifting to something different (e.g.
    one band locking onto a competing circular edge that happens to be
    stronger in that particular band).
    """
    results = [find_dish(render_band_gray8(cube, b), **find_dish_kwargs) for b in bands]
    arr = np.array(results, dtype=float)
    med = np.median(arr, axis=0)
    spread = float(np.max(np.hypot(arr[:, 0] - med[0], arr[:, 1] - med[1])))
    return int(round(med[0])), int(round(med[1])), int(round(med[2])), spread, results


# ── projection lattice (ported from the reference) ───────────────────────────
def _projection_sharpness(mask, rotation, center, size):
    """Variance of the row+col projections at a trial rotation (peaks when aligned)."""
    w, h = size
    rot = cv2.getRotationMatrix2D(center, rotation, 1.0)
    a = cv2.warpAffine(mask, rot, (w, h)) > 0
    return a.sum(0).astype(float).var() + a.sum(1).astype(float).var()


def refine_rotation(mask, seed_deg, center, size):
    """Refine the grid angle from the cutouts themselves (±2° around the marker seed)."""
    def best_in(angles):
        return max(angles, key=lambda a: _projection_sharpness(mask, a, center, size))
    coarse = best_in(np.arange(seed_deg - 2.0, seed_deg + 2.0 + 1e-9, 0.25))
    return float(best_in(np.arange(coarse - 0.25, coarse + 0.25 + 1e-9, 0.05)))


def find_bands(profile, threshold, min_length):
    """Runs where a 1-D projection stays above threshold -> (start,end,center,length)."""
    above = profile > threshold
    bands, run = [], None
    for i, on in enumerate(above):
        if on and run is None:
            run = i
        elif not on and run is not None:
            if i - run >= min_length:
                bands.append((run, i, (run + i) // 2, i - run))
            run = None
    if run is not None and len(above) - run >= min_length:
        bands.append((run, len(above), (run + len(above)) // 2, len(above) - run))
    return bands


def regular_lattice(centers, low, high):
    """Clean evenly-spaced node positions from noisy band centers. -> (nodes, pitch)."""
    centers = np.array(sorted(centers), dtype=float)
    pitch = float(np.median(np.diff(centers)))
    indices = np.round((centers - centers[0]) / pitch)
    phase = float(np.median(centers - indices * pitch))
    nodes, step = [], 0
    while phase + step * pitch <= high:
        nodes.append(phase + step * pitch)
        step += 1
    step = -1
    while phase + step * pitch >= low:
        nodes.append(phase + step * pitch)
        step -= 1
    return sorted(nodes), pitch


# ── grid construction ────────────────────────────────────────────────────────
def cutout_mask(gray8, cx, cy, r):
    """Dark cutout-interior mask inside the dish.

    Empirically the grid walls read bright and the cell interiors read dark (a
    kernel clipped in a cell does not fill it with light), so the cells are the
    dark rectangles. Threshold at the in-dish DARK_PCT percentile, keep only
    pixels inside DISH_RADIUS_FRAC*r, then open to despeckle. Each cell becomes a
    solid dark rectangle, so its x/y projection bumps at every column/row.
    """
    H, W = gray8.shape
    inside = np.zeros((H, W), np.uint8)
    cv2.circle(inside, (cx, cy), int(r * DISH_RADIUS_FRAC), 255, -1)
    thr = np.percentile(gray8[inside > 0], DARK_PCT)
    mask = ((gray8 < thr) & (inside > 0)).astype(np.uint8) * 255
    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def build_grid(gray8, cube=None, band_k=5, agree_px=10.0):
    """Build the cutout lattice. Returns (cells, meta). See module docstring.

    `cube` (optional): the full multi-band cube backing `gray8`. When given,
    the mean-image dish estimate is treated as a bootstrap and refined by
    select_discriminative_bands + consensus_dish -- see those docstrings.
    Refinement is only accepted when the per-band results agree tightly
    (within `agree_px`); otherwise the bootstrap is kept, since a wide
    disagreement means at least one band is seeing something other than the
    true rim and a median would just be an uninterpretable compromise.
    """
    H, W = gray8.shape
    markers = detect_markers(gray8)
    seed_deg = grid_rotation_deg(markers)
    cx, cy, r = find_dish(gray8)
    dish_bands, dish_spread = None, None

    if cube is not None:
        bands = select_discriminative_bands(cube, cx, cy, r, k=band_k)
        if len(bands) >= 3:
            mcx, mcy, mr, spread, _ = consensus_dish(cube, bands)
            if spread <= agree_px:
                print(f"  multiband consensus ({len(bands)} bands {bands}): "
                      f"({mcx},{mcy}) r={mr}  spread={spread:.1f}px -> using it")
                cx, cy, r = mcx, mcy, mr
                dish_bands, dish_spread = bands, spread
            else:
                print(f"  multiband consensus disagreed (spread={spread:.1f}px "
                      f"> {agree_px}px); keeping mean-image dish {cx,cy,r}")
        else:
            print(f"  only {len(bands)} usable discriminative band(s); keeping mean-image dish")

    mask = cutout_mask(gray8, cx, cy, r)
    rotation = refine_rotation(mask, seed_deg, (cx, cy), (W, H))

    rot = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
    rot_inv = cv2.invertAffineTransform(rot)
    aligned = cv2.warpAffine(mask, rot, (W, H))

    col_prof = (aligned > 0).sum(0).astype(float)
    row_prof = (aligned > 0).sum(1).astype(float)
    min_band = int(r * MIN_BAND_FRAC)
    col_bands = find_bands(col_prof, col_prof.max() * BAND_PEAK_FRAC, min_band)
    row_bands = find_bands(row_prof, row_prof.max() * BAND_PEAK_FRAC, min_band)
    if not col_bands or not row_bands:
        sys.exit("projection found no bands — check cutout-mask polarity/threshold")

    cutout_w = float(np.median([L for *_, L in col_bands])*1)
    cutout_h = float(np.median([L for *_, L in row_bands])*1.2)

    col_x, pitch_x = regular_lattice([c for *_, c, _ in col_bands], cx - r, cx + r)
    row_y, pitch_y = regular_lattice([c for *_, c, _ in row_bands], cy - r, cy + r)

    markers_aligned = {mid: rot @ np.array([c.mean(0)[0], c.mean(0)[1], 1.0])
                       for mid, c in markers.items()}

    cells = []
    hw, hh = cutout_w / 2, cutout_h / 2
    rim_tol = RIM_TOLERANCE_FRAC * min(cutout_w, cutout_h)
    inclusion_r = r * CELL_INCLUSION_RADIUS_FRAC
    for ri, y in enumerate(row_y):
        for ci, x in enumerate(col_x):
            box = np.array([[x - hw, y - hh], [x + hw, y - hh],
                            [x + hw, y + hh], [x - hw, y + hh]])
            corners = np.array([rot_inv @ [bx, by, 1.0] for bx, by in box])
            center = corners.mean(0)
            if math.hypot(center[0] - cx, center[1] - cy) > inclusion_r:
                continue
            cr = np.hypot(corners[:, 0] - cx, corners[:, 1] - cy)
            in_frame = ((corners[:, 0] >= 1) & (corners[:, 0] < W - 1) &
                        (corners[:, 1] >= 1) & (corners[:, 1] < H - 1))
            clipped = bool((cr > r + rim_tol).any() or not in_frame.all())
            marker_id = None
            for mid, (mx, my) in markers_aligned.items():
                if abs(mx - x) < hw and abs(my - y) < hh:
                    marker_id = mid
            cells.append({"row": ri, "col": ci,
                          "corners": corners.round(1).tolist(),
                          "center": center.round(1).tolist(),
                          "clipped": clipped, "marker_id": marker_id})

    meta = {"dish": [cx, cy, r], "angle_deg": round(rotation, 3),
            "marker_seed_deg": round(seed_deg, 3),
            "pitch_px": [round(pitch_x, 1), round(pitch_y, 1)],
            "cell_px": [round(cutout_w, 1), round(cutout_h, 1)],
            "n_rows": len(row_y), "n_cols": len(col_x),
            "dish_bands": dish_bands, "dish_band_spread_px":
                round(dish_spread, 2) if dish_spread is not None else None,
            "markers": {str(mid): c.round(1).tolist() for mid, c in markers.items()}}
    return cells, meta


# ── marker-locked labelling (ported from step5) ──────────────────────────────
def marker_axis_mapping(cells, anchor_corners):
    """(drow,dcol)->(rel_row,rel_col) in the anchor marker's own frame."""
    A = np.array([[c["col"], c["row"], 1.0] for c in cells])
    B = np.array([c["center"] for c in cells], dtype=float)
    g_col, g_row, _ = np.linalg.lstsq(A, B, rcond=None)[0]
    g_col /= np.linalg.norm(g_col)
    g_row /= np.linalg.norm(g_row)
    TL, TR, _BR, BL = np.array(anchor_corners, dtype=float)
    u = (TR - TL) / np.linalg.norm(TR - TL)
    v = (BL - TL) / np.linalg.norm(BL - TL)
    if abs(g_col @ u) >= abs(g_row @ u):
        col_from_col = True
        col_sign, row_sign = np.sign(g_col @ u), np.sign(g_row @ v)
    else:
        col_from_col = False
        col_sign, row_sign = np.sign(g_row @ u), np.sign(g_col @ v)

    def to_rel(drow, dcol):
        if col_from_col:
            return int(row_sign * drow), int(col_sign * dcol)
        return int(row_sign * dcol), int(col_sign * drow)
    return to_rel


def label_cells(cells, meta):
    """Add marker-locked 'label'/rel_row/rel_col to each cell (needs >=1 marker)."""
    with_marker = {c["marker_id"]: c for c in cells if c["marker_id"] is not None}
    if not with_marker:
        for c in cells:                       # no anchor: fall back to raw indices
            c["rel_row"], c["rel_col"] = c["row"], c["col"]
            c["label"] = f"R{c['row']}C{c['col']}"
        return None
    anchor_id = min(with_marker)
    a = with_marker[anchor_id]
    to_rel = marker_axis_mapping(cells, meta["markers"][str(anchor_id)])
    for c in cells:
        rr, rc = to_rel(c["row"] - a["row"], c["col"] - a["col"])
        c["rel_row"], c["rel_col"] = rr, rc
        c["label"] = f"R{rr:+d}C{rc:+d}"
    return anchor_id


# ── drawing ──────────────────────────────────────────────────────────────────
def draw_grid(gray8, cells, meta):
    """Dish circle (cyan) + usable cells (green, labelled); marker cells (red)."""
    vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    cx, cy, r = meta["dish"]
    cv2.circle(vis, (cx, cy), r, (255, 255, 0), 2)
    for c in cells:
        corners = np.array(c["corners"], np.int32)
        if c["marker_id"] is not None:
            cv2.polylines(vis, [corners], True, (0, 0, 255), 2)
            continue
        if c["clipped"]:
            continue
        cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
        lx, ly = np.array(c["center"], int)
        cv2.putText(vis, c.get("label", f"{c['row']},{c['col']}"), (lx - 26, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, c.get("label", f"{c['row']},{c['col']}"), (lx - 26, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?", default=DEFAULT_CUBE,
                   help="Corrected *_cube.npy.")
    p.add_argument("--out", default=None, help="Output prefix (default: from src).")
    p.add_argument("--debug", action="store_true", help="Dump dish/mask stages too.")
    p.add_argument("--no-multiband", action="store_true",
                   help="Skip the multiband dish-detection refinement (mean-image only).")
    args = p.parse_args()

    gray8 = load_gray(args.src)
    print(f"Greyscale {gray8.shape[1]}x{gray8.shape[0]}.")
    cube = None if args.no_multiband else np.load(args.src, mmap_mode="r")
    cells, meta = build_grid(gray8, cube=cube)
    anchor = label_cells(cells, meta)

    n_use = sum(not c["clipped"] and c["marker_id"] is None for c in cells)
    n_clip = sum(c["clipped"] for c in cells)
    n_mark = sum(c["marker_id"] is not None for c in cells)
    print(f"Dish center {meta['dish'][:2]} r={meta['dish'][2]}  rotation {meta['angle_deg']}°")
    print(f"Lattice {meta['n_rows']}x{meta['n_cols']}  pitch {meta['pitch_px']}  "
          f"cell {meta['cell_px']} px")
    print(f"Cells in dish: {len(cells)} (usable {n_use}, clipped {n_clip}, marker {n_mark}); "
          f"anchor marker {anchor}")

    prefix = Path(args.out) if args.out else Path(str(args.src).removesuffix("_cube.npy"))
    overlay_path = prefix.with_name(prefix.name + "_grid_overlay.png")
    cells_path = prefix.with_name(prefix.name + "_grid_cells.json")
    cv2.imwrite(str(overlay_path), draw_grid(gray8, cells, meta))
    with open(cells_path, "w") as f:
        json.dump({"meta": meta, "cells": cells}, f, indent=2)
    print(f"wrote {overlay_path}")
    print(f"wrote {cells_path}")

    if args.debug:
        cx, cy, r = meta["dish"]
        d = prefix.with_name(prefix.name + "_grid_dbg")
        cv2.imwrite(f"{d}_blur.png", cv2.GaussianBlur(gray8, (9, 9), 0))
        cv2.imwrite(f"{d}_cutmask.png", cutout_mask(gray8, cx, cy, r))
        print(f"wrote {d}_blur.png, {d}_cutmask.png")


if __name__ == "__main__":
    main()
