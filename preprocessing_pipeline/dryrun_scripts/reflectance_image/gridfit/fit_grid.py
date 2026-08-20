"""
Robustly fit the analytical 7x4 plate grid to a corrected reflectance cube, so
every kernel well can be addressed by its plate cell rather than by hand-picked
pixel coordinates.

Pipeline:
  1. localize the two ArUco fiducials (localize_markers -- candidate+NCC, works
     on push-broom blur where cv2.aruco's decode fails). id0@R3C3, id1@R2C6.
  2. Radon angle: de-rotate, maximise row+col projection variance -> grid
     rotation, robust even on a ~45deg-rotated dish.
  3. orient with NOMINAL pitch (a known rigid-plate constant): markers place the
     origin and pick which of 4 orientations is correct (by origin agreement).
     Nominal pitch (not marker-derived) keeps a weak marker from stretching the
     lattice at init.
  4. robust refine: snap each KERNEL cell centre onto its WALL FRAME (a matched
     template that locks to the bright walls / dark interior, unbiased unlike the
     dark-well centroid), then trimmed least-squares over the well snaps only,
     dropping outliers. Markers are NOT in the geometry (their ArUco pattern
     centre sits a systematic ~15px off the cell centre) -- they orient the grid
     and serve as an independent post-fit cross-check.
  5. cross-validate -> confidence -> status: pass / needs_review / fail. The
     well-fit rms + inlier count gate the geometry; the marker cross-check
     (coarse, ~1-pitch tolerance) catches gross mis-indexing. Never silently
     emit a wrong grid.

Because the lattice is anchored to the markers and the plate model is fixed,
cell (row, col) -- and therefore the cell index used to name outputs -- means the
same physical well on every capture, whatever the plate's rotation. That is the
whole point: it is what makes a well trackable across days.

Degrades gracefully: one marker still solvable (orientation picked by well
agreement, flagged needs_review); zero markers -> fail.

Vendored from reflectance_image_pipeline_production/grid_sandbox/fit_grid.py.
Library only -- the standalone CLI, the file writes and the process_image import
were dropped so process_image.py can import this without a circular import:
`fit()` takes the gray8 render instead of loading and rendering a cube itself.
Added here (not in the sandbox original): kernel_cells(), cell_mask() and
draw_grid_overlay(), which the sandbox handled in extract_cells.py instead.
Keep in sync with the sandbox original if that one is retuned.
"""
import cv2
import numpy as np

from localize_markers import localize_markers  # noqa: F401  (re-exported for callers)

# ---- analytical plate model (fixed rigid plate; confirmed 2026-07-24) --------
N_ROWS, N_COLS = 4, 7
MARKER_CELLS = {0: (3, 3), 1: (2, 6)}          # id -> (row, col); verified on 716_2
CLIPPED_CORNERS = {(0, 0), (0, 6), (3, 0), (3, 6)}  # rim-clipped, always empty
# nominal pitch: a plate constant (px 73-77, py 121-126 measured across captures).
# used only for init/orientation + the degradation path; refine measures it exactly.
PITCH_X_NOM, PITCH_Y_NOM = 76.0, 125.0
CANON_PX = (64.0, 88.0)      # refined-pitch acceptance band (~+-15%) for confidence
CANON_PY = (108.0, 142.0)

# ---- snap + fit tuning -------------------------------------------------------
SNAP_SEARCH_FRAC = 0.38      # search window as a fraction of the larger pitch
SNAP_STEP = 2.0              # px grid step for the wall-frame search
SNAP_MIN_SCORE = 22.0        # min (wall_mean - interior_mean) in gray8 to accept a well
FIT_RESID_FLOOR = 8.0        # px; trimmed-LSQ inlier floor (above robust MAD gate)

# ---- confidence thresholds ---------------------------------------------------
RMS_PASS = 6.0               # px refine rms for a clean pass
RMS_REVIEW = 11.0            # px above which -> needs_review
MIN_WELL_INLIERS_PASS = 14   # 22 kernel wells; want most snapping cleanly
MIN_WELL_INLIERS_FAIL = 8    # below this we can't trust the geometry
# marker centre (ArUco pattern) sits a systematic ~15-20px off the geometric cell
# centre, so this is a COARSE orientation/indexing check (gross error = wrong cell,
# ~>=1 pitch), NOT a precise metric gate -- rms + inliers gate the geometry.
MARKER_RESID_TOL = 35.0

# A cell IS its lattice rectangle, corner to corner -- so the default inset is
# zero and cell_mask() rasterises exactly the quad drawn on the overlay. The cells
# then tile the plate contiguously and every one is identifiable by its corners,
# with nothing clipped off a kernel that happens to sit hard against a wall.
#
# extract_cells.py insets by 0.16 instead, but for a different job: it needs the
# bright wall kept out of its Otsu brightness sample before segmenting the kernel.
# That is a segmentation concern, not a definition of the cell, so it does not
# belong in the default here -- pass --cell-inset when you want the well interior.
CELL_INSET_FRAC = 0.0

N_KERNEL_CELLS = N_ROWS * N_COLS - len(CLIPPED_CORNERS) - len(MARKER_CELLS)  # 22


def unit(deg):
    r = np.radians(deg)
    return np.array([np.cos(r), np.sin(r)])


# ------------------------------------------------------------------ rotation --
def radon_angle(mask, coarse=np.arange(-50, 50, 1.0), fine_half=1.0, fine_step=0.1):
    """Grid angle: the de-rotation that maximises row+col projection variance."""
    h, w = mask.shape
    cen = (w / 2.0, h / 2.0)

    def score(a):
        M = cv2.getRotationMatrix2D(cen, a, 1.0)
        rot = cv2.warpAffine(mask, M, (w, h))
        return rot.sum(axis=0).var() + rot.sum(axis=1).var()

    best = max(coarse, key=score)
    fine = np.arange(best - fine_half, best + fine_half + 1e-9, fine_step)
    return float(max(fine, key=score))


def dark_mask(gray8, pct=25):
    return (gray8 < np.percentile(gray8, pct)).astype(np.float32)


# ---------------------------------------------------------------- orientation --
def _frame(origin, base):
    ec = PITCH_X_NOM * unit(base)
    er = PITCH_Y_NOM * unit(base + 90.0)
    return {"origin": np.asarray(origin, float), "e_col": ec, "e_row": er,
            "px": PITCH_X_NOM, "py": PITCH_Y_NOM, "angle": base % 180.0}


def orient_two_markers(markers, angle):
    """Pick the orientation (of 4) whose two nominal-pitch origin estimates agree.

    The asymmetric id0->id1 grid offset makes the correct orientation's origin
    disagreement small (~pitch error) while the wrong ones are hugely off.
    """
    id0c = np.asarray(markers[0]["center"], float)
    id1c = np.asarray(markers[1]["center"], float)
    (r0, c0), (r1, c1) = MARKER_CELLS[0], MARKER_CELLS[1]
    best, best_dis = None, np.inf
    for base in (angle, angle + 90.0, angle + 180.0, angle + 270.0):
        ec, er = PITCH_X_NOM * unit(base), PITCH_Y_NOM * unit(base + 90.0)
        o0 = id0c - c0 * ec - r0 * er
        o1 = id1c - c1 * ec - r1 * er
        dis = float(np.linalg.norm(o0 - o1))
        if dis < best_dis:
            best_dis, best = dis, _frame(0.5 * (o0 + o1), base)
    best["origin_disagree"] = best_dis
    return best


def orient_one_marker(markers, angle, gray_f):
    """One marker: pick the orientation with the most well snaps (well agreement)."""
    mid = 0 if 0 in markers else 1
    r, c = MARKER_CELLS[mid]
    mc = np.asarray(markers[mid]["center"], float)
    best, best_n = None, -1
    for base in (angle, angle + 90.0, angle + 180.0, angle + 270.0):
        ec, er = PITCH_X_NOM * unit(base), PITCH_Y_NOM * unit(base + 90.0)
        fr = _frame(mc - c * ec - r * er, base)
        n = sum(1 for cell in cells_from_frame(fr) if cell["kind"] != "clipped"
                and snap_wall_frame(gray_f, cell["center_px"], ec, er)[1] > SNAP_MIN_SCORE)
        if n > best_n:
            best_n, best = n, fr
    best["origin_disagree"] = float("nan")
    return best


def cells_from_frame(frame):
    """All 28 lattice cells, row-major. `index` numbers the 22 kernel cells only.

    The numbering is row-major over the fixed plate model, so cell index N always
    denotes the same physical well regardless of how the plate is rotated in the
    frame -- clipped corners and marker cells are skipped and carry index None.
    """
    o, ec, er = frame["origin"], frame["e_col"], frame["e_row"]
    cells, idx = [], 0
    for r in range(N_ROWS):
        for c in range(N_COLS):
            center = o + c * ec + r * er
            corners = [o + (c + dc) * ec + (r + dr) * er
                       for dc, dr in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
            kind = "clipped" if (r, c) in CLIPPED_CORNERS else "kernel"
            for mid, cell in MARKER_CELLS.items():
                if cell == (r, c):
                    kind = f"marker{mid}"
            index = None
            if kind == "kernel":
                index, idx = idx, idx + 1
            cells.append({
                "index": index,
                "row": r, "col": c, "kind": kind,
                "center_px": [float(center[0]), float(center[1])],
                "corners_px": [[float(x), float(y)] for x, y in corners],
            })
    return cells


def kernel_cells(cells):
    """The 22 kernel cells, ordered by their stable cell index."""
    return sorted((c for c in cells if c["kind"] == "kernel"), key=lambda c: c["index"])


# --------------------------------------------------------------- wall snap ----
def _sample(gray_f, pts):
    """Nearest-neighbour sample of gray_f at float points (...,2) as (x,y)."""
    h, w = gray_f.shape
    xi = np.clip(np.rint(pts[..., 0]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.rint(pts[..., 1]).astype(np.int32), 0, h - 1)
    return gray_f[yi, xi]


def snap_wall_frame(gray_f, center, e_col, e_row):
    """Lock a cell onto its wall frame: the offset maximising (bright walls -
    dark interior). Unbiased vs the dark-well centroid. Returns (center, score)."""
    center = np.asarray(center, float)
    px, py = np.linalg.norm(e_col), np.linalg.norm(e_row)
    ec, er = e_col / px, e_row / py

    f = np.linspace(-0.3, 0.3, 5)
    interior = np.array([a * px * ec + b * py * er for a in f for b in f])  # (25,2)
    t = np.linspace(-0.25, 0.25, 5)
    walls = []
    for s in (-0.5, 0.5):
        walls += [s * px * ec + u * py * er for u in t]   # left/right walls
        walls += [u * px * ec + s * py * er for u in t]   # top/bottom walls
    walls = np.array(walls)  # (20,2)

    search = SNAP_SEARCH_FRAC * max(px, py)
    g = np.arange(-search, search + 1e-9, SNAP_STEP)
    offs = np.array([[ox, oy] for oy in g for ox in g])   # (K,2)
    cand = center[None, :] + offs                          # (K,2)

    gi = _sample(gray_f, cand[:, None, :] + interior[None, :, :]).mean(axis=1)
    gw = _sample(gray_f, cand[:, None, :] + walls[None, :, :]).mean(axis=1)
    score = gw - gi
    k = int(np.argmax(score))
    return cand[k], float(score[k])


# ------------------------------------------------------------------ robust fit --
def _affine(idx, obs, keep):
    A = np.column_stack([idx[keep, 1], idx[keep, 0], np.ones(int(keep.sum()))])  # (c,r,1)
    sol, *_ = np.linalg.lstsq(A, obs[keep], rcond=None)  # (3,2): d/dcol, d/drow, origin
    return sol


def fit_affine_robust(idx_pairs, obs):
    """Trimmed LSQ affine (col,row,1)->(x,y); iteratively drop outliers.
    Returns (sol, keep_mask, residuals, rms)."""
    idx = np.asarray(idx_pairs, float)
    obs = np.asarray(obs, float)
    Aall = np.column_stack([idx[:, 1], idx[:, 0], np.ones(len(idx))])
    keep = np.ones(len(idx), bool)
    resid = np.zeros(len(idx))
    for _ in range(5):
        sol = _affine(idx, obs, keep)
        resid = np.linalg.norm(obs - Aall @ sol, axis=1)
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med)) + 1e-6
        gate = max(FIT_RESID_FLOOR, med + 3.0 * mad)
        newkeep = resid <= gate
        if newkeep.sum() < 4 or np.array_equal(newkeep, keep):
            break
        keep = newkeep
    rms = float(np.sqrt((resid[keep] ** 2).mean()))
    return sol, keep, resid, rms


def _cell_center(frame, r, c):
    return frame["origin"] + c * frame["e_col"] + r * frame["e_row"]


def _snap_wells(gray_f, frame):
    """Wall-snap the KERNEL wells (skip clipped corners AND marker cells)."""
    labels, idx_pairs, obs = [], [], []
    for cell in cells_from_frame(frame):
        if cell["kind"] != "kernel":
            continue
        c, score = snap_wall_frame(gray_f, cell["center_px"], frame["e_col"], frame["e_row"])
        if score >= SNAP_MIN_SCORE:
            labels.append(f"R{cell['row']}C{cell['col']}")
            idx_pairs.append((cell["row"], cell["col"]))
            obs.append(c)
    return labels, idx_pairs, obs


def robust_fit(gray_f, frame, markers, iters=1):
    """Wall-snap the kernel wells and trimmed-LSQ refit the affine.

    Single-pass by default: with the nominal-pitch marker init this is both
    accurate (rms ~3px) and stable. Re-snap iteration was tried and REMOVED --
    the unconstrained affine can drift its scale to fit mis-snapped cells, and
    re-snapping then chases the drift (it collapsed the 45deg case). Markers are
    NOT in the geometry (pattern-centre ~15px off the cell centre); they're an
    independent post-fit cross-check. Returns (new_frame, diag)."""
    cur = frame
    labels, idx_pairs, obs, sol, keep, resid, rms = [], [], [], None, None, None, None
    for it in range(iters):
        labels, idx_pairs, obs = _snap_wells(gray_f, cur)
        if len(idx_pairs) < 4:
            return cur, {"n_well": len(idx_pairs), "well_inliers": 0, "rms": None,
                         "marker_resid": {}, "rejected": labels}
        sol, keep, resid, rms = fit_affine_robust(idx_pairs, obs)
        new = {"origin": sol[2], "e_col": sol[0], "e_row": sol[1],
               "px": float(np.linalg.norm(sol[0])), "py": float(np.linalg.norm(sol[1])),
               "angle": float(np.degrees(np.arctan2(sol[0][1], sol[0][0])) % 180.0),
               "origin_disagree": frame.get("origin_disagree", float("nan"))}
        moved = float(np.linalg.norm(new["origin"] - cur["origin"]))
        cur = new
        if it > 0 and moved < 1.0:
            break

    marker_resid = {}
    for mid, (r, c) in MARKER_CELLS.items():
        if mid in markers:
            pred = _cell_center(cur, r, c)
            marker_resid[str(mid)] = float(np.linalg.norm(pred - np.asarray(markers[mid]["center"], float)))

    rejected = [labels[i] for i in range(len(labels)) if not keep[i]]
    diag = {"n_well": len(idx_pairs), "well_inliers": int(keep.sum()), "rms": rms,
            "marker_resid": marker_resid, "rejected": rejected}
    return cur, diag


# ------------------------------------------------------------------ assess ----
def assess(frame, diag, n_markers):
    """status in {pass, needs_review, fail} + human-readable reasons."""
    reasons = []
    wi, rms = diag["well_inliers"], diag["rms"]
    px, py = frame["px"], frame["py"]

    if rms is None or wi < MIN_WELL_INLIERS_FAIL:
        return "fail", [f"only {wi} well inliers / fit failed"]

    if not (CANON_PX[0] <= px <= CANON_PX[1] and CANON_PY[0] <= py <= CANON_PY[1]):
        reasons.append(f"pitch out of band (px={px:.1f} py={py:.1f})")
    if rms > RMS_REVIEW:
        reasons.append(f"high rms {rms:.1f}px")
    if wi < MIN_WELL_INLIERS_PASS:
        reasons.append(f"few well inliers ({wi})")
    if n_markers < 2:
        reasons.append("only one marker (degraded)")
    bad_markers = [m for m, r in diag["marker_resid"].items() if r > MARKER_RESID_TOL]
    if len(bad_markers) >= 2:
        reasons.append("both markers disagree with lattice")
    elif bad_markers:
        reasons.append(f"marker {bad_markers[0]} off by "
                       f"{diag['marker_resid'][bad_markers[0]]:.1f}px")

    if not reasons and rms <= RMS_PASS:
        return "pass", []
    # a lone slightly-off marker with an otherwise clean lattice is tolerable
    tolerable = all("marker" in r or "one marker" in r for r in reasons) \
        and wi >= MIN_WELL_INLIERS_PASS and rms <= RMS_REVIEW \
        and CANON_PX[0] <= px <= CANON_PX[1] and CANON_PY[0] <= py <= CANON_PY[1] \
        and len(bad_markers) < 2
    return ("pass" if tolerable else "needs_review"), reasons


# -------------------------------------------------------------- cell masks ----
def _inset_corners(cell, inset_frac):
    """The cell rectangle shrunk toward its centre by inset_frac on every edge."""
    center = np.asarray(cell["center_px"], float)
    corners = np.asarray(cell["corners_px"], float)
    return center + (corners - center) * (1.0 - 2.0 * inset_frac)


def cell_mask(cell, width, n_lines, inset_frac=CELL_INSET_FRAC):
    """Boolean (width, n_lines) mask of one cell.

    At the default inset of 0 this is the lattice rectangle itself -- the same
    quad the overlay outlines -- so the mask covers the whole cell and clips
    nothing. A non-zero inset_frac shrinks it toward the well interior.

    Returned in CUBE orientation, so it indexes the cube directly:
    `cube[mask]` -> (n_masked, bands). The lattice is rotated in general, so this
    has to be a rasterised polygon rather than a slice.
    """
    # grid orientation is (n_lines, width): row=y=scan, col=x=width, and cv2 takes
    # points as (x, y) -- which is exactly how corners_px is stored.
    m = np.zeros((n_lines, width), np.uint8)
    cv2.fillPoly(m, [np.round(_inset_corners(cell, inset_frac)).astype(np.int32)], 1)
    return m.astype(bool).T


# ------------------------------------------------------------------ overlay ---
STATUS_COLOR = {"pass": (0, 220, 0), "needs_review": (0, 200, 255), "fail": (0, 0, 255)}
KIND_COLOR = {"kernel": (0, 255, 0), "clipped": (120, 120, 120),
              "marker0": (0, 0, 255), "marker1": (0, 165, 255)}


def _to_preview(pts):
    """Grid (x=width, y=scan) points -> cv2 (col, row) points on the preview.

    The preview image is the cube's own (width, n_lines) layout, i.e. row=width
    and col=scan -- the transpose of the orientation the grid is fitted in. So a
    grid point's coordinates simply swap. Drawing in the preview's own layout
    (rather than transposing the finished overlay) keeps the text upright.
    """
    return np.asarray(pts, float)[..., ::-1]


def draw_grid_overlay(preview_gray8, cells, markers, frame, diag, status, reasons,
                      inset_frac=CELL_INSET_FRAC):
    """Annotate the (width, n_lines) intensity preview with the fitted grid.

    Kernel cells are outlined at exactly the footprint cell_mask() rasterises, so
    the overlay never promises a region the files do not contain. At the default
    zero inset that is the full lattice rectangle, so the cells tile the plate
    corner to corner and each is identifiable by its corners. Marker and clipped
    cells have no mask and are always drawn at full rectangle, as context.

    Returns a BGR image the same size as the preview, so the written PNG stays
    pixel-for-pixel aligned with the saved cube.
    """
    vis = cv2.cvtColor(preview_gray8, cv2.COLOR_GRAY2BGR)
    for cell in cells:
        outline = (_inset_corners(cell, inset_frac) if cell["kind"] == "kernel"
                   else cell["corners_px"])
        pts = np.round(_to_preview(outline)).astype(np.int32)
        cv2.polylines(vis, [pts], True, KIND_COLOR.get(cell["kind"], (255, 255, 0)), 1)
        cx, cy = _to_preview(cell["center_px"])
        if cell["kind"] == "kernel":
            label = f"c{cell['index']}"
        elif cell["kind"].startswith("marker"):
            label = cell["kind"].replace("marker", "id")
        else:
            label = "x"
        for txt, dy, sc in ((label, -3, 0.42), (f"R{cell['row']}C{cell['col']}", 11, 0.32)):
            org = (int(cx) - 14, int(cy) + dy)
            cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 0), 1, cv2.LINE_AA)
    for mid, m in markers.items():
        mx, my = _to_preview(m["center"])
        cv2.drawMarker(vis, (int(mx), int(my)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    rms_s = f"{diag['rms']:.2f}" if diag.get("rms") is not None else "n/a"
    mres = " ".join(f"m{k}={v:.1f}" for k, v in diag.get("marker_resid", {}).items())
    line1 = (f"GRID {status.upper()}  angle={frame['angle']:.1f} px={frame['px']:.1f} "
             f"py={frame['py']:.1f} wells={diag.get('well_inliers')}/{diag.get('n_well')} "
             f"rms={rms_s} {mres}")
    for i, txt in enumerate((line1, "; ".join(reasons))):
        if not txt:
            continue
        org = (8, 18 + i * 20)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, STATUS_COLOR[status], 1,
                    cv2.LINE_AA)
    return vis


# ------------------------------------------------------------------- entry ----
def fit(gray8):
    """Fit the plate grid to a (n_lines, width) gray8 render. No file writes.

    Returns a dict with keys: status, reasons, frame, cells, markers, diag,
    angle, n_markers. `frame` is None (and `cells` empty) only when no marker was
    localizable, which is the one case the grid cannot be placed at all.
    """
    gray_f = gray8.astype(np.float32)
    markers = localize_markers(gray8)
    n_markers = sum(1 for mid in (0, 1) if mid in markers)
    base = {"markers": markers, "n_markers": n_markers}

    if n_markers == 0:
        return {**base, "status": "fail", "reasons": ["no ArUco markers localized"],
                "frame": None, "cells": [], "diag": {}, "angle": None}

    angle = radon_angle(dark_mask(gray8))
    frame = (orient_two_markers(markers, angle) if n_markers == 2
             else orient_one_marker(markers, angle, gray_f))
    rframe, diag = robust_fit(gray_f, frame, markers)
    status, reasons = assess(rframe, diag, n_markers)
    return {**base, "status": status, "reasons": reasons, "frame": rframe,
            "cells": cells_from_frame(rframe), "diag": diag, "angle": angle}
