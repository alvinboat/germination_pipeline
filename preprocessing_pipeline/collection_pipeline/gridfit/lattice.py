"""Fit the plate lattice to one capture and decide whether to believe it.

    seeds + fiducials  ->  basis  ->  integer lattice  ->  pose  ->  refine  ->  gates

basis      the well cloud's own two periods, from the pairwise vectors between
           neighbouring wells. Nothing about the plate is assumed except roughly
           how big a cell is; the pitch, the angle and the skew are measured.
integer    every seed gets an (i, j) lattice index, then the basis is re-fitted
           to those indices and the seeds re-indexed until it stops changing.
pose       four relabelings map that anonymous lattice onto the plate model
           (flip either axis or neither). Two of them reverse handedness, which
           is what a ventral capture -- the plate physically turned over -- needs.
           The winner is chosen by evidence, not by the folder name: which cells
           have wells, which are empty, and which cell each identified fiducial
           lands in. A translation search of +-2 cells runs alongside, so a whole
           missing edge row cannot shift the labelling by one.
refine     trimmed least squares over the seeds the pose matched, dropping
           outliers. Fiducials are deliberately NOT in the fit -- an ArUco
           pattern centre sits a systematic ~15-25 px off its cell centre, so
           including them would bias every cell -- they are the cross-check.
gates      pass / needs_review / fail. A wrong grid is worse than no grid: it
           mislabels which physical kernel a spectrum came from, produces a
           perfectly clean-looking onverlay, and nothing downstream can detect it.
           So the fit is allowed to fail, and callers must honour that.

Everything is in the working frame (see render.py): x = column, y = row of the
(n_lines, width) render.
"""
import numpy as np

from . import plate

MIN_SEEDS = 8               # below this the basis itself is not measurable
MIN_MATCHED_PASS = 14       # of 22 wells; most should land on the lattice
MIN_MATCHED_FAIL = 8
# Two seeds on one cell means over-detection, not mis-indexing, as long as it is
# rare: a germinating kernel throws a shoot and a root that each threshold out as
# their own dark blob, which is why this fires on the later days and on the
# ventral face. The extra blob is dropped (the one further from the node) and
# noted. Pervasive doubling is different -- that is a lattice at the wrong pitch
# folding two columns onto one -- so it still fails.
MAX_DUPLICATE_FRAC = 0.25
RMS_PASS = 6.0              # px, seed-to-node
RMS_REVIEW = 12.0
POSE_MARGIN_FAIL = 4.0      # below: the labelling is a coin toss, refuse it
POSE_MARGIN_PASS = 8.0
MARKER_OFFSET_TOL = 40.0    # px; coarse -- catches a whole-cell error, not mm
FIT_RESID_FLOOR = 8.0       # px; trimmed-LSQ inlier floor above the MAD gate
OFFSET_SEARCH = 2           # cells, each way, in the pose translation search


# ------------------------------------------------------------------- basis --
def _pair_vectors(P):
    D = P[None, :, :] - P[:, None, :]          # D[i, j] = P[j] - P[i]
    L = np.hypot(D[..., 0], D[..., 1])
    return D, L


def _dominant(vs):
    """Median of vectors after folding them into a common half-plane.

    Folded by the doubled-angle mean, which is the circular mean of an axis
    (a direction defined only up to sign) rather than of a direction.
    """
    a = np.arctan2(vs[:, 1], vs[:, 0])
    m = np.angle(np.mean(np.exp(2j * a))) / 2.0
    u = np.array([np.cos(m), np.sin(m)])
    s = np.sign(vs @ u)
    s[s == 0] = 1.0
    return np.median(vs * s[:, None], axis=0)


def basis_from_seeds(P):
    """(e_col, e_row) from the well cloud alone, or None if it has no lattice.

    e_col is the short period (across the plate's 7 columns), e_row the long one.
    Every pair of wells about one pitch apart votes, not just nearest
    neighbours, so a few missing or merged wells cannot tilt the estimate.
    """
    if len(P) < MIN_SEEDS:
        return None
    D, L = _pair_vectors(P)
    Ln = L.copy()
    np.fill_diagonal(Ln, np.inf)
    d0 = float(np.median(Ln.min(axis=1)))
    if not np.isfinite(d0) or d0 <= 1.0:
        return None

    near = (L > 0.70 * d0) & (L < 1.35 * d0)
    if near.sum() < 6:
        return None
    e_col = _dominant(D[near])
    px = float(np.hypot(*e_col))
    if px <= 1.0:
        return None
    uc = e_col / px

    cosang = np.abs(D @ uc) / np.maximum(L, 1e-9)
    perp = (L > 1.35 * px) & (L < 2.45 * px) & (cosang < 0.45)
    if perp.sum() >= 6:
        w = np.array([-uc[1], uc[0]])
        vs = D[perp]
        s = np.sign(vs @ w)
        s[s == 0] = 1.0
        e_row = np.median(vs * s[:, None], axis=0)
    else:
        # No second period visible (a plate cropped to one or two rows). Fall
        # back to the plate's nominal shape rather than giving up: the pose
        # search will not clear its margin, so this can only end in a flag.
        e_row = np.array([-uc[1], uc[0]]) * px * (plate.PITCH_Y_NOM / plate.PITCH_X_NOM)

    if np.hypot(*e_row) < np.hypot(*e_col):
        e_col, e_row = e_row, e_col
    return np.asarray(e_col, float), np.asarray(e_row, float)


# ------------------------------------------------------- integer assignment --
def _solve_frame(idx, P, keep=None):
    """Least-squares (origin, e_col, e_row) from index->point pairs."""
    if keep is None:
        keep = np.ones(len(P), bool)
    A = np.column_stack([np.ones(int(keep.sum())), idx[keep, 0], idx[keep, 1]])
    sol, *_ = np.linalg.lstsq(A, P[keep], rcond=None)
    return sol[0], sol[1], sol[2]


def index_seeds(P, e_col, e_row, iters=4):
    """Integer (i, j) per seed, with the basis re-fitted to them. -> (idx, frame).

    The initial phase comes from the circular mean of the fractional lattice
    coordinates, so the integer grid lands on the wells rather than between them
    whatever arbitrary point the basis was measured from.
    """
    B = np.column_stack([e_col, e_row])
    origin = P.mean(axis=0)
    t = np.linalg.solve(B, (P - origin).T).T
    for k in (0, 1):
        t[:, k] -= np.angle(np.mean(np.exp(2j * np.pi * t[:, k]))) / (2 * np.pi)
    idx = np.rint(t).astype(int)

    for _ in range(iters):
        origin, e_col, e_row = _solve_frame(idx.astype(float), P)
        B = np.column_stack([e_col, e_row])
        t = np.linalg.solve(B, (P - origin).T).T
        new = np.rint(t).astype(int)
        if np.array_equal(new, idx):
            break
        idx = new
    idx = idx - idx.min(axis=0)
    origin, e_col, e_row = _solve_frame(idx.astype(float), P)
    return idx, (origin, e_col, e_row)


def _index_of(point, frame):
    origin, e_col, e_row = frame
    t = np.linalg.solve(np.column_stack([e_col, e_row]), np.asarray(point, float) - origin)
    return tuple(int(v) for v in np.rint(t))


# -------------------------------------------------------------------- pose --
SEED_ON_KERNEL = 1.0
SEED_OFF_KERNEL = -2.0      # a well where the model says fiducial, rim or nothing
KERNEL_EMPTY = -2.0         # a model well with no seed
DUPLICATE = -2.0
MARKER_RIGHT_CELL = 8.0     # an identified fiducial in its own cell
MARKER_WRONG_CELL = -8.0


def _score_pose(pose, dr, dc, seed_idx, marker_idx):
    cells = []
    for i, j in seed_idx:
        r, c = plate.apply_pose(pose, int(j), int(i))
        cells.append((r + dr, c + dc))
    occ = set(cells)
    kernels = set(plate.KERNEL_CELLS)
    score = (SEED_ON_KERNEL * len(occ & kernels)
             + SEED_OFF_KERNEL * len(occ - kernels)
             + KERNEL_EMPTY * len(kernels - occ)
             + DUPLICATE * (len(cells) - len(occ)))
    hits = 0
    for mid, (mi, mj) in marker_idx.items():
        r, c = plate.apply_pose(pose, int(mj), int(mi))
        if (r + dr, c + dc) == plate.MARKER_CELL[mid]:
            score += MARKER_RIGHT_CELL
            hits += 1
        else:
            score += MARKER_WRONG_CELL
    return score, cells, hits


def choose_pose(seed_idx, marker_idx):
    """Best (pose, dr, dc) by evidence. -> dict with cells, score, margin.

    Margin over the runner-up is the number that matters: a plate is a nearly
    symmetric object and the two fiducials are what break the symmetry, so a
    thin margin means the evidence did not actually decide and the labelling
    must not be trusted.
    """
    results = []
    rng = range(-OFFSET_SEARCH, OFFSET_SEARCH + 1)
    for pose in plate.POSES:
        for dr in rng:
            for dc in rng:
                score, cells, hits = _score_pose(pose, dr, dc, seed_idx, marker_idx)
                results.append((score, pose, dr, dc, cells, hits))
    results.sort(key=lambda t: -t[0])
    best = results[0]
    margin = best[0] - results[1][0] if len(results) > 1 else float("inf")
    return {"pose": best[1][0], "flip_row": best[1][1], "flip_col": best[1][2],
            "dr": best[2], "dc": best[3], "cells": best[4],
            "score": float(best[0]), "margin": float(margin),
            "marker_cell_hits": best[5]}


# ------------------------------------------------------------------ refine --
def refine(cells, P):
    """Trimmed least squares (row, col) -> (x, y) over the matched seeds.

    -> (frame, keep_mask, residuals, rms). Outliers are dropped on a robust MAD
    gate with a floor, so one badly-centred blob cannot drag the lattice.
    """
    idx = np.array([[c, r] for r, c in cells], float)      # design order: col, row
    keep = np.ones(len(P), bool)
    resid = np.zeros(len(P))
    origin = e_col = e_row = None
    for _ in range(5):
        origin, e_col, e_row = _solve_frame(idx, P, keep)
        pred = origin + idx[:, 0:1] * e_col + idx[:, 1:2] * e_row
        resid = np.linalg.norm(P - pred, axis=1)
        med = float(np.median(resid[keep]))
        mad = float(np.median(np.abs(resid[keep] - med))) + 1e-6
        new = resid <= max(FIT_RESID_FLOOR, med + 3.0 * mad)
        if new.sum() < 4 or np.array_equal(new, keep):
            break
        keep = new
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
    return (origin, e_col, e_row), keep, resid, rms


def _resolve_duplicates(matched, resid):
    """Keep one seed per cell -- the one nearest the node. -> (kept, dropped)."""
    by_cell = {}
    for i, (rc, k) in enumerate(matched):
        by_cell.setdefault(rc, []).append((float(resid[i]), i))
    keep_i, dropped = set(), []
    for rc, group in by_cell.items():
        group.sort()
        keep_i.add(group[0][1])
        dropped += [matched[i] for _, i in group[1:]]
    return [m for i, m in enumerate(matched) if i in keep_i], dropped


def cell_center(frame, r, c):
    origin, e_col, e_row = frame
    return origin + c * e_col + r * e_row


def cell_corners(frame, r, c):
    """The cell's lattice rectangle: centre +- half a pitch on each axis."""
    _, e_col, e_row = frame
    ctr = cell_center(frame, r, c)
    a, b = e_col / 2.0, e_row / 2.0
    return np.array([ctr - a - b, ctr + a - b, ctr + a + b, ctr - a + b])


# ------------------------------------------------------------------- entry --
def fit(seeds, markers, side_hint=None):
    """Fit the lattice. -> result dict; `frame` is None when it must not be used.

    `side_hint` is the folder's dorsal/ventral label. It is never used to decide
    anything -- the fiducial ids say which face this is -- but disagreeing with
    it is reported, because a capture filed under the wrong side is exactly the
    kind of error that survives every other check.
    """
    res = {"status": "fail", "reasons": [], "flags": [], "side_flags": [], "info": [],
           "frame": None,
           "n_seeds": len(seeds), "markers": markers, "side_hint": side_hint,
           "confirmed_cells": 0, "n_duplicate": 0, "duplicate_cells": [],
           "unmatched_seeds": len(seeds), "n_matched": 0}

    P = np.array([s["center"] for s in seeds], float) if seeds else np.zeros((0, 2))
    ids = sorted(markers)
    res["marker_ids"] = ids
    res["side_observed"] = _side_from_ids(ids)
    res["side_flags"] = side_notes(ids, res["side_observed"], side_hint)

    if len(P) < MIN_SEEDS:
        res["reasons"].append(f"only {len(P)} well seeds, need {MIN_SEEDS}")
        return assess(res)

    basis = basis_from_seeds(P)
    if basis is None:
        res["reasons"].append("no lattice period found in the well cloud")
        return assess(res)
    seed_idx, frame0 = index_seeds(P, *basis)

    marker_idx = {mid: _index_of(m["center"], frame0) for mid, m in markers.items()}
    pose = choose_pose(seed_idx, marker_idx)
    res["pose"] = {k: pose[k] for k in
                   ("pose", "flip_row", "flip_col", "dr", "dc", "score", "margin",
                    "marker_cell_hits")}

    kernels = set(plate.KERNEL_CELLS)
    matched = [(rc, k) for k, rc in enumerate(pose["cells"]) if rc in kernels]
    if len(matched) < MIN_MATCHED_FAIL:
        res["n_matched"] = len(matched)
        res["reasons"].append(f"only {len(matched)} seeds landed on plate wells")
        return assess(res)

    frame, keep, resid, rms = refine([rc for rc, _ in matched], P[[k for _, k in matched]])
    matched, dropped = _resolve_duplicates(matched, resid)
    if dropped:
        frame, keep, resid, rms = refine([rc for rc, _ in matched],
                                         P[[k for _, k in matched]])

    res["n_matched"] = len(matched)
    res["n_duplicate"] = len(dropped)
    res["duplicate_cells"] = sorted({plate.cell_name(*rc) for rc, _ in dropped})
    res["unmatched_seeds"] = int(len(P) - len(matched))
    res["empty_cells"] = sorted(plate.cell_name(*rc)
                                for rc in kernels - {rc for rc, _ in matched})
    res["rms_px"] = rms
    res["max_resid_px"] = float(resid.max()) if len(resid) else 0.0
    res["n_inliers"] = int(keep.sum())

    e_col, e_row = frame[1], frame[2]
    px, py = float(np.hypot(*e_col)), float(np.hypot(*e_row))
    cross = float(e_col[0] * e_row[1] - e_col[1] * e_row[0])
    res["geometry"] = {
        "pitch_x_px": px, "pitch_y_px": py, "ratio": py / max(px, 1e-9),
        "angle_deg": float(np.degrees(np.arctan2(e_col[1], e_col[0])) % 360.0),
        "chirality": "left" if cross < 0 else "right",
        "skew_deg": float(np.degrees(np.arccos(np.clip(
            abs(e_col @ e_row) / (px * py), -1, 1))) - 90.0),
    }

    res["marker_offsets_px"] = {}
    for mid, m in markers.items():
        r, c = plate.MARKER_CELL[mid]
        d = float(np.hypot(*(np.asarray(m["center"], float) - cell_center(frame, r, c))))
        res["marker_offsets_px"][str(mid)] = d

    res["frame"] = frame
    assess(res)
    return res


def _side_from_ids(ids):
    sides = {plate.ID_SIDE[i] for i in ids if i in plate.ID_SIDE}
    return sides.pop() if len(sides) == 1 else None


def side_notes(ids, side_observed, side_hint):
    """What the codes say about which face this is, against what the folder says.

    A capture filed under the wrong side is the error that survives everything
    else: the cube is fine, the grid is fine, the cell names are fine, and every
    spectrum is attributed to the opposite face of the kernel. The plate answers
    it directly -- ids 0/1 are only ever moulded on the dorsal face and 2/3 only
    on the ventral -- so this is a read, not an inference.
    """
    if len(ids) > 1 and side_observed is None:
        return [f"fiducial ids {ids} span both faces of the plate"]
    if side_hint and side_observed and side_observed != side_hint:
        return [f"fiducial ids {ids} say {side_observed}, but this capture is "
                f"filed as {side_hint}"]
    return []


def assess(res):
    """Set status / reasons / flags on a fitted result, in place.

    Separate from `fit` and idempotent, because evidence can arrive after the
    lattice is placed: re-reading the fiducials at their predicted cells needs a
    lattice to predict from, and what it finds bears on whether to trust that
    lattice. Callers re-run this once that read is in `res["confirmed_cells"]`.
    """
    reasons, flags, info = [], list(res.get("side_flags", [])), []
    res["status"] = "fail"
    if res.get("frame") is None and not res.get("geometry"):
        res["reasons"] = res.get("reasons", [])
        res["flags"], res["info"] = flags, info
        return res
    g = res["geometry"]
    pose, rms = res["pose"], res["rms_px"]
    confirmed = int(res.get("confirmed_cells", 0))

    if pose["margin"] < POSE_MARGIN_FAIL:
        reasons.append(f"pose margin {pose['margin']:.1f} over the runner-up "
                       f"labelling -- the evidence does not pick one")
    dup_frac = res["n_duplicate"] / max(res["n_matched"], 1)
    if dup_frac > MAX_DUPLICATE_FRAC:
        reasons.append(f"{res['n_duplicate']} of {res['n_matched']} wells doubled up "
                       f"on a cell -- the lattice pitch is wrong, not the detection")
    if rms > RMS_REVIEW:
        reasons.append(f"well-snap rms {rms:.1f} px > {RMS_REVIEW}")
    if not (plate.PITCH_X_RANGE[0] <= g["pitch_x_px"] <= plate.PITCH_X_RANGE[1]):
        reasons.append(f"pitch_x {g['pitch_x_px']:.1f} px out of band")
    if not (plate.PITCH_Y_RANGE[0] <= g["pitch_y_px"] <= plate.PITCH_Y_RANGE[1]):
        reasons.append(f"pitch_y {g['pitch_y_px']:.1f} px out of band")
    if not (plate.PITCH_RATIO_RANGE[0] <= g["ratio"] <= plate.PITCH_RATIO_RANGE[1]):
        reasons.append(f"pitch ratio {g['ratio']:.2f} out of band")
    bad_markers = [m for m, d in res["marker_offsets_px"].items() if d > MARKER_OFFSET_TOL]
    if len(bad_markers) >= 2:
        reasons.append("both fiducials sit a cell or more from where the lattice puts them")
    if reasons:
        res["reasons"], res["flags"], res["info"] = reasons, flags, info
        res["frame"] = None
        return res

    if rms > RMS_PASS:
        flags.append(f"loose fit (rms {rms:.1f} px)")
    if res["n_matched"] < MIN_MATCHED_PASS:
        flags.append(f"only {res['n_matched']}/{plate.N_KERNEL_CELLS} wells matched")
    if confirmed < 2:
        if not res["marker_ids"]:
            flags.append("no fiducial identified -- the labelling rests on well "
                         "occupancy alone")
        elif len(res["marker_ids"]) < 2:
            flags.append(f"only fiducial id{res['marker_ids'][0]} identified")
        # A thin pose margin is only worrying while the fiducials are unaccounted
        # for. Once BOTH marker cells have been read and hold the code the model
        # expects, the labelling is pinned by the codes themselves and the
        # occupancy margin is no longer what is carrying it.
        if pose["margin"] < POSE_MARGIN_PASS:
            flags.append(f"thin pose margin ({pose['margin']:.1f})")
    if bad_markers:
        mid = bad_markers[0]
        flags.append(f"fiducial id{mid} sits {res['marker_offsets_px'][mid]:.0f} px "
                     f"from its cell centre")
    # Informational only. A second dark blob on a cell is over-detection, not a
    # threat to the labelling -- a germinating kernel throws a shoot that
    # thresholds out on its own, which is why these cluster on the later days.
    # The lattice never used it, so the indexing is unaffected and the capture
    # does not need a human. Pervasive doubling already failed above.
    if res["n_duplicate"]:
        info.append(f"{res['n_duplicate']} extra blob(s) on an already-matched cell "
                    f"({', '.join(res['duplicate_cells'])}) -- kept the nearer one")
    if res["unmatched_seeds"] - res["n_duplicate"] > 0:
        info.append(f"{res['unmatched_seeds'] - res['n_duplicate']} detected well(s) "
                    f"off the plate model")
    res["reasons"], res["flags"], res["info"] = reasons, flags, info
    res["status"] = "needs_review" if flags else "pass"
    return res
