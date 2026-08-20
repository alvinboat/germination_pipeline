"""
Robustly fit the analytical 7x4 plate grid to a corrected reflectance cube.

Pipeline (see memory kernel-grid-task / plate-physical-layout):
  1. localize the two ArUco fiducials (localize_markers -- candidate+NCC, works
     on push-broom blur where cv2.aruco's decode fails). id0@R3C3, id1@R2C6.
  2. Radon angle: de-rotate, maximise row+col projection variance -> grid
     rotation, robust even on the ~45deg-rotated dish (716_6).
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
  5. cross-validate -> confidence -> status: pass / needs_review / fail, with a
     diagnostic overlay. The well-fit rms + inlier count gate the geometry; the
     marker cross-check (coarse, ~1-pitch tolerance) catches gross mis-indexing.
     Never silently emit a wrong grid.

Degrades gracefully: one marker still solvable (orientation picked by well
agreement, flagged needs_review); zero markers -> fail.

This still only *fits + overlays* the grid (no extraction yet).

Usage:
    python3 fit_grid.py                 # all default captures
    python3 fit_grid.py <cube.npy> ...  # specific cubes
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from process_image import intensity_preview  # noqa: E402
from localize_markers import localize_markers  # noqa: E402

PROD = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

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


def load_gray8(cube_path):
    cube = np.load(cube_path)
    return intensity_preview(cube).T  # (n_lines, width): rows=y=scan, cols=x=width


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
    o, ec, er = frame["origin"], frame["e_col"], frame["e_row"]
    cells = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            center = o + c * ec + r * er
            corners = [o + (c + dc) * ec + (r + dr) * er
                       for dc, dr in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))]
            kind = "clipped" if (r, c) in CLIPPED_CORNERS else "kernel"
            for mid, cell in MARKER_CELLS.items():
                if cell == (r, c):
                    kind = f"marker{mid}"
            cells.append({
                "row": r, "col": c, "kind": kind,
                "center_px": [float(center[0]), float(center[1])],
                "corners_px": [[float(x), float(y)] for x, y in corners],
            })
    return cells


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


# ------------------------------------------------------------------ overlay ---
STATUS_COLOR = {"pass": (0, 220, 0), "needs_review": (0, 200, 255), "fail": (0, 0, 255)}


def draw_overlay(gray8, cells, markers, frame, diag, status, reasons, scale=2):
    h, w = gray8.shape
    vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    color = {"kernel": (0, 255, 0), "clipped": (120, 120, 120),
             "marker0": (0, 0, 255), "marker1": (0, 165, 255)}
    for cell in cells:
        pts = (np.array(cell["corners_px"]) * scale).astype(np.int32)
        cv2.polylines(vis, [pts], True, color.get(cell["kind"], (255, 255, 0)), 2)
        cx, cy = np.array(cell["center_px"]) * scale
        cv2.putText(vis, f"R{cell['row']}C{cell['col']}", (int(cx) - 24, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"R{cell['row']}C{cell['col']}", (int(cx) - 24, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for mid, m in markers.items():
        mx, my = np.array(m["center"]) * scale
        cv2.drawMarker(vis, (int(mx), int(my)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    mres = " ".join(f"m{k}={v:.1f}" for k, v in diag["marker_resid"].items())
    line1 = (f"{status.upper()}  angle={frame['angle']:.1f} px={frame['px']:.1f} "
             f"py={frame['py']:.1f} well_inliers={diag['well_inliers']}/{diag['n_well']} "
             f"rms={diag['rms']:.2f} {mres}")
    line2 = ("; ".join(reasons)) if reasons else ""
    for i, txt in enumerate((line1, line2)):
        if not txt:
            continue
        y = 24 + i * 26
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_COLOR[status], 1, cv2.LINE_AA)
    return vis


def fit(cube_path, cube=None):
    """Fit the grid to one cube and return everything (no file writes).

    Pass a preloaded `cube` to avoid a redundant np.load (e.g. when the caller
    also needs the full cube for extraction). Returns a dict with keys: name,
    status, reasons, frame, cells, markers, diag, angle, gray8, n_markers.
    """
    name = Path(cube_path).stem
    if cube is None:
        cube = np.load(cube_path)
    gray8 = intensity_preview(cube).T
    gray_f = gray8.astype(np.float32)
    markers = localize_markers(gray8)
    n_markers = sum(1 for mid in (0, 1) if mid in markers)
    base = {"name": name, "markers": markers, "n_markers": n_markers, "gray8": gray8}

    if n_markers == 0:
        return {**base, "status": "fail", "reasons": ["no markers"],
                "frame": None, "cells": [], "diag": {}, "angle": None}

    angle = radon_angle(dark_mask(gray8))
    frame = orient_two_markers(markers, angle) if n_markers == 2 \
        else orient_one_marker(markers, angle, gray_f)
    rframe, diag = robust_fit(gray_f, frame, markers)
    status, reasons = assess(rframe, diag, n_markers)
    cells = cells_from_frame(rframe)
    return {**base, "status": status, "reasons": reasons, "frame": rframe,
            "cells": cells, "diag": diag, "angle": angle}


def process(cube_path):
    res = fit(cube_path)
    name, status, reasons = res["name"], res["status"], res["reasons"]
    if res["frame"] is None:
        print(f"[{name}] FAIL: {'; '.join(reasons)}")
        return res
    gray8, markers = res["gray8"], res["markers"]
    rframe, diag, cells, angle = res["frame"], res["diag"], res["cells"], res["angle"]

    rms_s = f"{diag['rms']:.2f}" if diag["rms"] is not None else "n/a"
    mres = " ".join(f"m{k}={v:.1f}" for k, v in diag["marker_resid"].items())
    print(f"[{name}] {status.upper():12s} radon={angle:+.1f} angle={rframe['angle']:.1f} "
          f"px={rframe['px']:.1f} py={rframe['py']:.1f} "
          f"well={diag['well_inliers']}/{diag['n_well']} rms={rms_s} {mres}"
          + (f"  <- {'; '.join(reasons)}" if reasons else ""))

    out_png = OUT_DIR / f"{name}_fitgrid_overlay.png"
    cv2.imwrite(str(out_png), draw_overlay(gray8, cells, markers, rframe, diag, status, reasons))
    out_json = OUT_DIR / f"{name}_fitgrid_cells.json"
    with open(out_json, "w") as f:
        json.dump({"status": status, "reasons": reasons, "angle": rframe["angle"],
                   "px": rframe["px"], "py": rframe["py"], "n_markers": res["n_markers"],
                   "well_inliers": diag["well_inliers"], "n_well": diag["n_well"],
                   "rms": diag["rms"], "marker_resid": diag["marker_resid"],
                   "rejected": diag["rejected"], "cells": cells}, f, indent=2)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cubes", nargs="*", help="Corrected .npy cubes (default: t10-t13 + 716_1..6).")
    args = p.parse_args()
    cubes = args.cubes
    if not cubes:
        cf = PROD / "corrected_file"
        cubes = [str(cf / f"7222026_ref_t{i}.npy") for i in (10, 11, 12, 13)]
        cubes += [str(cf / f"716_{i}.npy") for i in range(1, 7)]
    results = [process(cube) for cube in cubes]
    n_pass = sum(r["status"] == "pass" for r in results)
    n_rev = sum(r["status"] == "needs_review" for r in results)
    n_fail = sum(r["status"] == "fail" for r in results)
    print(f"\nsummary: {n_pass} pass, {n_rev} needs_review, {n_fail} fail (of {len(results)})")


if __name__ == "__main__":
    main()
