"""ArUco-anchored well grid for a corrected TRANSMISSION cube.

Same method as the reflectance pipeline -- markers anchor the cell indexing so a
well means the same physical well on every capture -- but standalone, and using
transmission's own strengths for the two detection steps.

    python3 fit_wells.py day3_dish0_trans

Why a cell index matters: the plate is a fixed rigid 4x7 grid and the two ArUco
markers sit in known cells (id0 at row 3 col 3, id1 at row 2 col 6). Anchor the
lattice to them and cell (row, col) addresses the same well whatever the plate's
rotation or where it landed on the stage. That is what makes a well trackable
across days.

Two things are done differently from the reflectance version, both because
transmission looks different, not because the method changed:

  * Marker candidates come from dark connected components, not from
    cv2.aruco's candidate finder. On this render cv2's finder proposes exactly
    ONE quad in the whole frame (measured), because the markers sit against
    railed wells ~4x brighter than they are and its adaptive threshold gives up.
    The markers ARE solid dark squares though, so thresholding for dark and
    filtering on squareness + interior structure finds both.

  * Well centres come from the SATURATION mask, not from a wall-frame template.
    In transmission an empty well is the brightest thing in the frame -- railed
    open beam straight through it -- so the clipped-voxel map hands the wells
    over directly. Reflectance needs a matched wall template because there it is
    a subtle bright-wall/dark-interior contrast. Here it is 22 obvious blobs.

Marker identification still needs cv2.aruco's dictionary but NOT its bit-decode,
which fails on push-broom blur. Instead each candidate is perspective-warped and
scored by normalized cross-correlation against the two reference patterns, all 4
quarter-turns, INVERTED -- the physical markers are printed inverted. Measured on
this capture: +0.50 against the inverted reference, -0.18 against the normal one,
so the polarity is not ambiguous.

The 4-fold orientation ambiguity that two markers leave (they fix the pitch and
the axis but not which way is "up") is resolved by trying all four frames and
keeping the one that lands the detected wells on lattice nodes. The lattice is
then refined by least squares on the well centres only -- the markers orient it
and cross-check it, they are not in the fit, because the ArUco pattern centre
sits a systematic ~15-20 px off its cell's geometric centre.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ---- rigid plate model (shared with the reflectance pipeline) ---------------
N_ROWS, N_COLS = 4, 7
MARKER_CELLS = {0: (3, 3), 1: (2, 6)}                # id -> (row, col)
RIM_CORNERS = {(0, 0), (0, 6), (3, 0), (3, 6)}       # clipped by the dish rim
N_KERNEL_CELLS = N_ROWS * N_COLS - len(RIM_CORNERS) - len(MARKER_CELLS)   # 22
# Pitch ratio py/px is a plate constant, so it survives any magnification change
# and is what lets two markers alone fix the lattice. Absolute pitch is measured.
PITCH_RATIO = 125.0 / 76.0
PITCH_X_RANGE = (55.0, 105.0)      # sanity band for the measured pitch, in px

# ---- marker detection ------------------------------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1)
REF_SIDE = 240
# "Dark" is taken as a percentile of the render rather than an absolute grey
# level, so a capture whose stretch lands differently still works. The band below
# spans grey ~60-110 on this capture; the markers separate cleanly from their
# railed neighbours anywhere in it, and below ~55 the marker merges into the dish
# shadow and stops being a square.
MARK_DARK_PCT = 30.0
MARK_DARK_MULTS = (1.0, 1.2, 1.4, 1.6, 1.8)
MARK_SIDE = (45.0, 110.0)   # marker side in px (6 ArUco cells incl. its border)
MARK_SQUARENESS = 0.78
MARK_MIN_INTERIOR_STD = 12.0   # a marker has pattern inside; a plain blob does not
MARK_MIN_SCORE = 0.30

# ---- well detection --------------------------------------------------------
WELL_CLIP_FRAC = 0.5        # a well pixel clips in at least this many kept bands
WELL_AREA = (1500, 14000)
WELL_W = (40, 110)
WELL_H = (55, 160)

# ---- acceptance gates ------------------------------------------------------
# A wrong grid is worse than no grid: it silently mislabels which physical well a
# spectrum came from, and nothing downstream can detect that. So the fit has to
# be able to fail. These three gates are independent, and measured on this
# capture they separate a good fit from a mis-indexed one by a wide margin --
# good: rms 3.5 px, 0 duplicate snaps, marker offsets 16/21 px; the same data with
# the two marker ids swapped: rms 42.8 px, 7 duplicates, offsets 55/53 px.
MIN_WELLS = 8               # below this the lattice starts drifting (measured)
RMS_PASS = 8.0              # px; well-snap rms for a clean pass
RMS_REVIEW = 15.0           # px; above this the geometry is not trustworthy
MAX_DUP_SNAPS = 0           # two wells claiming one cell == mis-indexing
# Coarse, not precise: the ArUco pattern centre sits a systematic 15-20 px off its
# cell's geometric centre, so this catches gross mis-indexing (>= ~1 pitch) only.
MARKER_OFFSET_TOL = 35.0


def _runs(ix):
    if len(ix) == 0:
        return "none"
    out, s, p = [], ix[0], ix[0]
    for i in ix[1:]:
        if i != p + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


# ------------------------------------------------------------------ renders --
def render(cube, masks):
    """(gray8, clip_frac) both indexed [scan, spatial], matching image convention.

    gray8 is the same log-stretched mean-band render process.py writes as the
    preview -- markers are a shape, so the display stretch is the right input.
    """
    kept = masks["band_kept"]
    bands = np.nonzero(kept)[0]
    acc = np.zeros(cube.shape[:2], np.float64)
    cnt = np.zeros(cube.shape[:2], np.int32)
    sat = np.zeros(cube.shape[:2], np.int32)
    for b in bands:
        pl = cube[:, :, b]
        m = np.isfinite(pl)
        acc[m] += pl[m]
        cnt[m] += 1
        sat += masks["saturated"][:, :, b]
    plane = np.full(cube.shape[:2], np.nan, np.float32)
    good = cnt > 0
    plane[good] = (acc[good] / cnt[good]).astype(np.float32)
    clip = (sat / max(len(bands), 1)).astype(np.float32)

    fin = np.isfinite(plane)
    v = np.log10(np.clip(plane, 1e-4, None))
    pop = fin & (clip <= 0.02)
    lo, hi = np.percentile(v[pop if pop.sum() > 1000 else fin], [0.5, 99.5])
    g = np.clip((v - lo) / (max(hi - lo, 1e-9)) * 255.0, 0, 255)
    g[~fin] = 0
    return g.astype(np.uint8).T, clip.T


# ------------------------------------------------------------------ markers --
def _reference_patterns():
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    # Inverted: the physical markers are printed inverted. Confirmed on this
    # capture -- +0.50 against these, -0.18 against the un-inverted render.
    return {m: 255.0 - cv2.aruco.generateImageMarker(d, m, REF_SIDE).astype(np.float32)
            for m in MARKER_IDS}


def _ncc(warped, ref):
    """Zero-mean normalized correlation, best over the 4 quarter-turns of ref."""
    w = warped.astype(np.float32)
    w = (w - w.mean()) / (w.std() + 1e-6)
    best = -1.0
    for k in range(4):
        r = np.rot90(ref, k)
        r = (r - r.mean()) / (r.std() + 1e-6)
        best = max(best, float((w * r).mean()))
    return best


def marker_candidates(gray):
    """Oriented square quads (4,2) that look like a marker: dark, square, patterned.

    Several dark thresholds are tried and all survivors kept -- the right cut
    depends on how bright the wells beside a given marker are, and scoring a few
    extra candidates costs nothing.
    """
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    out = []
    base = np.percentile(gray, MARK_DARK_PCT)
    for thr in (base * f for f in MARK_DARK_MULTS):
        dark = (blur < thr).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cn in cnts:
            (cx, cy), (w, h), _ = rect = cv2.minAreaRect(cn)
            if not (MARK_SIDE[0] <= w <= MARK_SIDE[1] and MARK_SIDE[0] <= h <= MARK_SIDE[1]):
                continue
            if min(w, h) / max(w, h) < MARK_SQUARENESS:
                continue
            quad = cv2.boxPoints(rect).astype(np.float32)
            x0, y0 = quad.min(axis=0).astype(int)
            x1, y1 = quad.max(axis=0).astype(int)
            inner = gray[max(y0, 0) + 4:y1 - 4, max(x0, 0) + 4:x1 - 4]
            if inner.size < 64 or inner.std() < MARK_MIN_INTERIOR_STD:
                continue
            out.append(quad)
    return out


def find_markers(gray):
    """{id: {"center", "corners", "score", "margin"}} for the markers found.

    Each candidate is warped to a square and scored against both references; the
    best-scoring candidate per id wins. Only the relative ranking is used to
    assign ids, so a small margin between id0 and id1 is tolerated -- the plate
    has exactly two markers, and the lattice fit re-checks the assignment.
    """
    refs = _reference_patterns()
    dst = np.array([[0, 0], [REF_SIDE, 0], [REF_SIDE, REF_SIDE], [0, REF_SIDE]], np.float32)
    best = {}
    for quad in marker_candidates(gray):
        scores = {m: -1.0 for m in MARKER_IDS}
        for shift in range(4):     # a contour's first corner is not the marker's TL
            M = cv2.getPerspectiveTransform(np.roll(quad, shift, axis=0), dst)
            warped = cv2.warpPerspective(gray.astype(np.float32), M, (REF_SIDE, REF_SIDE))
            for m, ref in refs.items():
                scores[m] = max(scores[m], _ncc(warped, ref))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        mid, score = ranked[0]
        if score < MARK_MIN_SCORE:
            continue
        if mid not in best or score > best[mid]["score"]:
            best[mid] = {"center": quad.mean(axis=0).tolist(),
                         "corners": quad.tolist(),
                         "score": score, "margin": score - ranked[1][1]}
    return best


# -------------------------------------------------------------------- wells --
def find_wells(clip):
    """Centres of the railed well blobs -> [(x, y), ...].

    An empty well passes the full beam, which is ~24x over the sensor's range at
    this exposure, so it rails in essentially every band. Kernels sit well inside
    range. That makes the clipped map a clean well detector -- and it is why this
    does not need reflectance's wall-frame template.
    """
    m = (clip >= WELL_CLIP_FRAC).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    wells = []
    for i in range(1, n):
        _, _, w, h, a = stats[i]      # left, top, width, height, area
        if not (WELL_AREA[0] <= a <= WELL_AREA[1]):
            continue
        if not (WELL_W[0] <= w <= WELL_W[1] and WELL_H[0] <= h <= WELL_H[1]):
            continue
        wells.append((float(cent[i][0]), float(cent[i][1])))
    return wells


# ------------------------------------------------------------------ lattice --
def _frame(origin, e_col, e_row):
    return {"origin": np.asarray(origin, float),
            "e_col": np.asarray(e_col, float), "e_row": np.asarray(e_row, float)}


def _node(fr, r, c):
    return fr["origin"] + c * fr["e_col"] + r * fr["e_row"]


def kernel_cells():
    return [(r, c) for r in range(N_ROWS) for c in range(N_COLS)
            if (r, c) not in RIM_CORNERS and (r, c) not in MARKER_CELLS.values()]


def frames_from_markers(m0, m1):
    """The 4 lattices consistent with two markers a known cell-vector apart.

    id0 is at (3,3) and id1 at (2,6), so the pixel vector between them is
    3*e_col - 1*e_row. With e_row perpendicular to e_col and |e_row|/|e_col|
    fixed by the rigid plate, that determines the pitch and the axis up to a
    180-degree flip and a handedness choice -- hence four candidates, not one.
    """
    d = np.asarray(m1, float) - np.asarray(m0, float)
    (r0, c0), (r1, c1) = MARKER_CELLS[0], MARKER_CELLS[1]
    dc, dr = c1 - c0, r1 - r0                       # +3, -1
    out = []
    for flip in (0.0, 180.0):
        for hand in (+1.0, -1.0):
            # d = px*(dc*u + dr*R*hand*u_perp); solve px and theta in complex form
            z = complex(dc, dr * PITCH_RATIO * hand)
            ang = np.degrees(np.angle(complex(d[0], d[1]) / z)) + flip
            px = float(np.hypot(*d) / abs(z))
            if not (PITCH_X_RANGE[0] <= px <= PITCH_X_RANGE[1]):
                continue
            t = np.radians(ang)
            e_col = px * np.array([np.cos(t), np.sin(t)])
            e_row = px * PITCH_RATIO * hand * np.array([-np.sin(t), np.cos(t)])
            origin = np.asarray(m0, float) - c0 * e_col - r0 * e_row
            out.append((_frame(origin, e_col, e_row), px, ang % 360.0, hand))
    return out


def _assign(fr, wells):
    """(total_sq_error, [(well_xy, (r, c)), ...]) snapping wells to kernel nodes."""
    cells = kernel_cells()
    nodes = np.array([_node(fr, r, c) for r, c in cells])
    pairs, tot = [], 0.0
    for w in wells:
        d = nodes - np.asarray(w)
        i = int(np.argmin((d ** 2).sum(axis=1)))
        tot += float((d[i] ** 2).sum())
        pairs.append((w, cells[i]))
    return tot, pairs


def fit_lattice(wells, markers, iters=3):
    """Best lattice over both id assignments, then gated. -> (frame|None, info).

    The NCC id margin can be thin -- DICT_4X4_50's ids 0 and 1 differ in few bits
    and push-broom blur eats the difference (measured margin as low as 0.020). So
    rather than trust it, BOTH assignments are fitted and the one the wells agree
    with wins. That is a far stronger discriminator: on this capture the correct
    assignment fits at 3.5 px rms and the swapped one at 42.8 px.

    info["status"] is "pass", "needs_review" or "fail". Callers must not use the
    frame on "fail" -- a mis-indexed grid mislabels which physical well a spectrum
    came from, and nothing downstream can notice.
    """
    if 0 not in markers or 1 not in markers:
        return None, {"status": "fail",
                      "reasons": [f"need both markers to orient the lattice; "
                                  f"found {sorted(markers)}"]}
    if len(wells) < MIN_WELLS:
        return None, {"status": "fail",
                      "reasons": [f"only {len(wells)} wells, need {MIN_WELLS}"]}

    tried = []
    for swap in (False, True):
        mk = {0: markers[1], 1: markers[0]} if swap else markers
        fr, info = _fit_one(wells, mk, iters)
        if fr is not None:
            info["ids_swapped"] = swap
            tried.append((info["rms_px"], fr, info))
    if not tried:
        return None, {"status": "fail", "reasons": ["no candidate lattice had a plausible pitch"]}
    tried.sort(key=lambda t: t[0])
    _, fr, info = tried[0]
    info["rms_runner_up_px"] = tried[1][0] if len(tried) > 1 else None

    reasons = []
    if info["rms_px"] > RMS_REVIEW:
        reasons.append(f"well-snap rms {info['rms_px']:.1f} px > {RMS_REVIEW}")
    if info["duplicate_snaps"] > MAX_DUP_SNAPS:
        reasons.append(f"{info['duplicate_snaps']} wells snapped to an already-claimed cell")
    for mid in (0, 1):
        off = info.get(f"marker{mid}_offset_px")
        if off is not None and off > MARKER_OFFSET_TOL:
            reasons.append(f"id{mid} sits {off:.1f} px from its cell centre "
                           f"> {MARKER_OFFSET_TOL} -- indexing is off")
    info["reasons"] = reasons
    info["status"] = ("fail" if reasons else
                      "needs_review" if info["rms_px"] > RMS_PASS else "pass")
    return (None if reasons else fr), info


def _fit_one(wells, markers, iters=3):
    """One lattice for one id assignment: orient from the markers, refine on wells.

    Markers orient the lattice and are then left out of the fit -- their ArUco
    pattern centre sits a systematic ~15-20 px off the cell's geometric centre, so
    including them would bias every cell. They are used again afterwards as an
    independent cross-check.
    """
    cands = frames_from_markers(markers[0]["center"], markers[1]["center"])
    if not cands:
        return None, {}

    scored = []
    for fr, px, ang, hand in cands:
        tot, pairs = _assign(fr, wells)
        scored.append((tot, fr, px, ang, hand, pairs))
    scored.sort(key=lambda t: t[0])
    tot, fr, px, ang, hand, pairs = scored[0]
    runner = scored[1][0] if len(scored) > 1 else float("inf")

    # Least squares for origin/e_col/e_row from the well snaps. Solved as two
    # independent linear problems (x and y), design row [1, c, r] per well.
    for _ in range(iters):
        _, pairs = _assign(fr, wells)
        seen = {}
        for w, rc in pairs:
            seen.setdefault(rc, []).append(w)       # a doubled node means a bad snap
        A = np.array([[1.0, c, r] for _, (r, c) in pairs])
        for k, axis in enumerate(("x", "y")):
            b = np.array([w[k] for w, _ in pairs])
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            if axis == "x":
                fr["origin"][0], fr["e_col"][0], fr["e_row"][0] = sol
            else:
                fr["origin"][1], fr["e_col"][1], fr["e_row"][1] = sol

    _, pairs = _assign(fr, wells)
    resid = [float(np.hypot(*(np.asarray(w) - _node(fr, *rc)))) for w, rc in pairs]
    dup = len(pairs) - len({rc for _, rc in pairs})
    info = {"px": float(np.hypot(*fr["e_col"])), "py": float(np.hypot(*fr["e_row"])),
            "angle_deg": float(np.degrees(np.arctan2(fr["e_col"][1], fr["e_col"][0])) % 360.0),
            "handedness": hand, "n_wells": len(wells),
            "rms_px": float(np.sqrt(np.mean(np.square(resid)))),
            "max_resid_px": float(max(resid)) if resid else 0.0,
            "duplicate_snaps": dup,
            "orientation_margin": float(runner / max(tot, 1e-9))}
    for mid, mk in markers.items():
        rc = MARKER_CELLS[mid]
        info[f"marker{mid}_offset_px"] = float(
            np.hypot(*(np.asarray(mk["center"]) - _node(fr, *rc))))
    return fr, info


# ------------------------------------------------------------------ outputs --
def cell_quad(fr, r, c):
    """The cell's 4 corners: its lattice rectangle, centre +/- half a pitch."""
    ctr = _node(fr, r, c)
    a, b = fr["e_col"] / 2.0, fr["e_row"] / 2.0
    return np.array([ctr - a - b, ctr + a - b, ctr + a + b, ctr - a + b])


def overlay(gray, fr, markers, wells, path):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            q = cell_quad(fr, r, c).astype(np.int32)
            if (r, c) in RIM_CORNERS:
                col, lab = (110, 110, 110), "rim"
            elif (r, c) in MARKER_CELLS.values():
                mid = [k for k, v in MARKER_CELLS.items() if v == (r, c)][0]
                col, lab = (0, 200, 255), f"id{mid}"
            else:
                col, lab = (0, 230, 0), f"{r}{c}"
            cv2.polylines(img, [q], True, col, 1, cv2.LINE_AA)
            ctr = _node(fr, r, c).astype(int)
            cv2.putText(img, lab, (ctr[0] - 12, ctr[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    for x, y in wells:
        cv2.drawMarker(img, (int(x), int(y)), (255, 90, 0), cv2.MARKER_CROSS, 9, 1)
    for mid, mk in markers.items():
        cv2.polylines(img, [np.array(mk["corners"], np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sample", help="e.g. day3_dish0_trans")
    p.add_argument("--in", dest="indir", default="corrected_file")
    p.add_argument("--out-img", default="corrected_image")
    args = p.parse_args()

    indir = Path(args.indir)
    cube = np.load(indir / f"{args.sample}.npy", mmap_mode="r")
    masks = np.load(indir / f"{args.sample}_masks.npz")
    print(f"Loading {args.sample}: cube {cube.shape}")
    gray, clip = render(cube, masks)
    print(f"  render {gray.shape} [scan, spatial]")

    print("Markers:")
    markers = find_markers(gray)
    for mid in sorted(markers):
        mk = markers[mid]
        print(f"  id{mid}: centre ({mk['center'][0]:.1f}, {mk['center'][1]:.1f}) "
              f"score {mk['score']:.3f} margin {mk['margin']:.3f}")
    missing = [m for m in MARKER_IDS if m not in markers]
    if missing:
        print(f"  WARNING: no candidate identified as {missing}")

    print("Wells:")
    wells = find_wells(clip)
    print(f"  {len(wells)} railed well blob(s) found; plate model expects {N_KERNEL_CELLS}")
    if len(wells) < 8:
        raise SystemExit("too few wells to fit a lattice.")

    print("Lattice:")
    fr, info = fit_lattice(wells, markers)
    if "px" in info:
        print(f"  pitch {info['px']:.2f} x {info['py']:.2f} px, angle {info['angle_deg']:.2f} deg")
        print(f"  well snap rms {info['rms_px']:.2f} px, worst {info['max_resid_px']:.2f} px, "
              f"{info['duplicate_snaps']} duplicate snap(s)")
        print(f"  orientation margin {info['orientation_margin']:.1f}x over the runner-up frame")
        if info.get("ids_swapped"):
            print(f"  NOTE: the wells preferred the SWAPPED marker ids -- NCC had them "
                  f"backwards. Fit rms {info['rms_px']:.2f} px vs "
                  f"{info['rms_runner_up_px']:.2f} px for the NCC order.")
        elif info.get("rms_runner_up_px"):
            print(f"  id assignment confirmed by geometry: {info['rms_px']:.2f} px vs "
                  f"{info['rms_runner_up_px']:.2f} px for the swapped order")
        for mid in sorted(markers):
            k = f"marker{mid}_offset_px"
            if k in info:
                print(f"  cross-check: id{mid} sits {info[k]:.1f} px from its cell centre "
                      f"(~15-20 px is expected -- the pattern is offset in the cell)")
    print(f"  status: {info['status']}")
    for r in info.get("reasons", []):
        print(f"    - {r}")
    if fr is None:
        raise SystemExit("refusing to write a grid that failed its checks -- a mis-indexed "
                         "grid mislabels which physical well each spectrum came from, and "
                         "nothing downstream can detect that.")
    if info["status"] == "needs_review":
        print("  WARNING: fit is loose. Check the overlay before trusting the cell indices.")

    cells = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            kind = ("rim" if (r, c) in RIM_CORNERS else
                    "marker" if (r, c) in MARKER_CELLS.values() else "kernel")
            cells.append({"row": r, "col": c, "kind": kind,
                          "center": _node(fr, r, c).tolist(),
                          "corners": cell_quad(fr, r, c).tolist()})
    out = {"sample": args.sample, "lattice": info,
           "markers": {str(k): v for k, v in markers.items()},
           "wells_detected": wells, "cells": cells,
           "axes": "x = spatial (0-639), y = scan; same as the preview PNG transposed"}
    jp = indir / f"{args.sample}_cells.json"
    jp.write_text(json.dumps(out, indent=2, default=float))
    print(f"Writing:\n  cells: {jp} ({len(cells)} cells, {N_KERNEL_CELLS} kernel)")
    op = Path(args.out_img) / f"{args.sample}_grid.png"
    overlay(gray, fr, markers, wells, op)
    print(f"  overlay: {op} (green = kernel cells, orange = marker cells, "
          f"grey = rim, red quad = detected marker, blue cross = detected well)")


if __name__ == "__main__":
    main()
