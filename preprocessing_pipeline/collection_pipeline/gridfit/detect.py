"""Find the two things the lattice fit needs: well seeds and ArUco fiducials.

Both are deliberately detected WITHOUT any prior on where the plate is. The fit
then has two independent pieces of evidence to reconcile -- a cloud of well
centres and up to two identified fiducials -- and can refuse when they disagree.
That is the only defence against the failure that matters here: a lattice that
is off by one cell is still a clean-looking grid, it just silently attributes
every spectrum to the wrong kernel, and nothing downstream can notice.

Seeds
    reflectance   dark wells on a bright plate.
    transmittance railed wells on the clipped-fraction map -- the open beam
                  through an empty well saturates nearly every band.
    Both are found by a sweep of thresholds rather than one, because the right
    cut depends on the dish, and both are filtered on size, aspect and fill so a
    fiducial square (roughly 1:1) and a pair of merged wells (roughly 2:1 the
    other way) cannot pose as a well (roughly 1:2).

Fiducials
    cv2.aruco's own bit decode fails on this camera's push-broom blur, so
    candidate squares are found by thresholding for dark, and identified by
    normalized cross-correlation against the four reference patterns over all
    four quarter-turns. A 4x4 bit read of the warped square runs alongside it:
    when the contrast is good it matches a reference exactly, which is far
    stronger evidence than any correlation score, and it is what established
    that the ventral face carries ids 2 and 3 rather than mirrored 0 and 1.
"""
import cv2
import numpy as np

from . import plate

# ------------------------------------------------------------------- seeds --
# Bounds as fractions of the nominal cell, so they follow the pitch rather than
# being px constants tied to one rig setting.
SEED_SHORT_FRAC = (0.40, 1.10)      # short side / pitch_x
SEED_LONG_FRAC = (0.50, 1.10)       # long side  / pitch_y
SEED_ASPECT = (1.30, 3.20)          # long/short; a fiducial is ~1.05, a well ~2.0
SEED_AREA_FRAC = (0.22, 1.05)       # area / (pitch_x * pitch_y)
SEED_FILL = 0.62                    # area / minAreaRect area; a well is solid
SEED_DEDUPE_PX = 26.0

REFL_DARK_PCTS = (16.0, 20.0, 24.0, 28.0, 32.0)
TRANS_CLIP_THRS = (0.35, 0.50, 0.65)

# ---------------------------------------------------------------- fiducials --
REF_SIDE = 240
MARK_SIDE_FRAC = (0.55, 1.35)       # fiducial side / pitch_x
MARK_SQUARENESS = 0.72
MARK_MIN_INTERIOR_STD = 8.0
MARK_MIN_SCORE = 0.30               # NCC floor, unless the bit read matches exactly
MARK_DEDUPE_PX = 30.0
MARK_DARK_PCTS = (14.0, 20.0, 26.0, 32.0, 38.0)

# Where a fiducial sits inside its cell, in cell units from the cell centre, and
# how big it is as a fraction of the column pitch. A plate constant: measured
# over 22 captures spanning both modes, both faces and all three days, the
# spread is +-0.01 of a cell (under a pixel) and the square is axis-aligned with
# the lattice. That is what makes the predicted read below possible.
MARKER_IN_CELL = {(3, 3): (0.128, -0.093), (2, 6): (0.181, -0.100)}
MARKER_SIDE_FRAC = 0.79
PREDICT_SCALES = (0.72, 0.79, 0.86, 0.93)
PREDICT_NUDGE = (-0.04, 0.0, 0.04)
# The predicted read is judged on how far ahead the winning code is, not on an
# absolute correlation. A transmittance fiducial can be blurred down to a 0.26
# correlation and still name its code by a clear 0.2 over the runner-up, which
# is the question actually being asked -- which of four patterns is this.
PREDICT_MIN_MARGIN = 0.08
PREDICT_MIN_SCORE = 0.15


# ==================================================================== seeds ==
def _contour_blobs(mask255):
    """Solid blobs of a binary mask -> [(cx, cy, area, short, long, fill)]."""
    m = cv2.morphologyEx(mask255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cn in cnts:
        area = float(cv2.contourArea(cn))
        if area < 200:
            continue
        (_, _), (w, h), _ = cv2.minAreaRect(cn)
        short, long_ = (w, h) if w <= h else (h, w)
        if short < 1 or long_ < 1:
            continue
        mo = cv2.moments(cn)
        if mo["m00"] <= 0:
            continue
        out.append((mo["m10"] / mo["m00"], mo["m01"] / mo["m00"],
                    area, short, long_, area / (short * long_)))
    return out


def _keep_well(b, px, py):
    _, _, area, short, long_, fill = b
    return (SEED_SHORT_FRAC[0] * px <= short <= SEED_SHORT_FRAC[1] * px
            and SEED_LONG_FRAC[0] * py <= long_ <= SEED_LONG_FRAC[1] * py
            and SEED_ASPECT[0] <= long_ / short <= SEED_ASPECT[1]
            and SEED_AREA_FRAC[0] * px * py <= area <= SEED_AREA_FRAC[1] * px * py
            and fill >= SEED_FILL)


def _dedupe(items, radius):
    """Keep the first of any group of near-coincident detections.

    The threshold sweep proposes the same well several times; which threshold
    found it says nothing about quality, so the ordering the caller supplies
    decides (best-scoring first for fiducials, largest first for wells).
    """
    kept = []
    for it in items:
        c = np.asarray(it["center"], float)
        if any(np.hypot(*(c - np.asarray(k["center"], float))) < radius for k in kept):
            continue
        kept.append(it)
    return kept


def find_seeds(gray, clip, mode, px=plate.PITCH_X_NOM, py=plate.PITCH_Y_NOM):
    """Well centres in the working frame -> [{"center", "area", "aspect"}].

    Not required to find all 22. A missing well costs the pose search a little
    margin and nothing else; the cell geometry comes from the fitted lattice, not
    from the blob that seeded it.
    """
    found = []
    if mode == "transmittance":
        if clip is None:
            raise ValueError("transmittance seeds need the clipped-fraction map")
        for thr in TRANS_CLIP_THRS:
            found += _contour_blobs((clip >= thr).astype(np.uint8) * 255)
    else:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        for pct in REFL_DARK_PCTS:
            found += _contour_blobs((blur < np.percentile(gray, pct)).astype(np.uint8) * 255)

    wells = [{"center": [b[0], b[1]], "area": b[2], "aspect": b[4] / b[3]}
             for b in found if _keep_well(b, px, py)]
    wells.sort(key=lambda w: -w["area"])
    return _dedupe(wells, SEED_DEDUPE_PX)


# ================================================================ fiducials ==
def _references():
    """{(id, mirrored): float32 REF_SIDExREF_SIDE} for the plate's four codes.

    Inverted, because the physical fiducials are moulded inverted -- correlation
    against the plain render is negative across the board and against the
    inverted one is a clean 0.5-0.9. The mirrored copies are scored too: they
    should never win, and a capture where they do means the plate was imaged
    through something that flips it, which is worth being told about rather than
    silently accommodating.
    """
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, plate.ARUCO_DICT_NAME))
    refs = {}
    for mid in sorted(plate.MARKER_CELL):
        img = 255.0 - cv2.aruco.generateImageMarker(d, mid, REF_SIDE).astype(np.float32)
        refs[(mid, False)] = img
        refs[(mid, True)] = img[:, ::-1].copy()
    return refs


def _bit_grid(square):
    """Mean of the inner half of each cell of the 6x6 ArUco layout."""
    cell = square.shape[0] / 6.0
    g = np.zeros((6, 6), np.float32)
    for r in range(6):
        for c in range(6):
            y0, y1 = int((r + 0.25) * cell), int((r + 0.75) * cell)
            x0, x1 = int((c + 0.25) * cell), int((c + 0.75) * cell)
            g[r, c] = square[y0:y1, x0:x1].mean()
    return g


def _canonical_bits(grid):
    """Rotation-invariant bit string: the smallest of the four quarter-turns."""
    b = grid < (float(grid.min()) + float(grid.max())) / 2.0
    return min("".join("1" if v else "0" for v in np.rot90(b, k).ravel()) for k in range(4))


_BIT_TABLE = None


def _bit_table():
    global _BIT_TABLE
    if _BIT_TABLE is None:
        _BIT_TABLE = {_canonical_bits(_bit_grid(ref)): k for k, ref in _references().items()}
    return _BIT_TABLE


def _ncc(square, ref):
    """Zero-mean normalized correlation, best over the four quarter-turns."""
    w = square.astype(np.float32)
    w = (w - w.mean()) / (w.std() + 1e-6)
    best = -1.0
    for k in range(4):
        r = np.rot90(ref, k)
        r = (r - r.mean()) / (r.std() + 1e-6)
        best = max(best, float((w * r).mean()))
    return best


def _marker_candidates(gray, px):
    """Dark, square, internally-patterned quads (4,2) -- decoded or not.

    Contrast-equalised first: on a transmittance capture a fiducial sits in the
    dish's own shadow, several stops below the railed wells beside it, and a
    global threshold that separates it there does not separate the other one.
    """
    eq = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    lo, hi = MARK_SIDE_FRAC[0] * px, MARK_SIDE_FRAC[1] * px
    out = []
    for src in (gray, eq):
        blur = cv2.GaussianBlur(src, (0, 0), 1.5)
        for pct in MARK_DARK_PCTS:
            dark = (blur < np.percentile(src, pct)).astype(np.uint8) * 255
            dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cn in cnts:
                rect = cv2.minAreaRect(cn)
                (_, _), (w, h), _ = rect
                if not (lo <= w <= hi and lo <= h <= hi):
                    continue
                if min(w, h) / max(w, h) < MARK_SQUARENESS:
                    continue
                q = cv2.boxPoints(rect).astype(np.float32)
                x0, y0 = np.maximum(q.min(axis=0), 0).astype(int)
                x1, y1 = q.max(axis=0).astype(int)
                inner = gray[y0 + 4:y1 - 4, x0 + 4:x1 - 4]
                if inner.size < 64 or inner.std() < MARK_MIN_INTERIOR_STD:
                    continue
                out.append(q)
    return out


def _identify(grayf, quad, refs, table):
    """Score one candidate square against all four codes. -> record dict."""
    dst = np.array([[0, 0], [REF_SIDE, 0], [REF_SIDE, REF_SIDE], [0, REF_SIDE]], np.float32)
    best = {k: -1.0 for k in refs}
    bits = None
    for shift in range(4):     # a contour's first corner is not the marker's
        M = cv2.getPerspectiveTransform(np.roll(quad, shift, axis=0), dst)
        warped = cv2.warpPerspective(grayf, M, (REF_SIDE, REF_SIDE))
        if bits is None:
            bits = _canonical_bits(_bit_grid(warped))
        for k, ref in refs.items():
            best[k] = max(best[k], _ncc(warped, ref))
    rank = sorted(best.items(), key=lambda kv: -kv[1])
    (mid, mirrored), score = rank[0]
    hit = table.get(bits)
    rec = {"center": quad.mean(axis=0).tolist(), "corners": quad.tolist(),
           "ncc_id": int(mid), "mirrored": bool(mirrored),
           "score": float(score), "margin": float(score - rank[1][1]),
           "bit_id": None if hit is None else int(hit[0]),
           "bit_mirrored": None if hit is None else bool(hit[1])}
    rec["id"] = rec["bit_id"] if rec["bit_id"] is not None else rec["ncc_id"]
    rec["exact"] = rec["bit_id"] is not None
    return rec


def one_side_only(found):
    """Drop identifications from the face that is not showing.

    A plate face carries exactly two codes -- {0,1} or {2,3} -- so a detection
    from the other pair is a false positive by construction. In a reflectance
    frame the usual culprit is one of the two printed checkerboard targets: dark,
    square and patterned, which is the whole candidate test. Weighing the two
    faces against each other and keeping the winner removes them without a
    hand-tuned score floor that would also throw away a genuine dim fiducial.

    An exact bit read counts for far more than any correlation: it is a
    bit-for-bit match to one of 50 dictionary patterns, which blur does not
    produce by accident.
    """
    weight = {}
    for m in found:
        side = plate.ID_SIDE.get(m["id"])
        if side:
            weight[side] = weight.get(side, 0.0) + (10.0 if m["exact"] else m["score"])
    if len(weight) < 2:
        return found, None
    win = max(weight, key=weight.get)
    return [m for m in found if plate.ID_SIDE.get(m["id"]) == win], weight


def find_markers(gray, px=plate.PITCH_X_NOM):
    """{id: {...}} for the fiducials identified, best candidate per id.

    Each entry carries `score` (NCC), `margin` over the runner-up id, `mirrored`
    (the winning reference's polarity) and `bit_id` (the exact 4x4 read, or
    None). `bit_id` overrides the correlation when the two disagree: ids 0 and 1
    differ in few bits and their NCC margin has been measured as low as 0.02,
    while a bit read either matches a dictionary pattern exactly or does not.
    """
    refs = _references()
    table = _bit_table()
    grayf = gray.astype(np.float32)
    scored = [_identify(grayf, q, refs, table) for q in _marker_candidates(gray, px)]

    # exact bit reads first, then correlation strength
    scored.sort(key=lambda m: (not m["exact"], -m["score"]))
    scored = _dedupe(scored, MARK_DEDUPE_PX)
    scored = [m for m in scored if m["exact"] or m["score"] >= MARK_MIN_SCORE]
    scored, _ = one_side_only(scored)

    out = {}
    for m in scored:
        out.setdefault(m["id"], m)
    return out


def read_marker_cells(gray, cell_quads, px=plate.PITCH_X_NOM):
    """Re-read the fiducials once the lattice says exactly where they are.

    The blind pass has to find a dark patterned square anywhere in a frame that
    also contains checkerboards, a dish rim and 22 wells. This one is handed a
    ~74x134 px window and asked only what is inside it, so it can equalise
    locally -- which is what the transmittance fiducials need, sitting as they do
    in the dish's own shadow several stops under the railed wells beside them.

    cell_quads: {(row, col): (4,2) corners}. -> {(row, col): record}, the record
    carrying the same fields find_markers returns, plus `center` in full-frame
    coordinates.
    """
    refs = _references()
    table = _bit_table()
    out = {}
    for rc, quad in cell_quads.items():
        q = np.asarray(quad, float)
        x0 = int(max(np.floor(q[:, 0].min()) - 8, 0))
        y0 = int(max(np.floor(q[:, 1].min()) - 8, 0))
        x1 = int(min(np.ceil(q[:, 0].max()) + 8, gray.shape[1]))
        y1 = int(min(np.ceil(q[:, 1].max()) + 8, gray.shape[0]))
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue
        win = gray[y0:y1, x0:x1]
        eqw = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(win)
        wf = win.astype(np.float32)
        centre = np.array([(x1 - x0) / 2.0, (y1 - y0) / 2.0])

        best = None
        for cand in _marker_candidates(eqw, px):
            off = float(np.hypot(*(cand.mean(axis=0) - centre)))
            if off > 0.45 * px + 0.25 * plate.PITCH_Y_NOM:
                continue
            rec = _identify(wf, cand, refs, table)
            rank = (not rec["exact"], -rec["score"])
            if best is None or rank < best[0]:
                best = (rank, rec)
        if best is None:
            continue
        rec = best[1]
        if not rec["exact"] and rec["score"] < MARK_MIN_SCORE:
            continue
        rec["center"] = [rec["center"][0] + x0, rec["center"][1] + y0]
        rec["corners"] = [[x + x0, y + y0] for x, y in rec["corners"]]
        rec["from_cell"] = [int(rc[0]), int(rc[1])]
        rec["source"] = "found"
        out[rc] = rec
    return out


def read_predicted_cells(gray, frame, want=None):
    """Read the fiducials at the exact square the lattice says they occupy.

    The last resort, for the fiducial that is there but cannot be segmented: in
    transmittance the R2C6 code sits in the dish's shadow with its black border
    touching the unlit area beside it, so no threshold cuts a square out of it,
    even though the pattern is plainly visible. Since the fiducial's placement
    inside its cell is a plate constant (MARKER_IN_CELL), the square can simply
    be constructed and warped -- no segmentation at all -- and a small offset and
    scale search absorbs the residual.

    Two rules keep this from confirming whatever it is pointed at:

      * Mirrored references are excluded. Both faces of this plate read in normal
        polarity, so a mirrored win is a misread; with all eight references in
        play a blurred fiducial does sometimes prefer one, at a score that means
        nothing.
      * Acceptance is on the MARGIN over the runner-up code, not on an absolute
        correlation, because the question is which of four patterns this is.

    This is not circular evidence for the pose. A kernel well holds a kernel, not
    an ArUco pattern, so if the pose were flipped this would be reading a well
    and would return nothing -- a confident read at a predicted marker cell is
    itself a reason to believe the cell is where the lattice put it. It is not
    evidence about the GEOMETRY though: the square came from the lattice, so its
    position must never be fed back as a cross-check on the lattice.
    """
    refs = {k: v for k, v in _references().items() if not k[1]}
    table = {b: k for b, k in _bit_table().items() if not k[1]}
    origin, e_col, e_row = frame
    px = float(np.hypot(*e_col))
    uc = e_col / px
    ur = np.array([-uc[1], uc[0]])
    dst = np.array([[0, 0], [REF_SIDE, 0], [REF_SIDE, REF_SIDE], [0, REF_SIDE]], np.float32)
    grayf = gray.astype(np.float32)

    out = {}
    for rc, (dc, dr) in MARKER_IN_CELL.items():
        if want is not None and rc not in want:
            continue
        base = origin + rc[1] * e_col + rc[0] * e_row
        best = None
        for scale in PREDICT_SCALES:
            for ndc in PREDICT_NUDGE:
                for ndr in PREDICT_NUDGE:
                    ctr = base + (dc + ndc) * e_col + (dr + ndr) * e_row
                    h = 0.5 * scale * px
                    quad = np.array([ctr - h * uc - h * ur, ctr + h * uc - h * ur,
                                     ctr + h * uc + h * ur, ctr - h * uc + h * ur],
                                    np.float32)
                    warped = cv2.warpPerspective(
                        grayf, cv2.getPerspectiveTransform(quad, dst), (REF_SIDE, REF_SIDE))
                    rank = sorted(((_ncc(warped, r), k[0]) for k, r in refs.items()),
                                  reverse=True)
                    if best is None or rank[0][0] > best[0][0]:
                        best = (rank[0], rank[1], scale, quad,
                                table.get(_canonical_bits(_bit_grid(warped))))
        if best is None:
            continue
        (score, mid), (second, _), scale, quad, hit = best
        margin = score - second
        exact = hit is not None and hit[0] == mid
        if not exact and (margin < PREDICT_MIN_MARGIN or score < PREDICT_MIN_SCORE):
            continue
        out[rc] = {"center": quad.mean(axis=0).tolist(), "corners": quad.tolist(),
                   "id": int(mid), "ncc_id": int(mid), "mirrored": False,
                   "score": float(score), "margin": float(margin),
                   "bit_id": None if hit is None else int(hit[0]), "exact": exact,
                   "scale": float(scale), "from_cell": [int(rc[0]), int(rc[1])],
                   "source": "predicted"}
    return out
