"""
Locate the dish's two ArUco markers (ids 0 and 1, DICT_4X4_50) on a corrected
reflectance cube's greyscale render -- WITHOUT relying on cv2.aruco's own
bit-decode, which fails on this camera's push-broom-blurred captures (confirmed:
candidate squares are found fine, decode against every dictionary/param combo is
not).

Instead: take every quad cv2.aruco's detector considers a marker candidate (both
successfully-decoded AND rejected), filter to ones sized/shaped like an actual
marker, then identify each by normalized cross-correlation against the two known
reference patterns (inverted polarity -- these markers are 3D-printed inverted,
confirmed empirically: correlation against the normal render is negative across
the board, against the inverted render it's a clean 0.5-0.65). Best-scoring
candidate per id wins.

x = width/spatial axis, y = scan-line axis.

Vendored from reflectance_image_pipeline_production/grid_sandbox/localize_markers.py.
Library only: the standalone CLI and the process_image import were dropped so
process_image.py can import this without a circular import -- callers pass the
gray8 render in rather than this module loading a cube and rendering it itself.
Keep in sync with the sandbox original if that one is retuned.
"""
import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1)
REF_SIDE = 240

# Empirical marker size in this cube's pixel grid (~80x77px on a 640-wide frame)
# -- candidates must be within this band to be considered, or every grid-cell
# corner/kernel-blob quad in the frame becomes a "candidate" too.
MIN_SIDE, MAX_SIDE = 55, 110
MIN_SQUARENESS = 0.75  # min(w,h)/max(w,h) of the candidate's bounding box
MIN_SCORE = 0.35       # correlation floor below which we don't trust an ID at all
# A real, ground-truth-confirmed id0 match on grain_ref_exp_2500 scored only a 0.024
# margin over id1 (vs. 716_2's id0 margin of 0.060) -- 0.03 silently dropped a genuine
# detection. Keep this loose; MIN_SCORE is the real quality gate.
MIN_MARGIN = 0.015


def _reference_patterns():
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    refs = {}
    for mid in MARKER_IDS:
        img = cv2.aruco.generateImageMarker(d, mid, REF_SIDE).astype(np.float32)
        refs[mid] = 255.0 - img  # physical markers are printed inverted
    return refs


def find_marker_candidates(gray8):
    """All quads (4,2) cv2.aruco's detector treats as marker-shaped, decoded or not."""
    params = cv2.aruco.DetectorParameters()
    params.detectInvertedMarker = True
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    corners, ids, rejected = det.detectMarkers(gray8)
    quads = list(rejected)
    if corners:
        quads.extend(corners)

    out = []
    for q in quads:
        q = q.reshape(4, 2)
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        w, h = x1 - x0, y1 - y0
        if not (MIN_SIDE <= w <= MAX_SIDE and MIN_SIDE <= h <= MAX_SIDE):
            continue
        if min(w, h) / max(w, h) < MIN_SQUARENESS:
            continue
        out.append(q)
    return out


def _best_correlation(warped, ref):
    """Best zero-mean normalized correlation over the 4 quarter-turns of ref."""
    w = warped.astype(np.float32)
    w = (w - w.mean()) / (w.std() + 1e-6)
    best = -1.0
    for k in range(4):
        rk = np.rot90(ref, k)
        rk = (rk - rk.mean()) / (rk.std() + 1e-6)
        score = float((w * rk).mean())
        best = max(best, score)
    return best


def identify_candidate(gray8, quad, refs):
    """{id: score} for one candidate quad, trying all 4 corner-order rotations
    (findContours/detectMarkers starting corner isn't guaranteed to be the
    marker's own TL corner for a REJECTED candidate, so we can't assume it)."""
    dst = np.array([[0, 0], [REF_SIDE, 0], [REF_SIDE, REF_SIDE], [0, REF_SIDE]], dtype=np.float32)
    scores = {mid: -1.0 for mid in refs}
    for shift in range(4):
        ordered = np.roll(quad, shift, axis=0).astype(np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(gray8, M, (REF_SIDE, REF_SIDE))
        for mid, ref in refs.items():
            scores[mid] = max(scores[mid], _best_correlation(warped, ref))
    return scores


def localize_markers(gray8):
    """{id: {"corners": (4,2) TL,TR,BR,BL-ish, "center": (x,y), "score": float,
    "margin": float}} for whichever of MARKER_IDS clear the confidence bar."""
    refs = _reference_patterns()
    candidates = find_marker_candidates(gray8)

    per_id_best = {}
    for quad in candidates:
        scores = identify_candidate(gray8, quad, refs)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_id, best_score = ranked[0]
        margin = best_score - ranked[1][1]
        if best_score < MIN_SCORE or margin < MIN_MARGIN:
            continue
        prev = per_id_best.get(best_id)
        if prev is None or best_score > prev["score"]:
            per_id_best[best_id] = {
                "corners": quad.tolist(),
                "center": quad.mean(axis=0).tolist(),
                "score": best_score,
                "margin": margin,
            }
    return per_id_best
