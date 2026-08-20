"""ArUco-anchored kernel indexing for the collection.

One call per capture:

    from gridfit import fit_capture
    res = fit_capture("real_data/reflectance_images/day1/dish0/dorsal",
                      mode="reflectance", side="dorsal")
    if res["status"] != "fail":
        for cell in res["cells"]:
            ...                       # cell["name"] is the same well every day

`fit_capture` renders the cube (cached), detects the wells and the fiducials,
fits the plate lattice, and gates the result. See plate.py for what the plate is
and why a cell name is stable, and lattice.py for how the fit decides.
"""
from . import cells, detect, lattice, overlay, plate, render   # noqa: F401


def _face_filter(*groups):
    """Pool every fiducial reading and keep the face the evidence points at.

    A plate face carries exactly two codes -- {0,1} or {2,3} -- so any reading
    from the other pair is a misread, wherever it came from. Deciding this once
    over ALL the readings, rather than per pass, is what stops a weak
    wrong-face read in one pass from pre-empting a good read in the next.
    """
    pooled = [m for g in groups for m in g]
    if not pooled:
        return set()
    kept, _ = detect.one_side_only(pooled)
    return {m["id"] for m in kept}


def _reads(*groups):
    """(cell, record) pairs from several read passes, ordered by cell.

    Sorted on the cell alone: a cell can appear in more than one pass (a re-read
    that named the wrong code still leaves the cell open to the predicted pass),
    and the records are dicts, which have no ordering. Later groups win when a
    dict is built from the result, so pass the weaker evidence first.
    """
    return sorted((kv for g in groups for kv in g.items()), key=lambda kv: kv[0])


def fit_capture(capture_dir, mode, side=None, cache_dir=None):
    """Render, detect, fit and gate one capture. -> result dict.

    The fiducials are looked for three times, each pass a weaker claim than the
    one before, and no pass is allowed to overrule a stronger one:

    1. BLIND -- find a dark patterned square anywhere in a frame that also holds
       two checkerboards, a dish rim and 22 wells. Independent of everything;
       enough to place the lattice.
    2. RE-READ -- with the lattice placed, re-read just the two fiducial cells,
       where the detector can equalise locally instead of competing with the
       railed wells beside it. Still a segmentation, so its position remains an
       independent cross-check on the lattice. If it disagrees with the blind
       pass the lattice is re-fitted on it, because the pose depends on it.
    3. PREDICTED -- for a cell still unread, construct the square from the
       lattice and the plate's known fiducial placement. Confirms the CODE (and
       so the face); says nothing about the geometry, and is kept out of
       `markers` so it can never be mistaken for a geometric check.

    Adds to `lattice.fit`'s result: `gray`, `clip`, `seeds`, `markers`,
    `predicted_markers`, `cells` (empty when the fit failed -- a failed fit must
    not hand out cell geometry) and `marker_pass` recording what each pass saw.
    """
    gray, clip = render.load(capture_dir, mode, cache_dir=cache_dir)
    seeds = detect.find_seeds(gray, clip, mode)

    blind = detect.find_markers(gray)
    res = lattice.fit(seeds, blind, side_hint=side)
    markers, reread, predicted = blind, {}, {}

    if res["frame"] is not None:
        quads = {rc: lattice.cell_corners(res["frame"], *rc) for rc in plate.MARKER_CELLS}
        reread = detect.read_marker_cells(gray, quads)
        face = _face_filter(blind.values(), reread.values())
        reread = {rc: m for rc, m in reread.items() if m["id"] in face}
        wrong_cell = [rc for rc, m in reread.items() if plate.MARKER_CELL.get(m["id"]) != rc]
        if reread and (set(m["id"] for m in reread.values()) != set(blind) or wrong_cell):
            ordered = sorted(reread.values(), key=lambda m: (not m["exact"], -m["score"]))
            refit_markers = {}
            for m in ordered:
                refit_markers.setdefault(m["id"], m)
            again = lattice.fit(seeds, refit_markers, side_hint=side)
            if again["frame"] is not None:
                res, markers = again, refit_markers

    if res["frame"] is not None:
        missing = plate.MARKER_CELLS - {rc for rc, m in reread.items()
                                        if plate.MARKER_CELL.get(m["id"]) == rc}
        if missing:
            predicted = detect.read_predicted_cells(gray, res["frame"], want=missing)
            face = _face_filter(markers.values(), reread.values(), predicted.values())
            predicted = {rc: m for rc, m in predicted.items() if m["id"] in face}

    confirmed = {rc: m for rc, m in _reads(predicted, reread)
                 if plate.MARKER_CELL.get(m["id"]) == rc}
    if res["frame"] is not None:
        res["confirmed_cells"] = len(confirmed)
        ids = sorted({m["id"] for m in confirmed.values()} | set(markers))
        res["marker_ids"] = ids
        res["side_observed"] = lattice._side_from_ids(ids)
        res["side_flags"] = lattice.side_notes(ids, res["side_observed"], side)
        lattice.assess(res)

    res["marker_pass"] = {
        "blind_ids": sorted(blind),
        "cell_read": {plate.cell_name(*rc) + ("" if m["source"] == "found" else "~pred"):
                      {"id": m["id"], "exact": m["exact"],
                                             "score": round(m["score"], 3),
                                             "source": m["source"]}
                      for rc, m in _reads(reread, predicted)},
        "confirmed_cells": sorted(plate.cell_name(*rc) for rc in confirmed),
        "refit_on_cell_read": markers is not blind,
    }
    res["gray"] = gray
    res["clip"] = clip
    res["seeds"] = seeds
    res["markers"] = markers
    res["predicted_markers"] = predicted
    res["cells"] = (cells.tag_marker_ids(cells.cells_from_frame(res["frame"]),
                                         {m["id"]: m for m in confirmed.values()} or markers)
                    if res["frame"] is not None else [])
    return res
