"""Deciding which gridfit well each hand-drawn mask belongs to.

This is the join that gives a mask a name -- `dish7_R2C6` -- that means the same
physical well on every capture, from either face of the plate, on any day. Get
it wrong by one cell and every spectrum is attributed to the wrong kernel while
every overlay still looks perfect, so the assignment is made by pixel overlap
against the fitted lattice and is required to be a bijection.

An audit of all 4381 day-1/day-9 masks found zero off-lattice masks, zero
straddling two cells and zero double-assignments, with a median containment of
1.000 and a minimum of 0.9915 (re-measured from the built index on 2026-08-20;
an earlier note said 0.981). Those are the tolerances encoded below.
"""
import json
import sys

import numpy as np

import config
from . import coco

sys.path.insert(0, str(config.COLLECTION))
from gridfit import cells as gcells  # noqa: E402

MIN_CONTAINMENT = 0.50      # a mask must sit at least this far inside one cell

MARKER_LABEL = 100
RIM_LABEL = 200


def load_cells(key):
    """cells.json for a capture, or None if grid_index.py never fitted it."""
    p = (config.GRID_VIEW / f"{key.mode}_images" / f"day{key.day}"
         / f"dish{key.dish}" / key.side / "cells.json")
    return json.loads(p.read_text()) if p.exists() else None


def cell_label_image(rec):
    """-> (label image in cube layout, {label: cell}) for one capture.

    Kernel cells take label `index + 1`; markers and rim corners take sentinel
    labels so a mask landing on one is reported as such rather than silently
    snapped to the nearest kernel.
    """
    n_lines, width = rec["capture"]["render_shape"]
    label = np.zeros((width, n_lines), np.int16)
    kernels = {}
    for c in rec["cells"]:
        m = gcells.cell_mask(c, width, n_lines)
        if c["kind"] == "kernel":
            lv = c["index"] + 1
            kernels[lv] = c
        else:
            lv = MARKER_LABEL if c["kind"] == "marker" else RIM_LABEL
        label[m] = lv
    return label, kernels


def assign_capture(cap, rec):
    """-> (rows, problems).

    rows      [{cell, ann, mask, containment, n_components_raw, dropped_px, mask_px}]
    problems  [(kind, detail)] -- anything that would make a row untrustworthy
    """
    problems = []
    n_lines, width = rec["capture"]["render_shape"]
    if (cap.height, cap.width) != (width, n_lines):
        problems.append(("SHAPE_MISMATCH",
                         f"png {cap.height}x{cap.width} vs cells {width}x{n_lines}"))
        return [], problems
    if rec["status"] == "fail":
        problems.append(("FIT_FAILED", "grid fit says do not use the cells"))
        return [], problems

    label, kernels = cell_label_image(rec)
    claimed, rows = {}, []

    for ann in cap.anns:
        mask = coco.decode_rle(ann.counts, ann.size)
        mask, n_comp, dropped = coco.keep_largest_component(mask)
        raw_px = int(mask.sum())
        if config.MASK_ERODE_PX:
            # Erosion can pinch a mask into two lobes, so take the largest
            # component again rather than carry a stray fragment forward.
            mask = coco.erode(mask, config.MASK_ERODE_PX)
            mask, _, _ = coco.keep_largest_component(mask)
        npx = int(mask.sum())
        if npx == 0:
            problems.append(("MASK_ERODED_AWAY" if raw_px else "EMPTY_MASK",
                             f"ann {ann.ann_id} ({raw_px} px before erosion)"))
            continue

        hist = np.bincount(label[mask], minlength=RIM_LABEL + 1)
        kernel_hist = np.zeros(max(kernels) + 1, np.int64)
        for lv in kernels:
            kernel_hist[lv] = hist[lv]
        if kernel_hist.sum() == 0:
            problems.append(("MASK_OFF_LATTICE",
                             f"ann {ann.ann_id} bg={hist[0] / npx:.2f} "
                             f"marker={hist[MARKER_LABEL] / npx:.2f} "
                             f"rim={hist[RIM_LABEL] / npx:.2f}"))
            continue
        best = int(kernel_hist.argmax())
        containment = float(kernel_hist[best]) / npx
        if containment < MIN_CONTAINMENT:
            problems.append(("MASK_STRADDLES_CELLS",
                             f"ann {ann.ann_id} best={kernels[best]['name']} "
                             f"containment={containment:.2f}"))
            continue
        if best in claimed:
            problems.append(("TWO_MASKS_ONE_CELL",
                             f"{kernels[best]['name']} <- anns "
                             f"{claimed[best]} and {ann.ann_id}"))
            continue
        claimed[best] = ann.ann_id

        rows.append({
            "cell": kernels[best], "ann": ann, "mask": mask,
            "containment": containment, "n_components_raw": n_comp,
            "dropped_px": dropped, "mask_px": npx, "mask_px_raw": raw_px,
        })

    missing = [c["name"] for lv, c in sorted(kernels.items()) if lv not in claimed]
    if missing:
        problems.append(("CELL_WITHOUT_MASK", ",".join(missing)))
    return rows, problems
