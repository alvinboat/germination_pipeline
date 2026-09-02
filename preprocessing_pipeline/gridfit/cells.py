"""The fit's product: one record per plate cell, and the mask that cuts it.

A cell record is what everything downstream keys on, so it carries the stable
name (`R2C6`) and kernel index alongside the pixel geometry. The name is the
same physical well on every day, both faces and both modes; the pixels are not.
"""
import cv2
import numpy as np

from . import lattice, plate, render


def cells_from_frame(frame):
    """All 28 lattice positions, row-major: the 24 plate cells plus the 4 rim
    corners, which are drawn for context and have no mask.

    `index` numbers the 22 kernel wells row-major and is None for everything
    else, so index i names the same well on every capture.
    """
    out = []
    for r in range(plate.N_ROWS):
        for c in range(plate.N_COLS):
            kind = plate.cell_kind(r, c)
            ctr = lattice.cell_center(frame, r, c)
            out.append({
                "name": plate.cell_name(r, c),
                "index": plate.CELL_INDEX.get((r, c)),
                "row": r, "col": c, "kind": kind,
                "marker_id": None,
                "center": [float(ctr[0]), float(ctr[1])],
                "corners": [[float(x), float(y)]
                            for x, y in lattice.cell_corners(frame, r, c)],
            })
    return out


def tag_marker_ids(cells, markers):
    """Record which fiducial id was actually seen in each marker cell."""
    for mid in markers:
        rc = plate.MARKER_CELL.get(mid)
        for cell in cells:
            if rc and (cell["row"], cell["col"]) == rc:
                cell["marker_id"] = int(mid)
    return cells


OCCUPANCY_INSET = 0.18      # keep the well wall out of the sample


def _interior(cell, shape, inset_frac=OCCUPANCY_INSET):
    ctr = np.asarray(cell["center"], float)
    pts = ctr + (np.asarray(cell["corners"], float) - ctr) * (1.0 - 2.0 * inset_frac)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.rint(pts).astype(np.int32)], 1)
    return m.astype(bool)


def measure_occupancy(gray, clip, mode, cells):
    """{cell name: fraction of the well interior that reads as kernel}.

    Not a segmentation -- a presence check, so a well whose kernel was lost
    between days shows up as a discontinuity rather than as 22 quietly
    mis-attributed spectra. An empty well reads near zero in both modes: in
    transmittance it rails across the band, so anything NOT railed is kernel; in
    reflectance it is the darkest thing on the plate, so a threshold pooled over
    all 22 interiors separates kernel from well floor.
    """
    kernels = [c for c in cells if c["kind"] == "kernel"]
    if not kernels:
        return {}
    shape = gray.shape
    masks = {c["name"]: _interior(c, shape) for c in kernels}

    if mode == "transmittance" and clip is not None:
        kernel_px = clip < 0.5
    else:
        pooled = np.concatenate([gray[m] for m in masks.values()])
        thr, _ = cv2.threshold(pooled.reshape(-1, 1).astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_px = gray > thr
    return {name: (float(kernel_px[m].mean()) if m.any() else None)
            for name, m in masks.items()}


def cell_mask(cell, width, n_lines, inset_frac=0.0):
    """Boolean (width, n_lines) mask of one cell, in CUBE orientation.

    Returned already converted out of the working frame, so `cube[mask]` gives
    (n_masked, bands) directly. At the default zero inset this is exactly the
    quad the overlay draws -- the overlay never promises a footprint the mask
    does not cut. A non-zero inset shrinks it towards the well interior, which
    is what a segmentation step wants so the bright wall stays out of its
    brightness sample.
    """
    ctr = np.asarray(cell["center"], float)
    pts = ctr + (np.asarray(cell["corners"], float) - ctr) * (1.0 - 2.0 * inset_frac)
    m = np.zeros((n_lines, width), np.uint8)
    cv2.fillPoly(m, [np.rint(pts).astype(np.int32)], 1)
    return render.to_cube_mask(m)
