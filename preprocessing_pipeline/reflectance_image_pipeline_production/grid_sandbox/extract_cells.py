"""
Stage 3: extract per-kernel data from a fitted plate grid.

For each PASS capture (fit_grid.fit -> status=="pass"), for each of the 22 kernel
cells, take the RAW cube crop (axis-aligned bbox of the rotated cell, exact
sensor spectra -- no resampling) and segment the grain seed with a floor-anchored,
brightness-based method (see kernel_seg):

  1. well mask -- the cell rectangle rasterised (modest inset).
  2. Otsu on in-well brightness -> bright (kernel + plastic) vs dark floor;
     floor = largest dark blob (the one wall/rim-free region in the scene).
  3. interior = the floor's convex hull, lightly dilated -- an image-MEASURED
     boundary that keeps the kernel sitting in the floor while excluding the
     wall/rim plastic outside it (plastic is bright AND spectrally kernel-like,
     so only geometry can remove it -- SAM cannot; see kernel_seg docstring).
  4. kernel = bright & interior, largest blob, holes filled, NEVER eroded so
     thin tips / germ survive.

This replaced a two-endmember hysteresis SAM (kernel_sam, retired): SAM is
magnitude-invariant, so it discarded the brightness cue that actually separates
kernel from the near-zero floor and leaked into noisy floor pixels ("drips").

Residual limit (flagged, not hidden): where the dish rim rides just inside the
top wall the floor hull can't bound it -- per-cell `floor_frac` /
`border_touch_frac` / `rim_suspect` surface those for review.

Output per capture: <name>_cells.npz (per cell: crop / kernel_mask / well_mask /
floor_mask / mean_spectrum / kernel_ref / floor_ref + metadata),
<name>_extract_meta.json, and <name>_extract_overlay.png (filled kernel masks)
for QC.

Usage:
    python3 extract_cells.py                 # all default captures
    python3 extract_cells.py <cube.npy> ...
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_grid as fg  # noqa: E402
import kernel_seg  # noqa: E402

PROD = fg.PROD
OUT_DIR = fg.OUT_DIR

# The well floor is dark; the kernel is a bright blob on it. A modest inset keeps
# the well polygon out of neighbour cells and the bulk of the wall out of the
# Otsu sample; the wall/rim that remains is excluded by the floor hull, not by
# insetting, so a kernel riding the wall is not clipped.
INSET_FRAC = 0.16          # inset each edge this fraction of pitch
CROP_DTYPE = np.float16    # stored crop dtype (segmentation runs in float32)

# Brightness segmentation (kernel_seg): complete kernel mask + rim diagnostics.
SEG_PARAMS = dict(min_contrast=1.6, border_px=4, hull_dilate=6)

# Residual-rim flags: a mask reaching well past the wall-free floor (a rim slash),
# heavily on the border ring, or in a well whose floor was largely eaten, is
# surfaced for review -- never hidden, never silently trimmed. Tuned so a genuine
# rim intrusion (e.g. R0C5) flags while a clean kernel poking past its floor does
# not (measured outside_floor: rim ~0.35-0.45 vs clean <=0.12).
RIM_OUTSIDE_FLOOR = 0.25   # mask fraction outside the floor hull above which -> suspect
RIM_BORDER_TOUCH = 0.30    # mask fraction on the border ring above which -> suspect
RIM_MIN_FLOOR_FRAC = 0.25  # floor/well fraction below which -> suspect


# ------------------------------------------------------------------ helpers ---
def cell_bbox(corners, W, H):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0 = max(0, int(np.floor(min(xs))))
    x1 = min(W, int(np.ceil(max(xs))))
    y0 = max(0, int(np.floor(min(ys))))
    y1 = min(H, int(np.ceil(max(ys))))
    return x0, y0, x1, y1


def well_mask_local(cell, x0, y0, w, h):
    """Rasterise the (slightly inset) cell rectangle into a local (h, w) mask."""
    center = np.array(cell["center_px"])
    corners = np.array(cell["corners_px"])
    inset = center + (corners - center) * (1.0 - 2.0 * INSET_FRAC)
    poly = np.round(inset - np.array([x0, y0])).astype(np.int32)  # (col,row) = (x-x0, y-y0)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)


def segment_kernel(sub_img, well_mask):
    """Floor-anchored brightness segmentation. Returns (seg, refs) where seg is a
    kernel_seg.KernelSeg and refs holds median kernel-body / floor spectra."""
    seg = kernel_seg.segment(sub_img, well_mask, **SEG_PARAMS)
    kern_ref = (np.median(sub_img[seg.mask], axis=0).astype(np.float32)
                if seg.mask.any() else np.full(sub_img.shape[2], np.nan, np.float32))
    floor_ref = (np.median(sub_img[seg.floor], axis=0).astype(np.float32)
                 if seg.floor.any() else np.full(sub_img.shape[2], np.nan, np.float32))
    return seg, {"kernel_ref": kern_ref, "floor_ref": floor_ref}


# ------------------------------------------------------------------ extract ---
def extract(cube, res):
    """Build per-kernel arrays + metadata for one PASS capture."""
    W, H = cube.shape[0], cube.shape[1]  # cube is (width=x, n_lines=y, bands)
    arrays, meta_cells = {}, []
    for cell in res["cells"]:
        if cell["kind"] != "kernel":
            continue
        r, c = cell["row"], cell["col"]
        tag = f"R{r}C{c}"
        x0, y0, x1, y1 = cell_bbox(cell["corners_px"], W, H)
        if x1 - x0 < 5 or y1 - y0 < 5:
            meta_cells.append({"cell": tag, "row": r, "col": c, "found": False,
                               "reason": "bbox off-image", "bbox_xyxy": [x0, y0, x1, y1]})
            continue
        sub = cube[x0:x1, y0:y1, :]                 # (nx, ny, B)
        sub_img = np.ascontiguousarray(sub.transpose(1, 0, 2)).astype(np.float32)  # (H,W,B)
        h, w = sub_img.shape[:2]
        wmask = well_mask_local(cell, x0, y0, w, h)
        seg, refs = segment_kernel(sub_img, wmask)
        kmask, found = seg.mask, seg.found
        mean_spec = (sub_img[kmask].mean(axis=0) if found and kmask.any()
                     else np.full(sub_img.shape[2], np.nan, np.float32)).astype(np.float32)
        rim_suspect = bool(found and (seg.outside_floor_frac > RIM_OUTSIDE_FLOOR
                                      or seg.border_touch_frac > RIM_BORDER_TOUCH
                                      or seg.floor_frac < RIM_MIN_FLOOR_FRAC))

        arrays[f"{tag}_crop"] = sub_img.astype(CROP_DTYPE)
        arrays[f"{tag}_kernel_mask"] = kmask
        arrays[f"{tag}_well_mask"] = wmask
        arrays[f"{tag}_floor_mask"] = seg.floor
        arrays[f"{tag}_mean_spectrum"] = mean_spec
        arrays[f"{tag}_kernel_ref"] = refs["kernel_ref"]
        arrays[f"{tag}_floor_ref"] = refs["floor_ref"]
        frac = float(kmask.sum()) / max(1, int(wmask.sum()))
        meta_cells.append({
            "cell": tag, "row": r, "col": c, "found": found,
            "bbox_xyxy": [x0, y0, x1, y1],
            "center_px": cell["center_px"], "corners_px": cell["corners_px"],
            "kernel_px": int(kmask.sum()), "well_px": int(wmask.sum()),
            "kernel_well_frac": round(frac, 3),
            "floor_frac": round(seg.floor_frac, 3),
            "border_touch_frac": round(seg.border_touch_frac, 3),
            "outside_floor_frac": round(seg.outside_floor_frac, 3),
            "rim_suspect": rim_suspect,
        })
    return arrays, meta_cells


def draw_extract_overlay(gray8, res, meta_cells, arrays, scale=2):
    h, w = gray8.shape
    base = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    base = cv2.resize(base, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    paint = base.copy()
    for mc in meta_cells:                       # filled kernel masks (completeness)
        tag = mc["cell"]
        if f"{tag}_kernel_mask" not in arrays:
            continue
        x0, y0, _, _ = mc["bbox_xyxy"]
        km = arrays[f"{tag}_kernel_mask"].astype(np.uint8)
        km_big = cv2.resize(km, (km.shape[1] * scale, km.shape[0] * scale),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        reg = paint[y0 * scale:y0 * scale + km_big.shape[0], x0 * scale:x0 * scale + km_big.shape[1]]
        if not mc["found"]:
            col = (255, 0, 255)          # magenta: nothing segmented
        elif mc.get("rim_suspect"):
            col = (0, 165, 255)          # orange: rim/border suspect -> review
        else:
            col = (0, 0, 255)            # red: clean kernel
        reg[km_big] = col
    vis = cv2.addWeighted(paint, 0.5, base, 0.5, 0)
    for cell in res["cells"]:                   # well polygons on top
        if cell["kind"] != "kernel":
            continue
        pts = (np.array(cell["corners_px"]) * scale).astype(np.int32)
        cv2.polylines(vis, [pts], True, (255, 200, 0), 1)
    n_susp = sum(m.get("rim_suspect", False) for m in meta_cells)
    txt = (f"{res['name']}  kernels found {sum(m['found'] for m in meta_cells)}/22"
           f"  rim_suspect {n_susp} (orange)")
    cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def process(cube_path):
    cube = np.load(cube_path)
    res = fg.fit(cube_path, cube=cube)
    name, status = res["name"], res["status"]
    if status != "pass":
        print(f"[{name}] SKIP ({status}: {'; '.join(res['reasons'])})")
        return {"name": name, "status": status, "n_found": None}

    arrays, meta_cells = extract(cube, res)
    n_found = sum(m["found"] for m in meta_cells)
    n_susp = sum(m.get("rim_suspect", False) for m in meta_cells)
    suspects = [m["cell"] for m in meta_cells if m.get("rim_suspect")]
    fracs = [m["kernel_well_frac"] for m in meta_cells if m.get("found")]
    frac_rng = f"{min(fracs):.2f}-{max(fracs):.2f}" if fracs else "n/a"

    npz_path = OUT_DIR / f"{name}_cells.npz"
    np.savez_compressed(npz_path, **arrays)
    meta = {"name": name, "status": status, "px": res["frame"]["px"], "py": res["frame"]["py"],
            "angle": res["frame"]["angle"], "n_kernels_found": n_found,
            "n_rim_suspect": n_susp, "rim_suspect_cells": suspects, "cells": meta_cells}
    with open(OUT_DIR / f"{name}_extract_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    cv2.imwrite(str(OUT_DIR / f"{name}_extract_overlay.png"),
                draw_extract_overlay(res["gray8"], res, meta_cells, arrays))

    size_mb = npz_path.stat().st_size / 1e6
    susp_s = f"  rim_suspect {n_susp} ({','.join(suspects)})" if n_susp else ""
    print(f"[{name}] PASS  kernels found {n_found}/22  kernel/well frac {frac_rng}"
          f"{susp_s}  npz={size_mb:.0f}MB")
    return {"name": name, "status": status, "n_found": n_found, "n_susp": n_susp}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cubes", nargs="*", help="Corrected .npy cubes (default: t10-t13).")
    args = p.parse_args()
    cubes = args.cubes or [str(PROD / "corrected_file" / f"7222026_ref_t{i}.npy") for i in (10, 11, 12, 13)]
    results = [process(c) for c in cubes]
    ok = [r for r in results if r["status"] == "pass"]
    print(f"\nsummary: {len(ok)}/{len(results)} extracted; "
          f"kernels found per capture: {[r['n_found'] for r in ok]}; "
          f"rim_suspect per capture: {[r['n_susp'] for r in ok]}")


if __name__ == "__main__":
    main()
