"""Cubes + COCO masks + gridfit cells -> the modelling dataset.

    python3 build_dataset.py --dry-run           # counts and bytes, writes nothing
    python3 build_dataset.py --dish 0 --dish 5   # two dishes, for a quick check
    python3 build_dataset.py                     # everything in scope, ~14 GB

Scope is day1 (0 h) and day9 (8 h), both modes, both sides -- see config.HOURS.
One row per (kernel, mode, side, hours) view.

WHAT IS STORED, AND WHAT IS NOT
Stored is per-pixel pseudo-absorbance, `A = -log10(x)`. SNV is deliberately NOT
baked in: every dataloader path applies it by default, so a model still sees
exactly -log10 -> SNV, but keeping it in barley/transforms.py means swapping it
for a Savitzky-Golay derivative, or changing how the mask is applied, costs
nothing. That matters because the masks are provisional and this file takes
most of an hour to rebuild.

The run is two passes. The first assigns every mask to a lattice cell and
refuses to continue if that assignment is not a clean bijection -- it touches no
cube and takes a few seconds, so a mistake costs nothing. Only then are the
memmaps sized and the cubes read.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import config
from barley import assign, coco, extract

FIELDS = ["row", "kernel_uid", "dish", "cell", "cell_index", "variety",
          "mode", "side", "day_folder", "hours",
          "coco_ann_id", "coco_image_id",
          "mask_px_raw", "mask_px", "patch_mask_px", "containment",
          "n_components_raw", "dropped_px",
          "clipped_frac", "sat_frac", "nan_frac", "fit_status", "source"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan(dishes):
    """Pass one: assign every mask to a cell. -> (plan rows, problems, skipped)."""
    captures = coco.load(config.COCO_JSON, config.HOURS)
    rows, problems, skipped = [], [], []

    for key in sorted(captures, key=lambda k: (k.mode, k.day, k.dish, k.side)):
        if dishes and key.dish not in dishes:
            continue
        cap = captures[key]
        rec = assign.load_cells(key)
        if rec is None:
            skipped.append((key, "no cells.json -- run grid_index.py"))
            continue
        got, probs = assign.assign_capture(cap, rec)
        problems += [(key, kind, detail) for kind, detail in probs]
        n_lines, width = rec["capture"]["render_shape"]
        for r in got:
            # Cuttability depends only on the quad and the capture shape, both
            # known here, so the row count below is exact and the memmaps are
            # sized once rather than allocated large and trimmed.
            q = extract.quad(r["cell"])
            if extract.bbox(q, width, n_lines) is None:
                problems.append((key, "CELL_OUTSIDE_FRAME", r["cell"]["name"]))
                continue
            rows.append({"key": key, "rec": rec, **r})
    return rows, problems, skipped


FATAL = {"SHAPE_MISMATCH", "FIT_FAILED", "MASK_OFF_LATTICE",
         "MASK_STRADDLES_CELLS", "TWO_MASKS_ONE_CELL", "EMPTY_MASK",
         "MASK_ERODED_AWAY"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dish", action="append", type=int, help="repeatable; default all")
    p.add_argument("--out", type=Path, default=config.DATASET)
    p.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    args = p.parse_args()

    t0 = time.time()
    print(f"scope: days {sorted(config.HOURS)} -> hours {sorted(config.HOURS.values())}, "
          f"modes {config.MODES}, sides {config.SIDES}")
    rows, problems, skipped = plan(set(args.dish or []))

    for key, why in skipped:
        print(f"  SKIP {key}: {why}")

    kinds = Counter(k for _, k, _ in problems)
    if problems:
        print(f"\n{len(problems)} assignment note(s):")
        for kind, n in kinds.most_common():
            print(f"  {kind}: {n}")
        for key, kind, detail in problems:
            if kind != "CELL_WITHOUT_MASK":
                print(f"    {key}  {kind}  {detail}")
    if any(kinds[k] for k in FATAL):
        raise SystemExit(
            "\nrefusing to build: the mask-to-cell assignment is not a clean "
            "bijection. Every row downstream would be attributed to a well that "
            "may not be the one it was drawn on.")

    # `CELL_WITHOUT_MASK` is expected, not fatal: dish13 R0C4 and dish21 R3C1 are
    # empty wells in every view, and three further masks are genuinely missing.
    n = len(rows)
    caps = len({r["key"] for r in rows})
    nbytes = n * config.PATCH_H * config.PATCH_W * config.N_BANDS * 2
    print(f"\n{caps} captures -> {n} kernel views")
    print(f"  patches {n}x{config.PATCH_H}x{config.PATCH_W}x{config.N_BANDS} "
          f"float16 = {nbytes / 1e9:.1f} GB")
    print(f"  bands {config.BAND_LO}-{config.BAND_HI - 1} = "
          f"{config.wavelengths()[0]:.0f}-{config.wavelengths()[-1]:.0f} nm")
    print(f"  kernels {len({(r['key'].dish, r['cell']['name']) for r in rows})}, "
          f"dishes {len({r['key'].dish for r in rows})}")
    if args.dry_run:
        print(f"\ndry run, nothing written ({time.time() - t0:.1f}s)")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(
        args.out / "patches.npy", mode="w+", dtype=np.float16,
        shape=(n, config.PATCH_H, config.PATCH_W, config.N_BANDS))
    Mk = np.lib.format.open_memmap(
        args.out / "masks.npy", mode="w+", dtype=bool,
        shape=(n, config.PATCH_H, config.PATCH_W))
    S = np.zeros((n, config.N_BANDS), np.float32)

    fh = (args.out / "index.csv").open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()

    # Group by capture so each cube is opened once.
    by_capture = {}
    for r in rows:
        by_capture.setdefault(r["key"], []).append(r)

    row_i, checked = 0, False
    for c_i, (key, group) in enumerate(sorted(
            by_capture.items(), key=lambda kv: (kv[0].mode, kv[0].day, kv[0].dish, kv[0].side)), 1):
        rec = group[0]["rec"]
        src = config.local_capture_dir(rec["capture"]["source"])
        cube = np.load(src / "capture.npy", mmap_mode="r")
        width = cube.shape[0]
        sat = None
        if key.mode == "transmittance":
            sat = np.load(src / "capture_masks.npz")["saturated"]
        if not checked:
            extract.verify_frame(cube, width)
            print("  working-frame slice verified against render.to_working")
            checked = True

        for r in group:
            cell = r["cell"]
            patch, mask, st = extract.cut(
                cube, sat, r["mask"], width, extract.quad(cell))
            if patch is None:
                # plan() already proved every planned quad is cuttable, so this
                # can only mean the cube disagrees with cells.json about shape.
                raise SystemExit(f"{key} {cell['name']}: {st} -- cube and "
                                 "cells.json disagree; refusing to write")
            # Derive the spectrum from the float16 patch that actually lands on
            # disk, not the float32 original, so verify_dataset can recompute it
            # exactly rather than within a tolerance that could hide a real bug.
            p16 = patch.astype(np.float16)
            X[row_i] = p16
            Mk[row_i] = mask
            S[row_i] = extract.mask_mean_spectrum(np.asarray(p16, np.float32), mask)
            w.writerow({
                "row": row_i,
                "kernel_uid": f"dish{key.dish}_{cell['name']}",
                "dish": key.dish, "cell": cell["name"], "cell_index": cell["index"],
                "variety": config.variety_of(key.dish),
                "mode": key.mode, "side": key.side,
                "day_folder": key.day, "hours": config.HOURS[key.day],
                "coco_ann_id": r["ann"].ann_id, "coco_image_id": r["ann"].image_id,
                "mask_px_raw": r["mask_px_raw"], "mask_px": r["mask_px"],
                "patch_mask_px": st["patch_mask_px"],
                "containment": round(r["containment"], 4),
                "n_components_raw": r["n_components_raw"], "dropped_px": r["dropped_px"],
                "clipped_frac": round(st["clipped_frac"], 5),
                "sat_frac": round(st["sat_frac"], 5),
                "nan_frac": round(st["nan_frac"], 5),
                "fit_status": rec["status"], "source": str(src),
            })
            row_i += 1
        del cube, sat
        print(f"[{c_i:3d}/{len(by_capture)}] {key} -> {row_i} views", flush=True)

    fh.close()
    X.flush()
    Mk.flush()

    assert row_i == n, f"wrote {row_i} rows, planned {n}"
    np.save(args.out / "spectra.npy", S)

    (args.out / "bands.json").write_text(json.dumps({
        "band_index": list(range(config.BAND_LO, config.BAND_HI)),
        "wavelength_nm": [round(float(x), 2) for x in config.wavelengths()],
        "stored": "per-pixel pseudo-absorbance A = -log10(max(x, eps)); "
                  "SNV is applied by barley.transforms at load time, not here",
        "patch": {"h": config.PATCH_H, "w": config.PATCH_W,
                  "cell_scale": config.CELL_SCALE,
                  "axes": "rows follow the plate row axis, cols the column axis"},
        "hours_by_day_folder": config.HOURS,
    }, indent=1))

    (args.out / "build_meta.json").write_text(json.dumps({
        "rows": row_i,
        "captures": len(by_capture),
        "coco_json": str(config.COCO_JSON),
        "coco_sha256": sha256(config.COCO_JSON),
        "grid_view": str(config.GRID_VIEW),
        "scope": {"hours_by_day_folder": config.HOURS, "dishes": sorted(args.dish or [])},
        "absorbance_eps": config.ABSORBANCE_EPS,
        "mask_erode_px": config.MASK_ERODE_PX,
        "band_range": [config.BAND_LO, config.BAND_HI],
        "patch": [config.PATCH_H, config.PATCH_W],
        "cell_scale": config.CELL_SCALE,
        "cells_without_mask": [f"{k} {d}" for k, kind, d in problems
                               if kind == "CELL_WITHOUT_MASK"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }, indent=1))

    print(f"\npatches : {args.out / 'patches.npy'} ({row_i}, {config.PATCH_H}, "
          f"{config.PATCH_W}, {config.N_BANDS})")
    print(f"spectra : {args.out / 'spectra.npy'} ({row_i}, {config.N_BANDS})")
    print(f"index   : {args.out / 'index.csv'} ({row_i} rows)")
    print(f"done in {time.time() - t0:.1f}s -- now run: python3 verify_dataset.py")


if __name__ == "__main__":
    sys.exit(main())
