"""Index every kernel in the collection and write grid_view/ to look at.

    python3 grid_index.py                        # everything under real_data
    python3 grid_index.py --day 1 --dish 0       # one dish
    python3 grid_index.py --mode reflectance --workers 8

For each capture it fits the plate lattice (see gridfit/), writes an overlay and
a per-cell record, and then checks the captures of one dish against each other.
The output mirrors the data tree:

    grid_view/<mode>_images/day<D>/dish<N>/<side>/grid.png     the overlay
                                                  cells.json    28 cells + fit
    grid_view/contact_sheets/<mode>_day<D>_<side>.png          25 dishes at a glance
    grid_view/summary.csv                                      one row per capture
    grid_view/tracking.json                                    per dish, per kernel
    grid_view/.cache/                                          renders; safe to delete

WHY THE CROSS-CHECKS
Each capture is fitted alone, so nothing stops day 2 of a dish from being
labelled differently to day 1 -- and a lattice that is off by one cell still
produces a clean overlay. The per-dish checks in tracking.json are what would
catch that: the fiducial ids must name the same face every time, the handedness
must be the same for every dorsal capture in the collection, the pitch must not
move, and a well that holds a kernel on day 1 must still hold one on day 9.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gridfit import cells as cellmod, fit_capture, overlay, plate   # noqa: E402

# Derived from this file's location, not hard-coded: the checkout moves.
# Override with BARLEY_CAPTURE_ROOT or --data.
DEFAULT_DATA = Path(os.environ.get("BARLEY_CAPTURE_ROOT")
                    or Path(__file__).resolve().parents[2]) / "real_data"
MODES = ("reflectance", "transmittance")
SIDES = ("dorsal", "ventral")
THUMB_W = 260


# ------------------------------------------------------------------ walking --
def discover(data_root, modes, days, dishes, sides):
    """Every capture directory under the data root, in a stable order."""
    out = []
    for mode in modes:
        base = data_root / f"{mode}_images"
        if not base.is_dir():
            continue
        for day_dir in sorted(base.glob("day*"), key=lambda p: int(p.name[3:])):
            day = int(day_dir.name[3:])
            if days and day not in days:
                continue
            for dish_dir in sorted(day_dir.glob("dish*"), key=lambda p: int(p.name[4:])):
                dish = int(dish_dir.name[4:])
                if dishes and dish not in dishes:
                    continue
                for side in sides:
                    d = dish_dir / side
                    if (d / "capture.npy").exists():
                        out.append({"mode": mode, "day": day, "dish": dish,
                                    "side": side, "dir": str(d)})
    return out


# ------------------------------------------------------------------ one fit --
def run_one(job):
    """Fit one capture and write its two files. -> a JSON-safe summary row."""
    spec, out_root, cache_root = job
    mode, day, dish, side = spec["mode"], spec["day"], spec["dish"], spec["side"]
    rel = Path(f"{mode}_images") / f"day{day}" / f"dish{dish}" / side
    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    row = {"mode": mode, "day": day, "dish": dish, "side": side}
    t0 = time.time()
    try:
        res = fit_capture(spec["dir"], mode, side, cache_dir=cache_root / rel)
    except Exception as exc:                                  # noqa: BLE001
        row.update({"status": "error", "reasons": f"{type(exc).__name__}: {exc}"})
        return row

    occ = (cellmod.measure_occupancy(res["gray"], res.get("clip"), mode, res["cells"])
           if res["cells"] else {})
    title = f"{mode} day{day} dish{dish} {side}"
    cv2.imwrite(str(out_dir / "grid.png"),
                overlay.draw(res["gray"], res, res["cells"], res["seeds"], title))

    record = {
        "capture": {"mode": mode, "day": day, "dish": dish, "side": side,
                    "source": spec["dir"], "render_shape": list(res["gray"].shape)},
        "status": res["status"], "reasons": res["reasons"], "flags": res["flags"],
        "info": res.get("info", []),
        "marker_ids": res["marker_ids"], "side_observed": res["side_observed"],
        "marker_pass": res["marker_pass"], "marker_offsets_px": res.get("marker_offsets_px"),
        "markers": {str(k): {f: v[f] for f in ("center", "corners", "id", "score",
                                               "margin", "exact", "mirrored")}
                    for k, v in res["markers"].items()},
        "geometry": res.get("geometry"), "pose": res.get("pose"),
        "fit": {k: res.get(k) for k in ("n_seeds", "n_matched", "n_inliers",
                                        "n_duplicate", "rms_px", "max_resid_px",
                                        "unmatched_seeds", "empty_cells")},
        "frame": None if res["frame"] is None else {
            "origin": [float(v) for v in res["frame"][0]],
            "e_col": [float(v) for v in res["frame"][1]],
            "e_row": [float(v) for v in res["frame"][2]]},
        "axes": ("x = column, y = row of the (n_lines, width) render; "
                 "gridfit.render.to_cube_mask converts a mask back to cube layout"),
        "cells": [{**c, "occupancy": occ.get(c["name"])} for c in res["cells"]],
    }
    (out_dir / "cells.json").write_text(json.dumps(record, indent=1, default=float))

    g = res.get("geometry") or {}
    row.update({
        "status": res["status"],
        "marker_ids": "+".join(str(i) for i in res["marker_ids"]),
        "side_observed": res["side_observed"] or "",
        "chirality": g.get("chirality", ""),
        "n_seeds": res.get("n_seeds"), "n_matched": res.get("n_matched"),
        "rms_px": _r(res.get("rms_px"), 2), "pitch_x_px": _r(g.get("pitch_x_px"), 2),
        "pitch_y_px": _r(g.get("pitch_y_px"), 2), "angle_deg": _r(g.get("angle_deg"), 2),
        "pose": (res.get("pose") or {}).get("pose", ""),
        "pose_margin": _r((res.get("pose") or {}).get("margin"), 1),
        "n_occupied": sum(1 for v in occ.values() if v is not None and v >= OCCUPIED_FRAC),
        "seconds": round(time.time() - t0, 1),
        "reasons": " | ".join(res["reasons"]), "flags": " | ".join(res["flags"]),
        "info": " | ".join(res.get("info", [])),
    })
    return row


OCCUPIED_FRAC = 0.12


def _r(v, n):
    return None if v is None else round(float(v), n)


# ----------------------------------------------------------------- tracking --
def track(rows, out_root):
    """Cross-check the captures of each dish against each other.

    Everything here is a statement that must hold if the indexing is right, so
    the interesting output is the exceptions list -- an empty one is the result
    we want.
    """
    per_dish = {}
    for r in rows:
        per_dish.setdefault(r["dish"], []).append(r)

    chir = {}
    for r in rows:
        if r.get("chirality"):
            chir.setdefault(r["side"], {}).setdefault(r["chirality"], 0)
            chir[r["side"]][r["chirality"]] += 1

    dishes = {}
    for dish, rs in sorted(per_dish.items()):
        problems = []
        usable = [r for r in rs if r["status"] != "fail" and r["status"] != "error"]
        for r in rs:
            if r["status"] in ("fail", "error"):
                problems.append(f"{r['mode']} day{r['day']} {r['side']}: "
                                f"{r['status']} -- {r.get('reasons', '')}")
            elif r["side_observed"] and r["side_observed"] != r["side"]:
                problems.append(f"{r['mode']} day{r['day']} {r['side']}: fiducials say "
                                f"{r['side_observed']}")
        for side in SIDES:
            seen = {r["chirality"] for r in usable if r["side"] == side and r["chirality"]}
            if len(seen) > 1:
                problems.append(f"{side} captures disagree on handedness: {sorted(seen)}")
        px = [r["pitch_x_px"] for r in usable if r["pitch_x_px"]]
        py = [r["pitch_y_px"] for r in usable if r["pitch_y_px"]]
        if px and (max(px) - min(px)) > 4.0:
            problems.append(f"pitch_x spread {min(px):.1f}-{max(px):.1f} px across captures")
        if py and (max(py) - min(py)) > 7.0:
            problems.append(f"pitch_y spread {min(py):.1f}-{max(py):.1f} px across captures")

        occ = _occupancy_table(out_root, rs)
        agree, agree_problems = occupancy_agreement(occ)
        problems += agree_problems
        for name, seen in sorted(occ.items()):
            vals = [v for v in seen.values() if v is not None]
            if not vals:
                continue
            filled = [k for k, v in seen.items() if v is not None and v >= OCCUPIED_FRAC]
            if filled and len(filled) != len(vals):
                empty = sorted(k for k, v in seen.items()
                               if v is not None and v < OCCUPIED_FRAC)
                problems.append(f"{name}: holds a kernel in {len(filled)}/{len(vals)} "
                                f"captures (empty in {', '.join(empty)})")

        dishes[f"dish{dish}"] = {
            "captures": len(rs),
            "usable": len(usable),
            "kernels": [f"dish{dish}_{plate.cell_name(*rc)}" for rc in plate.KERNEL_CELLS],
            "occupancy": occ,
            "occupancy_agreement": agree,
            "exceptions": problems,
        }

    n_ok = sum(1 for d in dishes.values() if not d["exceptions"])
    return {
        "kernel_id": "dish<N>_R<row>C<col> -- 22 per dish, the same physical well "
                     "on every day, both faces and both modes",
        "captures": len(rows),
        "by_status": _count(rows, "status"),
        "chirality_by_side": chir,
        "dishes_clean": n_ok, "dishes_total": len(dishes),
        "dishes": dishes,
    }


def _wrong_pose_perms():
    """The three mislabelings of the plate, as permutations of the 22 wells.

    If a capture were indexed under the wrong pose, its cell names would be one
    of these permutations of the truth -- so testing a capture's content against
    them is a direct test of the labelling that uses no fiducial at all.
    """
    pos = {rc: i for i, rc in enumerate(plate.KERNEL_CELLS)}
    out = {}
    for label, flip_r, flip_c in plate.POSES:
        if label == "identity":
            continue
        out[label] = [pos.get(((plate.N_ROWS - 1 - r) if flip_r else r,
                               (plate.N_COLS - 1 - c) if flip_c else c), -1)
                      for r, c in plate.KERNEL_CELLS]
    return out


WRONG_POSES = _wrong_pose_perms()
AGREE_MIN = 0.45            # a capture this uncorrelated with its dish is not tracking
AGREE_MARGIN_MIN = 0.05     # true labelling must beat every mislabelling by this


def _zscore(v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 4 or np.nanstd(v[ok]) < 1e-9:
        return None
    z = np.full(v.shape, np.nan)
    z[ok] = (v[ok] - v[ok].mean()) / v[ok].std()
    return z


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")


def occupancy_agreement(table):
    """Does every capture of this dish agree about WHICH well is which?

    The fiducials say a cell name means the same physical well on every capture.
    This checks it against the content instead: how much of each well is kernel
    is a property of that kernel, so the 22-value profile of one capture should
    match every other capture of the same dish -- and should match it BETTER than
    any of the three mislabelings would. Measured on the collection, the true
    labelling correlates 0.93-0.97 across faces while the mirrored one manages
    only 0.59-0.84, so the comparison genuinely separates them.

    This is what would catch a dish indexed backwards on one day. It uses no
    fiducial, so it is not confirming the thing that placed the grid.
    """
    names = [plate.cell_name(*rc) for rc in plate.KERNEL_CELLS]
    keys = sorted({k for n in names for k in table.get(n, {})})
    rows = {}
    for k in keys:
        z = _zscore([table.get(n, {}).get(k) for n in names])
        if z is not None:
            rows[k] = z
    if len(rows) < 2:
        return {"captures": len(rows), "checked": False}, []

    out, problems, weak = {}, [], []
    for k, z in rows.items():
        ref = np.nanmedian(np.array([v for j, v in rows.items() if j != k]), axis=0)
        true = _corr(z, ref)
        wrong = {label: _corr(np.array([z[i] if i >= 0 else np.nan for i in perm]), ref)
                 for label, perm in WRONG_POSES.items()}
        best_wrong = max((v for v in wrong.values() if np.isfinite(v)), default=float("nan"))
        hit = max(wrong, key=lambda w: wrong[w] if np.isfinite(wrong[w]) else -9)
        margin = true - best_wrong
        out[k] = {"corr": round(true, 3),
                  "best_wrong_pose": hit,
                  "best_wrong_pose_corr": None if not np.isfinite(best_wrong)
                  else round(best_wrong, 3),
                  "margin": None if not np.isfinite(margin) else round(margin, 3)}
        if not np.isfinite(true) or true < AGREE_MIN:
            problems.append(f"{k}: well contents correlate only {true:.2f} with the rest "
                            f"of the dish -- this capture may not be indexed with them")
        elif np.isfinite(margin) and margin < 0:
            problems.append(f"{k}: a '{hit}' relabelling fits the rest of the dish BETTER "
                            f"than the labelling used ({best_wrong:.2f} vs {true:.2f})")
        elif np.isfinite(margin) and margin < AGREE_MARGIN_MIN:
            # Not a disagreement -- the labelling still wins, just barely. It
            # means this dish's kernels are too alike for their contents to tell
            # the poses apart, so here the fiducials are carrying the labelling
            # unaided rather than being corroborated.
            weak.append(f"{k}: contents only just prefer the labelling used over "
                        f"'{hit}' ({true:.2f} vs {best_wrong:.2f}) -- too symmetric "
                        f"to corroborate the fiducials")
    return ({"captures": len(rows), "checked": True,
             "min_margin": round(min((v["margin"] for v in out.values()
                                      if v["margin"] is not None), default=float("nan")), 3),
             "weak": weak, "per_capture": out}, problems)


def _occupancy_table(out_root, rs):
    """{cell name: {"<mode> day<D> <side>": occupancy or None}} for one dish."""
    table = {}
    for r in rs:
        key = f"{r['mode'][:5]} day{r['day']} {r['side'][:3]}"
        p = (out_root / f"{r['mode']}_images" / f"day{r['day']}" / f"dish{r['dish']}"
             / r["side"] / "cells.json")
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        for c in rec["cells"]:
            if c["kind"] != "kernel":
                continue
            table.setdefault(c["name"], {})[key] = c.get("occupancy")
    return table


def _count(rows, field):
    out = {}
    for r in rows:
        out[r.get(field, "")] = out.get(r.get(field, ""), 0) + 1
    return dict(sorted(out.items()))


# ----------------------------------------------------------- contact sheets --
BADGE = {"pass": (90, 200, 90), "needs_review": (0, 190, 240), "fail": (50, 50, 230),
         "error": (50, 50, 230)}


def contact_sheets(rows, out_root):
    """One sheet per mode/day/side: every dish small enough to scan in one look."""
    sheet_dir = out_root / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["day"], r["side"]), []).append(r)

    made = []
    for (mode, day, side), rs in sorted(groups.items()):
        tiles = []
        for r in sorted(rs, key=lambda r: r["dish"]):
            p = (out_root / f"{mode}_images" / f"day{day}" / f"dish{r['dish']}" / side
                 / "grid.png")
            img = cv2.imread(str(p)) if p.exists() else None
            if img is None:
                img = np.zeros((int(THUMB_W * 1.5), THUMB_W, 3), np.uint8)
            else:
                h = max(int(img.shape[0] * THUMB_W / img.shape[1]), 1)
                img = cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA)
            bar = np.zeros((22, THUMB_W, 3), np.uint8)
            bar[:] = BADGE.get(r["status"], (90, 90, 90))
            cv2.putText(bar, f"dish{r['dish']}  {r['status']}", (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, img]))

        h = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.zeros((h - t.shape[0], THUMB_W, 3), np.uint8)])
                 for t in tiles]
        per_row = 5
        grid = [np.hstack(tiles[i:i + per_row] + [np.zeros((h, THUMB_W, 3), np.uint8)]
                          * ((-len(tiles)) % per_row if i + per_row >= len(tiles) else 0))
                for i in range(0, len(tiles), per_row)]
        path = sheet_dir / f"{mode}_day{day}_{side}.png"
        cv2.imwrite(str(path), np.vstack(grid))
        made.append(path)
    return made


# --------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the corrected-cube tree")
    p.add_argument("--out", type=Path, default=ROOT / "grid_view")
    p.add_argument("--mode", action="append", choices=MODES, help="repeatable; default both")
    p.add_argument("--side", action="append", choices=SIDES, help="repeatable; default both")
    p.add_argument("--day", action="append", type=int, help="repeatable; default all")
    p.add_argument("--dish", action="append", type=int, help="repeatable; default all")
    p.add_argument("--workers", type=int, default=4,
                   help="captures fitted in parallel; the first pass is disk-bound "
                        "on ~0.6 GB per cube, re-runs come off the render cache")
    p.add_argument("--no-sheets", action="store_true", help="skip the contact sheets")
    args = p.parse_args()

    jobs = discover(args.data, args.mode or MODES, set(args.day or []),
                    set(args.dish or []), args.side or SIDES)
    if not jobs:
        raise SystemExit(f"no captures found under {args.data}")
    out_root, cache_root = args.out, args.out / ".cache"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} capture(s) -> {out_root}")

    t0 = time.time()
    rows, payload = [], [(j, out_root, cache_root) for j in jobs]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(run_one, payload), 1):
                rows.append(row)
                _progress(i, len(jobs), row)
    else:
        for i, item in enumerate(payload, 1):
            rows.append(run_one(item))
            _progress(i, len(jobs), rows[-1])

    rows.sort(key=lambda r: (r["mode"], r["day"], r["dish"], r["side"]))
    fields = ["mode", "day", "dish", "side", "status", "marker_ids", "side_observed",
              "chirality", "n_seeds", "n_matched", "n_occupied", "rms_px", "pitch_x_px",
              "pitch_y_px", "angle_deg", "pose", "pose_margin", "seconds",
              "reasons", "flags", "info"]
    with (out_root / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    tracking = track(rows, out_root)
    (out_root / "tracking.json").write_text(json.dumps(tracking, indent=1, default=float))
    if not args.no_sheets:
        for s in contact_sheets(rows, out_root):
            print(f"  sheet: {s}")

    print("=" * 78)
    for k, v in tracking["by_status"].items():
        print(f"  {k:14s} {v}")
    print(f"  handedness: " + "; ".join(
        f"{s} {dict(v)}" for s, v in sorted(tracking["chirality_by_side"].items())))
    print(f"  dishes with no exception: {tracking['dishes_clean']}/{tracking['dishes_total']}")
    print(f"  summary : {out_root / 'summary.csv'}")
    print(f"  tracking: {out_root / 'tracking.json'}")
    print(f"Done in {time.time() - t0:.0f}s.")


def _progress(i, n, row):
    tail = row.get("reasons") or row.get("flags") or ""
    print(f"[{i:3d}/{n}] {row['mode'][:5]} day{row['day']} dish{row['dish']:<2d} "
          f"{row['side']:7s} {row['status']:12s} "
          f"rms={row.get('rms_px')} ids={row.get('marker_ids')} {tail[:60]}")


if __name__ == "__main__":
    main()
