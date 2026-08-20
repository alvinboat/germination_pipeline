"""Draw, for every capture, exactly the pixels the models are trained on.

    python3 explore/review_masks.py                       # all 200 captures -> mask_review/
    python3 explore/review_masks.py --dish 7 --dish 13    # two dishes, for a quick look
    python3 explore/review_masks.py --mode transmittance --day 9

One PNG per (mode, day, dish, side). Each PNG has two halves:

  left    the capture as gridfit rendered it, with every mask that reached the
          dataset outlined on it and every extraction quad drawn. This answers
          "is this the right cube, and did the mask land on the well it is
          named after?"
  right   the 4x7 plate laid out as wells, each holding the patch that is
          actually stored in patches.npy with the mask that is actually stored
          in masks.npy. Outside the mask is dimmed, because outside the mask is
          what every model path discards. This answers "is the row we train on
          the grain, the whole grain, and nothing but the grain?"

Every panel is titled with its germination label, so a mask silently attributed
to the neighbouring well shows up as a label that does not match the kernel
under it -- the one failure that no amount of downstream metric can reveal.

`config.EXCLUDED_VARIETIES` is applied by default -- there is nothing to verify
about kernels no model will see. `--include-excluded` draws them anyway.

Nothing here recomputes a patch. The images are read straight out of the built
dataset, so what you see is the file the trainers open, not a re-derivation of
it that could differ. The context panel is the only thing read from source, and
it comes from gridfit's own render cache, keyed to the cube's size and mtime.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt              # noqa: E402
from matplotlib.patches import Polygon           # noqa: E402

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                     # noqa: E402
from barley import assign, coco, extract, germination    # noqa: E402
from barley import index as index_mod             # noqa: E402
from reporting import style as S                  # noqa: E402

sys.path.insert(0, str(config.COLLECTION))
from gridfit import render                        # noqa: E402

PLATE_ROWS, PLATE_COLS = 4, 7

MASK_INK = "#ff3b30"        # the annotator's mask, on both panels
QUAD_INK = "#2a78d6"        # the CELL_SCALE quad that becomes the patch frame
DEAD_INK = "#898781"        # rim and marker cells: never a kernel, never a row
HAZE = (0.60, 0.62, 0.66)   # what pixels outside the mask are washed towards


def germ_ink(day):
    """Colour for a germination outcome. Sequential for the days, apart for the
    two states that are not days -- never-germinated is a real observation and
    unscored is a missing label, and the two must never read as neighbours."""
    if day is germination.UNSCORED:
        return DEAD_INK
    if day is None:
        return S.series(1)                      # never germinated
    # Only the darker half of the ramp: day 1 in the palest blue would be
    # unreadable on the light surface, and every one of these is a text label.
    ramp = S.SEQ_BLUE[5:]
    i = int(round((day - 1) / max(len(config.GERMINATION_DAYS) - 1, 1) * (len(ramp) - 1)))
    return ramp[min(max(i, 0), len(ramp) - 1)]


def germ_text(day):
    if day is germination.UNSCORED:
        return "unscored"
    if day is None:
        return "never"
    return f"day {int(day)}"


# --------------------------------------------------------------------- data --
def group_by_capture(idx):
    """-> {CaptureKey: {cell_name: row dict}} over the built index."""
    out = defaultdict(dict)
    for i in range(len(idx)):
        key = coco.CaptureKey(mode=str(idx["mode"][i]), day=int(idx["day_folder"][i]),
                              dish=int(idx["dish"][i]), side=str(idx["side"][i]))
        out[key][str(idx["cell"][i])] = {
            "row": int(idx["row"][i]), "kernel_uid": str(idx["kernel_uid"][i]),
            "cell_index": int(idx["cell_index"][i]),
            "ann_id": int(idx["coco_ann_id"][i]),
            "patch_mask_px": int(idx["patch_mask_px"][i]),
            "mask_px": int(idx["mask_px"][i]),
            "nan_frac": float(idx["nan_frac"][i]), "sat_frac": float(idx["sat_frac"][i]),
            "clipped_frac": float(idx["clipped_frac"][i]),
            "containment": float(idx["containment"][i]), "source": str(idx["source"][i]),
        }
    return out


def context_image(key, rec):
    """gridfit's render of the capture, in the working frame. -> (h, w) uint8."""
    rel = Path(f"{key.mode}_images") / f"day{key.day}" / f"dish{key.dish}" / key.side
    gray, _ = render.load(config.local_capture_dir(rec["capture"]["source"]), key.mode,
                          cache_dir=config.GRID_VIEW / ".cache" / rel)
    return gray


def working_masks(cap, wanted_ann_ids, width):
    """{ann_id: working-frame bool mask} for the anns that reached the dataset.

    Decoded and reduced by the same two calls build_dataset used, so an outline
    drawn here is the outline that was warped -- not a fresh decode that might
    keep a speck the build dropped.
    """
    out = {}
    for ann in cap.anns:
        if ann.ann_id not in wanted_ann_ids:
            continue
        m, _, _ = coco.keep_largest_component(coco.decode_rle(ann.counts, ann.size))
        out[ann.ann_id] = m[::-1].T             # cube layout -> working frame
    return out


def patch_image(patch, mask):
    """(H, W, B) stored absorbance -> (H, W) float in 0..1, band-mean, stretched.

    Stretched on the masked pixels, not the whole patch. The well is far
    brighter than the grain in reflectance absorbance and far darker in
    transmittance; either way a whole-patch stretch spends the entire range on
    the part that gets discarded and flattens the grain to one flat tone --
    which is exactly the part that has to be inspected.
    """
    p = np.asarray(patch, np.float32)
    with np.errstate(invalid="ignore"):
        img = np.nanmean(np.where(np.isfinite(p).any(2, keepdims=True), p, 0.0), axis=2)
    ref = img[mask] if mask.any() else img
    lo, hi = np.nanpercentile(ref, [2, 98])
    pad = 0.15 * max(hi - lo, 1e-9)
    stretched = np.clip((img - lo + pad) / max(hi - lo + 2 * pad, 1e-9), 0, 1)
    invalid = 1.0 - float(np.isfinite(p)[mask].mean()) if mask.any() else 1.0
    return stretched, invalid


# ------------------------------------------------------------------ drawing --
def draw_context(ax, gray, rec, rows, masks):
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for c in rec["cells"]:
        if c["kind"] != "kernel":
            ax.add_patch(Polygon(c["corners"], closed=True, fill=False,
                                 ec=DEAD_INK, lw=0.6, ls=":", zorder=2))
            continue
        r = rows.get(c["name"])
        # The scaled quad, not the raw cell: this rectangle is what the
        # homography maps onto the 128x64 patch, so it is the patch's footprint.
        ax.add_patch(Polygon(extract.quad(c), closed=True, fill=False,
                             ec=QUAD_INK if r else DEAD_INK,
                             lw=0.7, alpha=0.9, zorder=2))
        # On the cell's own top edge, not its centre: over the centre the label
        # sits on the grain it is there to identify.
        x, y = (np.asarray(c["corners"][0], float) + c["corners"][1]) / 2
        ax.text(x, y, c["name"], fontsize=4.2, ha="center", va="center",
                color=QUAD_INK if r else DEAD_INK, zorder=4,
                bbox={"fc": "#ffffff", "ec": "none", "alpha": 0.72, "pad": 0.8})
    for m in masks.values():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cs:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=MASK_INK, lw=0.8, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(S.ink("axis"))


def draw_cell_panel(ax, cell, row, patches, masks_npy, day, empty):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    if cell is None or cell["kind"] != "kernel":
        kind = "off plate" if cell is None else cell["kind"]
        ax.set_facecolor(S.ink("plane"))
        ax.text(0.5, 0.5, kind, transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color=S.ink("muted"))
        ax.set_title(cell["name"] if cell else "", fontsize=6.5,
                     color=S.ink("muted"), pad=3)
        return

    if row is None:
        # Two different things, and conflating them would hide the one that
        # matters: a well config lists as permanently empty has nothing to
        # annotate, while any other gap is an annotation the export is missing
        # and a kernel this view does not contribute.
        ink = DEAD_INK if empty else S.series(1)
        ax.set_facecolor(S.ink("plane"))
        ax.text(0.5, 0.5, "empty well\n(known, no kernel)" if empty
                else "no mask\nnot trained on", transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, color=ink)
        ax.set_title(cell["name"], fontsize=6.5, color=ink, pad=3)
        return

    mask = np.asarray(masks_npy[row["row"]])
    img, invalid = patch_image(patches[row["row"]], mask)
    # Outside the mask is washed out rather than cut. Cutting it would hide the
    # well wall, which is the evidence that the mask is on the grain; darkening
    # it would not read as discarded, because in absorbance the grain is the
    # DARK side in reflectance and the bright side in transmittance. A flat,
    # low-contrast haze reads as discarded under either polarity.
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    rgb[~mask] = rgb[~mask] * 0.28 + np.float32(HAZE)
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    ax.contour(mask.astype(float), levels=[0.5], colors=[MASK_INK], linewidths=0.9)

    ink = germ_ink(day)
    ax.set_title(f"{cell['name']}  {germ_text(day)}", fontsize=6.5, color=ink, pad=3)
    # The invalid fraction is measured INSIDE the mask. index.csv's nan_frac and
    # sat_frac are over the whole patch, where a transmittance well is 40% railed
    # open beam by construction -- alarming, and about pixels no model reads. What
    # matters is how much of the grain itself is missing.
    note = f"{row['patch_mask_px']} px"
    ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color=S.ink("muted"))
    if invalid > 0.005:
        ax.text(0.5, -0.115, f"invalid {invalid:.0%} of mask", transform=ax.transAxes,
                ha="center", va="top", fontsize=5.5, color=S.series(1))


def draw_capture(key, rec, rows, gray, masks, patches, masks_npy, first_day, out):
    cells = {c["name"]: c for c in rec["cells"]}
    n_kernels = sum(c["kind"] == "kernel" for c in rec["cells"])

    # The ratios are the two panels' own aspects, so neither letterboxes: the
    # context is 928x640, the grid is 4 rows of 2:1 patches over 7 columns.
    fig = plt.figure(figsize=(13.4, 10.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.09], left=0.025, right=0.985,
                             top=0.868, bottom=0.075, wspace=0.05)
    draw_context(fig.add_subplot(outer[0, 0]), gray, rec, rows, masks)
    inner = outer[0, 1].subgridspec(PLATE_ROWS, PLATE_COLS, hspace=0.34, wspace=0.06)

    for r in range(PLATE_ROWS):
        for c in range(PLATE_COLS):
            cell = cells.get(f"R{r}C{c}")
            row = rows.get(f"R{r}C{c}")
            day = first_day.get(row["kernel_uid"], germination.UNSCORED) if row \
                else germination.UNSCORED
            empty = f"dish{key.dish}_R{r}C{c}" in config.KNOWN_EMPTY_WELLS
            draw_cell_panel(fig.add_subplot(inner[r, c]), cell, row,
                            patches, masks_npy, day, empty)

    hours = config.HOURS[key.day]
    variety = config.VARIETY_NAMES[config.variety_of(key.dish) - 1]
    gaps = [n for n, c in cells.items() if c["kind"] == "kernel" and n not in rows]
    empty = sorted(n for n in gaps if f"dish{key.dish}_{n}" in config.KNOWN_EMPTY_WELLS)
    missing = sorted(n for n in gaps if n not in empty)
    counts = Counter(germ_text(first_day.get(r["kernel_uid"], germination.UNSCORED))
                     for r in rows.values())
    order = [germ_text(d) for d in (*config.GERMINATION_DAYS, None, germination.UNSCORED)]
    germ = "  ".join(f"{k} {counts[k]}" for k in order if counts.get(k))
    fig.text(0.035, 0.965,
             f"{key.mode}  ·  day{key.day} = {hours:g} h  ·  dish{key.dish} "
             f"({variety})  ·  {key.side}",
             fontsize=15, color=S.ink("primary"), va="top")
    fig.text(0.035, 0.929,
             f"{len(rows)}/{n_kernels} kernel views in the dataset"
             + (f"   ·   empty well: {', '.join(empty)}" if empty else "")
             + (f"   ·   NO MASK: {', '.join(missing)}" if missing else "")
             + f"   ·   grid fit {rec['status']}"
             + f"   ·   germination  {germ}",
             fontsize=9, color=S.ink("secondary"), va="top")
    fig.text(0.035, 0.902,
             f"patch {config.PATCH_H}x{config.PATCH_W}x{config.N_BANDS} bands "
             f"{config.BAND_LO}-{config.BAND_HI - 1}, cell_scale {config.CELL_SCALE}, "
             f"band-mean pseudo-absorbance; washed out = outside the mask, discarded"
             f"   ·   {rec['capture']['source']}",
             fontsize=7.5, color=S.ink("muted"), va="top")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, facecolor=S.ink("surface"),
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return {"missing": missing, "empty": empty, "counts": counts}


# --------------------------------------------------------------------- main --
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=lambda s: config.ROOT / s, default=config.DATASET)
    p.add_argument("--out", type=Path, default=config.ROOT / "mask_review")
    p.add_argument("--dish", action="append", type=int, help="repeatable; default all")
    p.add_argument("--mode", choices=config.MODES)
    p.add_argument("--side", choices=config.SIDES)
    p.add_argument("--day", action="append", type=int, help="capture folder, not hours")
    p.add_argument("--include-excluded", action="store_true",
                   help="also draw config.EXCLUDED_VARIETIES, which no model sees")
    args = p.parse_args()

    t0 = time.time()
    S.use("light")
    # Two views of the index. `full` is every row the build wrote, and it is what
    # the label parser has to see: the annotator scored the whole plate, so a
    # cell map missing the excluded dishes would make their perfectly good labels
    # look like scores on wells that hold no kernel, and the parser would reject
    # the file. `idx` is what gets drawn.
    full = index_mod.Index.load(args.dataset / "index.csv", include_excluded=True)
    idx = full if args.include_excluded else index_mod.Index.load(
        args.dataset / "index.csv")
    patches = np.load(args.dataset / "patches.npy", mmap_mode="r")
    masks_npy = np.load(args.dataset / "masks.npy", mmap_mode="r")
    print(f"dataset {args.dataset}: {idx.describe()}")

    table = germination.read_germination_labels(kernel_uids=set(full["kernel_uid"]),
                                             cell_map=germination.build_cell_map(full))
    scored = len(set(idx["kernel_uid"]) & set(table))
    print(f"germination labels: {scored}/{len(set(idx['kernel_uid']))} kernels scored "
          f"({sum(table[u] is None for u in set(idx['kernel_uid']) & set(table))} never)")

    captures = coco.load(config.COCO_JSON, config.HOURS)
    by_capture = group_by_capture(idx)
    keys = [k for k in sorted(by_capture, key=lambda k: (k.mode, k.day, k.dish, k.side))
            if (not args.dish or k.dish in args.dish)
            and (not args.mode or k.mode == args.mode)
            and (not args.side or k.side == args.side)
            and (not args.day or k.day in args.day)]
    if not keys:
        raise SystemExit("those filters select no capture")
    filters = ", ".join(f"{k}={v}" for k, v in
                        [("mode", args.mode), ("day", args.day),
                         ("side", args.side), ("dish", args.dish)] if v)
    if config.EXCLUDED_VARIETIES and not args.include_excluded:
        filters = ", ".join(filter(None, [
            filters, f"excluding {', '.join(config.excluded_names())}"]))
    scope = f"filtered to {filters}" if filters else "every capture in the index"
    if filters:
        print(f"filtered run: summary.csv and README.md will describe {len(keys)} "
              f"of the {len(by_capture)} captures, not the whole folder")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, key in enumerate(keys, 1):
        rows = by_capture[key]
        rec = assign.load_cells(key)
        if rec is None:
            raise SystemExit(f"{key}: no cells.json, yet the index has rows for it")
        cap = captures[key]
        gray = context_image(key, rec)
        masks = working_masks(cap, {r["ann_id"] for r in rows.values()},
                              rec["capture"]["render_shape"][1])
        hours = config.HOURS[key.day]
        out = (args.out / key.mode / f"day{key.day}_{hours:g}h"
               / f"dish{key.dish:02d}_{key.side}.png")
        info = draw_capture(key, rec, rows, gray, masks, patches, masks_npy, table, out)
        summary.append({
            "mode": key.mode, "day_folder": key.day, "hours": hours, "dish": key.dish,
            "side": key.side, "variety": config.VARIETY_NAMES[config.variety_of(key.dish) - 1],
            "kernel_views": len(rows), "fit_status": rec["status"],
            "empty_wells": " ".join(info["empty"]),
            "cells_without_mask": " ".join(info["missing"]),
            "unscored": info["counts"].get("unscored", 0),
            "never": info["counts"].get("never", 0),
            "median_mask_px": int(np.median([r["patch_mask_px"] for r in rows.values()])),
            "image": str(out.relative_to(args.out)),
            "source": rec["capture"]["source"],
        })
        print(f"[{i:3d}/{len(keys)}] {key} -> {out.relative_to(args.out)}", flush=True)

    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    unexpected = sorted({f"dish{s['dish']}_{c}" for s in summary
                         for c in s["cells_without_mask"].split() if c})
    # Written on every run, filtered or not, with the scope stated -- a partial
    # re-render must not leave a summary that reads as if it covered the folder.
    (args.out / "README.md").write_text(REVIEW_README.format(
        n=len(summary), dataset=args.dataset, scope=scope,
        scored=scored, kernels=len(set(idx["kernel_uid"])),
        empty=", ".join(sorted(config.KNOWN_EMPTY_WELLS)),
        gaps=", ".join(unexpected) or "none",
        built=time.strftime("%Y-%m-%dT%H:%M:%S")))

    print(f"\n{len(summary)} capture sheets -> {args.out}")
    print(f"summary : {args.out / 'summary.csv'}")
    if unexpected:
        print(f"kernel wells with no mask in at least one view: "
              f"{', '.join(unexpected)}")
    print(f"done in {time.time() - t0:.1f}s")


REVIEW_README = """# mask review

{n} capture sheets, one per (mode, day, dish, side), written from `{dataset}`
on {built}, covering {scope}. Regenerate with `python3 explore/review_masks.py`.

Each sheet shows the pixels the models are actually trained on:

* **left** the capture, from gridfit's render of the same cube `index.csv`
  names, with every dataset mask outlined in red and every extraction quad in
  blue. Rim and marker cells are dotted grey; they never become rows.
* **right** the plate as a 4x7 well grid. Each panel is the patch stored in
  `patches.npy` and the mask stored in `masks.npy` -- read from those files,
  not recomputed -- averaged over the stored bands and contrast-stretched on
  the masked pixels. Everything outside the mask is washed out, because every
  model path discards it: the CNN zeroes it, the PLS never sees it. Where more
  than 0.5% of a mask's voxels are invalid, the panel says so; that fraction is
  measured inside the mask, unlike index.csv's whole-patch `nan_frac`.

Each panel is titled with the kernel's germination label, so a mask attributed
to the wrong well shows up as a label that does not match the grain beneath it.
That is the failure this folder exists to catch: it changes every downstream
number and shows up in no metric.

`summary.csv` has one row per sheet: view count, fit status, median mask area,
and any kernel well that is empty or has no mask in that view.

Labels: {scored} of {kernels} kernels are scored.
Permanently empty wells, expected to have no mask in every view: {empty}.
Other kernel wells missing a mask in at least one sheet drawn here: {gaps}.
"""


if __name__ == "__main__":
    sys.exit(main())
