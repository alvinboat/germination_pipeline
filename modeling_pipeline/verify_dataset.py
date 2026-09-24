"""The gate. Nothing trains against a dataset this refuses.

    python3 verify_dataset.py
    python3 verify_dataset.py --dataset dataset_subset --contact-sheet

Every check here exists because its failure is *invisible downstream*: a
mirrored patch, a mask attributed to the neighbouring well, a spectrum that no
longer matches the pixels it claims to summarise. All of them still produce a
clean loss curve and a plausible confusion matrix. The only way to know is to
measure the property directly, against the source data, every time.

Exit code is 0 only if every check passes.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

import config
from barley import assign, coco, extract


class Report:
    def __init__(self):
        self.lines, self.failed = [], 0

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        self.failed += not ok
        line = f"[{tag}] {name}" + (f"\n       {detail}" if detail else "")
        print(line, flush=True)
        self.lines.append(line)

    def note(self, text):
        print(f"       {text}")
        self.lines.append(f"       {text}")


def load_index(path):
    return list(csv.DictReader(path.open()))


# ------------------------------------------------------------------ checks --
def check_provenance(rep, ds, rows):
    meta = json.loads((ds / "build_meta.json").read_text())
    ok = meta["rows"] == len(rows)
    rep.check("index row count matches build_meta", ok,
              f"index {len(rows)}, build_meta {meta['rows']}")

    live = None
    if config.COCO_JSON.exists():
        import hashlib
        h = hashlib.sha256()
        with open(config.COCO_JSON, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        live = h.hexdigest()
    same = live == meta.get("coco_sha256")
    rep.check("annotations on disk are the ones this dataset was built from", same,
              "" if same else "instances_default.json has changed since the build -- "
                              "re-run build_dataset.py before trusting these rows")
    rep.note(f"eps={meta['absorbance_eps']} erode={meta.get('mask_erode_px')}px "
             f"scale={meta['cell_scale']} bands={meta['band_range']}")
    return meta


def check_rle_decoder(rep, n=400):
    """decode_rle must reproduce the exporter's own area and bbox."""
    captures = coco.load(config.COCO_JSON, config.HOURS)
    bad_area = bad_bbox = seen = 0
    for cap in list(captures.values()):
        for ann in cap.anns:
            if seen >= n:
                break
            m = coco.decode_rle(ann.counts, ann.size)
            bad_area += int(m.sum()) != int(ann.area)
            ys, xs = np.nonzero(m)
            bb = [float(xs.min()), float(ys.min()),
                  float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]
            bad_bbox += bb != [float(v) for v in ann.bbox]
            seen += 1
        if seen >= n:
            break
    rep.check("RLE decoder reproduces exporter area and bbox", not (bad_area or bad_bbox),
              f"{seen} annotations: {bad_area} area mismatch, {bad_bbox} bbox mismatch")


def check_assignment(rep, rows):
    """Re-derive the mask-to-cell join from source and require it to agree."""
    captures = coco.load(config.COCO_JSON, config.HOURS)
    want = {(r["mode"], int(r["day_folder"]), int(r["dish"]), r["side"],
             int(r["coco_ann_id"])): r["cell"] for r in rows}
    dishes = {int(r["dish"]) for r in rows}

    problems, disagree, rederived = Counter(), [], 0
    for key, cap in captures.items():
        if key.dish not in dishes:
            continue
        rec = assign.load_cells(key)
        if rec is None:
            problems["NO_CELLS_JSON"] += 1
            continue
        got, probs = assign.assign_capture(cap, rec)
        for kind, _ in probs:
            problems[kind] += 1
        for g in got:
            k = (key.mode, key.day, key.dish, key.side, g["ann"].ann_id)
            rederived += 1
            if want.get(k) != g["cell"]["name"]:
                disagree.append((k, want.get(k), g["cell"]["name"]))

    fatal = {k: v for k, v in problems.items() if k in
             {"SHAPE_MISMATCH", "FIT_FAILED", "MASK_OFF_LATTICE", "MASK_STRADDLES_CELLS",
              "TWO_MASKS_ONE_CELL", "EMPTY_MASK", "MASK_ERODED_AWAY"}}
    rep.check("every mask lands on exactly one kernel cell", not fatal,
              "; ".join(f"{k}={v}" for k, v in fatal.items()) or
              f"{rederived} masks, bijection clean")
    rep.check("stored cell assignment matches a fresh re-derivation", not disagree,
              f"{len(disagree)} disagreement(s): {disagree[:3]}" if disagree
              else f"{rederived} masks agree")
    if problems.get("CELL_WITHOUT_MASK"):
        rep.note(f"{problems['CELL_WITHOUT_MASK']} capture(s) have a well with no mask "
                 "-- expected: two wells are empty in every view, plus 3 known misses")


def check_frame(rep, rows):
    """The cube slice must equal gridfit's own working-frame reference."""
    # Re-anchored: index.csv records the absolute path from the machine that
    # built the dataset, which is not necessarily this one.
    src = config.local_capture_dir(sorted({r["source"] for r in rows})[0])
    cube = np.load(src / "capture.npy", mmap_mode="r")
    try:
        extract.verify_frame(cube, cube.shape[0])
        rep.check("cube slice agrees with gridfit.render.to_working", True, str(src))
    except SystemExit as e:
        rep.check("cube slice agrees with gridfit.render.to_working", False, str(e))


def check_warp(rep, rows, masks):
    """The mask must survive the warp: non-empty, connected, off the border."""
    empty = int((masks.reshape(len(masks), -1).sum(1) == 0).sum())
    rep.check("no mask warps to empty", empty == 0, f"{empty} empty patch masks")

    multi = border = 0
    step = max(1, len(masks) // 300)
    sample = range(0, len(masks), step)
    for i in sample:
        m = masks[i]
        n, _, _, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        multi += (n - 1) > 1
        edge = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        border += bool(edge)
    rep.check("warped masks are a single component", multi == 0,
              f"{multi}/{len(list(sample))} sampled masks are fragmented")
    rep.check("warped masks do not touch the patch border", border == 0,
              f"{border}/{len(list(sample))} sampled masks reach the patch edge -- "
              "the cell quad is clipping the kernel; raise config.CELL_SCALE"
              if border else f"0/{len(list(sample))} sampled masks reach the edge")

    ratio = np.array([int(r["patch_mask_px"]) / max(int(r["mask_px"]), 1) for r in rows])
    lo, hi = np.percentile(ratio, [0.5, 99.5])
    out = int(((ratio < lo / 2) | (ratio > hi * 2)).sum())
    rep.check("warp area scaling is consistent across views", out == 0,
              f"scale median {np.median(ratio):.2f}x, p0.5-p99.5 {lo:.2f}-{hi:.2f}, "
              f"{out} outlier(s)")


def check_spectra(rep, spectra, patches, masks):
    finite = np.isfinite(spectra)
    rep.check("no NaN or Inf in spectra", finite.all(),
              f"{int((~finite).sum())} non-finite cells in "
              f"{int((~finite.all(1)).sum())} row(s)")

    step = max(1, len(spectra) // 200)
    bad, worst = 0, 0.0
    for i in range(0, len(spectra), step):
        got = extract.mask_mean_spectrum(np.asarray(patches[i], np.float32), masks[i])
        d = np.nanmax(np.abs(got - spectra[i])) if np.isfinite(got).any() else np.inf
        worst = max(worst, float(d))
        bad += not np.array_equal(np.nan_to_num(got, nan=-999),
                                  np.nan_to_num(spectra[i], nan=-999))
    rep.check("spectra reproduce exactly from patches + masks", bad == 0,
              f"{bad} of {len(range(0, len(spectra), step))} sampled rows differ "
              f"(max abs {worst:.2e})")


def check_index(rep, rows, meta):
    n_views = Counter()
    for r in rows:
        n_views[r["kernel_uid"]] += 1
    expected = len(config.HOURS) * len(config.MODES) * len(config.SIDES)
    odd = {k: v for k, v in n_views.items() if v != expected}
    unexpected = {k: v for k, v in odd.items()
                  if k not in config.KNOWN_INCOMPLETE_KERNELS}
    rep.check(f"every kernel has {expected} views, or is a known gap", not unexpected,
              f"{len(unexpected)} kernel(s) not on the known list: "
              f"{dict(list(unexpected.items())[:5])}" if unexpected
              else f"{len(n_views)} kernels, {len(odd)} known-incomplete")
    if odd:
        rep.note("known-incomplete (one view missed by the annotator, flagged for "
                 f"the fact-check): {dict(sorted(odd.items()))}")

    rep.check("row numbers are 0..N-1 in order",
              [int(r["row"]) for r in rows] == list(range(len(rows))))

    for uid in config.KNOWN_EMPTY_WELLS:
        rep.check(f"known empty well {uid} is absent", uid not in n_views)

    var = Counter(int(r["variety"]) for r in rows)
    dish_var = defaultdict(set)
    for r in rows:
        dish_var[int(r["dish"])].add(int(r["variety"]))
    bad = {d: v for d, v in dish_var.items() if len(v) != 1}
    rep.check("each dish maps to exactly one variety", not bad, str(bad) if bad else
              f"{len(dish_var)} dishes, varieties {dict(sorted(var.items()))}")

    hours = Counter(float(r["hours"]) for r in rows)
    rep.check("hours come from the day map, not the folder name",
              set(hours) == set(config.HOURS.values()),
              f"{dict(sorted(hours.items()))} (day1=0h, day9=8h)")

    cont = np.array([float(r["containment"]) for r in rows])
    rep.note(f"containment: min {cont.min():.3f} median {np.median(cont):.3f}")
    ero = 1 - np.array([int(r["mask_px"]) for r in rows]) / np.array(
        [int(r["mask_px_raw"]) for r in rows])
    rep.note(f"erosion cost: median {100 * np.median(ero):.1f}% of mask area, "
             f"max {100 * ero.max():.1f}%")


def check_germination(rep, rows, path=None):
    """The germination label file, if it has been delivered.

    Absent is NOT a failure: the dataset has to verify green before the scoring
    is done, and the scoring is partial by design while it is in progress.
    Present-and-wrong IS a failure, on the same footing as a bad mask.

    The notes are the point of running this before training. Coverage says how
    much of the plate is scored; the per-variety breakdown says whether the
    target is confounded with variety, which -- because variety is confounded
    with dish, and folds hold out whole dishes -- decides whether a good score
    means anything at all.
    """
    from barley import germination

    path = path or config.GERMINATION_LABELS
    if not path.exists():
        rep.note(f"no germination labels at {path.name} yet -- survival target "
                 "not checked (this is not a failure)")
        return

    uids = {r["kernel_uid"] for r in rows}
    cell_map = {(int(r["dish"]), int(r["cell_index"])): r["kernel_uid"]
                for r in rows}
    try:
        table = germination.read_germination_labels(path, kernel_uids=uids,
                                                 cell_map=cell_map)
    except SystemExit as e:
        rep.check(f"{path.name} parses and joins to the index", False, str(e))
        return

    n_total = len(uids)
    n_scored = len(table)
    rep.check(f"{path.name} parses and joins to the index", True,
              f"{n_scored}/{n_total} kernels scored "
              f"({n_scored / n_total:.0%}); no label names a well the index "
              f"does not have")

    if n_scored == 0:
        rep.check("at least one kernel is scored", False, "the file is empty")
        return

    never = [u for u, d in table.items() if d is None]
    by_day = Counter(d for d in table.values() if d is not None)
    rep.check("germination outcome has both classes among scored kernels",
              0 < len(never) < n_scored,
              f"{n_scored - len(never)} germinated "
              f"({1 - len(never) / n_scored:.1%}), {len(never)} never")
    rep.note(f"first germination by assay day: {dict(sorted(by_day.items()))} "
             f"(day k = {[int(h) for h in config.germination_hours()]} h)")
    rep.check("more than one germination day is populated", len(by_day) > 1,
              f"{len(by_day)} of {len(config.GERMINATION_DAYS)} days populated"
              if len(by_day) > 1 else
              "a single populated day means the grid carries no timing "
              "information and only the ever/never call is learnable")

    # Coverage, by dish. A partly-scored dish is fine; the point is to say which
    # dishes a run will actually train on.
    per_dish_total = defaultdict(int)
    per_dish_scored = defaultdict(int)
    for u in uids:
        per_dish_total[u.split("_")[0]] += 1
    for u in table:
        per_dish_scored[u.split("_")[0]] += 1
    full = [d for d in per_dish_total if per_dish_scored[d] == per_dish_total[d]]
    none_ = [d for d in per_dish_total if per_dish_scored[d] == 0]
    partial = [d for d in per_dish_total
               if 0 < per_dish_scored[d] < per_dish_total[d]]
    rep.note(f"dishes fully scored {len(full)}, partly {len(partial)}, "
             f"not at all {len(none_)}"
             + (f" ({', '.join(sorted(none_, key=lambda s: int(s[4:])))})"
                if none_ else ""))
    if partial:
        rep.note("  partly scored: "
                 + ", ".join(f"{d} {per_dish_scored[d]}/{per_dish_total[d]}"
                             for d in sorted(partial, key=lambda s: int(s[4:]))))
    rep.check("at least two dishes are scored, so folds can hold one out",
              len(full) + len(partial) >= 2,
              f"{len(full) + len(partial)} dish(es) have any label")

    # Per-dish and per-variety rate. The variety line is the one that matters.
    rate = {d: 1 - sum(table[u] is None for u in table if u.startswith(d + "_"))
            / per_dish_scored[d]
            for d in per_dish_total if per_dish_scored[d]}
    lo = min(rate.items(), key=lambda x: x[1])
    hi = max(rate.items(), key=lambda x: x[1])
    rep.note(f"per-dish germination rate: {lo[1]:.2f} ({lo[0]}) to "
             f"{hi[1]:.2f} ({hi[0]}) over {len(rate)} scored dishes")

    var_n = defaultdict(int)
    var_never = defaultdict(int)
    for u, d in table.items():
        v = config.variety_of(int(u.split("_")[0][4:]))
        var_n[v] += 1
        var_never[v] += d is None
    rep.note("germination by variety (scored kernels only):")
    degenerate = []
    for v in sorted(var_n):
        name = config.VARIETY_NAMES[v - 1]
        frac = var_never[v] / var_n[v]
        rep.note(f"  variety {v} {name:<10s} {var_n[v]:3d} scored, "
                 f"{var_never[v]:3d} never ({frac:.0%})")
        if var_n[v] >= 20 and frac in (0.0, 1.0):
            degenerate.append((v, name, frac))
    for v, name, frac in degenerate:
        rep.note(f"  !! variety {v} ({name}) is {frac:.0%} never-germinating "
                 f"across every scored kernel. Variety is perfectly confounded "
                 f"with dish and folds hold out whole dishes, so a model can "
                 f"score well on the ever/never call by recognising the plate "
                 f"rather than the grain. Read the dish-identity control before "
                 f"believing any germination result.")


def check_splits(rep, rows, ds):
    p = ds / "splits.json"
    if not p.exists():
        rep.note("no splits.json yet -- run barley/splits.py before training")
        return
    folds = json.loads(p.read_text())["folds"]
    dish_of = {r["kernel_uid"]: int(r["dish"]) for r in rows}
    bad = []
    for i, f in enumerate(folds):
        tr, te = set(f["train"]), set(f["test"])
        if tr & te:
            bad.append(f"fold {i}: {len(tr & te)} shared kernels")
        if {dish_of[k] for k in tr} & {dish_of[k] for k in te}:
            bad.append(f"fold {i}: dish appears in both train and test")
    rep.check("no kernel or dish spans train and test", not bad, "; ".join(bad))


def contact_sheet(ds, rows, patches, masks, out, n=20):
    """Twenty patches with their mask outline. A mirrored, transposed or
    misindexed patch is obvious here and invisible in a loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(config.SEED)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    fig, axes = plt.subplots(4, 5, figsize=(14, 15))
    for ax, i in zip(axes.ravel(), pick):
        p = np.asarray(patches[i], np.float32)
        with np.errstate(invalid="ignore"):
            img = np.nanmean(np.where(np.isfinite(p).any(2, keepdims=True), p, 0.0), axis=2)
        lo, hi = np.nanpercentile(img, [1, 99])
        ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1), cmap="gray")
        ax.contour(masks[i].astype(float), levels=[0.5], colors="#ff3b30", linewidths=1.2)
        r = rows[i]
        ax.set_title(f"{r['kernel_uid']}\n{r['mode'][:5]} {r['side'][:3]} {r['hours']}h",
                     fontsize=8)
        ax.axis("off")
    fig.suptitle("kernel patches (band-mean pseudo-absorbance) with mask outline")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\ncontact sheet -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=lambda s: config.ROOT / s, default=config.DATASET)
    p.add_argument("--contact-sheet", action="store_true")
    args = p.parse_args()
    ds = args.dataset

    if not (ds / "index.csv").exists():
        raise SystemExit(f"no dataset at {ds} -- run build_dataset.py first")

    rows = load_index(ds / "index.csv")
    patches = np.load(ds / "patches.npy", mmap_mode="r")
    masks = np.load(ds / "masks.npy", mmap_mode="r")
    spectra = np.load(ds / "spectra.npy")

    print(f"verifying {ds}  ({len(rows)} rows)\n" + "=" * 70)
    rep = Report()
    meta = check_provenance(rep, ds, rows)
    rep.check("array lengths agree",
              len(patches) == len(masks) == len(spectra) == len(rows),
              f"patches {len(patches)} masks {len(masks)} "
              f"spectra {len(spectra)} index {len(rows)}")
    check_rle_decoder(rep)
    check_assignment(rep, rows)
    check_frame(rep, rows)
    check_warp(rep, rows, masks)
    check_spectra(rep, spectra, patches, masks)
    check_index(rep, rows, meta)
    check_germination(rep, rows)
    check_splits(rep, rows, ds)

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "verify.txt").write_text("\n".join(rep.lines) + "\n")
    if args.contact_sheet:
        contact_sheet(ds, rows, patches, masks,
                      config.REPORTS / "patch_contact_sheet.png")

    print("=" * 70)
    if rep.failed:
        print(f"{rep.failed} CHECK(S) FAILED -- do not train on this dataset")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
