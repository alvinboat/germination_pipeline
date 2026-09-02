"""Reading the germination labels, and the folds the germination models use.

THE THREE STATES, WHICH ARE NOT TWO
The annotator's sheet distinguishes them and so must everything downstream:

    1..5   germinated by that scoring visit
    -1     scored, never germinated -- a real observation, KEEP IT
    blank  nobody has scored that dish yet -- a missing label, DROP IT

Reading blank as -1 would train the model to predict dormancy from unscored
plates. Reading -1 as blank would throw away the kernels the whole question is
about. Internally `None` is never-germinated and the `UNSCORED` sentinel is
not-scored; only the caller can tell them apart, and it must.

WHAT "DAY k" ACTUALLY MEANS
Not a germination time. The kernel was seen ungerminated at visit k-1 and
germinated at visit k, so all that is known is an INTERVAL -- and the visits
are not 24 h apart and not the same for every dish. `barley.timing` has the
measured brackets; this module only reads which visit first saw it up.

SCOPE
Binary germination: did this kernel come up by the last scoring visit? The
time-to-germination regressor that used to live beside this was dropped -- see
the note in CLAUDE.md.
"""
import csv

import numpy as np

import config

# Cells that mean "this kernel never germinated" rather than "somebody forgot".
NEVER_TOKENS = ("-1", "never", "none", "n")
UNSCORED_TOKENS = ("", "-", "na", "n/a", "null", "?", "tbd")

_TABLE = None


def read_germination_labels(path=None, kernel_uids=None, cell_map=None):
    """-> {kernel_uid: first_germinated_day or None}, SCORED kernels only.

    None means "scored, never germinated". A kernel that has not been scored is
    absent from the mapping entirely -- callers must not read absence as never.

    Accepts the annotator's .xlsx (a dish x kernel-index grid) or a .csv with
    kernel_uid,first_germinated_day columns. `cell_map` is {(dish, cell_index):
    kernel_uid}, required for the .xlsx form because the sheet is indexed
    positionally.

    Strict on purpose: every failure here is one that otherwise produces a clean
    loss curve and a wrong answer.
    """
    path = path or config.GERMINATION_LABELS
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist.\n"
            "Expected either the annotator spreadsheet (dish rows x kernel-index "
            "columns 0..21) or a CSV with columns:\n"
            "  kernel_uid            e.g. dish7_R2C6 -- gridfit's own well name\n"
            f"  first_germinated_day  {list(config.GERMINATION_DAYS)}, or "
            f"{config.GERMINATION_NEVER} for never; blank = not scored.")

    if path.suffix.lower() in (".xlsx", ".xlsm"):
        table = _read_xlsx(path, cell_map)
    else:
        table = _read_csv(path)

    bad = sorted(set(table) & set(config.KNOWN_EMPTY_WELLS))
    if bad:
        raise SystemExit(
            f"{path}: {bad} are permanently empty wells and must not be scored.\n"
            "A value there means a blind 22-well template was applied, so every "
            "label in the file is suspect.")

    if kernel_uids is not None:
        orphan = set(table) - set(kernel_uids)
        if orphan:
            raise SystemExit(f"{len(orphan)} label(s) match no kernel in the "
                             f"index, e.g. {sorted(orphan)[:5]}")
    return table

def _read_xlsx(path, cell_map):
    from . import germination_sheet

    if cell_map is None:
        cell_map = build_cell_map()
    grid, sheet = germination_sheet.read_grid(path)

    table, off_plate = {}, []
    for (dish, kidx), v in grid.items():
        uid = cell_map.get((dish, kidx))
        if uid is None:
            # A value in a well the index says holds no kernel. The two known
            # empty wells are the expected case and are reported by name.
            off_plate.append((dish, kidx))
            continue
        table[uid] = _coerce_day(v, uid, path)
    if off_plate:
        named = [f"dish{d} kernel {k}" for d, k in sorted(off_plate)[:5]]
        raise SystemExit(
            f"{path}: {len(off_plate)} scored cell(s) name a well with no kernel "
            f"in index.csv, e.g. {named}. Either the sheet is shifted by a column "
            f"or it was filled from a blind template.")
    return table

def _read_csv(path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")
    if "kernel_uid" not in rows[0]:
        raise SystemExit(f"{path} has no kernel_uid column; got {list(rows[0])}")
    if "first_germinated_day" not in rows[0]:
        raise SystemExit(
            f"{path} has no first_germinated_day column; got {list(rows[0])}.\n"
            "A bare germinated 0/1 column is the old schema and throws away the "
            "timing the scoring photos were taken to capture.")

    table, dupes = {}, []
    for r in rows:
        uid = (r["kernel_uid"] or "").strip()
        if not uid:
            raise SystemExit(f"{path}: a row has an empty kernel_uid")
        raw = (r["first_germinated_day"] or "").strip()
        if raw.lower() in UNSCORED_TOKENS:
            continue                       # not scored; absent, not never
        # The obvious dict comprehension silently keeps the LAST duplicate, so a
        # double-scored kernel would quietly overwrite its own first call.
        if uid in table:
            dupes.append(uid)
        table[uid] = _coerce_day(raw, uid, path)
    if dupes:
        raise SystemExit(f"{path}: {len(dupes)} duplicate kernel_uid(s), "
                         f"e.g. {sorted(set(dupes))[:5]}")
    return table

def _coerce_day(raw, uid, path):
    """-> int day, or None for 'scored, never germinated'."""
    s = str(raw).strip()
    if s.lower() in NEVER_TOKENS:
        return None
    try:
        d = int(float(s))
    except ValueError:
        raise SystemExit(f"{path}: {uid} has first_germinated_day={raw!r}, which "
                         f"is neither a day number nor "
                         f"{config.GERMINATION_NEVER}")
    if d == config.GERMINATION_NEVER:
        return None
    if d not in config.GERMINATION_DAYS:
        raise SystemExit(
            f"{path}: {uid} has first_germinated_day={d}; valid values are "
            f"{list(config.GERMINATION_DAYS)} (day k = {24}k hours) or "
            f"{config.GERMINATION_NEVER} for never germinated.")
    return d

def build_cell_map(idx=None):
    """{(dish, cell_index): kernel_uid} from the dataset index.

    The spreadsheet is indexed positionally -- dish row, kernel column 0..21 --
    which is gridfit's `cell_index`, the same numbering stamped on
    labels/germination_photos/index_reference_*.png. Wells with no kernel simply
    have no entry, which is how a value in one gets caught.
    """
    from . import index as index_mod
    # include_excluded: the sheet scores the whole plate, so a well left out of
    # the modelling path still has to resolve to a name here -- otherwise its
    # perfectly good label reads as "a value in a well with no kernel" and the
    # parser rejects the entire file.
    idx = idx if idx is not None else index_mod.Index.load(include_excluded=True,
                                                           quiet=True)
    out = {}
    for d, c, u in zip(idx["dish"], idx["cell_index"], idx["kernel_uid"]):
        out[(int(d), int(c))] = u
    return out

def first_day_array(idx, table):
    """(N,) first day per view. UNSCORED is a distinct sentinel, not None.

    None means "scored, never germinated" and produces an all-zero staircase
    row. UNSCORED produces an all -1 row, which `usable` filters out.
    """
    return [table.get(u, UNSCORED) for u in idx["kernel_uid"]]


class _Unscored:
    __slots__ = ()

    def __repr__(self):
        return "UNSCORED"


UNSCORED = _Unscored()


def _table(path=None):
    global _TABLE
    if _TABLE is None:
        _TABLE = read_germination_labels(path)
    return _TABLE


# --------------------------------------------------------------- the folds --
def germinated(idx, table=None):
    """(N,) 0/1 per row: did this kernel ever come up? -1 where unscored."""
    table = _table() if table is None else table
    return np.array([-1 if u not in table else int(table[u] is not None)
                     for u in idx["kernel_uid"]], np.int64)


def stratified_dish_folds(dishes, varieties, rates, n_folds, seed=config.SEED):
    """Assign whole dishes to folds, balancing variety AND germination rate.

    -> {dish: fold}

    Two things have to be spread and only one of them used to be. Germination
    rate matters because 48 never-germinators over 20 dishes is few enough that
    an unlucky split leaves one fold with three and another with fourteen.
    Variety matters just as much: it is perfectly confounded with dish, the four
    cultivars differ in timing, and a fold that draws three prospect1 dishes and
    no laureate1 is testing on a plate the training set is simultaneously short
    of. Stratifying on rate alone did exactly that -- one fold came out 75% one
    variety with two varieties absent.

    So it is a Latin square. Within each variety the dishes are ranked by their
    own germination rate, and rank r goes to fold `(r + offset) % n_folds` with
    a different random offset per variety. Every fold then gets one dish of
    every variety, and across varieties a spread of rate ranks rather than all
    the low-rate plates landing together.

    Degrades cleanly: a variety with fewer dishes than folds simply misses some,
    one with more puts two in a fold.
    """
    rng = np.random.default_rng(seed)
    dishes = np.asarray(dishes)
    varieties = np.asarray(varieties)
    rates = np.asarray(rates, float)

    uniq_v = np.unique(varieties)
    offsets = rng.permutation(len(uniq_v))
    fold_of = {}
    for off, v in zip(offsets, uniq_v):
        m = varieties == v
        d_v, r_v = dishes[m], rates[m]
        order = np.lexsort((rng.random(len(d_v)), r_v))     # ties broken at random
        for rank, i in enumerate(order):
            fold_of[d_v[i].item()] = int((rank + off) % n_folds)
    return fold_of


def build_folds(idx, table=None, n_folds=config.N_FOLDS, seed=config.SEED):
    """Dish-grouped folds stratified on each dish's GERMINATION RATE.

    Not on variety, which is what dataset/splits.json encodes. Fold-to-fold
    variance here is driven almost entirely by how many never-germinators a
    held-out fold happens to draw -- 48 of them across 20 dishes, so a careless
    split leaves one fold with three and another with fourteen. Variety
    stratification does not control that at all.

    This uses outcome information at the DISH level, which is what stratified
    CV always does. No per-kernel information crosses the boundary: the stratum
    is one number per dish and whole dishes are held out.

    Balanced on variety as well -- see `stratified_dish_folds` for why and how.
    """
    y = germinated(idx, table)
    keep = y >= 0
    idx, y = idx.take(keep), y[keep]
    groups = idx["dish"]

    uniq = np.array(sorted(set(groups.tolist())))
    per_dish = np.array([y[groups == g].mean() for g in uniq])
    var_of = {int(d): int(v) for d, v in zip(idx["dish"], idx["variety"])}
    fold_of = stratified_dish_folds(uniq, [var_of[int(g)] for g in uniq],
                                    per_dish, n_folds, seed)

    assigned = np.array([fold_of[g] for g in groups.tolist()])
    kernels = idx["kernel_uid"]
    folds = [{"train": sorted(set(kernels[assigned != k].tolist())),
              "test": sorted(set(kernels[assigned == k].tolist()))}
             for k in range(n_folds)]
    return {
        "task": "germinated", "group_col": "dish", "n_folds": n_folds,
        "seed": seed, "stratified_on": "variety x per-dish germination rate",
        "dish_rate": {str(g): float(r) for g, r in zip(uniq.tolist(), per_dish)},
        "group_fold": {str(g): int(f) for g, f in sorted(fold_of.items(),
                                                        key=lambda x: str(x[0]))},
        "folds": folds,
    }


def splits_path():
    return config.DATASET / "splits_germination.json"


def main():
    import argparse
    import json
    from . import index as index_mod

    p = argparse.ArgumentParser(
        description="write dataset/splits_germination.json (dish-grouped, "
                    "stratified on per-dish germination rate)")
    p.add_argument("--folds", type=int, default=config.N_FOLDS)
    p.add_argument("--out", type=lambda s: config.ROOT / s, default=None)
    args = p.parse_args()

    idx = index_mod.Index.load()
    table = _table()
    spec = build_folds(idx, table, args.folds)
    out = args.out or splits_path()
    out.write_text(json.dumps(spec, indent=1))

    print(f"germinated: {args.folds} folds grouped by dish, stratified on "
          f"{spec['stratified_on']}")
    print(f"labels: {config.GERMINATION_LABELS}")
    for i, f in enumerate(spec["folds"]):
        dishes = sorted({k.split("_")[0] for k in f["test"]})
        ev = sum(table[u] is not None for u in f["test"] if u in table)
        print(f"  fold {i}: {len(f['train']):3d} train / {len(f['test']):3d} test "
              f"kernels | {ev:3d} events ({ev / len(f['test']):.0%}) | "
              f"held out {', '.join(dishes)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
